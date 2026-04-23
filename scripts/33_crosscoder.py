#!/usr/bin/env python3
"""Crosscoder between Mamba-1 L32 and Pythia-2.8B L16 residual streams.

Extract both models' residuals on the SAME 5M token sequence (tokenized with
Pythia's GPT-NeoX tokenizer, filtered to tokens valid in both models' vocabs),
then train a shared-feature autoencoder that reconstructs both. If shared
features exist across architecture families, this is a direct test of the
"universal features across architectures" hypothesis (Lindsey et al. 2024 —
crosscoders — but simplified for a quick proof-of-concept).

Architecture (simplified crosscoder):
  Encoder:       Linear(d_mamba + d_pythia = 5120, d_hidden=40960) + TopK(k=64)
  Decoder_m:     Linear(d_hidden, d_mamba)    — reconstructs Mamba residual
  Decoder_p:     Linear(d_hidden, d_pythia)   — reconstructs Pythia residual
  Loss:          MSE(mamba_recon) + MSE(pythia_recon) + sparsity(TopK handles it)

A "shared feature" fires on both models' activations at the same token position
and has meaningful decoder directions in both models' output spaces.

Output:
  $STORAGE/activations/crosscoder/mamba_L32.fp16.npy  (matched-token)
  $STORAGE/activations/crosscoder/pythia_L16.fp16.npy (matched-token)
  $STORAGE/checkpoints_normed/crosscoder_mambaL32_pythiaL16.pt
  $STORAGE/results_phase4/crosscoder.json
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
CKPT_DIR = STORAGE / "checkpoints_normed"
ACTS_DIR = STORAGE / "activations"
CC_DIR = ACTS_DIR / "crosscoder"
CC_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = STORAGE / "results_phase4"

MAMBA_NAME = "state-spaces/mamba-2.8b-hf"
PYTHIA_NAME = "EleutherAI/pythia-2.8b"
MAMBA_LAYER = 32
PYTHIA_LAYER = 16
D = 2560  # both d_model
EXPANSION = 16
K = 64
SEQ_LEN = 256
N_TOKENS = 5_000_000


def extract_matched_activations(args, device):
    """Tokenize Pile with Pythia tokenizer (superset vocab). Forward both models;
    capture residuals at specified layers; stream-write matched pairs to disk."""
    in_path = CC_DIR / "mamba_L32.fp16.npy"
    py_path = CC_DIR / "pythia_L16.fp16.npy"
    meta_path = CC_DIR / "meta.json"
    if in_path.exists() and py_path.exists() and meta_path.exists():
        print(f"[skip] matched activations exist at {CC_DIR}")
        return in_path, py_path, meta_path

    from transformers import AutoTokenizer, MambaForCausalLM, AutoModelForCausalLM
    print(f"Loading tokenizer (Pythia's GPT-NeoX)...")
    tok = AutoTokenizer.from_pretrained(PYTHIA_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading Mamba-1 {MAMBA_NAME}...")
    mamba = MambaForCausalLM.from_pretrained(MAMBA_NAME, torch_dtype=torch.float16).to(device).eval()
    mamba_vocab = mamba.config.vocab_size
    print(f"  Mamba vocab: {mamba_vocab}")

    print(f"Loading Pythia {PYTHIA_NAME}...")
    pythia = AutoModelForCausalLM.from_pretrained(PYTHIA_NAME, torch_dtype=torch.float16).to(device).eval()
    pythia_vocab = pythia.config.vocab_size
    print(f"  Pythia vocab: {pythia_vocab}")
    safe_vocab = min(mamba_vocab, pythia_vocab)
    print(f"  Using safe vocab: {safe_vocab}")

    # Tokenize streaming until we have enough tokens (only using tokens < safe_vocab)
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    it = iter(ds)
    sequences = []
    total_tokens = 0
    pbar = tqdm(total=N_TOKENS, desc="Tokenizing")
    while total_tokens < N_TOKENS:
        try:
            ex = next(it)
        except StopIteration:
            break
        text = ex.get("text", "")
        if not text or len(text.strip()) < 200:
            continue
        ids = tok(text, return_tensors="pt", truncation=True,
                   max_length=SEQ_LEN)["input_ids"][0]
        if ids.shape[0] < SEQ_LEN:
            continue
        # Filter: only sequences with all tokens < safe_vocab
        if (ids >= safe_vocab).any().item():
            continue
        sequences.append(ids)
        total_tokens += ids.shape[0]
        pbar.update(ids.shape[0])
    pbar.close()
    n_seqs = len(sequences)
    total_tokens = min(total_tokens, N_TOKENS)
    n_total = n_seqs * SEQ_LEN
    print(f"\nCollected {n_seqs} sequences, {n_total:,} tokens")

    sequences = torch.stack(sequences)  # (N, SEQ_LEN)

    # Preallocate fp16 memmaps
    mmap_m = np.memmap(in_path, dtype="float16", mode="w+", shape=(n_total, D))
    mmap_p = np.memmap(py_path, dtype="float16", mode="w+", shape=(n_total, D))

    cursor_m = {"v": 0}
    cursor_p = {"v": 0}
    def m_hook(mod, ins, out):
        r = out[0] if isinstance(out, tuple) else out
        flat = r.reshape(-1, D).to(torch.float16).cpu().numpy()
        n = flat.shape[0]
        mmap_m[cursor_m["v"]:cursor_m["v"] + n] = flat
        cursor_m["v"] += n
    def p_hook(mod, ins, out):
        r = out[0] if isinstance(out, tuple) else out
        flat = r.reshape(-1, D).to(torch.float16).cpu().numpy()
        n = flat.shape[0]
        mmap_p[cursor_p["v"]:cursor_p["v"] + n] = flat
        cursor_p["v"] += n

    # Register hooks on both models' target layers
    m_handle = mamba.backbone.layers[MAMBA_LAYER].register_forward_hook(m_hook)
    p_handle = pythia.gpt_neox.layers[PYTHIA_LAYER].register_forward_hook(p_hook)

    batch_size = 4
    n_batches = (n_seqs + batch_size - 1) // batch_size
    start = time.time()
    for i in tqdm(range(n_batches), desc="Dual-forward"):
        batch = sequences[i * batch_size:(i + 1) * batch_size].to(device)
        with torch.no_grad():
            mamba(batch)
        with torch.no_grad():
            pythia(batch)

    m_handle.remove(); p_handle.remove()
    mmap_m.flush(); mmap_p.flush()

    meta = {"n_total": n_total, "n_seqs": n_seqs, "d_model": D,
            "mamba_layer": MAMBA_LAYER, "pythia_layer": PYTHIA_LAYER,
            "safe_vocab": int(safe_vocab)}
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"Extraction done in {(time.time()-start)/60:.1f}min")

    del mamba, pythia
    import gc; gc.collect(); torch.cuda.empty_cache()
    return in_path, py_path, meta_path


class Crosscoder(nn.Module):
    def __init__(self, d_in_each, d_hidden, k):
        super().__init__()
        self.d_in_each = d_in_each
        self.d_hidden = d_hidden
        self.k = k
        # Shared encoder on concatenated inputs
        self.encoder = nn.Linear(2 * d_in_each, d_hidden, bias=True)
        # Separate decoders
        self.decoder_m = nn.Linear(d_hidden, d_in_each, bias=False)
        self.decoder_p = nn.Linear(d_hidden, d_in_each, bias=False)
        # Initialize
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.decoder_m.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.decoder_p.weight, a=5**0.5)

    def forward(self, x_m, x_p):
        x = torch.cat([x_m, x_p], dim=-1)
        pre = self.encoder(x)
        # TopK on pre-activations (per token)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, torch.relu(topk_vals))
        rec_m = self.decoder_m(z)
        rec_p = self.decoder_p(z)
        return rec_m, rec_p, z


def train_crosscoder(args, device, in_path, py_path, meta_path):
    meta = json.load(open(meta_path))
    n_total = meta["n_total"]
    d = meta["d_model"]

    arr_m = np.memmap(in_path, dtype="float16", mode="r", shape=(n_total, d))
    arr_p = np.memmap(py_path, dtype="float16", mode="r", shape=(n_total, d))

    # Normalize: compute mean/std on sample
    sample_m = torch.from_numpy(arr_m[:500_000].astype(np.float32))
    sample_p = torch.from_numpy(arr_p[:500_000].astype(np.float32))
    mean_m = sample_m.mean(dim=0, keepdim=True).to(device)
    std_m = sample_m.std(dim=0, keepdim=True).clamp(min=1e-6).to(device)
    mean_p = sample_p.mean(dim=0, keepdim=True).to(device)
    std_p = sample_p.std(dim=0, keepdim=True).clamp(min=1e-6).to(device)
    del sample_m, sample_p

    d_hidden = d * EXPANSION
    cc = Crosscoder(d, d_hidden, K).to(device)
    optim = torch.optim.Adam(cc.parameters(), lr=3e-4)

    batch_size = 4096
    n_steps = args.n_steps
    history = {"loss_m": [], "loss_p": [], "fve_m": [], "fve_p": []}
    start = time.time()
    for step in tqdm(range(n_steps), desc="Crosscoder training"):
        idx = torch.randint(0, n_total, (batch_size,))
        x_m = torch.from_numpy(arr_m[idx].astype(np.float32)).to(device)
        x_p = torch.from_numpy(arr_p[idx].astype(np.float32)).to(device)
        x_m = (x_m - mean_m) / std_m
        x_p = (x_p - mean_p) / std_p

        rec_m, rec_p, z = cc(x_m, x_p)
        loss_m = ((rec_m - x_m) ** 2).mean()
        loss_p = ((rec_p - x_p) ** 2).mean()
        loss = loss_m + loss_p
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cc.parameters(), 1.0)
        optim.step()

        if (step + 1) % 500 == 0:
            with torch.no_grad():
                var_m = x_m.var(dim=0).mean()
                var_p = x_p.var(dim=0).mean()
                fve_m = 1 - loss_m / var_m
                fve_p = 1 - loss_p / var_p
                history["loss_m"].append(loss_m.item())
                history["loss_p"].append(loss_p.item())
                history["fve_m"].append(float(fve_m))
                history["fve_p"].append(float(fve_p))
                tqdm.write(f"  step {step+1:>5d}  loss_m={loss_m:.4f}  loss_p={loss_p:.4f}  "
                            f"FVE_m={float(fve_m):.3f}  FVE_p={float(fve_p):.3f}")

    elapsed = time.time() - start
    print(f"Training done in {elapsed/60:.1f}min")

    # Save checkpoint + eval
    run_key = f"crosscoder_mambaL{MAMBA_LAYER}_pythiaL{PYTHIA_LAYER}_x{EXPANSION}_k{K}"
    ckpt_path = CKPT_DIR / f"{run_key}.pt"
    torch.save({
        "state_dict": cc.state_dict(),
        "mean_m": mean_m, "std_m": std_m, "mean_p": mean_p, "std_p": std_p,
        "d_hidden": d_hidden, "k": K,
    }, ckpt_path)

    # Eval on fresh batch + analyze shared features
    cc.eval()
    n_eval = 100_000
    idx = torch.arange(n_eval)
    x_m = torch.from_numpy(arr_m[idx].astype(np.float32)).to(device)
    x_p = torch.from_numpy(arr_p[idx].astype(np.float32)).to(device)
    x_m = (x_m - mean_m) / std_m
    x_p = (x_p - mean_p) / std_p
    with torch.no_grad():
        rec_m, rec_p, z = cc(x_m, x_p)
    var_m = x_m.var(dim=0).mean().item()
    var_p = x_p.var(dim=0).mean().item()
    final_fve_m = 1 - ((rec_m - x_m) ** 2).mean().item() / var_m
    final_fve_p = 1 - ((rec_p - x_p) ** 2).mean().item() / var_p

    # Feature-level analysis: "shared" features have non-trivial decoder magnitude
    # in BOTH models. "Mamba-only" features have large decoder_m and small decoder_p.
    with torch.no_grad():
        dec_m_norm = cc.decoder_m.weight.norm(dim=0)  # (d_hidden,)
        dec_p_norm = cc.decoder_p.weight.norm(dim=0)
    # Normalize per decoder
    dec_m_rel = dec_m_norm / dec_m_norm.max()
    dec_p_rel = dec_p_norm / dec_p_norm.max()
    # A feature is "shared" if its decoder magnitude is > 0.1 (relative to max) in BOTH
    shared_mask = (dec_m_rel > 0.1) & (dec_p_rel > 0.1)
    mamba_only_mask = (dec_m_rel > 0.1) & (dec_p_rel <= 0.1)
    pythia_only_mask = (dec_m_rel <= 0.1) & (dec_p_rel > 0.1)

    print(f"\n=== Crosscoder results ===")
    print(f"  Final FVE (Mamba L32):  {final_fve_m:.4f}")
    print(f"  Final FVE (Pythia L16): {final_fve_p:.4f}")
    print(f"  Shared features:      {int(shared_mask.sum())} / {d_hidden} ({100*float(shared_mask.float().mean()):.1f}%)")
    print(f"  Mamba-only features:  {int(mamba_only_mask.sum())} ({100*float(mamba_only_mask.float().mean()):.1f}%)")
    print(f"  Pythia-only features: {int(pythia_only_mask.sum())} ({100*float(pythia_only_mask.float().mean()):.1f}%)")

    result = {
        "mamba_layer": MAMBA_LAYER,
        "pythia_layer": PYTHIA_LAYER,
        "n_train_tokens": n_total,
        "n_steps": n_steps,
        "d_hidden": d_hidden,
        "k": K,
        "final_fve_mamba": final_fve_m,
        "final_fve_pythia": final_fve_p,
        "n_shared_features": int(shared_mask.sum()),
        "n_mamba_only_features": int(mamba_only_mask.sum()),
        "n_pythia_only_features": int(pythia_only_mask.sum()),
        "frac_shared": float(shared_mask.float().mean()),
        "history": history,
    }
    out_path = RESULTS_DIR / "crosscoder.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"Wrote {out_path}")
    print(f"Checkpoint: {ckpt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_steps", type=int, default=20000)
    ap.add_argument("--extract_only", action="store_true")
    args = ap.parse_args()
    device = args.device

    in_path, py_path, meta_path = extract_matched_activations(args, device)
    if args.extract_only:
        print("Extraction only — done.")
        return
    train_crosscoder(args, device, in_path, py_path, meta_path)


if __name__ == "__main__":
    main()
