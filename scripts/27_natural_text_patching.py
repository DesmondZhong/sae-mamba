#!/usr/bin/env python3
"""Natural-text patching sweep: does L30 x_proj.C localization hold on real Pile text?

Construction:
  natural clean = Pile document with a naturally repeated bigram
                  (first occurrence at position i-1..i, second at position j-1..j,
                   with j - i >= 32)
  natural corrupted = SAME text, but tokens at positions j..j+PATTERN_LEN replaced
                      with random ones (or with some other non-matching tokens)

For each such pair, run the x_proj patching (full + slice) at L30 and compare
induction-feature activation at the second-occurrence positions.

Output: $STORAGE/results_phase4/natural_text_patching.json
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.activation_cache import get_model_and_tokenizer
from src.mamba_internals import force_slow_forward
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
CKPT_DIR = STORAGE / "checkpoints_normed"
ACTS_DIR = STORAGE / "activations"
RESULTS_DIR = STORAGE / "results_phase4"

MODEL_NAME = "state-spaces/mamba-2.8b-hf"
MODEL_KEY = "mamba1_2.8b"
D_MODEL = 2560
MID_LAYER = 32
LOCUS_LAYER = 30
SAE_EXPANSION = 16
SAE_K = 64
SEQ_LEN = 256
PATTERN_LEN = 8   # number of tokens we treat as the second "pattern"
MIN_GAP = 32


def load_sae_and_norm(device):
    d_hidden = D_MODEL * SAE_EXPANSION
    run_key = f"{MODEL_KEY}_L{MID_LAYER}_x{SAE_EXPANSION}_k{SAE_K}_normed"
    sae = create_sae(D_MODEL, d_hidden, sae_type="topk", k=SAE_K).to(device)
    sae.load_state_dict(torch.load(CKPT_DIR / f"{run_key}.pt",
                                   map_location=device, weights_only=True))
    sae.eval()
    acts_path = ACTS_DIR / MODEL_KEY / f"layer_{MID_LAYER}.pt"
    t = torch.load(acts_path, map_location="cpu", weights_only=True, mmap=True)
    sample = t[:10000].clone().float()
    act_mean = sample.mean(dim=0).to(device)
    act_std = sample.std(dim=0).clamp(min=1e-6).to(device)
    del t, sample
    return sae, act_mean, act_std


def find_first_qualifying_repeat(ids, min_gap=MIN_GAP, min_first=8, max_k=None):
    """Find earliest (i, j) where bigram (t[j-1], t[j]) == (t[i-1], t[i]),
    j - i >= min_gap, i >= min_first, and j + PATTERN_LEN - 1 < len(ids)."""
    L = len(ids)
    if max_k is None:
        max_k = L - PATTERN_LEN
    seen = {}
    for k in range(1, max_k):
        bg = (ids[k - 1].item(), ids[k].item())
        if k >= min_first and bg in seen:
            for i in seen[bg]:
                if k - i >= min_gap:
                    return (i, k)
        seen.setdefault(bg, []).append(k)
    return None


def build_pairs(tokenizer, n_pairs, device, vocab_limit=50277):
    """Stream Pile; build (clean, corrupted) pairs where corrupted replaces
    PATTERN_LEN tokens starting at the second-occurrence position j with
    random tokens (not matching the first pattern)."""
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    it = iter(ds)
    clean_list, corr_list, j_list = [], [], []
    rng = np.random.default_rng(0)
    pbar = tqdm(desc="building pairs", total=n_pairs)
    n_docs = 0
    while len(clean_list) < n_pairs:
        try:
            ex = next(it)
        except StopIteration:
            break
        text = ex.get("text", "")
        if not text or len(text) < 600:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=SEQ_LEN)["input_ids"][0]
        if ids.shape[0] < SEQ_LEN - 1:
            continue
        pair = find_first_qualifying_repeat(ids, min_gap=MIN_GAP)
        if pair is None:
            continue
        i, j = pair
        clean = ids.clone()
        corr = ids.clone()
        # Replace tokens at positions j .. j+PATTERN_LEN-1 with random non-matching tokens
        original = ids[j:j + PATTERN_LEN].clone()
        while True:
            replacement = torch.from_numpy(
                rng.integers(0, vocab_limit, PATTERN_LEN).astype(np.int64))
            if not torch.equal(replacement, original):
                break
        corr[j:j + PATTERN_LEN] = replacement
        clean_list.append(clean)
        corr_list.append(corr)
        j_list.append(j)
        n_docs += 1
        pbar.update(1)
    pbar.close()
    if not clean_list:
        raise RuntimeError("no pairs found")
    clean = torch.stack(clean_list).to(device)
    corr = torch.stack(corr_list).to(device)
    return clean, corr, torch.tensor(j_list, device=device)


def capture_xproj_output(model, tokens, layer):
    captured = {}
    def hook(mod, ins, out):
        captured["o"] = out.detach().clone()
    h = model.backbone.layers[layer].mixer.x_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["o"]


def encode_at_positions_with_slice_patch(model, tokens, sae, act_mean, act_std,
                                           mid_layer, j_list, patch_value=None,
                                           slice_range=None):
    captured = {}
    def res_hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()
    hooks = [model.backbone.layers[mid_layer].register_forward_hook(res_hook)]
    if patch_value is not None:
        s, e = slice_range
        def patch_fn(mod, ins, out):
            new_out = out.clone()
            repl_slice = patch_value[..., s:e].to(new_out.dtype)
            if repl_slice.shape[0] != new_out.shape[0]:
                if repl_slice.shape[0] == 1:
                    repl_slice = repl_slice.expand(new_out.shape[0], -1, -1)
                else:
                    repl_slice = repl_slice[:new_out.shape[0]]
            new_out[..., s:e] = repl_slice
            return new_out
        hooks.append(model.backbone.layers[LOCUS_LAYER].mixer.x_proj.register_forward_hook(patch_fn))

    with torch.no_grad():
        model(tokens)
    for h in hooks:
        h.remove()
    res = captured["r"]
    normed = (res.float() - act_mean) / act_std
    _, z, *_ = sae(normed)
    # For each sample i, average over its PATTERN_LEN positions starting at j_list[i]
    B = z.shape[0]
    n_feat = z.shape[-1]
    out_per_sample = torch.zeros(B, n_feat, device=z.device, dtype=z.dtype)
    for i in range(B):
        j = j_list[i].item()
        out_per_sample[i] = z[i, j:j + PATTERN_LEN].mean(dim=0)
    return out_per_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    force_slow_forward(model)
    sae, act_mean, act_std = load_sae_and_norm(device)

    ind = json.load(open(RESULTS_DIR / "induction_features.json"))
    top_feats = torch.tensor(ind["feature"][:10], device=device)

    print(f"\nBuilding {args.n_pairs} natural-text pairs...")
    clean, corr, j_list = build_pairs(tokenizer, args.n_pairs, device)
    print(f"  pairs: clean shape={tuple(clean.shape)}, corrupted shape={tuple(corr.shape)}")

    # Baseline and corrupted
    z_clean = encode_at_positions_with_slice_patch(
        model, clean, sae, act_mean, act_std, MID_LAYER, j_list)
    z_corr = encode_at_positions_with_slice_patch(
        model, corr, sae, act_mean, act_std, MID_LAYER, j_list)
    baseline_act = z_clean[:, top_feats].mean().item()
    corrupted_act = z_corr[:, top_feats].mean().item()
    gap = baseline_act - corrupted_act
    print(f"baseline={baseline_act:.4f}, corrupted={corrupted_act:.4f}, gap={gap:.4f}")

    # Capture corrupted x_proj output at L30
    # Note: different samples have different induction positions, but we capture
    # the FULL x_proj output for each; patching will replace the C slice at ALL positions.
    xp_corr = capture_xproj_output(model, corr, LOCUS_LAYER)

    mixer = model.backbone.layers[LOCUS_LAYER].mixer
    dt_rank = mixer.time_step_rank
    state_size = mixer.ssm_state_size
    slices = {
        "full_xproj": (0, dt_rank + 2 * state_size),
        "delta_pre":  (0, dt_rank),
        "B_matrix":   (dt_rank, dt_rank + state_size),
        "C_matrix":   (dt_rank + state_size, dt_rank + 2 * state_size),
    }

    results = {}
    for name, sl in slices.items():
        z_patched = encode_at_positions_with_slice_patch(
            model, clean, sae, act_mean, act_std, MID_LAYER, j_list,
            patch_value=xp_corr, slice_range=sl,
        )
        patched_act = z_patched[:, top_feats].mean().item()
        damage = 1.0 - (patched_act - corrupted_act) / gap if abs(gap) > 1e-8 else 0.0
        results[name] = {
            "slice": list(sl),
            "patched_act": patched_act,
            "patch_damage": damage,
        }
        print(f"  {name:<12s} slice[{sl[0]}:{sl[1]}] patched={patched_act:.4f}  "
              f"damage={damage:+.4f}")

    out = {
        "n_pairs": args.n_pairs,
        "baseline_act": baseline_act,
        "corrupted_act": corrupted_act,
        "gap": gap,
        "results": results,
    }
    out_path = RESULTS_DIR / "natural_text_patching.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
