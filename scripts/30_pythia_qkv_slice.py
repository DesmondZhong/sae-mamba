#!/usr/bin/env python3
"""Pythia attention Q/K/V slice patching — parallel to my Mamba-1 x_proj slice
experiment.

GPTNeoX attention.query_key_value is a Linear(hidden=2560, 3*num_heads*head_dim=7680).
The reshape is (B, L, num_heads=32, 3*head_dim=240), then sliced into Q / K / V
by the last dim: Q=[0:80], K=[80:160], V=[160:240]. So Q, K, V are PER-HEAD
interleaved in the output — NOT a contiguous block.

This script patches each of (Q, K, V) separately at Pythia's peak induction
layer (L10), measures patch_damage using the Pythia L16 SAE induction features.

Output: $STORAGE/results_phase4/pythia_qkv_slice.json
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.activation_cache import get_model_and_tokenizer
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/path/to/storage"))
CKPT_DIR = STORAGE / "checkpoints_normed"
ACTS_DIR = STORAGE / "activations"
RESULTS_DIR = STORAGE / "results_phase4"

MODEL_NAME = "EleutherAI/pythia-2.8b"
MODEL_KEY = "pythia_2.8b"
D_MODEL = 2560
MID_LAYER = 16
LOCUS_LAYERS = [2, 6, 10, 12]   # Pythia induction sites from Phase C
SAE_EXPANSION = 16
SAE_K = 64
PATTERN_LEN = 8
PREFIX_LEN = 8
MID_LEN = 32


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


def make_batch(tokenizer, n_pairs, seed, device):
    rng = np.random.default_rng(seed)
    vocab = tokenizer.vocab_size
    seq_len = PREFIX_LEN + PATTERN_LEN + MID_LEN + PATTERN_LEN
    clean = np.zeros((n_pairs, seq_len), dtype=np.int64)
    corr = np.zeros_like(clean)
    ind_start = PREFIX_LEN + PATTERN_LEN + MID_LEN
    ind_end = ind_start + PATTERN_LEN
    for i in range(n_pairs):
        prefix = rng.integers(0, vocab, PREFIX_LEN)
        P = rng.integers(0, vocab, PATTERN_LEN)
        mid = rng.integers(0, vocab, MID_LEN)
        while True:
            Pp = rng.integers(0, vocab, PATTERN_LEN)
            if not np.array_equal(Pp, P):
                break
        clean[i, :PREFIX_LEN] = prefix
        clean[i, PREFIX_LEN:PREFIX_LEN + PATTERN_LEN] = P
        clean[i, PREFIX_LEN + PATTERN_LEN:ind_start] = mid
        clean[i, ind_start:ind_end] = P
        corr[i, :ind_start] = clean[i, :ind_start]
        corr[i, ind_start:ind_end] = Pp
    return (torch.from_numpy(clean).to(device),
            torch.from_numpy(corr).to(device),
            ind_start, ind_end)


def capture_qkv(model, tokens, layer):
    captured = {}
    def hook(mod, ins, out):
        captured["qkv"] = out.detach().clone()
    attn = model.gpt_neox.layers[layer].attention.query_key_value
    h = attn.register_forward_hook(hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["qkv"]


def encode_at_positions(model, tokens, sae, act_mean, act_std, mid_layer, positions,
                         qkv_replacement=None, qkv_layer=None, slice_spec=None):
    """Forward; optionally patch query_key_value output at `qkv_layer`.
    slice_spec: {"q": (0, head_dim), "k": (head_dim, 2*head_dim), ...} — the
    range on the LAST dim of the per-head reshaped tensor.
    """
    captured = {}
    def res_hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()

    hooks = [model.gpt_neox.layers[mid_layer].register_forward_hook(res_hook)]
    if qkv_replacement is not None:
        def patch_fn(mod, ins, out):
            # out shape: (B, L, 3 * num_heads * head_dim) = (B, L, 7680)
            # Reshape to (B, L, num_heads, 3 * head_dim) = (B, L, 32, 240)
            new_out = out.clone()
            num_heads = mod.out_features // 240  # dummy; we'll compute below
            head_dim = 80  # pythia-2.8b head_dim
            n_heads = new_out.shape[-1] // (3 * head_dim)
            shaped = new_out.view(new_out.shape[0], new_out.shape[1], n_heads, 3 * head_dim)
            repl = qkv_replacement.view(qkv_replacement.shape[0],
                                          qkv_replacement.shape[1], n_heads, 3 * head_dim)
            # Expand batch dim if needed
            if repl.shape[0] != shaped.shape[0]:
                if repl.shape[0] == 1:
                    repl = repl.expand(shaped.shape[0], -1, -1, -1)
                else:
                    repl = repl[:shaped.shape[0]]
            s, e = slice_spec
            shaped[..., s:e] = repl[..., s:e].to(shaped.dtype)
            new_out = shaped.reshape(out.shape)
            return new_out
        attn = model.gpt_neox.layers[qkv_layer].attention.query_key_value
        hooks.append(attn.register_forward_hook(patch_fn))

    with torch.no_grad():
        model(tokens)
    for h in hooks:
        h.remove()
    res = captured["r"]
    normed = (res.float() - act_mean) / act_std
    _, z, *_ = sae(normed)
    return z[:, positions[0]:positions[1]].mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_pairs", type=int, default=128)
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    sae, act_mean, act_std = load_sae_and_norm(device)

    # Pythia-2.8b: num_heads=32, head_dim=80
    head_dim = 80
    slices = {
        "Q":        (0, head_dim),
        "K":        (head_dim, 2 * head_dim),
        "V":        (2 * head_dim, 3 * head_dim),
        "QK":       (0, 2 * head_dim),
        "full":     (0, 3 * head_dim),
    }

    clean, corr, ind_start, ind_end = make_batch(tokenizer, args.n_pairs, 0, device)

    # Identify Pythia induction features
    z_clean = encode_at_positions(model, clean, sae, act_mean, act_std,
                                    MID_LAYER, (ind_start, ind_end))
    z_corr = encode_at_positions(model, corr, sae, act_mean, act_std,
                                   MID_LAYER, (ind_start, ind_end))
    score = (z_clean - z_corr).mean(dim=0).detach()
    top_vals, top_idx = score.topk(10)
    ti = top_idx
    baseline_act = z_clean[:, ti].mean().item()
    corrupted_act = z_corr[:, ti].mean().item()
    gap = baseline_act - corrupted_act
    print(f"\nbaseline={baseline_act:.3f}, corrupted={corrupted_act:.3f}, gap={gap:.3f}")
    print(f"Top-10 Pythia induction features: {top_idx.cpu().tolist()}")

    all_results = {}
    for L in LOCUS_LAYERS:
        # Capture corrupted QKV at L
        qkv_corr = capture_qkv(model, corr, L)

        layer_results = {}
        for slice_name, (s, e) in slices.items():
            z_patched = encode_at_positions(
                model, clean, sae, act_mean, act_std,
                MID_LAYER, (ind_start, ind_end),
                qkv_replacement=qkv_corr, qkv_layer=L,
                slice_spec=(s, e),
            )
            patched_act = z_patched[:, ti].mean().item()
            damage = 1.0 - (patched_act - corrupted_act) / gap if abs(gap) > 1e-8 else 0.0
            layer_results[slice_name] = damage
            print(f"  L{L:>2} {slice_name:<6s} [{s}:{e}]  patch_damage={damage:+.4f}")
        all_results[f"L{L}"] = layer_results

    # Find top QKV slice across all layers
    print("\n=== Top (layer, slice) sites ===")
    all_sites = []
    for lk, slices_data in all_results.items():
        for s, d in slices_data.items():
            all_sites.append((lk, s, d))
    all_sites.sort(key=lambda x: -x[2])
    for lk, s, d in all_sites[:10]:
        print(f"  {lk} {s:<6s}  damage={d:+.4f}")

    out = {
        "model": MODEL_NAME,
        "mid_layer": MID_LAYER,
        "locus_layers": LOCUS_LAYERS,
        "head_dim": head_dim,
        "baseline_act": baseline_act,
        "corrupted_act": corrupted_act,
        "gap": gap,
        "induction_features": top_idx.cpu().tolist(),
        "per_layer_slice_damage": all_results,
    }
    out_path = RESULTS_DIR / "pythia_qkv_slice.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
