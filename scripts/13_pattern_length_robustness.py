#!/usr/bin/env python3
"""Pattern-length robustness: re-run L30 x_proj patching at pattern_len=4, 8, 16.

For each pattern length:
  1. Generate 128 induction pairs with that pattern length.
  2. Identify top-10 induction features from clean-vs-corrupted contrast.
  3. Patch L30 x_proj output from corrupted into clean; measure patch_damage
     against those features.

If the L30 x_proj locus is robust, patch_damage stays high across lengths and
the top-10 feature sets are similar.

Output: $STORAGE/results_phase4/pattern_length_robustness.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.activation_cache import get_model_and_tokenizer
from src.mamba_internals import (
    MambaInternalCapture, MambaInternalPatcher, force_slow_forward,
)
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/path/to/storage"))
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


def make_batch(tokenizer, n_pairs, pattern_len, seed, device):
    rng = np.random.default_rng(seed)
    vocab = tokenizer.vocab_size
    seq_len = PREFIX_LEN + pattern_len + MID_LEN + pattern_len
    clean = np.zeros((n_pairs, seq_len), dtype=np.int64)
    corr = np.zeros_like(clean)
    ind_start = PREFIX_LEN + pattern_len + MID_LEN
    ind_end = ind_start + pattern_len
    for i in range(n_pairs):
        prefix = rng.integers(0, vocab, PREFIX_LEN)
        P = rng.integers(0, vocab, pattern_len)
        mid = rng.integers(0, vocab, MID_LEN)
        while True:
            Pp = rng.integers(0, vocab, pattern_len)
            if not np.array_equal(Pp, P):
                break
        clean[i, :PREFIX_LEN] = prefix
        clean[i, PREFIX_LEN:PREFIX_LEN + pattern_len] = P
        clean[i, PREFIX_LEN + pattern_len:ind_start] = mid
        clean[i, ind_start:ind_end] = P
        corr[i, :ind_start] = clean[i, :ind_start]
        corr[i, ind_start:ind_end] = Pp
    return (torch.from_numpy(clean).to(device),
            torch.from_numpy(corr).to(device),
            ind_start, ind_end)


def encode_at_positions(model, tokens, sae, act_mean, act_std, layer, positions,
                         patches=None):
    captured = {}
    def hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()
    h = model.backbone.layers[layer].register_forward_hook(hook)
    if patches is not None:
        with MambaInternalPatcher(model, patches=patches):
            with torch.no_grad():
                model(tokens)
    else:
        with torch.no_grad():
            model(tokens)
    h.remove()
    res = captured["r"]
    normed = (res.float() - act_mean) / act_std
    _, z, *_ = sae(normed)
    return z[:, positions[0]:positions[1]].mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_pairs", type=int, default=128)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4, 8, 16])
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    force_slow_forward(model)
    print("Loading L32 SAE...")
    sae, act_mean, act_std = load_sae_and_norm(device)

    out = {"lengths": args.lengths, "results": {}}
    for plen in args.lengths:
        print(f"\n=== PATTERN_LEN = {plen} ===")
        clean, corr, ind_start, ind_end = make_batch(
            tokenizer, args.n_pairs, plen, seed=0, device=device)
        print(f"  seq_len={clean.shape[1]}, ind positions {ind_start}..{ind_end}")

        # Identify top-10 induction features at this pattern length
        z_clean = encode_at_positions(model, clean, sae, act_mean, act_std,
                                       MID_LAYER, (ind_start, ind_end))
        z_corr = encode_at_positions(model, corr, sae, act_mean, act_std,
                                      MID_LAYER, (ind_start, ind_end))
        score = (z_clean - z_corr).mean(dim=0).detach()
        top_vals, top_idx = score.topk(10)
        top_feats = top_idx.cpu().tolist()
        print(f"  top-10 features: {top_feats}")

        # Measure baseline / corrupted target-feature activation
        ti = top_idx
        baseline_act = z_clean[:, ti].mean().item()
        corrupted_act = z_corr[:, ti].mean().item()
        total = baseline_act - corrupted_act
        print(f"  baseline={baseline_act:.3f}, corrupted={corrupted_act:.3f}")

        # Patch L30 x_proj
        with MambaInternalCapture(model, sites=[(LOCUS_LAYER, "x_proj")]) as cap:
            with torch.no_grad():
                model(corr)
        corr_xproj = cap.captured[(LOCUS_LAYER, "x_proj")]
        z_patched = encode_at_positions(
            model, clean, sae, act_mean, act_std,
            MID_LAYER, (ind_start, ind_end),
            patches={(LOCUS_LAYER, "x_proj"): corr_xproj},
        )
        patched_act = z_patched[:, ti].mean().item()
        patch_damage = 1.0 - (patched_act - corrupted_act) / total if abs(total) > 1e-8 else 0.0
        print(f"  L30 x_proj patch_damage = {patch_damage:+.4f}")

        out["results"][str(plen)] = {
            "pattern_len": plen,
            "top_features": top_feats,
            "top_scores": top_vals.cpu().tolist(),
            "baseline_act": baseline_act,
            "corrupted_act": corrupted_act,
            "patched_act": patched_act,
            "patch_damage": patch_damage,
        }

    # Overlap analysis
    sets = [set(out["results"][str(l)]["top_features"]) for l in args.lengths]
    overlap = {}
    for i, li in enumerate(args.lengths):
        for j, lj in enumerate(args.lengths):
            if i < j:
                inter = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                overlap[f"{li}_vs_{lj}"] = {"intersection": inter, "union": union,
                                              "jaccard": inter / union if union else 0}
    out["feature_set_overlap"] = overlap

    out_path = RESULTS_DIR / "pattern_length_robustness.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")
    print("\n=== Summary ===")
    for l in args.lengths:
        r = out["results"][str(l)]
        print(f"  plen={l:>2d}  patch_damage={r['patch_damage']:+.4f}")
    for k, v in overlap.items():
        print(f"  {k}: Jaccard = {v['jaccard']:.2f}")


if __name__ == "__main__":
    main()
