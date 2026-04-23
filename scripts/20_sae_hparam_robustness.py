#!/usr/bin/env python3
"""SAE-hyperparameter robustness: does L30 x_proj dominance hold across different
Mamba-1 L32 SAEs?

Tests k ∈ {32, 64, 128} at x16 and x ∈ {8, 16, 32} at k64 — the 5 Mamba-1 L32
SAEs available. For each, identify induction features, measure patch_damage at
L30 x_proj. If the locus is robust, patch_damage stays high across all SAEs.

Output: $STORAGE/results_phase4/sae_hparam_robustness.json
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.activation_cache import get_model_and_tokenizer
from src.mamba_internals import force_slow_forward, MambaInternalCapture, MambaInternalPatcher
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
PATTERN_LEN = 8
PREFIX_LEN = 8
MID_LEN = 32


def load_sae_norm(device, expansion, k):
    d_hidden = D_MODEL * expansion
    run_key = f"{MODEL_KEY}_L{MID_LAYER}_x{expansion}_k{k}_normed"
    path = CKPT_DIR / f"{run_key}.pt"
    if not path.exists():
        return None, None, None
    sae = create_sae(D_MODEL, d_hidden, sae_type="topk", k=k).to(device)
    sae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
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


def encode_at(model, tokens, sae, act_mean, act_std, mid_layer, positions,
               patches=None):
    captured = {}
    def hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()
    h = model.backbone.layers[mid_layer].register_forward_hook(hook)
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
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    force_slow_forward(model)

    # Pre-capture corrupted x_proj at L30 once (doesn't depend on SAE)
    clean, corr, ind_start, ind_end = make_batch(
        tokenizer, args.n_pairs, seed=0, device=device)
    with MambaInternalCapture(model, sites=[(LOCUS_LAYER, "x_proj")]) as cap:
        with torch.no_grad():
            model(corr)
    corr_xproj = cap.captured[(LOCUS_LAYER, "x_proj")]

    cases = [("x16", 32), ("x16", 64), ("x16", 128), ("x8", 64), ("x32", 64)]
    results = {}
    for exp_str, k in cases:
        expansion = int(exp_str[1:])
        sae, act_mean, act_std = load_sae_norm(device, expansion, k)
        if sae is None:
            print(f"[skip] missing checkpoint for x{expansion} k{k}")
            continue
        d_hidden = D_MODEL * expansion
        # Identify induction features with THIS SAE
        z_clean = encode_at(model, clean, sae, act_mean, act_std,
                             MID_LAYER, (ind_start, ind_end))
        z_corr = encode_at(model, corr, sae, act_mean, act_std,
                            MID_LAYER, (ind_start, ind_end))
        score = (z_clean - z_corr).mean(dim=0).detach()
        top_vals, top_idx = score.topk(10)
        ti = top_idx
        baseline_act = z_clean[:, ti].mean().item()
        corrupted_act = z_corr[:, ti].mean().item()
        gap = baseline_act - corrupted_act

        # Patch L30 x_proj with pre-captured corrupted version
        z_patched = encode_at(
            model, clean, sae, act_mean, act_std,
            MID_LAYER, (ind_start, ind_end),
            patches={(LOCUS_LAYER, "x_proj"): corr_xproj},
        )
        patched_act = z_patched[:, ti].mean().item()
        patch_damage = 1.0 - (patched_act - corrupted_act) / gap if abs(gap) > 1e-8 else 0.0

        key = f"x{expansion}_k{k}"
        results[key] = {
            "expansion": expansion, "k": k, "d_hidden": d_hidden,
            "top_features": top_idx.cpu().tolist(),
            "baseline_act": baseline_act, "corrupted_act": corrupted_act,
            "gap": gap,
            "patched_act": patched_act,
            "patch_damage_L30_xproj": patch_damage,
        }
        print(f"  {key:<10s} gap={gap:.3f}  L30 x_proj patch_damage = {patch_damage:+.4f}  "
              f"top_feats[:3]={top_idx[:3].cpu().tolist()}")

        del sae

    # Feature-set overlap across SAEs (canonical x16 k64 vs others; different d_hidden so only sensible for same-width comparisons)
    by_expansion = {}
    for key, r in results.items():
        by_expansion.setdefault(r["expansion"], []).append((key, r["top_features"]))

    overlap = {}
    # K-sweep at x16
    if 16 in by_expansion and len(by_expansion[16]) > 1:
        refs = {k_: tf for k_, tf in by_expansion[16]}
        ref_k64 = set(refs.get("x16_k64", []))
        for k_, tf in refs.items():
            other = set(tf)
            j = len(ref_k64 & other) / len(ref_k64 | other) if ref_k64 or other else 0
            overlap[f"{k_}_vs_x16_k64"] = j

    out = {
        "n_pairs": args.n_pairs,
        "results": results,
        "feature_overlap_k_sweep_at_x16": overlap,
    }
    json.dump(out, open(RESULTS_DIR / "sae_hparam_robustness.json", "w"), indent=2)
    print(f"\nWrote {RESULTS_DIR / 'sae_hparam_robustness.json'}")
    print("\nSummary:")
    print(f"{'config':<12s}  {'gap':>6s}  {'L30 x_proj damage':>18s}")
    for key, r in results.items():
        print(f"{key:<12s}  {r['gap']:>6.3f}  {r['patch_damage_L30_xproj']:>+18.4f}")
    for k_, j in overlap.items():
        print(f"{k_}: Jaccard = {j:.2f}")


if __name__ == "__main__":
    main()
