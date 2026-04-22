#!/usr/bin/env python3
"""Validation experiments to pressure-test Phase-B induction-localization claims.

Runs three sanity checks:
  (A) Null patching: patch CLEAN-run activations back into CLEAN runs. If the
      pipeline is correctly measuring induction-specific damage, this should give
      patch_damage ≈ 0 (since we're not removing anything).
  (B) Random-feature baseline: reuse the same patching machinery but track a
      random set of 10 NON-induction features; compute the same patch_damage
      metric. If random features show comparably large damage at L30 x_proj,
      the metric isn't specific to induction.
  (C) Multi-seed robustness: re-identify induction features with different
      random seeds for the induction-pair generator; measure how much the
      top-10 feature set overlaps across seeds (higher = more robust).

Outputs $STORAGE/results_phase4/validation.json with per-check numbers.
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
from tqdm import tqdm

from src.activation_cache import get_model_and_tokenizer
from src.mamba_internals import (
    MambaInternalCapture, MambaInternalPatcher, force_slow_forward,
)
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
CKPT_DIR = STORAGE / "checkpoints_normed"
ACTS_DIR = STORAGE / "activations"
RESULTS_DIR = STORAGE / "results_phase4"

MODEL_NAME = "state-spaces/mamba-2.8b-hf"
MODEL_KEY = "mamba1_2.8b"
D_MODEL = 2560
MID_LAYER = 32

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


def make_induction_batch(tokenizer, n_pairs=64, seed=0, device="cuda:0",
                         pattern_len=PATTERN_LEN):
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


def encode_residual_at(model, tokens, sae, act_mean, act_std, layer, positions):
    captured = {}
    def hook(mod, ins, out):
        captured["out"] = out[0].detach() if isinstance(out, tuple) else out.detach()
    h = model.backbone.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    res = captured["out"]
    normed = (res.float() - act_mean) / act_std
    _, z, *_ = sae(normed)
    # z shape: (B, L, d_hidden)
    return z[:, positions[0]:positions[1]].mean(dim=1)  # (B, d_hidden)


def identify_induction_features(model, tokenizer, sae, act_mean, act_std, device,
                                 seed=0, top_k=10, n_pairs=64):
    clean, corr, ind_start, ind_end = make_induction_batch(
        tokenizer, n_pairs=n_pairs, seed=seed, device=device)
    clean_z = encode_residual_at(model, clean, sae, act_mean, act_std,
                                  MID_LAYER, (ind_start, ind_end))
    corr_z = encode_residual_at(model, corr, sae, act_mean, act_std,
                                 MID_LAYER, (ind_start, ind_end))
    score = (clean_z - corr_z).mean(dim=0)
    top_vals, top_idx = score.topk(top_k)
    return top_idx.cpu().tolist(), top_vals.cpu().tolist(), score.detach()


def mean_target_act(model, tokens, sae, act_mean, act_std, target_idx,
                    ind_start, ind_end, patches=None):
    if patches is not None:
        with MambaInternalPatcher(model, patches=patches):
            z = encode_residual_at(model, tokens, sae, act_mean, act_std,
                                    MID_LAYER, (ind_start, ind_end))
    else:
        z = encode_residual_at(model, tokens, sae, act_mean, act_std,
                                MID_LAYER, (ind_start, ind_end))
    return z[:, target_idx].mean().item()


def validate_null_patching(model, tokenizer, sae, act_mean, act_std, top_feats,
                            device, seed=0):
    """Run the full pipeline but patch CLEAN activations into CLEAN run.
    Expect patch_damage ≈ 0."""
    clean, corr, ind_start, ind_end = make_induction_batch(
        tokenizer, n_pairs=32, seed=seed, device=device)
    ti = torch.tensor(top_feats, device=device)
    baseline_act = mean_target_act(model, clean, sae, act_mean, act_std, ti,
                                    ind_start, ind_end)
    corrupted_act = mean_target_act(model, corr, sae, act_mean, act_std, ti,
                                     ind_start, ind_end)
    total = baseline_act - corrupted_act
    results = []
    for (layer, comp) in [(30, "x_proj"), (30, "conv1d"), (30, "in_proj"), (32, "in_proj")]:
        with MambaInternalCapture(model, sites=[(layer, comp)]) as cap:
            with torch.no_grad():
                model(clean)  # capture from CLEAN run
        clean_internal = cap.captured[(layer, comp)]
        patched_act = mean_target_act(model, clean, sae, act_mean, act_std, ti,
                                       ind_start, ind_end,
                                       patches={(layer, comp): clean_internal})
        patch_damage = 1.0 - (patched_act - corrupted_act) / total if abs(total) > 1e-8 else 0.0
        results.append({"layer": layer, "component": comp,
                        "baseline_act": baseline_act, "corrupted_act": corrupted_act,
                        "patched_act": patched_act, "patch_damage": patch_damage})
        print(f"  NULL-PATCH L{layer} {comp}: patch_damage={patch_damage:+.4f} "
              f"(baseline={baseline_act:.3f}, patched={patched_act:.3f})")
    return results


def random_feature_baseline(model, tokenizer, sae, act_mean, act_std,
                             top_feats, device, d_hidden, seed=42, n_random=10):
    """Pick random features NOT in top-50 induction; compute patch_damage for the
    same top sites as Phase B. If patch_damage is comparable, the metric is not
    induction-specific."""
    rng = np.random.default_rng(seed)
    ind_set = set(top_feats[:50]) if len(top_feats) >= 50 else set(top_feats)
    # Sample random features
    pool = [i for i in range(d_hidden) if i not in ind_set]
    rand_feats = rng.choice(pool, n_random, replace=False).tolist()

    clean, corr, ind_start, ind_end = make_induction_batch(
        tokenizer, n_pairs=32, seed=0, device=device)
    ti = torch.tensor(rand_feats, device=device)
    baseline_act = mean_target_act(model, clean, sae, act_mean, act_std, ti,
                                    ind_start, ind_end)
    corrupted_act = mean_target_act(model, corr, sae, act_mean, act_std, ti,
                                     ind_start, ind_end)
    total = baseline_act - corrupted_act
    print(f"  RANDOM-FEAT baseline={baseline_act:.3f}, corrupted={corrupted_act:.3f}, "
          f"total={total:.3f}")
    results = []
    for (layer, comp) in [(30, "x_proj"), (30, "conv1d"), (30, "in_proj")]:
        with MambaInternalCapture(model, sites=[(layer, comp)]) as cap:
            with torch.no_grad():
                model(corr)
        corr_internal = cap.captured[(layer, comp)]
        patched_act = mean_target_act(model, clean, sae, act_mean, act_std, ti,
                                       ind_start, ind_end,
                                       patches={(layer, comp): corr_internal})
        patch_damage = 1.0 - (patched_act - corrupted_act) / total if abs(total) > 1e-8 else 0.0
        results.append({"layer": layer, "component": comp, "random_features": rand_feats,
                        "baseline_act": baseline_act, "corrupted_act": corrupted_act,
                        "patched_act": patched_act, "patch_damage": patch_damage})
        print(f"  RANDOM-FEAT L{layer} {comp}: patch_damage={patch_damage:+.4f}")
    return results


def multi_seed_robustness(model, tokenizer, sae, act_mean, act_std, device,
                           top_k=10, seeds=(0, 1, 2, 3, 4)):
    sets = []
    scores = []
    for s in seeds:
        feats, _, full_score = identify_induction_features(
            model, tokenizer, sae, act_mean, act_std, device,
            seed=s, top_k=top_k, n_pairs=128)
        sets.append(feats)
        scores.append(full_score.cpu().numpy())
        print(f"  seed={s} top-{top_k}: {feats}")
    # Jaccard overlap across all pairs
    overlaps = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = len(set(sets[i]) & set(sets[j]))
            union = len(set(sets[i]) | set(sets[j]))
            overlaps.append(inter / union)
    # Also: union of top-10 across seeds (how many distinct features appear)
    distinct = set().union(*sets)
    # Correlation of full score vectors across seeds
    score_stack = np.stack(scores, axis=0)
    corrs = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            c = np.corrcoef(score_stack[i], score_stack[j])[0, 1]
            corrs.append(c)
    return {
        "seeds": list(seeds),
        "sets": sets,
        "pairwise_jaccard_mean": float(np.mean(overlaps)),
        "pairwise_jaccard_all": overlaps,
        "n_distinct_union": len(distinct),
        "full_score_corr_mean": float(np.mean(corrs)),
        "full_score_corr_all": corrs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_seed", action="store_true")
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    force_slow_forward(model)

    print("Loading L32 SAE...")
    sae, act_mean, act_std = load_sae_and_norm(device)

    # Use existing top-10 induction features
    ind = json.load(open(RESULTS_DIR / "induction_features.json"))
    top_feats = ind["feature"][:10]
    print(f"Using top-10 induction features: {top_feats}")
    d_hidden = D_MODEL * SAE_EXPANSION

    print("\n=== (A) NULL PATCHING (clean→clean, expect ≈0) ===")
    null_results = validate_null_patching(model, tokenizer, sae, act_mean, act_std,
                                           top_feats, device)

    print("\n=== (B) RANDOM-FEATURE BASELINE (expect ≪ induction) ===")
    rand_results = random_feature_baseline(model, tokenizer, sae, act_mean, act_std,
                                            top_feats, device, d_hidden)

    seed_results = None
    if not args.skip_seed:
        print("\n=== (C) MULTI-SEED ROBUSTNESS OF INDUCTION-FEATURE ID ===")
        seed_results = multi_seed_robustness(model, tokenizer, sae, act_mean, act_std,
                                              device, top_k=10, seeds=[0, 1, 2, 3, 4])

    out = {
        "null_patching": null_results,
        "random_feature_baseline": rand_results,
        "multi_seed_robustness": seed_results,
    }
    out_path = RESULTS_DIR / "validation.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
