#!/usr/bin/env python3
"""Validate that induction features identified from synthetic patterns also fire
on natural repeated n-grams in Pile text.

Pipeline:
  1. Load Mamba-1, L32 SAE, normalization stats.
  2. Stream Pile; for each chunk, find positions (i, j) where the bigram
     (tok_{j-1}, tok_j) also appears at some earlier position (i-1, i) with
     i <= j - 32 (at least 32-token gap) and i >= 16.
  3. Collect the activation of the top induction features from the existing
     induction_features.json at the "second occurrence" position (j).
  4. Compare to a random-position baseline (same texts, position uniformly
     sampled outside the repeat).
  5. Report: mean/median activation at repeat vs baseline, and the fraction of
     top induction features that fire >2x stronger at repeat positions.

Outputs:
  $STORAGE/results_phase4/real_text_induction.json
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
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/path/to/storage"))
CKPT_DIR = STORAGE / "checkpoints_normed"
ACTS_DIR = STORAGE / "activations"
RESULTS_DIR = STORAGE / "results_phase4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "state-spaces/mamba-2.8b-hf"
MODEL_KEY = "mamba1_2.8b"
D_MODEL = 2560
MID_LAYER = 32
SAE_EXPANSION = 16
SAE_K = 64
SEQ_LEN = 512


def load_sae_and_norm(device):
    d_hidden = D_MODEL * SAE_EXPANSION
    run_key = f"{MODEL_KEY}_L{MID_LAYER}_x{SAE_EXPANSION}_k{SAE_K}_normed"
    ckpt_path = CKPT_DIR / f"{run_key}.pt"
    sae = create_sae(D_MODEL, d_hidden, sae_type="topk", k=SAE_K).to(device)
    sae.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    sae.eval()

    acts_path = ACTS_DIR / MODEL_KEY / f"layer_{MID_LAYER}.pt"
    t = torch.load(acts_path, map_location="cpu", weights_only=True, mmap=True)
    sample = t[:10000].clone().float()
    act_mean = sample.mean(dim=0).to(device)
    act_std = sample.std(dim=0).clamp(min=1e-6).to(device)
    del t, sample
    return sae, act_mean, act_std


def capture_residual_at_layer(model, tokens, layer_idx):
    captured = {}
    def hook(mod, ins, out):
        captured["out"] = out[0].detach() if isinstance(out, tuple) else out.detach()
    h = model.backbone.layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["out"]


def find_repeat_positions(token_ids, min_gap=32, min_first=16, max_second=None):
    """Return list of (i, j) where bigram (t[j-1], t[j]) == (t[i-1], t[i]),
    j - i >= min_gap, i >= min_first, j <= max_second."""
    L = len(token_ids)
    if max_second is None:
        max_second = L
    repeats = []
    seen = {}  # bigram -> list of i positions
    for k in range(1, L):
        bg = (token_ids[k - 1].item(), token_ids[k].item())
        if k >= min_first and bg in seen:
            for i in seen[bg]:
                if k - i >= min_gap and k < max_second:
                    repeats.append((i, k))
                    break  # only take first qualifying prior occurrence
        seen.setdefault(bg, []).append(k)
    return repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=500,
                    help="Number of Pile documents to scan for natural repeats.")
    ap.add_argument("--top_features", type=int, default=10,
                    help="Number of top induction features to track.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device

    # Load induction-feature list from Phase B
    ind = json.load(open(RESULTS_DIR / "induction_features.json"))
    top_feats = ind["feature"][:args.top_features]
    ind_scores = ind["score"][:args.top_features]
    print(f"Tracking top-{args.top_features} induction features: {top_feats}")

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)

    print("Loading L32 SAE...")
    sae, act_mean, act_std = load_sae_and_norm(device)

    # Stream Pile for documents
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    all_repeat_acts = []   # activation of top features at j (repeat position)
    all_baseline_acts = [] # activation at a non-repeat position (random)
    n_repeats_found = 0
    n_docs_processed = 0

    it = iter(ds)
    pbar = tqdm(desc="docs", total=args.n_docs)
    while n_docs_processed < args.n_docs:
        try:
            ex = next(it)
        except StopIteration:
            break
        text = ex.get("text", "")
        if not text or len(text) < 400:
            continue
        # Tokenize at SEQ_LEN
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=SEQ_LEN)["input_ids"][0]
        if ids.shape[0] < 64:
            continue
        repeats = find_repeat_positions(ids, min_gap=32, min_first=16,
                                        max_second=ids.shape[0])
        if not repeats:
            n_docs_processed += 1
            pbar.update(1)
            continue

        # Forward pass
        with torch.no_grad():
            res = capture_residual_at_layer(model, ids.unsqueeze(0).to(device),
                                            MID_LAYER)
            # res shape: (1, L, d_model)
            normed = (res.float() - act_mean) / act_std
            _, z, *_ = sae(normed)
            # z shape: (1, L, d_hidden)

        # Positions where repeat ends (j indices)
        j_positions = torch.tensor([j for _, j in repeats], device=device)
        # Sample random baseline positions that are NOT j and far from start
        forbidden = set(j for _, j in repeats)
        candidates = [p for p in range(16, ids.shape[0]) if p not in forbidden]
        if not candidates:
            n_docs_processed += 1
            pbar.update(1)
            continue
        rng = np.random.default_rng(n_docs_processed)
        baseline_pos = rng.choice(candidates, size=min(len(repeats), len(candidates)),
                                   replace=False)
        baseline_pos = torch.tensor(baseline_pos, device=device, dtype=torch.long)

        top_feats_t = torch.tensor(top_feats, device=device, dtype=torch.long)
        rep_z = z[0, j_positions][:, top_feats_t]         # (n_rep, n_feat)
        base_z = z[0, baseline_pos][:, top_feats_t]        # (n_base, n_feat)

        all_repeat_acts.append(rep_z.cpu().numpy())
        all_baseline_acts.append(base_z.cpu().numpy())
        n_repeats_found += len(repeats)
        n_docs_processed += 1
        pbar.update(1)
        pbar.set_postfix(repeats_found=n_repeats_found)
    pbar.close()

    rep = np.concatenate(all_repeat_acts, axis=0) if all_repeat_acts else np.zeros((0, len(top_feats)))
    base = np.concatenate(all_baseline_acts, axis=0) if all_baseline_acts else np.zeros((0, len(top_feats)))
    print(f"\n=== Stats ({rep.shape[0]} repeat positions, {base.shape[0]} baselines) ===")
    per_feat = []
    for i, (f, s) in enumerate(zip(top_feats, ind_scores)):
        rep_mean = float(rep[:, i].mean()) if rep.shape[0] else 0.0
        base_mean = float(base[:, i].mean()) if base.shape[0] else 0.0
        rep_frac_active = float((rep[:, i] > 0).mean()) if rep.shape[0] else 0.0
        base_frac_active = float((base[:, i] > 0).mean()) if base.shape[0] else 0.0
        ratio = rep_mean / base_mean if base_mean > 1e-6 else float("inf") if rep_mean > 1e-6 else 1.0
        per_feat.append({
            "feature": int(f),
            "synth_score": float(s),
            "real_repeat_mean_act": rep_mean,
            "real_baseline_mean_act": base_mean,
            "real_repeat_ratio_to_baseline": ratio,
            "real_repeat_frac_active": rep_frac_active,
            "real_baseline_frac_active": base_frac_active,
        })
        print(f"  feat {f:>6d}  synth={s:.3f}  rep_mean={rep_mean:.4f}  "
              f"base_mean={base_mean:.4f}  ratio={ratio:.2f}x  "
              f"rep_active={rep_frac_active:.2%} vs base={base_frac_active:.2%}")

    out = {
        "n_docs_processed": n_docs_processed,
        "n_repeats_found": n_repeats_found,
        "n_repeat_positions": int(rep.shape[0]),
        "n_baseline_positions": int(base.shape[0]),
        "top_features": top_feats,
        "per_feature": per_feat,
        "summary": {
            "mean_ratio_repeat_over_baseline": float(np.mean(
                [p["real_repeat_ratio_to_baseline"]
                 for p in per_feat
                 if np.isfinite(p["real_repeat_ratio_to_baseline"])])),
            "frac_features_above_2x": float(np.mean(
                [p["real_repeat_ratio_to_baseline"] >= 2.0 for p in per_feat])),
        },
    }
    json.dump(out, open(RESULTS_DIR / "real_text_induction.json", "w"), indent=2)
    print(f"\nWrote {RESULTS_DIR / 'real_text_induction.json'}")
    print(f"Mean ratio repeat/baseline: {out['summary']['mean_ratio_repeat_over_baseline']:.2f}x")
    print(f"Fraction of features with >=2x ratio: {out['summary']['frac_features_above_2x']:.0%}")


if __name__ == "__main__":
    main()
