#!/usr/bin/env python3
"""Real-text induction at varying gap distances between first and second
occurrences of a bigram.

The real-text validation (§D of report_phase4) uses min_gap=32. This script
sweeps min_gap ∈ {16, 32, 64, 128, 256} to check whether Mamba's induction
features still fire strongly at long-range repeats.

Output: $STORAGE/results_phase4/gap_sweep.json
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


def find_repeat_positions(ids, min_gap, max_gap, min_first=8):
    repeats = []
    seen = {}
    for k in range(1, len(ids)):
        bg = (ids[k - 1].item(), ids[k].item())
        if k >= min_first and bg in seen:
            for i in seen[bg]:
                gap = k - i
                if min_gap <= gap <= max_gap:
                    repeats.append((i, k, gap))
                    break
        seen.setdefault(bg, []).append(k)
    return repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=600)
    ap.add_argument("--gap_bins", type=int, nargs="+",
                    default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device

    ind = json.load(open(RESULTS_DIR / "induction_features.json"))
    top_feats = ind["feature"][:10]
    ti = torch.tensor(top_feats, device=device)

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    sae, act_mean, act_std = load_sae_and_norm(device)

    # Bin structure: [low, high) ranges defined by consecutive gap_bins
    bin_edges = args.gap_bins
    # Build bins as [b0, b1), [b1, b2), ...
    per_bin_rep_acts = {f"[{bin_edges[i]},{bin_edges[i+1]})": []
                        for i in range(len(bin_edges) - 1)}
    per_bin_base_acts = {k: [] for k in per_bin_rep_acts}

    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    it = iter(ds)
    pbar = tqdm(desc="docs", total=args.n_docs)
    n_processed = 0

    while n_processed < args.n_docs:
        try:
            ex = next(it)
        except StopIteration:
            break
        text = ex.get("text", "")
        if not text or len(text) < 800:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=SEQ_LEN)["input_ids"][0]
        if ids.shape[0] < 128:
            continue

        # Find repeats in any bin
        all_repeats = find_repeat_positions(ids, min_gap=bin_edges[0],
                                              max_gap=bin_edges[-1])
        if not all_repeats:
            n_processed += 1; pbar.update(1); continue

        # Forward + encode
        captured = {}
        def hook(mod, ins, out):
            captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()
        h = model.backbone.layers[MID_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            model(ids.unsqueeze(0).to(device))
        h.remove()

        normed = (captured["r"].float() - act_mean) / act_std
        _, z, *_ = sae(normed)
        z = z[0].detach()

        # Bin repeats by gap
        for (i, k, gap) in all_repeats:
            bin_idx = None
            for b in range(len(bin_edges) - 1):
                if bin_edges[b] <= gap < bin_edges[b + 1]:
                    bin_idx = b
                    break
            if bin_idx is None:
                continue
            bin_name = f"[{bin_edges[bin_idx]},{bin_edges[bin_idx+1]})"
            rep_act = z[k, ti].cpu().numpy()
            per_bin_rep_acts[bin_name].append(rep_act)

        # Sample baseline positions
        rng = np.random.default_rng(n_processed)
        baseline_pos = rng.integers(bin_edges[0], ids.shape[0],
                                     size=min(20, ids.shape[0] - bin_edges[0]))
        base_acts = z[baseline_pos, :][:, ti].cpu().numpy()  # (n_base, n_feat)
        # Add each baseline sample to ALL bins equally (we use the same sample)
        for bin_name in per_bin_base_acts:
            per_bin_base_acts[bin_name].extend(list(base_acts))

        n_processed += 1
        pbar.update(1)
    pbar.close()

    results = {}
    print("\n=== Gap-binned activation ratios ===")
    for bin_name in per_bin_rep_acts:
        rep = np.array(per_bin_rep_acts[bin_name])
        base = np.array(per_bin_base_acts[bin_name])
        if rep.size == 0 or base.size == 0:
            continue
        rep_means = rep.mean(axis=0)
        base_means = base.mean(axis=0)
        ratios = [r / b if b > 1e-6 else float("inf")
                  for r, b in zip(rep_means, base_means)]
        results[bin_name] = {
            "n_repeats": int(rep.shape[0]),
            "n_baseline": int(base.shape[0]),
            "per_feature_ratio": ratios,
            "mean_ratio": float(np.mean([r for r in ratios if np.isfinite(r)])),
            "median_ratio": float(np.median([r for r in ratios if np.isfinite(r)])),
            "rep_mean_act": rep_means.tolist(),
            "base_mean_act": base_means.tolist(),
        }
        print(f"  gap {bin_name:<14s}  n_rep={rep.shape[0]:>6d}  mean_ratio={results[bin_name]['mean_ratio']:.2f}x")

    out = {
        "top_features": top_feats,
        "n_docs_processed": n_processed,
        "bin_edges": bin_edges,
        "results": results,
    }
    json.dump(out, open(RESULTS_DIR / "gap_sweep.json", "w"), indent=2)
    print(f"\nWrote {RESULTS_DIR / 'gap_sweep.json'}")


if __name__ == "__main__":
    main()
