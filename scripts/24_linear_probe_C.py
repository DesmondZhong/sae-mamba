#!/usr/bin/env python3
"""Linear probe on the 16-dim C output of L30 x_proj.

Direct test of "induction lives in C": train a logistic regression on the 16
C-dims only (at induction positions of synthetic pairs) to predict whether
the sequence is a clean induction pair or corrupted. If the probe reaches
near-perfect accuracy, that's strong evidence the induction signal is
linearly decodable from just those 16 dims.

Also compares:
- Δ_pre (160 dim) probe accuracy
- B (16 dim) probe accuracy
- full x_proj output (192 dim) probe accuracy
- the full d_inner=5120 input to x_proj probe accuracy (upper bound)

Output: $STORAGE/results_phase4/linear_probe.json
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.activation_cache import get_model_and_tokenizer

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
RESULTS_DIR = STORAGE / "results_phase4"

MODEL_NAME = "state-spaces/mamba-2.8b-hf"
LOCUS_LAYER = 30
PATTERN_LEN = 8
PREFIX_LEN = 8
MID_LEN = 32


def make_induction_batch(tokenizer, n_pairs, seed, device):
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


def capture_xproj_output(model, tokens, layer):
    captured = {}
    def hook(mod, ins, out):
        captured["o"] = out.detach().clone()
    h = model.backbone.layers[layer].mixer.x_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["o"]


def capture_xproj_input(model, tokens, layer):
    captured = {}
    def pre_hook(mod, ins):
        captured["x"] = ins[0].detach().clone()
    h = model.backbone.layers[layer].mixer.x_proj.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["x"]


def probe_acc(X, y, name):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=4)
    return float(scores.mean()), float(scores.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_pairs", type=int, default=512)
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)

    clean, corr, ind_start, ind_end = make_induction_batch(
        tokenizer, args.n_pairs, seed=0, device=device)

    xo_clean = capture_xproj_output(model, clean, LOCUS_LAYER)
    xo_corr = capture_xproj_output(model, corr, LOCUS_LAYER)
    xi_clean = capture_xproj_input(model, clean, LOCUS_LAYER)
    xi_corr = capture_xproj_input(model, corr, LOCUS_LAYER)

    # Build X, y datasets at induction positions
    # Stack all (pair × induction_position) × feature_dim
    def flatten(t, ind_start, ind_end):
        # t: (B, L, D) → (B * PL, D)
        return t[:, ind_start:ind_end, :].reshape(-1, t.shape[-1]).cpu().float().numpy()

    mixer = model.backbone.layers[LOCUS_LAYER].mixer
    dt_rank = mixer.time_step_rank
    state_size = mixer.ssm_state_size

    X_out_clean = flatten(xo_clean, ind_start, ind_end)  # (n_pairs*8, 192)
    X_out_corr = flatten(xo_corr, ind_start, ind_end)
    X_in_clean = flatten(xi_clean, ind_start, ind_end)   # (n_pairs*8, 5120)
    X_in_corr = flatten(xi_corr, ind_start, ind_end)

    n_per_class = X_out_clean.shape[0]
    y = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)])  # 1=clean, 0=corrupted

    slices = {
        "C_matrix_out": np.s_[:, dt_rank + state_size:dt_rank + 2 * state_size],
        "B_matrix_out": np.s_[:, dt_rank:dt_rank + state_size],
        "delta_pre_out": np.s_[:, :dt_rank],
        "full_xproj_out": np.s_[:, :],  # 192 dim
    }
    results = {}
    print("\n=== Linear probe on x_proj OUTPUT slices ===")
    for name, sl in slices.items():
        X = np.vstack([X_out_clean[sl], X_out_corr[sl]])
        mean, std = probe_acc(X, y, name)
        results[name] = {"dim": X.shape[1], "accuracy_mean": mean, "accuracy_std": std}
        print(f"  {name:<16s}  dim={X.shape[1]:>4d}  acc={mean:.4f} ± {std:.4f}")

    print("\n=== Linear probe on x_proj INPUT (d_inner=5120) ===")
    X = np.vstack([X_in_clean, X_in_corr])
    mean, std = probe_acc(X, y, "xproj_input")
    results["xproj_input"] = {"dim": X.shape[1], "accuracy_mean": mean, "accuracy_std": std}
    print(f"  xproj_input     dim={X.shape[1]:>4d}  acc={mean:.4f} ± {std:.4f}")

    # Control: a random 16-dim slice of d_inner
    rng = np.random.default_rng(42)
    rand_dims = rng.choice(X_in_clean.shape[1], 16, replace=False)
    X_rand = np.vstack([X_in_clean[:, rand_dims], X_in_corr[:, rand_dims]])
    mean, std = probe_acc(X_rand, y, "random_16")
    results["random_16dim_from_xproj_input"] = {
        "dim": 16, "accuracy_mean": mean, "accuracy_std": std}
    print(f"  random_16 (d_inner) dim={16}  acc={mean:.4f} ± {std:.4f}")

    out = {
        "n_pairs": args.n_pairs,
        "n_positions_per_class": n_per_class,
        "results": results,
    }
    json.dump(out, open(RESULTS_DIR / "linear_probe.json", "w"), indent=2)
    print(f"\nWrote {RESULTS_DIR / 'linear_probe.json'}")


if __name__ == "__main__":
    main()
