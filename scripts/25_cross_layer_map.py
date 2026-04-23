#!/usr/bin/env python3
"""Cross-layer feature-to-feature map: L30 internal SAE features → L32 residual SAE induction features.

The hypothesized circuit is:
  pre-x_proj features (d_inner space, internal SAE at L30)
    → x_proj(·) → (Δ, B, C) parameters at L30
      → selective scan uses state h_{0..29} + new params
        → output residual at L30 added to residual stream
          → L32 residual SAE features (induction features)

If this is (approximately) linear, a regressor from L30 internal SAE features →
L32 induction features should have high R². If not, there's essential non-
linearity (which the selective scan provides via exp(A·Δ)).

This script:
  1. Generate induction pairs (clean + corrupted).
  2. Capture L30 x_proj INPUT → encode through L30 internal SAE → 40,960 features.
  3. Capture L32 residual → encode through L32 SAE → 40,960 features.
  4. Train Ridge regression from (1) → (subset of 2, the top-10 induction features).
  5. Report R² and per-feature R².

Output: $STORAGE/results_phase4/cross_layer_map.json
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.activation_cache import get_model_and_tokenizer
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
CKPT_DIR = STORAGE / "checkpoints_normed"
ACTS_DIR = STORAGE / "activations"
RESULTS_DIR = STORAGE / "results_phase4"

MODEL_NAME = "state-spaces/mamba-2.8b-hf"
MODEL_KEY = "mamba1_2.8b"
D_MODEL = 2560
D_IN = 5120
L30 = 30
L32 = 32
INTERNAL_EXP = 8
L32_EXP = 16
K = 64
PATTERN_LEN = 8
PREFIX_LEN = 8
MID_LEN = 32


def load_saes(device):
    d_hidden_int = D_IN * INTERNAL_EXP
    sae_int = create_sae(D_IN, d_hidden_int, sae_type="topk", k=K).to(device)
    sae_int.load_state_dict(torch.load(
        CKPT_DIR / f"{MODEL_KEY}_L{L30}_xprojin_x{INTERNAL_EXP}_k{K}_normed.pt",
        map_location=device, weights_only=True))
    sae_int.eval()
    np_int = torch.load(
        CKPT_DIR / f"{MODEL_KEY}_L{L30}_xprojin_x{INTERNAL_EXP}_k{K}_normed_normparams.pt",
        map_location=device, weights_only=True)

    d_hidden_l32 = D_MODEL * L32_EXP
    sae_l32 = create_sae(D_MODEL, d_hidden_l32, sae_type="topk", k=K).to(device)
    sae_l32.load_state_dict(torch.load(
        CKPT_DIR / f"{MODEL_KEY}_L{L32}_x{L32_EXP}_k{K}_normed.pt",
        map_location=device, weights_only=True))
    sae_l32.eval()
    acts_path = ACTS_DIR / MODEL_KEY / f"layer_{L32}.pt"
    t = torch.load(acts_path, map_location="cpu", weights_only=True, mmap=True)
    sample = t[:10000].clone().float()
    l32_mean = sample.mean(dim=0).to(device)
    l32_std = sample.std(dim=0).clamp(min=1e-6).to(device)
    del t, sample
    return (sae_int, np_int["act_mean"].to(device), np_int["act_std"].to(device),
            sae_l32, l32_mean, l32_std)


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


def capture(model, tokens):
    captured = {}
    def pre_hook_30(mod, ins):
        captured["x30"] = ins[0].detach()
    def post_hook_32(mod, ins, out):
        captured["r32"] = (out[0] if isinstance(out, tuple) else out).detach()
    h1 = model.backbone.layers[L30].mixer.x_proj.register_forward_pre_hook(pre_hook_30)
    h2 = model.backbone.layers[L32].register_forward_hook(post_hook_32)
    with torch.no_grad():
        model(tokens)
    h1.remove(); h2.remove()
    return captured["x30"], captured["r32"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_pairs", type=int, default=512)
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    sae_int, int_mean, int_std, sae_l32, l32_mean, l32_std = load_saes(device)

    # Load the L32 SAE induction features
    ind = json.load(open(RESULTS_DIR / "induction_features.json"))
    top_L32_feats = torch.tensor(ind["feature"][:10], device=device)

    # Build matched (X, Y) pairs: use clean AND corrupted, both at induction positions
    Xs, Ys = [], []
    for seed, tokens in [(0, "clean"), (0, "corr"), (1, "clean"), (1, "corr"),
                         (2, "clean"), (2, "corr"), (3, "clean"), (3, "corr")]:
        clean, corr, ind_start, ind_end = make_batch(tokenizer, args.n_pairs // 4, seed, device)
        toks = clean if tokens == "clean" else corr
        x30, r32 = capture(model, toks)

        # Encode x30 through internal SAE
        n_int = (x30.float() - int_mean) / int_std
        _, z_int, *_ = sae_int(n_int.reshape(-1, D_IN))
        z_int = z_int.reshape(toks.shape[0], toks.shape[1], -1)

        # Encode r32 through L32 SAE
        n_l32 = (r32.float() - l32_mean) / l32_std
        _, z_l32, *_ = sae_l32(n_l32)

        # Take only the induction positions (48..55)
        Xi = z_int[:, ind_start:ind_end].reshape(-1, z_int.shape[-1])       # (N, 40960)
        Yi = z_l32[:, ind_start:ind_end][:, :, top_L32_feats].reshape(-1, len(top_L32_feats))
        Xs.append(Xi.detach().cpu().numpy().astype(np.float32))
        Ys.append(Yi.detach().cpu().numpy().astype(np.float32))

    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    print(f"X shape: {X.shape}  (internal SAE features)")
    print(f"Y shape: {Y.shape}  (L32 SAE top-10 induction features)")

    # 5-fold CV Ridge regression
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    per_feat_r2 = np.zeros(Y.shape[1])
    total_r2 = 0.0

    for fold_idx, (tr, te) in enumerate(kf.split(X)):
        reg = Ridge(alpha=1.0)
        reg.fit(X[tr], Y[tr])
        pred = reg.predict(X[te])
        # Per-feature R²
        ss_res = ((Y[te] - pred) ** 2).sum(axis=0)
        ss_tot = ((Y[te] - Y[te].mean(axis=0)) ** 2).sum(axis=0)
        ss_tot = np.where(ss_tot > 1e-12, ss_tot, 1.0)
        r2_per = 1 - ss_res / ss_tot
        per_feat_r2 += r2_per / kf.get_n_splits()
        # Overall R² (multi-output, flatten)
        ss_res_all = ((Y[te] - pred) ** 2).sum()
        ss_tot_all = ((Y[te] - Y[te].mean(axis=0)) ** 2).sum()
        total_r2 += (1 - ss_res_all / max(ss_tot_all, 1e-12)) / kf.get_n_splits()

    # Sparse regression: use only top-k internal features as predictors (via Lasso)
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=0.01, max_iter=3000)
    lasso.fit(X, Y.mean(axis=1))  # predict the mean induction activation
    nonzero = int((np.abs(lasso.coef_) > 1e-4).sum())

    result = {
        "n_samples": int(X.shape[0]),
        "n_internal_features": int(X.shape[1]),
        "top_L32_feats": [int(f) for f in top_L32_feats.cpu().tolist()],
        "overall_R2_5fold": float(total_r2),
        "per_feat_R2": per_feat_r2.tolist(),
        "lasso_nonzero_predictors": nonzero,
    }
    out_path = RESULTS_DIR / "cross_layer_map.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")
    print(f"\n=== 5-fold CV Ridge regression: L30 internal SAE → L32 induction features ===")
    print(f"  Overall R² (multi-output avg):  {total_r2:.4f}")
    print(f"  Per-feature R²:")
    for f, r2 in zip(top_L32_feats.cpu().tolist(), per_feat_r2.tolist()):
        print(f"    L32 feat {f:>6d}: R² = {r2:+.4f}")
    print(f"\n  Lasso (α=0.01) nonzero predictors: {nonzero} / {X.shape[1]}")


if __name__ == "__main__":
    main()
