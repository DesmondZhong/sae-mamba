#!/usr/bin/env python3
"""Plot top-10 induction feature activations by sequence position.

Sanity check: the induction features we identified (by scoring clean-minus-
corrupted at positions 48-55) should fire ONLY at those positions, not at
prefix / first-pattern / mid-filler positions. If features fire broadly,
they're not induction-specific — they're general "something interesting here"
detectors.

Runs 1,024 synthetic induction pairs through Mamba-1 + L32 SAE, collects
feature activations across all 56 sequence positions, and saves mean activation
profiles for the top-10 induction features.

Output: $STORAGE/results_phase4/features_by_position.json + figure.
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.activation_cache import get_model_and_tokenizer
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
CKPT_DIR = STORAGE / "checkpoints_normed"
ACTS_DIR = STORAGE / "activations"
RESULTS_DIR = STORAGE / "results_phase4"
FIGS = RESULTS_DIR / "figures"

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_pairs", type=int, default=256)
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    sae, act_mean, act_std = load_sae_and_norm(device)

    ind = json.load(open(RESULTS_DIR / "induction_features.json"))
    top_feats = ind["feature"][:10]

    clean, corr, ind_start, ind_end = make_batch(tokenizer, args.n_pairs, 0, device)
    seq_len = clean.shape[1]

    # Forward clean + corrupted, capture L32 residual, encode through SAE
    def residuals(toks):
        captured = {}
        def hook(mod, ins, out):
            captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()
        h = model.backbone.layers[MID_LAYER].register_forward_hook(hook)
        # Process in chunks to fit in memory
        all_z = []
        batch_size = 32
        for i in range(0, toks.shape[0], batch_size):
            batch = toks[i:i + batch_size]
            with torch.no_grad():
                model(batch)
            res = captured["r"]
            normed = (res.float() - act_mean) / act_std
            _, z, *_ = sae(normed)
            # z shape: (B, L, d_hidden)
            all_z.append(z[:, :, top_feats].detach().cpu())
        h.remove()
        return torch.cat(all_z, dim=0)  # (n_pairs, seq_len, 10)

    print("Forward clean...")
    z_clean = residuals(clean)
    print("Forward corrupted...")
    z_corr = residuals(corr)

    # Mean and std per (feature, position)
    clean_mean = z_clean.mean(dim=0).numpy()   # (seq_len, 10)
    clean_std = z_clean.std(dim=0).numpy()
    corr_mean = z_corr.mean(dim=0).numpy()

    # Plot: per feature, activation by position, clean vs corrupted
    fig, axes = plt.subplots(5, 2, figsize=(13, 14), sharex=True)
    positions = np.arange(seq_len)
    for i, f in enumerate(top_feats):
        ax = axes[i // 2, i % 2]
        ax.plot(positions, clean_mean[:, i], color="#c03", label="clean", linewidth=1.5)
        ax.fill_between(positions,
                         clean_mean[:, i] - clean_std[:, i],
                         clean_mean[:, i] + clean_std[:, i],
                         alpha=0.2, color="#c03")
        ax.plot(positions, corr_mean[:, i], color="#888", label="corrupted", linewidth=1.0, linestyle=":")
        ax.axvspan(PREFIX_LEN, PREFIX_LEN + PATTERN_LEN, alpha=0.15, color="#36c", label="P1 (first pattern)")
        ax.axvspan(ind_start, ind_end, alpha=0.15, color="#c03", label="P2 (induction)" if i == 0 else None)
        ax.set_title(f"feat {f}")
        ax.set_ylabel("activation")
        if i >= 8:
            ax.set_xlabel("sequence position")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Top-10 induction features: activation by sequence position\n"
                 "(mean ± 1 std over 256 pairs; shaded regions mark P1 and P2)",
                  y=0.99)
    plt.tight_layout()
    fig_path = FIGS / "features_by_position.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")

    # Summary: does each feature peak at induction positions?
    summary = []
    for i, f in enumerate(top_feats):
        ind_mean = clean_mean[ind_start:ind_end, i].mean()
        elsewhere_mean = np.concatenate([clean_mean[:ind_start, i],
                                           clean_mean[ind_end:, i]]).mean()
        peak_pos = int(clean_mean[:, i].argmax())
        ratio = ind_mean / elsewhere_mean if elsewhere_mean > 1e-6 else float("inf")
        summary.append({
            "feature": int(f),
            "ind_mean": float(ind_mean),
            "non_ind_mean": float(elsewhere_mean),
            "ratio": float(ratio),
            "peak_position": peak_pos,
            "peaks_at_induction": 48 <= peak_pos < 56,
        })

    print("\n=== Per-feature peak location ===")
    for s in summary:
        print(f"  feat {s['feature']:>6d}  ind_mean={s['ind_mean']:.3f}  "
              f"elsewhere={s['non_ind_mean']:.3f}  ratio={s['ratio']:.2f}x  "
              f"peak@pos{s['peak_position']:>2d}  {'✓' if s['peaks_at_induction'] else '✗'}")

    json.dump({
        "ind_start": int(ind_start), "ind_end": int(ind_end),
        "per_feature": summary,
        "clean_mean": clean_mean.tolist(),
        "corr_mean": corr_mean.tolist(),
    }, open(RESULTS_DIR / "features_by_position.json", "w"), indent=2)
    print(f"Wrote {RESULTS_DIR / 'features_by_position.json'}")


if __name__ == "__main__":
    main()
