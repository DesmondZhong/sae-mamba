#!/usr/bin/env python3
"""Train a TopK SAE on L30 x_proj INPUT activations (d_in=5120), then identify
induction-specific features via clean-vs-corrupted contrast.

Depends on scripts/07_extract_xproj.py having written
  $STORAGE/activations/mamba1_2.8b/layer_30_xproj_in.fp16.npy
and an induction-pair generator (replicated from scripts/04_induction_circuit.py).

Outputs:
  $STORAGE/checkpoints_normed/mamba1_2.8b_L30_xprojin_x8_k64_normed.pt
  $STORAGE/results_phase4/xproj_internal_sae_induction_features.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.train_sae import train_sae
from src.sae import create_sae
from src.activation_cache import get_model_and_tokenizer

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
ACTS_DIR = STORAGE / "activations" / "mamba1_2.8b"
CKPT_DIR = STORAGE / "checkpoints_normed"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = STORAGE / "results_phase4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "state-spaces/mamba-2.8b-hf"
LAYER = int(os.environ.get("XPROJ_LAYER", "30"))
D_IN = 5120
EXPANSION = 8
K = 64


# --- Replicated from 04_induction_circuit.py for feature-id step ---

N_VOCAB_SAFE = 50277  # pythia/mamba tokenizer vocab size (not used here but safe)
PATTERN_LEN = 8
PREFIX_LEN = 8
MID_LEN = 32


def make_induction_batch(tokenizer, n_pairs=64, pattern_len=PATTERN_LEN,
                         prefix_len=PREFIX_LEN, mid_len=MID_LEN,
                         seed=0, device="cuda:0"):
    """Build clean + corrupted batches. Seq structure:
        [PREFIX][P][MID][P]        (clean)
        [PREFIX][P][MID][P']       (corrupted, P' different from P)
    """
    rng = np.random.default_rng(seed)
    vocab = tokenizer.vocab_size
    clean = np.zeros((n_pairs, prefix_len + pattern_len + mid_len + pattern_len), dtype=np.int64)
    corr = np.zeros_like(clean)
    ind_start = prefix_len + pattern_len + mid_len  # inclusive
    ind_end = ind_start + pattern_len  # exclusive
    for i in range(n_pairs):
        prefix = rng.integers(0, vocab, prefix_len)
        P = rng.integers(0, vocab, pattern_len)
        mid = rng.integers(0, vocab, mid_len)
        # P' must differ; resample until different
        while True:
            Pp = rng.integers(0, vocab, pattern_len)
            if not np.array_equal(Pp, P):
                break
        clean[i, :prefix_len] = prefix
        clean[i, prefix_len:prefix_len + pattern_len] = P
        clean[i, prefix_len + pattern_len:ind_start] = mid
        clean[i, ind_start:ind_end] = P
        corr[i, :ind_start] = clean[i, :ind_start]
        corr[i, ind_start:ind_end] = Pp
    return (torch.from_numpy(clean).to(device),
            torch.from_numpy(corr).to(device),
            ind_start, ind_end)


def capture_xproj_input(model, tokens, layer):
    captured = {}
    def pre_hook(mod, ins):
        captured["x"] = ins[0].detach().clone()
    h = model.backbone.layers[layer].mixer.x_proj.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["x"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_steps", type=int, default=30000)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train", action="store_true", default=True)
    ap.add_argument("--analyze", action="store_true", default=True)
    args = ap.parse_args()

    in_path = ACTS_DIR / f"layer_{LAYER}_xproj_in.fp16.npy"
    meta_path = ACTS_DIR / f"layer_{LAYER}_xproj_meta.json"
    if not in_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing {in_path} or {meta_path} — run 07_extract_xproj.py first.")

    meta = json.load(open(meta_path))
    n_tokens, d_in = meta["n_tokens_written"], meta["d_in"]
    assert d_in == D_IN, f"Expected d_in={D_IN}, got {d_in}"
    d_hidden = D_IN * EXPANSION

    run_key = f"mamba1_2.8b_L{LAYER}_xprojin_x{EXPANSION}_k{K}_normed"
    ckpt_path = CKPT_DIR / f"{run_key}.pt"

    if args.train and not ckpt_path.exists():
        print(f"[Train] {run_key}  (d_in={D_IN}, d_hidden={d_hidden}, K={K}, tokens={n_tokens:,})")
        # Memmap as a torch tensor (fp16 backing; upcast in train_sae batch loop)
        arr = np.memmap(in_path, dtype="float16", mode="r", shape=(n_tokens, D_IN))
        activations = torch.from_numpy(arr)  # shares memory with mmap
        # Compute normalization stats from 500K sample
        sample = activations[:500_000].float()
        act_mean = sample.mean(dim=0, keepdim=True)
        act_std = sample.std(dim=0, keepdim=True).clamp(min=1e-6)
        del sample

        start = time.time()
        sae, history, summary = train_sae(
            activations, d_hidden, sae_type="topk", k=K,
            n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr,
            device=args.device, save_path=str(ckpt_path),
            act_mean=act_mean, act_std=act_std,
        )
        elapsed = time.time() - start
        print(f"[Done] FVE={summary['final_fve']:.4f} L0={summary['final_l0']:.1f} "
              f"dead={summary['final_dead_features']} ({elapsed/60:.1f}min)")
        # Save norm stats alongside ckpt
        torch.save({"act_mean": act_mean, "act_std": act_std},
                   CKPT_DIR / f"{run_key}_normparams.pt")
        del sae, activations
        import gc; gc.collect()
        torch.cuda.empty_cache()

    if args.analyze:
        # Reload
        print("Reloading SAE + model for induction-feature ID...")
        sae = create_sae(D_IN, d_hidden, sae_type="topk", k=K).to(args.device)
        sae.load_state_dict(torch.load(ckpt_path, map_location=args.device, weights_only=True))
        sae.eval()
        np_params = torch.load(CKPT_DIR / f"{run_key}_normparams.pt",
                               map_location=args.device, weights_only=True)
        act_mean = np_params["act_mean"].to(args.device)
        act_std = np_params["act_std"].to(args.device)

        model, tokenizer = get_model_and_tokenizer(MODEL_NAME, args.device)

        # Generate induction pairs and score features
        all_clean_acts = []
        all_corr_acts = []
        n_total_pairs = 0
        for seed in range(16):  # 16 × 64 = 1024 pairs
            clean, corr, ind_start, ind_end = make_induction_batch(
                tokenizer, n_pairs=64, seed=seed, device=args.device)
            xp_clean = capture_xproj_input(model, clean, LAYER)  # (B, L, 5120)
            xp_corr = capture_xproj_input(model, corr, LAYER)
            # Normalize + encode through SAE
            normed_c = (xp_clean.float() - act_mean) / act_std
            normed_d = (xp_corr.float() - act_mean) / act_std
            # Just get z (activations) — use the SAE encoder path
            with torch.no_grad():
                _, z_clean, *_ = sae(normed_c.reshape(-1, D_IN))
                _, z_corr, *_ = sae(normed_d.reshape(-1, D_IN))
            # Reshape back to (B, L, d_hidden) and take induction positions
            B, L = clean.shape
            z_clean = z_clean.reshape(B, L, d_hidden)
            z_corr = z_corr.reshape(B, L, d_hidden)
            # Average over induction positions (ind_start..ind_end)
            clean_ind = z_clean[:, ind_start:ind_end].mean(dim=1)  # (B, d_hidden)
            corr_ind = z_corr[:, ind_start:ind_end].mean(dim=1)
            all_clean_acts.append(clean_ind.cpu())
            all_corr_acts.append(corr_ind.cpu())
            n_total_pairs += B

        all_clean = torch.cat(all_clean_acts, dim=0)
        all_corr = torch.cat(all_corr_acts, dim=0)
        # Score = mean(clean - corrupted) per feature
        score = (all_clean - all_corr).mean(dim=0)  # (d_hidden,)
        # Top features
        top_k = 10
        top_vals, top_idx = score.topk(top_k)
        # Also report contrast / mean-clean ratio for each
        result = {
            "layer": LAYER,
            "model": MODEL_NAME,
            "site": "x_proj input",
            "d_in": D_IN,
            "d_hidden": d_hidden,
            "k": K,
            "n_pairs": n_total_pairs,
            "top_features": {
                "feature": top_idx.tolist(),
                "score": top_vals.tolist(),
                "clean_mean": all_clean[:, top_idx].mean(dim=0).tolist(),
                "corr_mean": all_corr[:, top_idx].mean(dim=0).tolist(),
            },
            "n_features_strongly_induction": int(((score > 0.5)).sum().item()),
            "max_score": float(score.max().item()),
            "median_score": float(score.median().item()),
            "fraction_pos_score": float((score > 0).float().mean().item()),
        }
        out_path = RESULTS_DIR / f"xproj_L{LAYER}_internal_sae_induction_features.json"
        json.dump(result, open(out_path, "w"), indent=2)
        print(f"Wrote {out_path}")
        print(f"Top induction features in L30 x_proj internal SAE:")
        for f, s, cm, dm in zip(top_idx.tolist(), top_vals.tolist(),
                                all_clean[:, top_idx].mean(dim=0).tolist(),
                                all_corr[:, top_idx].mean(dim=0).tolist()):
            print(f"  feat {f:>6d}  score={s:+.4f}  clean={cm:.4f}  corr={dm:.4f}")


if __name__ == "__main__":
    main()
