#!/usr/bin/env python3
"""Mamba-2 pattern-length sweep: is the weak induction signal (gap=0.22 at
plen=8) an artifact of short patterns, or intrinsic to the architecture?

Tests plen ∈ {4, 8, 16, 32} with the Mamba-2 L32 SAE. Reports baseline,
corrupted, and gap at each length.
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.activation_cache import get_model_and_tokenizer
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
CKPT_DIR = STORAGE / "checkpoints_normed"
ACTS_DIR = STORAGE / "activations"
RESULTS_DIR = STORAGE / "results_phase4"

MODEL_NAME = "state-spaces/mamba2-2.7b"
MODEL_KEY = "mamba2_2.7b"
D_MODEL = 2560
MID_LAYER = 32
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


def make_batch(tokenizer, n_pairs, plen, seed, device):
    rng = np.random.default_rng(seed)
    vocab = tokenizer.vocab_size
    seq_len = PREFIX_LEN + plen + MID_LEN + plen
    clean = np.zeros((n_pairs, seq_len), dtype=np.int64)
    corr = np.zeros_like(clean)
    ind_start = PREFIX_LEN + plen + MID_LEN
    ind_end = ind_start + plen
    for i in range(n_pairs):
        prefix = rng.integers(0, vocab, PREFIX_LEN)
        P = rng.integers(0, vocab, plen)
        mid = rng.integers(0, vocab, MID_LEN)
        while True:
            Pp = rng.integers(0, vocab, plen)
            if not np.array_equal(Pp, P):
                break
        clean[i, :PREFIX_LEN] = prefix
        clean[i, PREFIX_LEN:PREFIX_LEN + plen] = P
        clean[i, PREFIX_LEN + plen:ind_start] = mid
        clean[i, ind_start:ind_end] = P
        corr[i, :ind_start] = clean[i, :ind_start]
        corr[i, ind_start:ind_end] = Pp
    return (torch.from_numpy(clean).to(device),
            torch.from_numpy(corr).to(device),
            ind_start, ind_end)


def encode(model, tokens, sae, act_mean, act_std, mid_layer, positions):
    captured = {}
    def hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()
    h = model.backbone.layers[mid_layer].register_forward_hook(hook)
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
    ap.add_argument("--n_pairs", type=int, default=64)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4, 8, 16, 32])
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    sae, act_mean, act_std = load_sae_and_norm(device)

    out = {"model": MODEL_NAME, "lengths": args.lengths, "results": {}}
    for plen in args.lengths:
        clean, corr, ind_start, ind_end = make_batch(
            tokenizer, args.n_pairs, plen, seed=0, device=device)
        z_clean = encode(model, clean, sae, act_mean, act_std, MID_LAYER, (ind_start, ind_end))
        z_corr = encode(model, corr, sae, act_mean, act_std, MID_LAYER, (ind_start, ind_end))
        score = (z_clean - z_corr).mean(dim=0).detach()
        top_vals, top_idx = score.topk(10)
        ti = top_idx
        baseline_act = z_clean[:, ti].mean().item()
        corrupted_act = z_corr[:, ti].mean().item()
        gap = baseline_act - corrupted_act
        out["results"][str(plen)] = {
            "pattern_len": plen,
            "top_features": top_idx.cpu().tolist(),
            "top_scores": top_vals.cpu().tolist(),
            "baseline_act": baseline_act,
            "corrupted_act": corrupted_act,
            "gap": gap,
        }
        print(f"  plen={plen:>2d}  baseline={baseline_act:.4f}  corrupted={corrupted_act:.4f}  gap={gap:.4f}")

    json.dump(out, open(RESULTS_DIR / "mamba2_plen_sweep.json", "w"), indent=2)
    print(f"\nWrote {RESULTS_DIR / 'mamba2_plen_sweep.json'}")


if __name__ == "__main__":
    main()
