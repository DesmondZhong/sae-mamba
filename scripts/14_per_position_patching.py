#!/usr/bin/env python3
"""Per-position patching of L30 x_proj output — high resolution across the sequence.

The existing Phase-B position-specific test groups positions into three bins
(all / induction-only / pre-induction-only). This script patches x_proj output
at EACH sequence position individually, giving a per-position attribution map.

Sequence layout (len 56):
  [0:8]    prefix
  [8:16]   first pattern P
  [16:48]  mid filler (32 tokens)
  [48:56]  second pattern P  (induction target)

Expected pattern, if Mamba is doing induction via L30 x_proj:
  - High patch_damage at positions 48-55 (the current induction positions)
  - Notable signal at positions 8-15 (where the prior pattern was encoded in state)
  - ~0 signal at positions 0-7 (prefix) and 16-47 (mid filler)

Output: $STORAGE/results_phase4/per_position_patching.json
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
from src.mamba_internals import force_slow_forward
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
            ind_start, ind_end, seq_len)


def capture_xproj_output(model, tokens, layer):
    captured = {}
    def hook(mod, ins, out):
        captured["o"] = out.detach().clone()
    h = model.backbone.layers[layer].mixer.x_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["o"]


def encode_at_positions_with_patch(model, tokens, sae, act_mean, act_std,
                                     mid_layer, positions, patch_pos, patch_value):
    """Forward tokens; at LOCUS_LAYER x_proj, replace output at position `patch_pos`
    (across all samples) with patch_value[..., patch_pos, :]."""
    captured = {}
    def res_hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()

    def patch_fn(mod, ins, out):
        new_out = out.clone()
        # out: (B, L, 192). Replace only position patch_pos
        repl = patch_value[:, patch_pos:patch_pos + 1, :].to(new_out.dtype)
        if repl.shape[0] != new_out.shape[0]:
            if repl.shape[0] == 1:
                repl = repl.expand(new_out.shape[0], -1, -1)
            else:
                repl = repl[:new_out.shape[0]]
        new_out[:, patch_pos:patch_pos + 1, :] = repl
        return new_out

    h_res = model.backbone.layers[mid_layer].register_forward_hook(res_hook)
    h_patch = model.backbone.layers[LOCUS_LAYER].mixer.x_proj.register_forward_hook(patch_fn)
    with torch.no_grad():
        model(tokens)
    h_res.remove(); h_patch.remove()
    res = captured["r"]
    normed = (res.float() - act_mean) / act_std
    _, z, *_ = sae(normed)
    return z[:, positions[0]:positions[1]].mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_pairs", type=int, default=64)
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    force_slow_forward(model)
    print("Loading L32 SAE + norm stats...")
    sae, act_mean, act_std = load_sae_and_norm(device)

    ind = json.load(open(RESULTS_DIR / "induction_features.json"))
    top_feats = ind["feature"][:10]
    ti = torch.tensor(top_feats, device=device)
    print(f"Using top-10 induction features: {top_feats}")

    clean, corr, ind_start, ind_end, seq_len = make_induction_batch(
        tokenizer, args.n_pairs, seed=0, device=device)
    print(f"seq_len={seq_len}, induction positions {ind_start}..{ind_end}")

    # Capture corrupted x_proj output
    xproj_corr = capture_xproj_output(model, corr, LOCUS_LAYER)

    # Baseline / corrupted target feature activation
    captured = {}
    def res_hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()
    h = model.backbone.layers[MID_LAYER].register_forward_hook(res_hook)
    with torch.no_grad():
        model(clean)
    h.remove()
    normed = (captured["r"].float() - act_mean) / act_std
    _, z_clean, *_ = sae(normed)

    captured.clear()
    h = model.backbone.layers[MID_LAYER].register_forward_hook(res_hook)
    with torch.no_grad():
        model(corr)
    h.remove()
    normed = (captured["r"].float() - act_mean) / act_std
    _, z_corr, *_ = sae(normed)

    baseline_act = z_clean[:, ind_start:ind_end, ti].mean().item()
    corrupted_act = z_corr[:, ind_start:ind_end, ti].mean().item()
    total = baseline_act - corrupted_act
    print(f"baseline={baseline_act:.4f}, corrupted={corrupted_act:.4f}")

    # For each position in 0..seq_len-1, patch corrupted→clean at that position
    per_position = []
    for p in tqdm(range(seq_len), desc="per-position patch"):
        z_patched_ind = encode_at_positions_with_patch(
            model, clean, sae, act_mean, act_std, MID_LAYER,
            (ind_start, ind_end), p, xproj_corr)
        patched_act = z_patched_ind[:, ti].mean().item()
        patch_damage = 1.0 - (patched_act - corrupted_act) / total if abs(total) > 1e-8 else 0.0
        per_position.append({"position": p, "patched_act": patched_act,
                             "patch_damage": patch_damage})

    # Summarize by region
    def region_stats(start, end):
        rows = [r for r in per_position if start <= r["position"] < end]
        if not rows:
            return {}
        d = [r["patch_damage"] for r in rows]
        return {"start": start, "end": end, "n": len(rows),
                "mean": float(np.mean(d)), "max": float(np.max(d)), "sum": float(np.sum(d))}

    regions = {
        "prefix": region_stats(0, PREFIX_LEN),
        "P1_first_pattern": region_stats(PREFIX_LEN, PREFIX_LEN + PATTERN_LEN),
        "mid_filler": region_stats(PREFIX_LEN + PATTERN_LEN, ind_start),
        "P2_induction": region_stats(ind_start, ind_end),
    }

    out = {
        "layer": LOCUS_LAYER,
        "reading_layer": MID_LAYER,
        "sequence_layout": {
            "prefix": [0, PREFIX_LEN],
            "P1_first_pattern": [PREFIX_LEN, PREFIX_LEN + PATTERN_LEN],
            "mid_filler": [PREFIX_LEN + PATTERN_LEN, ind_start],
            "P2_induction": [ind_start, ind_end],
        },
        "n_pairs": args.n_pairs,
        "baseline_act": baseline_act,
        "corrupted_act": corrupted_act,
        "per_position": per_position,
        "region_stats": regions,
    }
    out_path = RESULTS_DIR / "per_position_patching.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")
    print("=== Region summary ===")
    for name, r in regions.items():
        if r:
            print(f"  {name:<22s} positions [{r['start']}, {r['end']}): "
                  f"n={r['n']}, mean_damage={r['mean']:+.4f}, max={r['max']:+.4f}")


if __name__ == "__main__":
    main()
