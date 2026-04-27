#!/usr/bin/env python3
"""Two-pattern stress test.

Tests whether the Mamba L30 C locus handles multiple induction patterns in
the same sequence. Construct:

  clean:     [prefix 8] P1(8) [mid1 16] P2(8) [mid2 16] P1(8) [mid3 16] P2(8)
  corrupted: [prefix 8] P1(8) [mid1 16] P2(8) [mid2 16] P1'(8) [mid3 16] P2'(8)

Two induction targets: positions 48-55 (P1 2nd occurrence) and 80-87 (P2 2nd).

Measure patch_damage at L30 C for:
  (i)  Using the original top-10 induction features from the standard (single
       pattern) sweep. How well do they generalize to two-pattern sequences?
  (ii) Damage on 2nd P1 positions vs 2nd P2 positions separately.

If L30 C is a general induction mechanism, it should carry both patterns'
signal; if it's specific to one induction event, it might not generalize.

Output: $STORAGE/results_phase4/two_pattern.json
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

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


def make_two_pattern_batch(tokenizer, n_pairs, seed, device):
    """
    Positions:
      0-7:   prefix (random)
      8-15:  P1 (first pattern)
      16-31: mid1 (random)
      32-39: P2 (first occurrence of second pattern)
      40-55: mid2 (random)
      56-63: P1' (corrupted: different pattern;  clean: same as P1)  *** wait, single P1 match at 56-63 not 48-55 ***
    Actually let me redesign:
      0-7:   prefix
      8-15:  P1 (8 tokens)
      16-31: mid1 (16 tokens)
      32-39: P2 (8 tokens, different pattern)
      40-55: mid2 (16 tokens)
      56-63: P1 again (second occurrence of P1) — induction target 1
      64-79: mid3 (16 tokens)
      80-87: P2 again (second occurrence of P2) — induction target 2
    Total length: 88 tokens.

    corrupted: replace positions 56-63 with P1' and 80-87 with P2'.
    """
    rng = np.random.default_rng(seed)
    vocab = tokenizer.vocab_size
    seq_len = 88
    clean = np.zeros((n_pairs, seq_len), dtype=np.int64)
    corr = np.zeros_like(clean)
    ind1_start, ind1_end = 56, 64  # positions 56-63
    ind2_start, ind2_end = 80, 88  # positions 80-87
    for i in range(n_pairs):
        prefix = rng.integers(0, vocab, 8)
        P1 = rng.integers(0, vocab, PATTERN_LEN)
        P2 = rng.integers(0, vocab, PATTERN_LEN)
        mid1 = rng.integers(0, vocab, 16)
        mid2 = rng.integers(0, vocab, 16)
        mid3 = rng.integers(0, vocab, 16)
        while True:
            P1p = rng.integers(0, vocab, PATTERN_LEN)
            if not np.array_equal(P1p, P1): break
        while True:
            P2p = rng.integers(0, vocab, PATTERN_LEN)
            if not np.array_equal(P2p, P2): break
        clean[i, 0:8] = prefix
        clean[i, 8:16] = P1
        clean[i, 16:32] = mid1
        clean[i, 32:40] = P2
        clean[i, 40:56] = mid2
        clean[i, ind1_start:ind1_end] = P1       # P1 match (induction 1)
        clean[i, 64:80] = mid3
        clean[i, ind2_start:ind2_end] = P2       # P2 match (induction 2)
        corr[i, :ind1_start] = clean[i, :ind1_start]
        corr[i, ind1_start:ind1_end] = P1p       # corrupt: P1'
        corr[i, 64:80] = mid3
        corr[i, ind2_start:ind2_end] = P2p       # corrupt: P2'
    return (torch.from_numpy(clean).to(device),
            torch.from_numpy(corr).to(device),
            (ind1_start, ind1_end), (ind2_start, ind2_end))


def capture_xproj(model, tokens, layer):
    captured = {}
    def hook(mod, ins, out):
        captured["o"] = out.detach().clone()
    h = model.backbone.layers[layer].mixer.x_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["o"]


def encode_at_positions(model, tokens, sae, act_mean, act_std, mid_layer, positions,
                         patch_value=None, slice_range=None):
    captured = {}
    def res_hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()
    hooks = [model.backbone.layers[mid_layer].register_forward_hook(res_hook)]
    if patch_value is not None:
        s, e = slice_range
        def patch_fn(mod, ins, out):
            new_out = out.clone()
            repl = patch_value[..., s:e].to(new_out.dtype)
            if repl.shape[0] != new_out.shape[0]:
                if repl.shape[0] == 1:
                    repl = repl.expand(new_out.shape[0], -1, -1)
                else:
                    repl = repl[:new_out.shape[0]]
            new_out[..., s:e] = repl
            return new_out
        hooks.append(model.backbone.layers[LOCUS_LAYER].mixer.x_proj.register_forward_hook(patch_fn))
    with torch.no_grad():
        model(tokens)
    for h in hooks:
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
    sae, act_mean, act_std = load_sae_and_norm(device)

    # Original single-pattern induction features from Phase B
    ind = json.load(open(RESULTS_DIR / "induction_features.json"))
    top_feats_single = torch.tensor(ind["feature"][:10], device=device)

    clean, corr, ind1, ind2 = make_two_pattern_batch(tokenizer, args.n_pairs, 0, device)
    print(f"Two-pattern seq length: {clean.shape[1]}")
    print(f"Induction target 1 (P1 2nd): positions {ind1}")
    print(f"Induction target 2 (P2 2nd): positions {ind2}")

    # Measure at each induction site
    mixer = model.backbone.layers[LOCUS_LAYER].mixer
    dt_rank = mixer.time_step_rank
    state_size = mixer.ssm_state_size
    C_slice = (dt_rank + state_size, dt_rank + 2 * state_size)
    full_slice = (0, dt_rank + 2 * state_size)

    xp_corr = capture_xproj(model, corr, LOCUS_LAYER)

    results = {}
    for label, positions in [("ind1_P1_second", ind1), ("ind2_P2_second", ind2)]:
        # Baseline, corrupted, and patched (C slice) at this induction target
        z_clean = encode_at_positions(model, clean, sae, act_mean, act_std,
                                        MID_LAYER, positions)
        z_corr = encode_at_positions(model, corr, sae, act_mean, act_std,
                                       MID_LAYER, positions)
        baseline = z_clean[:, top_feats_single].mean().item()
        corrupted = z_corr[:, top_feats_single].mean().item()
        gap = baseline - corrupted
        if abs(gap) < 1e-4:
            print(f"  WARNING: gap near zero at {label} — features may not fire here")

        # Patch L30 x_proj C at all positions (global patch)
        z_patched_full = encode_at_positions(
            model, clean, sae, act_mean, act_std, MID_LAYER, positions,
            patch_value=xp_corr, slice_range=full_slice)
        z_patched_C = encode_at_positions(
            model, clean, sae, act_mean, act_std, MID_LAYER, positions,
            patch_value=xp_corr, slice_range=C_slice)

        patched_full = z_patched_full[:, top_feats_single].mean().item()
        patched_C = z_patched_C[:, top_feats_single].mean().item()
        damage_full = 1.0 - (patched_full - corrupted) / gap if abs(gap) > 1e-8 else 0.0
        damage_C = 1.0 - (patched_C - corrupted) / gap if abs(gap) > 1e-8 else 0.0

        results[label] = {
            "positions": list(positions),
            "baseline": baseline,
            "corrupted": corrupted,
            "gap": gap,
            "patch_damage_full_xproj": damage_full,
            "patch_damage_C_matrix": damage_C,
        }
        print(f"\n  {label}: baseline={baseline:.3f}, corrupted={corrupted:.3f}, gap={gap:.3f}")
        print(f"    full x_proj patch damage: {damage_full:+.4f}")
        print(f"    C slice patch damage:     {damage_C:+.4f}")

    out = {
        "n_pairs": args.n_pairs,
        "results_per_target": results,
    }
    json.dump(out, open(RESULTS_DIR / "two_pattern.json", "w"), indent=2)
    print(f"\nWrote {RESULTS_DIR / 'two_pattern.json'}")


if __name__ == "__main__":
    main()
