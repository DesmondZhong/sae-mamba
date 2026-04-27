#!/usr/bin/env python3
"""Slice-wise patching of L30 x_proj output in Mamba-1.

x_proj output of shape (B, L, 192) is the concatenation of:
  [0:160]      = time_step (dt_rank = 160), feeds dt_proj → discrete_time_step Δ
  [160:176]    = B                (state_size = 16)
  [176:192]    = C                (state_size = 16)

We patch ONLY one of {Δ_pre, B, C} from the corrupted run into the clean run,
and measure patch_damage separately. This tests whether the induction signal
lives in the time-step generation or in the state-space matrices themselves.

Sanity tie-ins:
- Full x_proj patch damages induction by 0.833 (Phase B).
- dt_proj patch (Δ after projection) damages by 0.00 (Phase B).
- So the signal must be in B or C, not Δ. This script quantifies that
  directly by slicing x_proj output at the pre-dt_proj level.

Output: $STORAGE/results_phase4/xproj_slice_patching.json
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
MID_LAYER = 32      # where we read features from
LOCUS_LAYER = 30    # where we patch x_proj
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


def capture_xproj_output(model, tokens, layer):
    captured = {}
    def hook(mod, ins, out):
        captured["o"] = out.detach().clone()
    h = model.backbone.layers[layer].mixer.x_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["o"]


def encode_residual_at(model, tokens, sae, act_mean, act_std, layer, positions,
                        slice_patch=None):
    """Optionally install a slice patch on LOCUS_LAYER's x_proj during the
    forward pass.

    slice_patch: None OR dict {"slice": (start, end), "value": tensor_of_corr_out}
    """
    captured = {}
    def residual_hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()

    patch_hook = None
    hooks = []
    h_res = model.backbone.layers[layer].register_forward_hook(residual_hook)
    hooks.append(h_res)

    if slice_patch is not None:
        s, e = slice_patch["slice"]
        repl = slice_patch["value"].to(next(model.parameters()).device)
        def patch_fn(mod, ins, out):
            # out: (B, L, 192). Replace only [..., s:e] with replacement's [..., s:e]
            new_out = out.clone()
            # shape-align: if replacement is 3d (B, L, 192), take the same slice
            repl_slice = repl[..., s:e].to(new_out.dtype)
            # If batch sizes differ (corrupt captured with different batch shape)
            if repl_slice.shape[0] != new_out.shape[0]:
                # broadcast batch-0 or tile
                if repl_slice.shape[0] == 1:
                    repl_slice = repl_slice.expand(new_out.shape[0], -1, -1)
                else:
                    repl_slice = repl_slice[:new_out.shape[0]]
            new_out[..., s:e] = repl_slice
            return new_out
        h_patch = model.backbone.layers[LOCUS_LAYER].mixer.x_proj.register_forward_hook(patch_fn)
        hooks.append(h_patch)

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

    print("Loading L32 SAE + norm stats...")
    sae, act_mean, act_std = load_sae_and_norm(device)

    ind = json.load(open(RESULTS_DIR / "induction_features.json"))
    top_feats = ind["feature"][:10]
    ti = torch.tensor(top_feats, device=device)
    print(f"Using top-10 induction features: {top_feats}")

    # Get x_proj dims
    mixer = model.backbone.layers[LOCUS_LAYER].mixer
    dt_rank = mixer.time_step_rank  # 160
    state_size = mixer.ssm_state_size  # 16
    x_out_dim = mixer.x_proj.out_features
    print(f"L{LOCUS_LAYER} x_proj: dt_rank={dt_rank}, state_size={state_size}, "
          f"total_out={x_out_dim}")
    assert x_out_dim == dt_rank + 2 * state_size

    slices = {
        "full_xproj":   (0, x_out_dim),                            # sanity: should match Phase B = 0.833
        "delta_pre":    (0, dt_rank),                              # time_step (pre-dt_proj)
        "B_matrix":     (dt_rank, dt_rank + state_size),
        "C_matrix":     (dt_rank + state_size, x_out_dim),
        "B_and_C":      (dt_rank, x_out_dim),                      # both state-space matrices
    }

    clean, corr, ind_start, ind_end = make_induction_batch(
        tokenizer, n_pairs=args.n_pairs, seed=0, device=device)

    # Baseline / corrupted: no patch
    z_clean = encode_residual_at(model, clean, sae, act_mean, act_std,
                                  MID_LAYER, (ind_start, ind_end))
    z_corr = encode_residual_at(model, corr, sae, act_mean, act_std,
                                 MID_LAYER, (ind_start, ind_end))
    baseline_act = z_clean[:, ti].mean().item()
    corrupted_act = z_corr[:, ti].mean().item()
    total = baseline_act - corrupted_act
    print(f"\nbaseline={baseline_act:.4f}, corrupted={corrupted_act:.4f}, total={total:.4f}")

    # Capture corrupted x_proj output at L30 for patching
    xproj_corr = capture_xproj_output(model, corr, LOCUS_LAYER)
    print(f"captured corrupted x_proj output shape: {tuple(xproj_corr.shape)}")

    results = {}
    for name, (s, e) in slices.items():
        z_patched = encode_residual_at(
            model, clean, sae, act_mean, act_std,
            MID_LAYER, (ind_start, ind_end),
            slice_patch={"slice": (s, e), "value": xproj_corr},
        )
        patched_act = z_patched[:, ti].mean().item()
        patch_damage = 1.0 - (patched_act - corrupted_act) / total if abs(total) > 1e-8 else 0.0
        results[name] = {
            "slice": [s, e],
            "dim": e - s,
            "baseline_act": baseline_act,
            "corrupted_act": corrupted_act,
            "patched_act": patched_act,
            "patch_damage": patch_damage,
        }
        print(f"  {name:<12s} slice[{s:>3d}:{e:>3d}] dim={e-s:>3d}  "
              f"patched={patched_act:.4f}  patch_damage={patch_damage:+.4f}")

    out_path = RESULTS_DIR / "xproj_slice_patching.json"
    json.dump({
        "layer": LOCUS_LAYER,
        "reading_layer": MID_LAYER,
        "dt_rank": dt_rank,
        "state_size": state_size,
        "n_pairs": args.n_pairs,
        "baseline_act": baseline_act,
        "corrupted_act": corrupted_act,
        "results": results,
    }, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
