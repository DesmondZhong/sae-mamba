#!/usr/bin/env python3
"""Residual-stream sufficiency: patch clean post-LN residual stream at layer K
into the corrupted run. Does induction get restored?

This decomposes "what's needed for induction" between:
  - a consistent state built by layers 0..K  (patched via residual)
  - the downstream mechanism layers K..N applying itself

If patching at K=29 restores induction (high rescue fraction), the induction
machinery at L30 is SUFFICIENT given the right state.
If K=32 is needed for full rescue, induction work continues past L30.

Output: $STORAGE/results_phase4/residual_sufficiency.json
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

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
CKPT_DIR = STORAGE / "checkpoints_normed"
ACTS_DIR = STORAGE / "activations"
RESULTS_DIR = STORAGE / "results_phase4"

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


def capture_residual(model, tokens, layer):
    captured = {}
    def hook(mod, ins, out):
        captured["r"] = (out[0] if isinstance(out, tuple) else out).detach().clone()
    h = model.backbone.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["r"]


def encode_with_residual_patch(model, tokens, sae, act_mean, act_std,
                                 patch_layer, patch_value, mid_layer, positions,
                                 patch_region=None):
    """Run forward; replace layer `patch_layer`'s output residual stream with
    patch_value (optionally only at `patch_region` positions).

    Note: HF Mamba layer returns (hidden, residual) tuple. We need to replace
    the FIRST element of the tuple (output of that block).
    """
    captured = {}
    def res_hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()

    def patch_fn(mod, ins, out):
        # out is the output of backbone.layers[patch_layer], shape (B, L, d_model)
        # OR a tuple. HF Mamba returns a single Tensor (residual stream).
        if isinstance(out, tuple):
            hs = out[0]
            new_hs = patch_value if patch_region is None else hs.clone()
            if patch_region is not None:
                s, e = patch_region
                new_hs[:, s:e] = patch_value[:, s:e].to(new_hs.dtype)
            return (new_hs,) + out[1:]
        else:
            if patch_region is None:
                return patch_value.to(out.dtype)
            new_out = out.clone()
            s, e = patch_region
            new_out[:, s:e] = patch_value[:, s:e].to(new_out.dtype)
            return new_out

    h_res = model.backbone.layers[mid_layer].register_forward_hook(res_hook)
    h_patch = model.backbone.layers[patch_layer].register_forward_hook(patch_fn)
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
    ap.add_argument("--n_pairs", type=int, default=128)
    ap.add_argument("--layers", type=int, nargs="+", default=[16, 24, 28, 29, 30, 31])
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    force_slow_forward(model)
    sae, act_mean, act_std = load_sae_and_norm(device)
    ind = json.load(open(RESULTS_DIR / "induction_features.json"))
    top_feats = ind["feature"][:10]
    ti = torch.tensor(top_feats, device=device)

    clean, corr, ind_start, ind_end = make_induction_batch(
        tokenizer, args.n_pairs, seed=0, device=device)

    # Baseline / corrupted
    res_clean_mid = capture_residual(model, clean, MID_LAYER)
    res_corr_mid = capture_residual(model, corr, MID_LAYER)
    z_clean = (res_clean_mid.float() - act_mean) / act_std
    z_corr = (res_corr_mid.float() - act_mean) / act_std
    _, zc, *_ = sae(z_clean); _, zx, *_ = sae(z_corr)
    baseline_act = zc[:, ind_start:ind_end, ti].mean().item()
    corrupted_act = zx[:, ind_start:ind_end, ti].mean().item()
    gap = baseline_act - corrupted_act
    print(f"baseline={baseline_act:.4f}, corrupted={corrupted_act:.4f}, gap={gap:.4f}")

    results = {}
    for L in args.layers:
        print(f"\n--- Patching CLEAN residual at L{L} into corrupted run ---")
        # Capture CLEAN residual at L
        clean_res_L = capture_residual(model, clean, L)
        # Rescue: full positions
        z_rescued = encode_with_residual_patch(
            model, corr, sae, act_mean, act_std,
            L, clean_res_L, MID_LAYER, (ind_start, ind_end))
        rescued_act = z_rescued[:, ti].mean().item()
        rescue = (rescued_act - corrupted_act) / gap if abs(gap) > 1e-8 else 0.0

        # Rescue: only ind positions
        z_rescued_ind = encode_with_residual_patch(
            model, corr, sae, act_mean, act_std,
            L, clean_res_L, MID_LAYER, (ind_start, ind_end),
            patch_region=(ind_start, ind_end))
        rescued_act_ind = z_rescued_ind[:, ti].mean().item()
        rescue_ind = (rescued_act_ind - corrupted_act) / gap if abs(gap) > 1e-8 else 0.0

        results[f"L{L}"] = {
            "layer": L,
            "rescue_all_positions": rescue,
            "rescue_ind_positions_only": rescue_ind,
            "rescued_all_act": rescued_act,
            "rescued_ind_act": rescued_act_ind,
        }
        print(f"  rescue (all pos): {rescue:+.4f}")
        print(f"  rescue (ind-only positions): {rescue_ind:+.4f}")

    out = {
        "baseline_act": baseline_act,
        "corrupted_act": corrupted_act,
        "gap": gap,
        "n_pairs": args.n_pairs,
        "layers_tested": args.layers,
        "results": results,
    }
    json.dump(out, open(RESULTS_DIR / "residual_sufficiency.json", "w"), indent=2)
    print(f"\nWrote {RESULTS_DIR / 'residual_sufficiency.json'}")


if __name__ == "__main__":
    main()
