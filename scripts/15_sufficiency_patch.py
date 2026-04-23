#!/usr/bin/env python3
"""Sufficiency test: patch CLEAN L30 x_proj (C-slice only) INTO corrupted run.
Does induction get restored?

This complements the necessity patching: Phase B showed that removing L30 x_proj's
C output destroys 80% of induction ("necessary"). This experiment tests the
other direction: putting CLEAN L30 x_proj C back into an otherwise-corrupted
forward pass — does induction come back ("sufficient")?

If patch_rescue ~1.0, C at L30 is sufficient. If intermediate, other layers
contribute too (e.g. L28 has some signal, L32 has some).

Output: $STORAGE/results_phase4/sufficiency_patch.json
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


def encode_with_slice_patch(model, tokens, sae, act_mean, act_std, mid_layer,
                              positions, patch_value, slice_range):
    captured = {}
    def res_hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()
    s, e = slice_range
    def patch_fn(mod, ins, out):
        new_out = out.clone()
        repl_slice = patch_value[..., s:e].to(new_out.dtype)
        if repl_slice.shape[0] != new_out.shape[0]:
            if repl_slice.shape[0] == 1:
                repl_slice = repl_slice.expand(new_out.shape[0], -1, -1)
            else:
                repl_slice = repl_slice[:new_out.shape[0]]
        new_out[..., s:e] = repl_slice
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
    ap.add_argument("--n_pairs", type=int, default=128)
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

    # baseline (clean all), corrupted (corrupted all)
    captured = {}
    def res_hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()
    h = model.backbone.layers[MID_LAYER].register_forward_hook(res_hook)
    with torch.no_grad():
        model(clean)
    h.remove()
    normed = (captured["r"].float() - act_mean) / act_std
    _, z_c, *_ = sae(normed)
    baseline_act = z_c[:, ind_start:ind_end, ti].mean().item()

    captured.clear()
    h = model.backbone.layers[MID_LAYER].register_forward_hook(res_hook)
    with torch.no_grad():
        model(corr)
    h.remove()
    normed = (captured["r"].float() - act_mean) / act_std
    _, z_x, *_ = sae(normed)
    corrupted_act = z_x[:, ind_start:ind_end, ti].mean().item()
    total = baseline_act - corrupted_act
    print(f"baseline={baseline_act:.4f}, corrupted={corrupted_act:.4f}, gap={total:.4f}")

    # capture CLEAN x_proj output
    xproj_clean = capture_xproj_output(model, clean, LOCUS_LAYER)

    dt_rank = model.backbone.layers[LOCUS_LAYER].mixer.time_step_rank
    state_size = model.backbone.layers[LOCUS_LAYER].mixer.ssm_state_size

    results = {}
    for name, sl in {
        "full_xproj": (0, dt_rank + 2 * state_size),
        "C_matrix":   (dt_rank + state_size, dt_rank + 2 * state_size),
        "B_and_C":    (dt_rank, dt_rank + 2 * state_size),
    }.items():
        z_rescued = encode_with_slice_patch(
            model, corr, sae, act_mean, act_std, MID_LAYER,
            (ind_start, ind_end), xproj_clean, sl,
        )
        rescued_act = z_rescued[:, ti].mean().item()
        # Sufficiency: how much of the gap got restored?
        rescue = (rescued_act - corrupted_act) / total if abs(total) > 1e-8 else 0.0
        results[name] = {"slice": list(sl), "rescued_act": rescued_act,
                         "rescue_fraction": rescue}
        print(f"  RESCUE ({name:<12s}  slice[{sl[0]}:{sl[1]}])  rescued={rescued_act:.4f}  "
              f"fraction={rescue:+.4f}")

    out = {
        "baseline_act": baseline_act,
        "corrupted_act": corrupted_act,
        "gap": total,
        "n_pairs": args.n_pairs,
        "sufficiency": results,
    }
    out_path = RESULTS_DIR / "sufficiency_patch.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
