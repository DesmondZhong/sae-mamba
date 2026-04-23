#!/usr/bin/env python3
"""State-level patching: capture and replace the SSM hidden state `h` inside
L30's selective scan.

The prior slice-level sufficiency test (§7) showed that patching clean C
into an otherwise-corrupted run rescues only 5.6% of induction — because C
multiplies the hidden state h, and h is wrong in the corrupted run. This
script tests the complementary sufficiency claim: if we patch in the CLEAN
hidden state h at L30 (everything else corrupted), is induction restored?

Approach:
  Monkey-patch MambaMixer.slow_forward with a version that exposes h during
  the recurrence, via `self._state_capture` (list to fill) or
  `self._state_replace` (dict of {position: state_tensor} to substitute).

Output: $STORAGE/results_phase4/state_patching.json
"""
import argparse, json, os, sys, types, inspect
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def state_aware_slow_forward(self, input_states, cache_params=None,
                              cache_position=None, attention_mask=None):
    """Reimplementation of MambaMixer.slow_forward with state capture/replace
    hooks via `self._state_capture` (defaultdict-style list) and
    `self._state_replace` (dict of {position: tensor}).
    """
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(1, 2)
    hidden_states, gate = projected_states.chunk(2, dim=1)
    # 2. Convolution sequence transformation
    hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
    # 3.a. x_proj input projection
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters,
        [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
    discrete_time_step = self.dt_proj(time_step)
    discrete_time_step = F.softplus(discrete_time_step).transpose(1, 2)
    # 3.b. Discretization
    A = -torch.exp(self.A_log.float())
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
    # 3.c. Recurrence with STATE HOOKS
    scan_outputs = []
    ssm_state = torch.zeros(
        (batch_size, self.intermediate_size, self.ssm_state_size),
        device=hidden_states.device, dtype=hidden_states.dtype)
    cap = getattr(self, "_state_capture", None)
    rep = getattr(self, "_state_replace", None)
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
        # State hooks: capture after recurrence, before readout
        if cap is not None:
            cap.append(ssm_state.detach().clone())
        if rep is not None and i in rep:
            ssm_state = rep[i].to(ssm_state.dtype).to(ssm_state.device)
        scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
        scan_outputs.append(scan_output[:, :, 0])
    scan_output = torch.stack(scan_outputs, dim=-1)
    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = scan_output * self.act(gate)
    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.transpose(1, 2))
    return contextualized_states


def install_state_hooks(model, layer_idx):
    mixer = model.backbone.layers[layer_idx].mixer
    mixer.forward = types.MethodType(state_aware_slow_forward, mixer)


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


def encode_residual(model, tokens, sae, act_mean, act_std, mid_layer,
                      ind_start, ind_end):
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
    return z[:, ind_start:ind_end].mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_pairs", type=int, default=64)
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    force_slow_forward(model)
    install_state_hooks(model, LOCUS_LAYER)
    sae, act_mean, act_std = load_sae_and_norm(device)

    ind = json.load(open(RESULTS_DIR / "induction_features.json"))
    top_feats = torch.tensor(ind["feature"][:10], device=device)

    clean, corr, ind_start, ind_end = make_batch(
        tokenizer, args.n_pairs, 0, device)

    locus_mixer = model.backbone.layers[LOCUS_LAYER].mixer

    # --- Capture CLEAN state trajectory at L30 ---
    locus_mixer._state_capture = []
    locus_mixer._state_replace = None
    z_clean = encode_residual(model, clean, sae, act_mean, act_std,
                                MID_LAYER, ind_start, ind_end)
    clean_states = list(locus_mixer._state_capture)
    locus_mixer._state_capture = None
    print(f"Captured {len(clean_states)} clean states at L{LOCUS_LAYER}, "
          f"each shape {tuple(clean_states[0].shape)}")

    # --- Capture CORRUPTED state for baseline ---
    locus_mixer._state_capture = []
    z_corr = encode_residual(model, corr, sae, act_mean, act_std,
                                MID_LAYER, ind_start, ind_end)
    corr_states = list(locus_mixer._state_capture)
    locus_mixer._state_capture = None

    baseline_act = z_clean[:, top_feats].mean().item()
    corrupted_act = z_corr[:, top_feats].mean().item()
    gap = baseline_act - corrupted_act
    print(f"baseline={baseline_act:.4f}, corrupted={corrupted_act:.4f}, gap={gap:.4f}")

    # --- Experiment A: patch clean state → corrupted run, at INDUCTION positions ---
    print("\n=== Experiment A: patch clean h at L30 positions 48..55 INTO corrupted run ===")
    replace_dict = {i: clean_states[i] for i in range(ind_start, ind_end)}
    locus_mixer._state_replace = replace_dict
    z_patched_A = encode_residual(model, corr, sae, act_mean, act_std,
                                    MID_LAYER, ind_start, ind_end)
    locus_mixer._state_replace = None
    patched_act_A = z_patched_A[:, top_feats].mean().item()
    rescue_A = (patched_act_A - corrupted_act) / gap if abs(gap) > 1e-8 else 0.0
    print(f"  rescued_act = {patched_act_A:.4f}  rescue_fraction = {rescue_A:+.4f}")

    # --- Experiment B: patch clean state across ALL positions ---
    print("\n=== Experiment B: patch clean h at ALL L30 positions INTO corrupted run ===")
    replace_dict = {i: clean_states[i] for i in range(len(clean_states))}
    locus_mixer._state_replace = replace_dict
    z_patched_B = encode_residual(model, corr, sae, act_mean, act_std,
                                    MID_LAYER, ind_start, ind_end)
    locus_mixer._state_replace = None
    patched_act_B = z_patched_B[:, top_feats].mean().item()
    rescue_B = (patched_act_B - corrupted_act) / gap if abs(gap) > 1e-8 else 0.0
    print(f"  rescued_act = {patched_act_B:.4f}  rescue_fraction = {rescue_B:+.4f}")

    # --- Experiment C: necessity variant — patch corrupted state into clean run ---
    print("\n=== Experiment C: patch corrupted h at L30 positions 48..55 INTO clean run ===")
    replace_dict = {i: corr_states[i] for i in range(ind_start, ind_end)}
    locus_mixer._state_replace = replace_dict
    z_patched_C = encode_residual(model, clean, sae, act_mean, act_std,
                                    MID_LAYER, ind_start, ind_end)
    locus_mixer._state_replace = None
    patched_act_C = z_patched_C[:, top_feats].mean().item()
    damage_C = 1.0 - (patched_act_C - corrupted_act) / gap if abs(gap) > 1e-8 else 0.0
    print(f"  patched_act = {patched_act_C:.4f}  patch_damage = {damage_C:+.4f}")

    # --- Experiment D: state at position 47 only (just-before-induction boundary) ---
    print("\n=== Experiment D: patch clean h at L30 position 47 (just before induction) INTO corrupted ===")
    replace_dict = {47: clean_states[47]}
    locus_mixer._state_replace = replace_dict
    z_patched_D = encode_residual(model, corr, sae, act_mean, act_std,
                                    MID_LAYER, ind_start, ind_end)
    locus_mixer._state_replace = None
    patched_act_D = z_patched_D[:, top_feats].mean().item()
    rescue_D = (patched_act_D - corrupted_act) / gap if abs(gap) > 1e-8 else 0.0
    print(f"  rescued_act = {patched_act_D:.4f}  rescue_fraction = {rescue_D:+.4f}")

    out = {
        "n_pairs": args.n_pairs,
        "baseline_act": baseline_act,
        "corrupted_act": corrupted_act,
        "gap": gap,
        "A_clean_state_ind_positions_into_corr": {
            "positions": [ind_start, ind_end],
            "rescued_act": patched_act_A,
            "rescue_fraction": rescue_A,
        },
        "B_clean_state_all_positions_into_corr": {
            "rescued_act": patched_act_B,
            "rescue_fraction": rescue_B,
        },
        "C_corrupted_state_ind_positions_into_clean": {
            "patched_act": patched_act_C,
            "patch_damage": damage_C,
        },
        "D_clean_state_position_47_into_corr": {
            "rescued_act": patched_act_D,
            "rescue_fraction": rescue_D,
        },
    }
    json.dump(out, open(RESULTS_DIR / "state_patching.json", "w"), indent=2)
    print(f"\nWrote {RESULTS_DIR / 'state_patching.json'}")


if __name__ == "__main__":
    main()
