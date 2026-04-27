#!/usr/bin/env python3
"""Logit lens on the 16-dim C matrix of L30 x_proj.

For each of the 16 "C columns" (state-readout directions), project the raw
direction through the downstream pathway to the vocabulary and report the
top tokens each direction predicts. Attempts two projection modes:
  (A) Simple: direction × out_proj.weight → residual delta → lm_head → logits.
  (B) With all-zero prior state: initialize state as zero, apply C[:, k] as
      the scan-output direction (y_k = 1 at dim k, else 0), pass through the
      rest of L30's mixer (post-scan: silu gate multiply, out_proj),
      contribute to residual, measure change in final logits.

Mode (A) is a one-shot projection; mode (B) approximates "what would a unit
activation of C column k contribute to the model's final prediction."

Output: $STORAGE/results_phase4/logit_lens_C.json
"""
import json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.activation_cache import get_model_and_tokenizer

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/path/to/storage"))
RESULTS_DIR = STORAGE / "results_phase4"
MODEL_NAME = "state-spaces/mamba-2.8b-hf"
LOCUS_LAYER = 30


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top_k", type=int, default=15)
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)

    mixer = model.backbone.layers[LOCUS_LAYER].mixer
    dt_rank = mixer.time_step_rank
    state_size = mixer.ssm_state_size   # 16
    d_inner = mixer.intermediate_size    # 5120

    # C lives at [dt_rank + state_size : dt_rank + 2*state_size] of x_proj OUTPUT.
    # But the x_proj output's C slice is the READOUT matrix for the SSM scan:
    #   y_t = ssm_state_t · C_t
    # where C_t is (B, L, state_size). ssm_state_t is (B, d_inner, state_size).
    # The output y_t has shape (B, L, d_inner).
    #
    # For logit lens, we want: "if C had a unit of activation at dimension k,
    # what's the downstream effect on logits?"
    #
    # Approximation: assume ssm_state has unit magnitude at all d_inner positions
    # in state direction k. Then y at time t for state direction k is basically
    # C[:,:,k] broadcast across d_inner. After silu*gate and out_proj, this maps
    # to residual stream.

    # Simpler direct lens: each C column k, when the scan state h is any vector,
    # contributes h · C[:,:,k] to y. If h is e_j (unit vector in state dim j),
    # y_j = C[:,:,k] at dim k. So the projection of the k-th C column into
    # residual space is:
    #   C_column_k(...) → through silu-gated out_proj → residual
    # But we don't know the gate. Approximate by using out_proj directly on a
    # d_inner-dim vector formed by broadcasting C[:, :, k] (state-size=16 scalar)
    # uniformly across d_inner=5120.

    # Actually the cleanest lens: take y = 1 at all d_inner positions (hypothetical
    # "all scan outputs fire") and project through (no gate) out_proj → residual
    # → unembedding. This gives the raw "per-state-dim" effect on vocab.

    # C matrix weights: The SSM readout uses C_t which is computed at each
    # position from x_proj. The structural "C matrix" is the weights of the
    # x_proj output slice that produces C. So W_x_proj[dt_rank + state_size :, :]
    # where x_proj is Linear(d_inner, dt_rank + 2*state_size).
    # Shape: (state_size, d_inner)

    x_proj_w = mixer.x_proj.weight.data  # (out_features, in_features) = (dt+2*state, d_inner)
    C_weights = x_proj_w[dt_rank + state_size:, :]  # (state_size=16, d_inner=5120)
    out_proj_w = mixer.out_proj.weight.data  # (d_model=2560, d_inner=5120)
    lm_head_w = model.lm_head.weight.data  # (vocab_size, d_model)

    print(f"C_weights shape: {tuple(C_weights.shape)}")
    print(f"out_proj weight shape: {tuple(out_proj_w.shape)}")
    print(f"lm_head weight shape: {tuple(lm_head_w.shape)}")

    # Method: for each of 16 C rows (state dimensions), use it as a "query"
    # direction in d_inner space. The scan produces y[:, :, d_inner] that, for
    # state dimension k, depends on how much h[:, d_inner, k] couples with
    # C[k, :]. So the d_inner-direction pooled by C row k is simply C[k, :]
    # itself as a weighting over d_inner dimensions.
    #
    # Equivalent: treat C[k] as a d_inner-dim direction. Project through out_proj
    # (d_inner → d_model), then through lm_head (d_model → vocab). Top tokens
    # are what this direction would "predict" if activated.

    results = []
    for k in range(state_size):
        direction = C_weights[k].float()  # (d_inner,)
        # Project: first through out_proj, then lm_head
        residual_contrib = out_proj_w.float() @ direction  # (d_model,)
        logits_contrib = lm_head_w.float() @ residual_contrib  # (vocab,)
        top_k_vals, top_k_idx = logits_contrib.topk(args.top_k)
        top_tokens = [tokenizer.decode([int(i)]) for i in top_k_idx.cpu().tolist()]
        bot_k_vals, bot_k_idx = (-logits_contrib).topk(args.top_k)
        bot_tokens = [tokenizer.decode([int(i)]) for i in bot_k_idx.cpu().tolist()]
        results.append({
            "dim": k,
            "direction_norm": float(direction.norm().item()),
            "top_positive_tokens": list(zip(top_tokens, top_k_vals.cpu().tolist())),
            "top_negative_tokens": list(zip(bot_tokens, (-bot_k_vals).cpu().tolist())),
        })

    print("\n=== Logit lens per C direction ===")
    for r in results:
        print(f"\n--- C dimension {r['dim']} (norm={r['direction_norm']:.2f}) ---")
        print("  TOP tokens (what this direction 'writes'):")
        for tok, val in r['top_positive_tokens'][:8]:
            print(f"    {tok!r:<20s}  {val:+.3f}")
        print("  TOP *suppressed* tokens (what this direction 'unwrites'):")
        for tok, val in r['top_negative_tokens'][:5]:
            print(f"    {tok!r:<20s}  {val:+.3f}")

    out_path = RESULTS_DIR / "logit_lens_C.json"
    json.dump(results, open(out_path, "w"), indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
