#!/usr/bin/env python3
"""Does patching L30 x_proj.C affect the model's actual next-token prediction?

All prior Phase 4 experiments measure SAE-feature activation (a proxy for
induction). This script measures the actual behavior: logit of the correct
next token at the induction positions.

For each synthetic induction pair:
  - clean[p] is the p-th token of the second pattern; it should match the
    token at p-PATTERN_LEN-MID_LEN (same token in first pattern).
  - The MODEL's prediction at position p-1 should be the token at p
    (induction = model predicts the next token from the earlier occurrence).

We compute:
  - baseline: clean logit at the correct next token
  - corrupted: corrupted logit at the correct-for-clean next token
  - patched: clean run with L30 x_proj.C patched from corrupted
  - damage: 1 - (patched - corrupted) / (baseline - corrupted)

Output: $STORAGE/results_phase4/next_token_damage.json
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.activation_cache import get_model_and_tokenizer
from src.mamba_internals import force_slow_forward

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/path/to/storage"))
RESULTS_DIR = STORAGE / "results_phase4"

MODEL_NAME = "state-spaces/mamba-2.8b-hf"
LOCUS_LAYER = 30
PATTERN_LEN = 8
PREFIX_LEN = 8
MID_LEN = 32


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


def forward_with_patch(model, tokens, slice_patch=None):
    """Return logits (B, L, V)."""
    hooks = []
    if slice_patch is not None:
        s, e = slice_patch["slice"]
        repl = slice_patch["value"]
        def patch_fn(mod, ins, out):
            new_out = out.clone()
            repl_slice = repl[..., s:e].to(new_out.dtype)
            if repl_slice.shape[0] != new_out.shape[0]:
                if repl_slice.shape[0] == 1:
                    repl_slice = repl_slice.expand(new_out.shape[0], -1, -1)
                else:
                    repl_slice = repl_slice[:new_out.shape[0]]
            new_out[..., s:e] = repl_slice
            return new_out
        hooks.append(model.backbone.layers[LOCUS_LAYER].mixer.x_proj.register_forward_hook(patch_fn))

    with torch.no_grad():
        out = model(tokens)
    for h in hooks:
        h.remove()
    return out.logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_pairs", type=int, default=256)
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    force_slow_forward(model)

    clean, corr, ind_start, ind_end = make_induction_batch(
        tokenizer, args.n_pairs, seed=0, device=device)

    # logit at induction positions. Position p predicts token at p+1.
    # So we look at logits[:, ind_start-1:ind_end-1] vs target tokens[:, ind_start:ind_end].
    target_positions = list(range(ind_start, ind_end))     # the tokens being predicted
    logit_positions = [p - 1 for p in target_positions]    # positions that predict them

    def target_logits(logits, target_tokens):
        # logits: (B, L, V)   target_tokens: (B, pattern_len)
        gathered = logits[:, logit_positions, :]            # (B, PL, V)
        return gathered.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)

    # Clean: model should predict clean[:, ind_start:ind_end] correctly (induction).
    clean_logits = forward_with_patch(model, clean)
    clean_target_logits = target_logits(clean_logits, clean[:, ind_start:ind_end])
    clean_mean_logit = clean_target_logits.mean().item()

    # Corrupted: model sees different second pattern; logit at the CLEAN target is low.
    corr_logits = forward_with_patch(model, corr)
    corr_target_logits = target_logits(corr_logits, clean[:, ind_start:ind_end])
    corr_mean_logit = corr_target_logits.mean().item()

    gap = clean_mean_logit - corr_mean_logit
    print(f"clean_logit={clean_mean_logit:.3f}, corrupted_logit={corr_mean_logit:.3f}, gap={gap:.3f}")

    # Capture corrupted x_proj output at L30
    xproj_corr = capture_xproj_output(model, corr, LOCUS_LAYER)

    # Slices
    mixer = model.backbone.layers[LOCUS_LAYER].mixer
    dt_rank = mixer.time_step_rank    # 160
    state_size = mixer.ssm_state_size # 16
    slices = {
        "full_xproj":  (0, dt_rank + 2 * state_size),
        "delta_pre":   (0, dt_rank),
        "B_matrix":    (dt_rank, dt_rank + state_size),
        "C_matrix":    (dt_rank + state_size, dt_rank + 2 * state_size),
        "B_and_C":     (dt_rank, dt_rank + 2 * state_size),
    }

    results = {
        "clean_mean_logit": clean_mean_logit,
        "corrupted_mean_logit": corr_mean_logit,
        "gap": gap,
        "n_pairs": args.n_pairs,
        "per_slice": {},
    }
    for name, (s, e) in slices.items():
        patched_logits = forward_with_patch(
            model, clean, slice_patch={"slice": (s, e), "value": xproj_corr},
        )
        patched_target_logits = target_logits(patched_logits,
                                                clean[:, ind_start:ind_end])
        patched_mean = patched_target_logits.mean().item()
        damage = 1.0 - (patched_mean - corr_mean_logit) / gap if abs(gap) > 1e-8 else 0.0
        results["per_slice"][name] = {
            "slice": [s, e],
            "patched_mean_logit": patched_mean,
            "next_token_damage": damage,
        }
        print(f"  {name:<12s} slice[{s}:{e}] patched_logit={patched_mean:.3f}  "
              f"next_token_damage={damage:+.4f}")

    json.dump(results, open(RESULTS_DIR / "next_token_damage.json", "w"), indent=2)
    print(f"\nWrote {RESULTS_DIR / 'next_token_damage.json'}")


if __name__ == "__main__":
    main()
