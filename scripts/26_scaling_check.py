#!/usr/bin/env python3
"""Scaling check: does L30 x_proj.C localization generalize to Mamba-130M / Mamba-370M?

Unlike the 2.8B experiments which use SAE feature activation as the proxy,
this script uses the model's actual next-token LOGIT on the correct target
token. Avoids needing to train new SAEs for the smaller models.

For each (layer, slice) of x_proj output, measure:
  patch_damage_logit = 1 - (patched - corrupted) / (clean - corrupted)
at the induction positions.

Finds the dominant (layer, slice) across a layer sweep.
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.activation_cache import get_model_and_tokenizer
from src.mamba_internals import force_slow_forward

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
RESULTS_DIR = STORAGE / "results_phase4"

PATTERN_LEN = 8
PREFIX_LEN = 8
MID_LEN = 32


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


def capture_xproj_output(model, tokens, layer):
    captured = {}
    def hook(mod, ins, out):
        captured["o"] = out.detach().clone()
    h = model.backbone.layers[layer].mixer.x_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["o"]


def forward_with_patch(model, tokens, slice_patch=None, locus_layer=None):
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
        hooks.append(model.backbone.layers[locus_layer].mixer.x_proj.register_forward_hook(patch_fn))
    with torch.no_grad():
        out = model(tokens)
    for h in hooks:
        h.remove()
    return out.logits


def target_logits_at_induction(logits, target_tokens, ind_start, ind_end):
    # Predict token at position p from representation at position p-1
    gathered = logits[:, ind_start - 1:ind_end - 1, :]
    return gathered.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_pairs", type=int, default=128)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--tag", required=True, help="short tag for output file")
    args = ap.parse_args()
    device = args.device

    print(f"Loading {args.model}...")
    model, tokenizer = get_model_and_tokenizer(args.model, device)
    force_slow_forward(model)

    mixer0 = model.backbone.layers[0].mixer
    dt_rank = mixer0.time_step_rank
    state_size = mixer0.ssm_state_size
    x_out = mixer0.x_proj.out_features
    d_model = model.backbone.embeddings.weight.shape[1] if hasattr(model.backbone, 'embeddings') else model.config.hidden_size
    n_layers = len(model.backbone.layers)
    print(f"Model: {args.model}  n_layers={n_layers}  d_model={d_model}  "
          f"dt_rank={dt_rank}  state_size={state_size}  x_proj_out={x_out}")

    slices = {
        "full_xproj": (0, x_out),
        "delta_pre":  (0, dt_rank),
        "B_matrix":   (dt_rank, dt_rank + state_size),
        "C_matrix":   (dt_rank + state_size, dt_rank + 2 * state_size),
    }

    clean, corr, ind_start, ind_end = make_batch(tokenizer, args.n_pairs, 0, device)

    # Baseline and corrupted logits
    clean_logits = forward_with_patch(model, clean)
    corr_logits = forward_with_patch(model, corr)
    # Target: clean tokens at induction positions (what the model should predict)
    tgt = clean[:, ind_start:ind_end]
    clean_mean = target_logits_at_induction(clean_logits, tgt, ind_start, ind_end).mean().item()
    corr_mean = target_logits_at_induction(corr_logits, tgt, ind_start, ind_end).mean().item()
    gap = clean_mean - corr_mean
    print(f"\nbaseline={clean_mean:.3f}, corrupted={corr_mean:.3f}, gap={gap:.3f}")
    if gap < 0.5:
        print(f"WARNING: small gap ({gap:.3f}) means weak induction behavior in this model.")

    per_site = []
    for L in tqdm(args.layers, desc="layer sweep"):
        xproj_corr = capture_xproj_output(model, corr, L)
        for slice_name, (s, e) in slices.items():
            patched_logits = forward_with_patch(
                model, clean,
                slice_patch={"slice": (s, e), "value": xproj_corr},
                locus_layer=L,
            )
            patched_mean = target_logits_at_induction(patched_logits, tgt,
                                                       ind_start, ind_end).mean().item()
            damage = 1.0 - (patched_mean - corr_mean) / gap if abs(gap) > 1e-8 else 0.0
            per_site.append({
                "layer": L, "slice": slice_name, "slice_range": [s, e],
                "patched_mean_logit": patched_mean,
                "logit_damage": damage,
            })

    per_site_sorted = sorted(per_site, key=lambda r: -r["logit_damage"])
    print(f"\n=== Top-10 {args.tag} logit_damage sites ===")
    for r in per_site_sorted[:10]:
        print(f"  L{r['layer']:>2}  {r['slice']:<12s}  logit_damage={r['logit_damage']:+.4f}")

    out = {
        "model": args.model,
        "n_layers": n_layers,
        "d_model": d_model,
        "dt_rank": dt_rank,
        "state_size": state_size,
        "n_pairs": args.n_pairs,
        "clean_mean_logit": clean_mean,
        "corrupted_mean_logit": corr_mean,
        "gap": gap,
        "layers_tested": args.layers,
        "per_site": per_site,
    }
    out_path = RESULTS_DIR / f"scaling_{args.tag}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
