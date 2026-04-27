#!/usr/bin/env python3
"""Pythia next-token logit damage — direct parallel to scripts/22 on Mamba.

Measure the logit drop on the correct induction-target token when patching
Pythia's attention_qkv at the strongest-single-site location (L10 from §4)
and at each slice (Q, K, V). Gives a behavioral number directly comparable
to Mamba-1's 47.5% next-token damage.

Output: $STORAGE/results_phase4/pythia_next_token_damage.json
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.activation_cache import get_model_and_tokenizer

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/path/to/storage"))
RESULTS_DIR = STORAGE / "results_phase4"

MODEL_NAME = "EleutherAI/pythia-2.8b"
LOCUS_LAYERS = [2, 6, 10, 12]
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


def capture_qkv(model, tokens, layer):
    captured = {}
    def hook(mod, ins, out):
        captured["qkv"] = out.detach().clone()
    attn = model.gpt_neox.layers[layer].attention.query_key_value
    h = attn.register_forward_hook(hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["qkv"]


def forward_with_qkv_patch(model, tokens, qkv_replacement=None, qkv_layer=None,
                             slice_spec=None, head_dim=80):
    hooks = []
    if qkv_replacement is not None:
        def patch_fn(mod, ins, out):
            new_out = out.clone()
            n_heads = new_out.shape[-1] // (3 * head_dim)
            shaped = new_out.view(new_out.shape[0], new_out.shape[1], n_heads, 3 * head_dim)
            repl = qkv_replacement.view(qkv_replacement.shape[0],
                                          qkv_replacement.shape[1], n_heads, 3 * head_dim)
            if repl.shape[0] != shaped.shape[0]:
                if repl.shape[0] == 1:
                    repl = repl.expand(shaped.shape[0], -1, -1, -1)
                else:
                    repl = repl[:shaped.shape[0]]
            s, e = slice_spec
            shaped[..., s:e] = repl[..., s:e].to(shaped.dtype)
            return shaped.reshape(out.shape)
        attn = model.gpt_neox.layers[qkv_layer].attention.query_key_value
        hooks.append(attn.register_forward_hook(patch_fn))
    with torch.no_grad():
        out = model(tokens)
    for h in hooks:
        h.remove()
    return out.logits


def target_logits(logits, target_tokens, ind_start, ind_end):
    logit_positions = list(range(ind_start - 1, ind_end - 1))
    gathered = logits[:, logit_positions, :]
    target = target_tokens[:, ind_start:ind_end]
    return gathered.gather(-1, target.unsqueeze(-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_pairs", type=int, default=256)
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)

    head_dim = 80
    slices = {
        "Q":    (0, head_dim),
        "K":    (head_dim, 2 * head_dim),
        "V":    (2 * head_dim, 3 * head_dim),
        "QK":   (0, 2 * head_dim),
        "full": (0, 3 * head_dim),
    }

    clean, corr, ind_start, ind_end = make_batch(tokenizer, args.n_pairs, 0, device)

    clean_logits = forward_with_qkv_patch(model, clean)
    corr_logits = forward_with_qkv_patch(model, corr)
    clean_mean = target_logits(clean_logits, clean, ind_start, ind_end).mean().item()
    corr_mean = target_logits(corr_logits, clean, ind_start, ind_end).mean().item()
    gap = clean_mean - corr_mean
    print(f"baseline={clean_mean:.3f}, corrupted={corr_mean:.3f}, gap={gap:.3f}")

    results = {}
    for L in LOCUS_LAYERS:
        qkv_corr = capture_qkv(model, corr, L)
        layer_results = {}
        for name, (s, e) in slices.items():
            patched_logits = forward_with_qkv_patch(
                model, clean, qkv_replacement=qkv_corr, qkv_layer=L,
                slice_spec=(s, e), head_dim=head_dim)
            patched_mean = target_logits(patched_logits, clean, ind_start, ind_end).mean().item()
            damage = 1.0 - (patched_mean - corr_mean) / gap if abs(gap) > 1e-8 else 0.0
            layer_results[name] = {"patched_mean": patched_mean, "damage": damage}
            print(f"  L{L:>2} {name:<5s}  patched={patched_mean:.3f}  damage={damage:+.4f}")
        results[f"L{L}"] = layer_results

    # Top sites
    all_sites = []
    for lk, d in results.items():
        for s, r in d.items():
            all_sites.append((lk, s, r["damage"]))
    all_sites.sort(key=lambda x: -x[2])
    print("\n=== Top 10 Pythia next-token damage sites ===")
    for lk, s, d in all_sites[:10]:
        print(f"  {lk} {s:<5s}  damage={d:+.4f}")

    out = {
        "model": MODEL_NAME,
        "n_pairs": args.n_pairs,
        "baseline_logit": clean_mean,
        "corrupted_logit": corr_mean,
        "gap": gap,
        "per_layer_slice": results,
    }
    out_path = RESULTS_DIR / "pythia_next_token_damage.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
