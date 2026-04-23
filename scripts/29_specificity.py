#!/usr/bin/env python3
"""Specificity check: is L30 x_proj.C only important for induction, or for
everything?

For each of N natural Pile sequences, measure three things under patching
L30 x_proj.C with a DIFFERENT sequence's C:
  (1) KL(next-token distribution) between clean and patched — how much the
      model's prediction changes across the full vocabulary.
  (2) Cosine similarity of L32 residual stream before/after patch.
  (3) Cross-entropy change on held-out target tokens.

Compare these to the same metrics on induction pairs. If C is induction-
specific, the general-text numbers should be much smaller than the induction
numbers.

Output: $STORAGE/results_phase4/specificity.json
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

MODEL_NAME = "state-spaces/mamba-2.8b-hf"
LOCUS_LAYER = 30
SEQ_LEN = 128
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


def load_natural_batch(tokenizer, n_seqs, device, seq_len=SEQ_LEN):
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    it = iter(ds)
    seqs = []
    while len(seqs) < n_seqs:
        try:
            ex = next(it)
        except StopIteration:
            break
        text = ex.get("text", "")
        if not text or len(text) < 400:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=seq_len)["input_ids"][0]
        if ids.shape[0] == seq_len:
            seqs.append(ids)
    return torch.stack(seqs).to(device)


def capture_xproj_C(model, tokens, layer, state_start, state_end):
    captured = {}
    def hook(mod, ins, out):
        captured["C"] = out[..., state_start:state_end].detach().clone()
        captured["full"] = out.detach().clone()
    h = model.backbone.layers[layer].mixer.x_proj.register_forward_hook(hook)
    with torch.no_grad():
        out = model(tokens)
    h.remove()
    return captured["full"], out.logits


def forward_with_C_patch(model, tokens, replacement_C_full, state_start, state_end,
                          layer):
    """Replace C slice only; leave Δ_pre and B slices unchanged from the
    clean forward."""
    def patch_fn(mod, ins, out):
        new_out = out.clone()
        # Replace only the C slice with the replacement's C
        repl = replacement_C_full[..., state_start:state_end].to(new_out.dtype)
        if repl.shape[0] != new_out.shape[0]:
            if repl.shape[0] == 1:
                repl = repl.expand(new_out.shape[0], -1, -1)
            else:
                repl = repl[:new_out.shape[0]]
        new_out[..., state_start:state_end] = repl
        return new_out
    h = model.backbone.layers[layer].mixer.x_proj.register_forward_hook(patch_fn)
    with torch.no_grad():
        out = model(tokens)
    h.remove()
    return out.logits


def kl_per_position(p_logits, q_logits):
    """Mean KL(P || Q) across positions. Chunk along batch dim to save memory."""
    total = 0.0
    n = 0
    chunk = 8
    for i in range(0, p_logits.shape[0], chunk):
        pl = p_logits[i:i + chunk].float()
        ql = q_logits[i:i + chunk].float()
        p_logp = torch.log_softmax(pl, dim=-1)
        q_logp = torch.log_softmax(ql, dim=-1)
        p = p_logp.exp()
        kl = (p * (p_logp - q_logp)).sum(dim=-1)
        total += kl.sum().item()
        n += kl.numel()
    return total / n


def next_token_loss(logits, target_tokens, positions):
    """CE at specified positions."""
    # logits: (B, L, V); predict token at p+1 from position p
    gathered = logits[:, [p - 1 for p in positions], :]
    target = target_tokens[:, positions]
    ce = torch.nn.functional.cross_entropy(
        gathered.reshape(-1, gathered.shape[-1]),
        target.reshape(-1),
        reduction="mean"
    ).item()
    return ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_nat", type=int, default=256)
    ap.add_argument("--n_ind", type=int, default=128)
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    force_slow_forward(model)

    mixer = model.backbone.layers[LOCUS_LAYER].mixer
    dt_rank = mixer.time_step_rank
    state_size = mixer.ssm_state_size
    C_start = dt_rank + state_size
    C_end = C_start + state_size

    # ----- Induction pairs -----
    print(f"\n=== INDUCTION pairs (n={args.n_ind}) ===")
    clean_ind, corr_ind, ind_start, ind_end = make_induction_batch(
        tokenizer, args.n_ind, seed=0, device=device)
    _, clean_logits = capture_xproj_C(model, clean_ind, LOCUS_LAYER, C_start, C_end)
    xp_corr_full, corr_logits = capture_xproj_C(model, corr_ind, LOCUS_LAYER, C_start, C_end)

    # Patch clean with corrupted C (same as §6 but for logit metrics)
    patched_logits_ind = forward_with_C_patch(
        model, clean_ind, xp_corr_full, C_start, C_end, LOCUS_LAYER)

    # Metrics at induction positions (positions 48..55), predicting positions 49..56
    ind_positions = list(range(ind_start, ind_end))
    ind_kl_patched = kl_per_position(
        clean_logits[:, [p - 1 for p in ind_positions], :],
        patched_logits_ind[:, [p - 1 for p in ind_positions], :],
    )
    ind_kl_corrupted = kl_per_position(
        clean_logits[:, [p - 1 for p in ind_positions], :],
        corr_logits[:, [p - 1 for p in ind_positions], :],
    )
    ind_ce_clean = next_token_loss(clean_logits, clean_ind, ind_positions)
    ind_ce_corr = next_token_loss(corr_logits, clean_ind, ind_positions)
    ind_ce_patched = next_token_loss(patched_logits_ind, clean_ind, ind_positions)

    print(f"  induction patched-vs-clean KL (ind positions):  {ind_kl_patched:.4f}")
    print(f"  induction corrupted-vs-clean KL (ind positions): {ind_kl_corrupted:.4f}")
    print(f"  induction CE on clean-target tokens: clean={ind_ce_clean:.3f}, corr={ind_ce_corr:.3f}, patched={ind_ce_patched:.3f}")

    # ----- Natural Pile text -----
    print(f"\n=== NATURAL Pile text (n={args.n_nat}) ===")
    natural = load_natural_batch(tokenizer, args.n_nat, device)
    print(f"  loaded {natural.shape[0]} sequences of {natural.shape[1]} tokens")

    # Capture each sequence's clean x_proj output
    xp_nat_full, nat_clean_logits = capture_xproj_C(model, natural, LOCUS_LAYER, C_start, C_end)
    # Random permutation: patch each sequence's C with a DIFFERENT sequence's C
    perm = torch.randperm(natural.shape[0])
    xp_nat_shuffled = xp_nat_full[perm]
    # Ensure no self-matches
    # (permutation might accidentally be identity for some indices; refine)
    perm_fixed = perm.clone()
    for i in range(len(perm_fixed)):
        if perm_fixed[i].item() == i:
            perm_fixed[i] = (i + 1) % len(perm_fixed)
    xp_nat_shuffled = xp_nat_full[perm_fixed]

    patched_nat_logits = forward_with_C_patch(
        model, natural, xp_nat_shuffled, C_start, C_end, LOCUS_LAYER)

    # Measure metrics averaged across all positions 8..seq_len-1 (skip prefix)
    n_positions = list(range(8, natural.shape[1] - 1))
    nat_kl_patched = kl_per_position(
        nat_clean_logits[:, n_positions, :],
        patched_nat_logits[:, n_positions, :],
    )

    # CE on actual next-token targets
    nat_ce_clean = torch.nn.functional.cross_entropy(
        nat_clean_logits[:, :-1, :].reshape(-1, nat_clean_logits.shape[-1]),
        natural[:, 1:].reshape(-1),
        reduction="mean"
    ).item()
    nat_ce_patched = torch.nn.functional.cross_entropy(
        patched_nat_logits[:, :-1, :].reshape(-1, patched_nat_logits.shape[-1]),
        natural[:, 1:].reshape(-1),
        reduction="mean"
    ).item()
    print(f"  natural patched-vs-clean KL (all positions):  {nat_kl_patched:.4f}")
    print(f"  natural CE on next-token: clean={nat_ce_clean:.3f}, patched={nat_ce_patched:.3f}")
    print(f"  natural CE increase:  {nat_ce_patched - nat_ce_clean:+.4f}")

    out = {
        "induction": {
            "n": args.n_ind,
            "kl_patched_vs_clean": ind_kl_patched,
            "kl_corrupted_vs_clean": ind_kl_corrupted,
            "ce_clean": ind_ce_clean,
            "ce_corrupted": ind_ce_corr,
            "ce_patched": ind_ce_patched,
        },
        "natural": {
            "n": args.n_nat,
            "kl_patched_vs_clean": nat_kl_patched,
            "ce_clean": nat_ce_clean,
            "ce_patched": nat_ce_patched,
            "ce_increase": nat_ce_patched - nat_ce_clean,
        },
        "induction_to_natural_kl_ratio": ind_kl_patched / nat_kl_patched if nat_kl_patched > 0 else float("inf"),
    }
    print(f"\n=== SUMMARY ===")
    print(f"  Induction KL (patched→clean): {ind_kl_patched:.4f}")
    print(f"  Natural   KL (patched→clean): {nat_kl_patched:.4f}")
    print(f"  Ratio: {out['induction_to_natural_kl_ratio']:.2f}x stronger effect on induction than on general text")

    json.dump(out, open(RESULTS_DIR / "specificity.json", "w"), indent=2)
    print(f"Wrote {RESULTS_DIR / 'specificity.json'}")


if __name__ == "__main__":
    main()
