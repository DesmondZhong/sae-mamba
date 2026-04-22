#!/usr/bin/env python3
"""Phase 4: Localize Mamba-1's attention-free induction mechanism.

Pipeline:
  1. Generate (clean, corrupted) induction pairs:
       clean     = [prefix] P [mid] P         (pattern P repeats → induction)
       corrupted = [prefix] P [mid] P'        (fresh pattern → no induction)
  2. Using the existing L32 SAE, find features with highest
       mean_activation(clean @ second-pattern) − mean_activation(corrupted @ second-pattern).
     These are our "induction features".
  3. For each (layer, component) site in {in_proj, conv1d, x_proj, dt_proj, out_proj_in}
     across a sweep of layers, run:
       a) capture the component's activation on the corrupted input
       b) run the clean input with that component's output replaced by (a)
       c) measure the drop in target-feature activation
     → `patch_damage` = 1 − (patched_act − corrupted_act) / (clean_act − corrupted_act)
     1.0 means this component fully carries the induction signal, 0.0 means it doesn't.

Outputs:
  $STORAGE/results_phase4/induction_features.json
  $STORAGE/results_phase4/patching_results.json
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
from tqdm import tqdm

from src.activation_cache import get_model_and_tokenizer
from src.mamba_internals import (
    ALL_COMPONENTS,
    MambaInternalCapture,
    MambaInternalPatcher,
    ResidualStreamCapture,
)
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/mnt/storage/desmond/excuse"))
CKPT_DIR = STORAGE / "checkpoints_normed"
ACTS_DIR = STORAGE / "activations"
RESULTS_DIR = STORAGE / "results_phase4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_KEY = "mamba1_2.8b"
MODEL_NAME = "state-spaces/mamba-2.8b-hf"
D_MODEL = 2560
MID_LAYER = 32

SAE_EXPANSION = 16
SAE_K = 64


def load_sae_and_norm(layer: int, device: str):
    """Load the normalized L32 SAE and the normalization stats used at training time."""
    d_hidden = D_MODEL * SAE_EXPANSION
    run_key = f"{MODEL_KEY}_L{layer}_x{SAE_EXPANSION}_k{SAE_K}_normed"
    ckpt_path = CKPT_DIR / f"{run_key}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    sae = create_sae(D_MODEL, d_hidden, sae_type="topk", k=SAE_K).to(device)
    sae.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    sae.eval()

    acts_path = ACTS_DIR / MODEL_KEY / f"layer_{layer}.pt"
    t = torch.load(acts_path, map_location="cpu", weights_only=True, mmap=True)
    sample = t[:10000].clone().float()
    act_mean = sample.mean(dim=0).to(device)
    act_std = sample.std(dim=0).clamp(min=1e-6).to(device)
    del t, sample
    return sae, act_mean, act_std


def make_induction_pair(
    vocab_size: int,
    n_pairs: int = 32,
    prefix_len: int = 40,
    pattern_len: int = 5,
    mid_len: int = 40,
    device: str = "cuda",
    seed: int = 42,
):
    """Build matched (clean, corrupted) induction sequences.

    clean:      [prefix] [pattern] [mid] [pattern]
    corrupted:  [prefix] [pattern] [mid] [alt_pattern]
    """
    rng = np.random.default_rng(seed)
    clean_seqs, corr_seqs = [], []
    for _ in range(n_pairs):
        prefix = rng.integers(10, vocab_size - 10, size=prefix_len).tolist()
        pattern = rng.integers(10, vocab_size - 10, size=pattern_len).tolist()
        mid = rng.integers(10, vocab_size - 10, size=mid_len).tolist()
        alt_pattern = rng.integers(10, vocab_size - 10, size=pattern_len).tolist()
        clean_seqs.append(prefix + pattern + mid + pattern)
        corr_seqs.append(prefix + pattern + mid + alt_pattern)

    clean = torch.tensor(clean_seqs, dtype=torch.long, device=device)
    corr = torch.tensor(corr_seqs, dtype=torch.long, device=device)

    second_start = prefix_len + pattern_len + mid_len
    induction_positions = list(range(second_start, second_start + pattern_len))
    return clean, corr, induction_positions


@torch.no_grad()
def encode_residual(model, tokens, sae, act_mean, act_std, layer, positions=None):
    """Forward `tokens`, grab layer's residual stream, encode with the SAE at `positions`.

    Returns tensor of shape (B, n_positions, d_hidden) or (B, L, d_hidden) if positions is None.
    """
    with ResidualStreamCapture(model, [layer]) as cap:
        model(tokens)
    h = cap.captured[layer].float()  # (B, L, D)
    B, L, D = h.shape
    h_flat = h.reshape(-1, D)
    h_normed = (h_flat - act_mean) / act_std
    z = sae.encode(h_normed)
    z = z.reshape(B, L, -1)
    if positions is not None:
        z = z[:, positions, :]
    return z


def identify_induction_features(
    model, tokenizer, sae, act_mean, act_std, device,
    top_k: int = 10, n_pairs: int = 64,
):
    """Find SAE features that activate on induction positions (clean − corrupted)."""
    clean, corr, ind_pos = make_induction_pair(
        tokenizer.vocab_size, n_pairs=n_pairs, device=device,
    )
    z_clean = encode_residual(model, clean, sae, act_mean, act_std, MID_LAYER, ind_pos)
    z_corr = encode_residual(model, corr, sae, act_mean, act_std, MID_LAYER, ind_pos)

    mean_clean = z_clean.reshape(-1, z_clean.shape[-1]).mean(dim=0)
    mean_corr = z_corr.reshape(-1, z_corr.shape[-1]).mean(dim=0)
    induction_score = mean_clean - mean_corr

    top_vals, top_idx = torch.topk(induction_score, top_k)
    info = {
        "feature": top_idx.cpu().tolist(),
        "score": top_vals.cpu().tolist(),
        "mean_clean": mean_clean[top_idx].cpu().tolist(),
        "mean_corr": mean_corr[top_idx].cpu().tolist(),
    }
    return top_idx.cpu().tolist(), info


def run_patching(
    model, tokenizer, sae, act_mean, act_std, device,
    target_features, patch_layers, components,
    n_pairs: int = 32,
):
    """Rank (layer, component) sites by fraction of induction signal destroyed."""
    clean, corr, ind_pos = make_induction_pair(
        tokenizer.vocab_size, n_pairs=n_pairs, device=device,
    )
    target_idx = torch.tensor(target_features, device=device, dtype=torch.long)

    def mean_target_act(tokens, patches=None):
        if patches is not None:
            with MambaInternalPatcher(model, patches=patches):
                z = encode_residual(model, tokens, sae, act_mean, act_std, MID_LAYER, ind_pos)
        else:
            z = encode_residual(model, tokens, sae, act_mean, act_std, MID_LAYER, ind_pos)
        return z[:, :, target_idx].mean().item()

    baseline_act = mean_target_act(clean)
    corrupted_act = mean_target_act(corr)
    total_effect = baseline_act - corrupted_act

    per_site = []
    for layer_idx in tqdm(patch_layers, desc="Layer sweep"):
        for component in components:
            site = (layer_idx, component)
            with MambaInternalCapture(model, sites=[site]) as cap:
                with torch.no_grad():
                    model(corr)
            corrupted_internal = cap.captured[site]

            patched_act = mean_target_act(clean, patches={site: corrupted_internal})

            patch_damage = (
                1.0 - (patched_act - corrupted_act) / total_effect
                if abs(total_effect) > 1e-8 else 0.0
            )

            per_site.append({
                "layer": layer_idx,
                "component": component,
                "baseline_act": baseline_act,
                "corrupted_act": corrupted_act,
                "patched_act": patched_act,
                "patch_damage": patch_damage,
            })
            print(f"  L{layer_idx:2d} {component:<12s} "
                  f"patch_damage={patch_damage:+.3f}  "
                  f"(base={baseline_act:.3f}, corr={corrupted_act:.3f}, patch={patched_act:.3f})",
                  flush=True)

    return {
        "baseline_act": baseline_act,
        "corrupted_act": corrupted_act,
        "total_effect": total_effect,
        "target_features": target_features,
        "per_site": per_site,
    }


def run_position_specific_patching(
    model, tokenizer, sae, act_mean, act_std, device,
    target_features, best_sites,
    n_pairs: int = 32,
):
    """For the top-ranked sites from the full-sequence sweep, also test patching only the
    second-pattern positions (induction positions). Disentangles 'signal AT induction
    positions' from 'signal flowing through the SSM state FROM earlier positions'.
    """
    clean, corr, ind_pos = make_induction_pair(
        tokenizer.vocab_size, n_pairs=n_pairs, device=device,
    )
    target_idx = torch.tensor(target_features, device=device, dtype=torch.long)

    baseline_act = encode_residual(
        model, clean, sae, act_mean, act_std, MID_LAYER, ind_pos
    )[:, :, target_idx].mean().item()
    corrupted_act = encode_residual(
        model, corr, sae, act_mean, act_std, MID_LAYER, ind_pos
    )[:, :, target_idx].mean().item()
    total_effect = baseline_act - corrupted_act

    results = []
    for site in best_sites:
        layer_idx, component = site
        with MambaInternalCapture(model, sites=[site]) as cap:
            with torch.no_grad():
                model(corr)
        corrupted_internal = cap.captured[site]

        for label, positions in [("all", None), ("ind_only", ind_pos), ("pre_ind_only",
                                 list(range(ind_pos[0])))]:
            with MambaInternalPatcher(
                model, patches={site: corrupted_internal}, positions=positions,
            ):
                z = encode_residual(
                    model, clean, sae, act_mean, act_std, MID_LAYER, ind_pos,
                )
            patched_act = z[:, :, target_idx].mean().item()
            patch_damage = (
                1.0 - (patched_act - corrupted_act) / total_effect
                if abs(total_effect) > 1e-8 else 0.0
            )
            results.append({
                "layer": layer_idx,
                "component": component,
                "patch_region": label,
                "baseline_act": baseline_act,
                "corrupted_act": corrupted_act,
                "patched_act": patched_act,
                "patch_damage": patch_damage,
            })
            print(f"  L{layer_idx:2d} {component:<12s} region={label:<12s} "
                  f"patch_damage={patch_damage:+.3f}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--layers", nargs="+", type=int,
        default=[4, 8, 12, 16, 20, 24, 28, 30, 31, 32],
        help="Layer indices at which to patch components.",
    )
    parser.add_argument("--top_features", type=int, default=10)
    parser.add_argument("--n_pairs", type=int, default=32)
    parser.add_argument(
        "--components", nargs="+", default=ALL_COMPONENTS,
        choices=ALL_COMPONENTS,
    )
    parser.add_argument(
        "--skip_position_sweep", action="store_true",
        help="Skip the position-specific patching step.",
    )
    args = parser.parse_args()

    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)

    print(f"Loading L{MID_LAYER} SAE...")
    sae, act_mean, act_std = load_sae_and_norm(MID_LAYER, device)

    print("\n=== Step 1: Identify induction features ===")
    top_feats, feat_info = identify_induction_features(
        model, tokenizer, sae, act_mean, act_std, device,
        top_k=args.top_features,
    )
    print(f"  Top {len(top_feats)} induction features (feature idx → score):")
    for f, s in zip(feat_info["feature"], feat_info["score"]):
        print(f"    feat {f:>6d}   score={s:.4f}")

    with open(RESULTS_DIR / "induction_features.json", "w") as f:
        json.dump(feat_info, f, indent=2, default=str)

    print("\n=== Step 2: Component patching sweep ===")
    patch_results = run_patching(
        model, tokenizer, sae, act_mean, act_std, device,
        target_features=top_feats,
        patch_layers=args.layers,
        components=args.components,
        n_pairs=args.n_pairs,
    )

    with open(RESULTS_DIR / "patching_results.json", "w") as f:
        json.dump(patch_results, f, indent=2, default=str)

    # Rank and report top sites.
    sites_sorted = sorted(
        patch_results["per_site"], key=lambda x: x["patch_damage"], reverse=True,
    )
    print("\n=== Ranked (layer, component) sites by induction-signal carriage ===")
    for s in sites_sorted[:15]:
        print(f"  L{s['layer']:2d} {s['component']:<12s}  patch_damage={s['patch_damage']:+.3f}")

    if not args.skip_position_sweep:
        print("\n=== Step 3: Position-specific patching on top-5 sites ===")
        top_sites = [(s["layer"], s["component"]) for s in sites_sorted[:5]]
        pos_results = run_position_specific_patching(
            model, tokenizer, sae, act_mean, act_std, device,
            target_features=top_feats,
            best_sites=top_sites,
            n_pairs=args.n_pairs,
        )
        with open(RESULTS_DIR / "patching_position_specific.json", "w") as f:
            json.dump(pos_results, f, indent=2, default=str)

    print("\nAll results written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
