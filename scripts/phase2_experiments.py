#!/usr/bin/env python3
"""
Phase 2 experiments: deeper analysis on trained SAEs to find novel insights.
Runs AFTER all normalized SAEs are trained.

Experiments:
1. Feature frequency distribution comparison (Zipf's law analysis)
   - Do Mamba features fire more uniformly or follow steeper power laws?

2. Feature geometry: decoder weight analysis
   - Are Mamba features more orthogonal? More clustered?
   - Cosine similarity distribution of decoder columns

3. Activation correlation structure
   - Do features co-activate more in one architecture?
   - Feature co-occurrence matrix analysis

4. Layer-to-layer feature similarity
   - How much do features change between adjacent layers?
   - Within-model CKA across depth

5. Effective dimensionality
   - Participation ratio of SAE activations
   - How many features actually matter?
"""

import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

STORAGE = Path("/path/to/storage")
ACTS_DIR = STORAGE / "activations"
CKPT_DIR = STORAGE / "checkpoints_normed"
RESULTS_DIR = STORAGE / "results_normed"
PHASE2_DIR = STORAGE / "results_phase2"
PHASE2_DIR.mkdir(exist_ok=True)

MODELS = {
    "mamba1_2.8b": {"n_layers": 64, "d_model": 2560},
    "pythia_2.8b": {"n_layers": 32, "d_model": 2560},
}


def get_layer_indices(n_layers):
    return list(range(0, n_layers, max(1, n_layers // 8)))


def load_sae(model_key, layer, expansion=16, k=64, device="cpu"):
    from src.sae import create_sae
    d_model = MODELS[model_key]["d_model"]
    d_hidden = d_model * expansion
    run_key = f"{model_key}_L{layer}_x{expansion}_k{k}_normed"
    path = CKPT_DIR / f"{run_key}.pt"
    if not path.exists():
        return None
    sae = create_sae(d_model, d_hidden, sae_type="topk", k=k).to(device)
    sae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    sae.eval()
    return sae


def load_acts_sample(model_key, layer, n=10000):
    """Load a subsample of activations, normalized."""
    path = ACTS_DIR / model_key / f"layer_{layer}.pt"
    if not path.exists():
        return None
    t = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    sample = t[:n].clone().float()
    # Normalize (same as training)
    mean = sample.mean(dim=0, keepdim=True)
    std = sample.std(dim=0, keepdim=True).clamp(min=1e-6)
    normed = (sample - mean) / std
    del t
    return normed


def experiment_1_feature_frequency(device="cpu"):
    """Compare feature firing frequency distributions across architectures."""
    print("=" * 60)
    print("Experiment 1: Feature Frequency Distribution")
    print("=" * 60)

    results = {}
    for model_key, info in MODELS.items():
        results[model_key] = {}
        for layer in get_layer_indices(info["n_layers"]):
            sae = load_sae(model_key, layer, device=device)
            acts = load_acts_sample(model_key, layer, n=50000)
            if sae is None or acts is None:
                continue

            # Get feature activations
            with torch.no_grad():
                z = sae.encode(acts.to(device)).cpu()

            # Feature frequency: fraction of examples each feature fires on
            freq = (z > 0).float().mean(dim=0).numpy()
            freq_nonzero = freq[freq > 0]

            # Fit power law: sort descending, check log-log slope
            sorted_freq = np.sort(freq_nonzero)[::-1]
            ranks = np.arange(1, len(sorted_freq) + 1)
            if len(sorted_freq) > 10:
                log_ranks = np.log(ranks)
                log_freq = np.log(sorted_freq + 1e-10)
                # Linear fit in log-log space = power law exponent
                slope, intercept = np.polyfit(log_ranks[:len(log_ranks)//2],
                                              log_freq[:len(log_freq)//2], 1)
            else:
                slope = 0

            # Gini coefficient (inequality measure)
            n = len(freq_nonzero)
            if n > 0:
                sorted_f = np.sort(freq_nonzero)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_f)) / (n * np.sum(sorted_f)) - (n + 1) / n
            else:
                gini = 0

            # Effective number of features (exponential of entropy)
            probs = freq_nonzero / freq_nonzero.sum() if freq_nonzero.sum() > 0 else freq_nonzero
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            effective_features = np.exp(entropy)

            depth = layer / max(info["n_layers"] - 1, 1)
            results[model_key][f"layer_{layer}"] = {
                "layer": layer,
                "depth": depth,
                "n_alive": int(len(freq_nonzero)),
                "n_dead": int(np.sum(freq == 0)),
                "mean_freq": float(freq_nonzero.mean()) if len(freq_nonzero) > 0 else 0,
                "median_freq": float(np.median(freq_nonzero)) if len(freq_nonzero) > 0 else 0,
                "max_freq": float(freq_nonzero.max()) if len(freq_nonzero) > 0 else 0,
                "zipf_slope": float(slope),
                "gini": float(gini),
                "effective_features": float(effective_features),
                "effective_ratio": float(effective_features / len(freq_nonzero)) if len(freq_nonzero) > 0 else 0,
            }
            print(f"  {model_key} L{layer}: alive={len(freq_nonzero)}, "
                  f"zipf={slope:.2f}, gini={gini:.3f}, "
                  f"effective={effective_features:.0f}/{len(freq_nonzero)}")

            del sae, acts, z

    with open(PHASE2_DIR / "feature_frequency.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved feature_frequency.json")
    return results


def experiment_2_decoder_geometry(device="cpu"):
    """Analyze decoder weight geometry: orthogonality, clustering."""
    print("\n" + "=" * 60)
    print("Experiment 2: Decoder Weight Geometry")
    print("=" * 60)

    results = {}
    for model_key, info in MODELS.items():
        results[model_key] = {}
        for layer in get_layer_indices(info["n_layers"]):
            sae = load_sae(model_key, layer, device=device)
            if sae is None:
                continue

            # Decoder columns: (d_model, d_hidden) — each column is a feature direction
            W = sae.decoder.weight.data.float()  # (d_model, d_hidden)
            W_normed = W / (W.norm(dim=0, keepdim=True) + 1e-8)

            # Pairwise cosine similarity (sample 2000 features for speed)
            n_features = min(2000, W.shape[1])
            W_sample = W_normed[:, :n_features]
            cos_sim = torch.mm(W_sample.T, W_sample).cpu()

            # Extract upper triangle (exclude diagonal)
            mask = torch.triu(torch.ones(n_features, n_features), diagonal=1).bool()
            pairwise = cos_sim[mask].numpy()

            # Mean absolute cosine similarity (0 = orthogonal, 1 = parallel)
            mean_abs_cos = float(np.abs(pairwise).mean())
            std_cos = float(pairwise.std())

            # Fraction of near-parallel pairs (|cos| > 0.5)
            frac_parallel = float((np.abs(pairwise) > 0.5).mean())
            frac_antiparallel = float((pairwise < -0.5).mean())

            # Effective rank of decoder matrix
            U, S, Vh = torch.linalg.svd(W[:, :n_features].cpu(), full_matrices=False)
            S_norm = S / S.sum()
            effective_rank = float(torch.exp(-torch.sum(S_norm * torch.log(S_norm + 1e-10))).item())

            depth = layer / max(info["n_layers"] - 1, 1)
            results[model_key][f"layer_{layer}"] = {
                "layer": layer,
                "depth": depth,
                "mean_abs_cosine": mean_abs_cos,
                "std_cosine": std_cos,
                "frac_parallel": frac_parallel,
                "frac_antiparallel": frac_antiparallel,
                "effective_rank": effective_rank,
                "n_features_sampled": n_features,
            }
            print(f"  {model_key} L{layer}: |cos|={mean_abs_cos:.4f}, "
                  f"parallel={frac_parallel:.3f}, rank={effective_rank:.0f}")

            del sae, W, cos_sim

    with open(PHASE2_DIR / "decoder_geometry.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved decoder_geometry.json")
    return results


def experiment_3_feature_coactivation(device="cpu"):
    """Analyze feature co-activation patterns."""
    print("\n" + "=" * 60)
    print("Experiment 3: Feature Co-activation Structure")
    print("=" * 60)

    results = {}
    for model_key, info in MODELS.items():
        results[model_key] = {}
        mid_layer = info["n_layers"] // 2
        # Only do middle layer (most interesting)
        for layer in [get_layer_indices(info["n_layers"])[0],
                      mid_layer,
                      get_layer_indices(info["n_layers"])[-1]]:
            sae = load_sae(model_key, layer, device=device)
            acts = load_acts_sample(model_key, layer, n=20000)
            if sae is None or acts is None:
                continue

            with torch.no_grad():
                z = sae.encode(acts.to(device)).cpu()

            # Binary activation matrix
            active = (z > 0).float()

            # Average pairwise co-activation (sample 1000 features)
            n_feat = min(1000, active.shape[1])
            active_sample = active[:, :n_feat]

            # Co-activation rate: P(both active) / P(either active)
            # Jaccard similarity on activation patterns
            intersection = torch.mm(active_sample.T, active_sample)
            row_sums = active_sample.sum(dim=0, keepdim=True)
            union = row_sums + row_sums.T - intersection
            jaccard = (intersection / (union + 1e-8))

            mask = torch.triu(torch.ones(n_feat, n_feat), diagonal=1).bool()
            jaccard_vals = jaccard[mask].numpy()

            # Correlation of activation magnitudes (for active features)
            # Sample fewer for speed
            n_corr = min(500, z.shape[1])
            z_sample = z[:, :n_corr]
            z_centered = z_sample - z_sample.mean(dim=0, keepdim=True)
            z_std = z_sample.std(dim=0, keepdim=True).clamp(min=1e-8)
            z_normed = z_centered / z_std
            corr = torch.mm(z_normed.T, z_normed) / z_normed.shape[0]
            corr_mask = torch.triu(torch.ones(n_corr, n_corr), diagonal=1).bool()
            corr_vals = corr[corr_mask].numpy()

            depth = layer / max(info["n_layers"] - 1, 1)
            results[model_key][f"layer_{layer}"] = {
                "layer": layer,
                "depth": depth,
                "mean_jaccard": float(jaccard_vals.mean()),
                "std_jaccard": float(jaccard_vals.std()),
                "mean_abs_correlation": float(np.abs(corr_vals).mean()),
                "frac_corr_above_0.3": float((np.abs(corr_vals) > 0.3).mean()),
            }
            print(f"  {model_key} L{layer}: jaccard={jaccard_vals.mean():.4f}, "
                  f"|corr|={np.abs(corr_vals).mean():.4f}")

            del sae, acts, z, active

    with open(PHASE2_DIR / "coactivation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved coactivation.json")
    return results


def experiment_4_within_model_cka(device="cpu"):
    """CKA between adjacent layers within each model — how fast do representations change?"""
    print("\n" + "=" * 60)
    print("Experiment 4: Within-Model Layer-to-Layer CKA")
    print("=" * 60)

    results = {}
    for model_key, info in MODELS.items():
        results[model_key] = {}
        layers = get_layer_indices(info["n_layers"])

        # Load SAE activations for all layers
        sae_acts = {}
        for layer in layers:
            sae = load_sae(model_key, layer, device=device)
            acts = load_acts_sample(model_key, layer, n=5000)
            if sae is None or acts is None:
                continue
            with torch.no_grad():
                z = sae.encode(acts.to(device)).cpu()
            sae_acts[layer] = z
            del sae, acts

        # Compute CKA between adjacent layer pairs
        sorted_layers = sorted(sae_acts.keys())
        for i in range(len(sorted_layers) - 1):
            la, lb = sorted_layers[i], sorted_layers[i + 1]
            a = sae_acts[la].float()
            b = sae_acts[lb].float()
            n = min(a.shape[0], b.shape[0])
            a, b = a[:n], b[:n]
            a = a - a.mean(dim=0, keepdim=True)
            b = b - b.mean(dim=0, keepdim=True)
            ka = torch.mm(a, a.T)
            kb = torch.mm(b, b.T)
            hsic_ab = (ka * kb).sum()
            hsic_aa = (ka * ka).sum()
            hsic_bb = (kb * kb).sum()
            cka = (hsic_ab / (torch.sqrt(hsic_aa * hsic_bb) + 1e-10)).item()

            depth_a = la / max(info["n_layers"] - 1, 1)
            results[model_key][f"L{la}_L{lb}"] = {
                "layer_a": la, "layer_b": lb,
                "depth_a": depth_a,
                "cka": cka,
            }
            print(f"  {model_key} L{la}→L{lb}: CKA={cka:.4f}")

        del sae_acts

    with open(PHASE2_DIR / "within_model_cka.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved within_model_cka.json")
    return results


def experiment_5_effective_dimensionality(device="cpu"):
    """Participation ratio: how many features effectively contribute?"""
    print("\n" + "=" * 60)
    print("Experiment 5: Effective Dimensionality (Participation Ratio)")
    print("=" * 60)

    results = {}
    for model_key, info in MODELS.items():
        results[model_key] = {}
        for layer in get_layer_indices(info["n_layers"]):
            sae = load_sae(model_key, layer, device=device)
            acts = load_acts_sample(model_key, layer, n=20000)
            if sae is None or acts is None:
                continue

            with torch.no_grad():
                z = sae.encode(acts.to(device)).cpu()

            # Participation ratio: (sum of variances)^2 / sum of (variances^2)
            # High PR = many features contribute equally
            # Low PR = few features dominate
            variances = z.var(dim=0).numpy()
            var_nonzero = variances[variances > 1e-10]

            if len(var_nonzero) > 0:
                pr = float((var_nonzero.sum() ** 2) / (var_nonzero ** 2).sum())
                # Also: fraction of total variance explained by top-k features
                sorted_var = np.sort(var_nonzero)[::-1]
                cumvar = np.cumsum(sorted_var) / sorted_var.sum()
                n_for_90 = int(np.searchsorted(cumvar, 0.9) + 1)
                n_for_50 = int(np.searchsorted(cumvar, 0.5) + 1)
            else:
                pr = 0
                n_for_90 = 0
                n_for_50 = 0

            depth = layer / max(info["n_layers"] - 1, 1)
            results[model_key][f"layer_{layer}"] = {
                "layer": layer,
                "depth": depth,
                "participation_ratio": pr,
                "n_alive_features": int(len(var_nonzero)),
                "features_for_90pct_var": n_for_90,
                "features_for_50pct_var": n_for_50,
                "pr_ratio": pr / len(var_nonzero) if len(var_nonzero) > 0 else 0,
            }
            print(f"  {model_key} L{layer}: PR={pr:.0f}, "
                  f"50%var={n_for_50}, 90%var={n_for_90}")

            del sae, acts, z

    with open(PHASE2_DIR / "effective_dim.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved effective_dim.json")
    return results


def main():
    print("=" * 60)
    print("PHASE 2: DEEP ANALYSIS EXPERIMENTS")
    print("=" * 60)

    experiment_1_feature_frequency("cpu")
    experiment_2_decoder_geometry("cpu")
    experiment_3_feature_coactivation("cpu")
    experiment_4_within_model_cka("cpu")
    experiment_5_effective_dimensionality("cpu")

    print("\n" + "=" * 60)
    print("ALL PHASE 2 EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
