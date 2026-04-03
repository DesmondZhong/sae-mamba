"""Analyze trained SAE features."""

import torch
import numpy as np
import json
from pathlib import Path
from src.sae import SparseAutoencoder


def find_max_activating_examples(sae, activations, token_texts, n_features=100,
                                  top_k=20, device="cuda"):
    """Find top-k activating examples for each feature."""
    sae.eval()
    batch_size = 4096
    n_samples = activations.shape[0]
    d_hidden = sae.d_hidden

    # Only analyze first n_features features (or fewer if dead)
    n_features = min(n_features, d_hidden)

    # Accumulate top activations
    top_vals = torch.zeros(n_features, top_k)
    top_indices = torch.zeros(n_features, top_k, dtype=torch.long)

    for i in range(0, n_samples, batch_size):
        batch = activations[i:i + batch_size].to(device)
        with torch.no_grad():
            x_centered = batch - sae.decoder.bias
            z = torch.relu(sae.encoder(x_centered))  # (batch, d_hidden)

        z_subset = z[:, :n_features].cpu()  # (batch, n_features)

        for feat in range(n_features):
            feat_acts = z_subset[:, feat]
            combined_vals = torch.cat([top_vals[feat], feat_acts])
            combined_idx = torch.cat([top_indices[feat], torch.arange(i, i + len(feat_acts))])
            topk = torch.topk(combined_vals, top_k)
            top_vals[feat] = topk.values
            top_indices[feat] = combined_idx[topk.indices]

    # Build feature info
    features = []
    for feat in range(n_features):
        examples = []
        for k in range(top_k):
            idx = top_indices[feat, k].item()
            val = top_vals[feat, k].item()
            if val > 0 and idx < len(token_texts):
                examples.append({"token_idx": idx, "activation": val, "text": token_texts[idx]})
        features.append({
            "feature_id": feat,
            "max_activation": top_vals[feat, 0].item(),
            "top_examples": examples,
        })

    return features


def compute_feature_stats(sae, activations, device="cuda", batch_size=4096):
    """Compute aggregate statistics about SAE features."""
    sae.eval()
    n_samples = activations.shape[0]
    d_hidden = sae.d_hidden

    # Track per-feature activation stats
    feature_active_count = torch.zeros(d_hidden)
    feature_act_sum = torch.zeros(d_hidden)
    l0_values = []
    recon_losses = []

    for i in range(0, n_samples, batch_size):
        batch = activations[i:i + batch_size].to(device)
        with torch.no_grad():
            x_hat, z, loss, metrics = sae(batch)

        z_cpu = z.cpu()
        active = (z_cpu > 0).float()
        feature_active_count += active.sum(dim=0)
        feature_act_sum += z_cpu.sum(dim=0)
        l0_values.append(active.sum(dim=1))  # per-example L0
        recon_losses.append(metrics["recon_loss"])

    l0_all = torch.cat(l0_values)
    dead_features = (feature_active_count == 0).sum().item()
    alive_features = d_hidden - dead_features

    stats = {
        "d_hidden": d_hidden,
        "n_samples": n_samples,
        "dead_features": dead_features,
        "alive_features": alive_features,
        "dead_frac": dead_features / d_hidden,
        "l0_mean": l0_all.mean().item(),
        "l0_median": l0_all.median().item(),
        "l0_std": l0_all.std().item(),
        "avg_recon_loss": np.mean(recon_losses),
        "feature_frequency": (feature_active_count / n_samples).tolist(),
    }

    return stats


def build_token_context(sequences, tokenizer, seq_len=512, context_window=10):
    """Build text contexts for each token position."""
    token_texts = []
    for seq_idx, seq in enumerate(sequences):
        tokens = seq.tolist()
        decoded = tokenizer.decode(tokens)
        for pos in range(len(tokens)):
            start = max(0, pos - context_window)
            end = min(len(tokens), pos + context_window + 1)
            context = tokenizer.decode(tokens[start:end])
            current = tokenizer.decode([tokens[pos]])
            token_texts.append(f"...{context}... [>{current}<]")
    return token_texts
