"""Analyze trained SAE features: CKA, monosemanticity, baselines, feature stats."""

import torch
import numpy as np
import json
from pathlib import Path


def compute_feature_stats(sae, activations, device="cuda", batch_size=4096):
    """Compute aggregate statistics about SAE features."""
    sae.eval()
    n_samples = activations.shape[0]
    d_hidden = sae.d_hidden

    feature_active_count = torch.zeros(d_hidden)
    feature_act_sum = torch.zeros(d_hidden)
    l0_values = []
    recon_losses = []
    total_var = []

    for i in range(0, n_samples, batch_size):
        batch = activations[i:i + batch_size].to(device)
        with torch.no_grad():
            x_hat, z, loss, metrics = sae(batch)

        z_cpu = z.cpu()
        active = (z_cpu > 0).float()
        feature_active_count += active.sum(dim=0)
        feature_act_sum += z_cpu.sum(dim=0)
        l0_values.append(active.sum(dim=1))
        recon_losses.append((batch - x_hat).pow(2).sum(dim=-1).cpu())
        total_var.append(batch.var(dim=-1).cpu())

    l0_all = torch.cat(l0_values)
    recon_all = torch.cat(recon_losses)
    var_all = torch.cat(total_var)
    dead_features = (feature_active_count == 0).sum().item()
    alive_features = d_hidden - dead_features

    # FVE: fraction of variance explained
    act_var = activations[:min(50000, len(activations))].to(device).var().item()
    avg_recon = recon_all.mean().item() / activations.shape[1]
    fve = 1.0 - avg_recon / max(act_var, 1e-10)

    stats = {
        "d_hidden": d_hidden,
        "n_samples": n_samples,
        "dead_features": dead_features,
        "alive_features": alive_features,
        "dead_frac": dead_features / d_hidden,
        "l0_mean": l0_all.mean().item(),
        "l0_median": l0_all.median().item(),
        "l0_std": l0_all.std().item(),
        "avg_recon_loss": avg_recon,
        "fve": fve,
        "feature_frequency": (feature_active_count / n_samples).tolist(),
    }

    return stats


def compute_cka(acts_a: torch.Tensor, acts_b: torch.Tensor, n_samples: int = 10000):
    """Compute CKA (Centered Kernel Alignment) between two activation matrices.

    acts_a: (n, d1) — SAE activations from model A on shared inputs
    acts_b: (n, d2) — SAE activations from model B on shared inputs
    Returns: float in [0, 1], where 1 = identical representations
    """
    n = min(n_samples, acts_a.shape[0], acts_b.shape[0])
    a = acts_a[:n].float()
    b = acts_b[:n].float()

    # Center
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)

    # Linear CKA = ||A^T B||_F^2 / (||A^T A||_F * ||B^T B||_F)
    ab = torch.mm(a.T, b)
    aa = torch.mm(a.T, a)
    bb = torch.mm(b.T, b)

    cka = (ab ** 2).sum() / (torch.sqrt((aa ** 2).sum() * (bb ** 2).sum()) + 1e-10)
    return cka.item()


def compute_cka_batched(acts_a: torch.Tensor, acts_b: torch.Tensor,
                        n_samples: int = 10000, device: str = "cuda"):
    """GPU-accelerated CKA for large feature dimensions."""
    n = min(n_samples, acts_a.shape[0], acts_b.shape[0])
    a = acts_a[:n].float().to(device)
    b = acts_b[:n].float().to(device)

    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)

    # Use kernel trick for high-dimensional features: K = XX^T
    # CKA = HSIC(K_a, K_b) / sqrt(HSIC(K_a, K_a) * HSIC(K_b, K_b))
    # With linear kernel: HSIC = 1/(n-1)^2 * tr(K_a H K_b H)
    # = 1/(n-1)^2 * ||A^T B||_F^2 (after centering)

    # For very large d, compute K = AA^T (n×n) instead of A^T A (d×d)
    if a.shape[1] > n:
        ka = torch.mm(a, a.T)
        kb = torch.mm(b, b.T)
        hsic_ab = (ka * kb).sum()
        hsic_aa = (ka * ka).sum()
        hsic_bb = (kb * kb).sum()
    else:
        ab = torch.mm(a.T, b)
        aa = torch.mm(a.T, a)
        bb = torch.mm(b.T, b)
        hsic_ab = (ab ** 2).sum()
        hsic_aa = (aa ** 2).sum()
        hsic_bb = (bb ** 2).sum()

    cka = hsic_ab / (torch.sqrt(hsic_aa * hsic_bb) + 1e-10)
    result = cka.item()
    del a, b
    torch.cuda.empty_cache()
    return result


def compute_random_baseline(activations: torch.Tensor, k: int, device: str = "cuda",
                            n_trials: int = 5, n_samples: int = 10000):
    """Compute FVE using random orthogonal directions as baseline."""
    n = min(n_samples, activations.shape[0])
    x = activations[:n].float().to(device)
    d = x.shape[1]

    fves = []
    for _ in range(n_trials):
        # Random orthogonal matrix (first k columns)
        random_matrix = torch.randn(d, k, device=device)
        q, _ = torch.linalg.qr(random_matrix)  # (d, k) orthonormal

        # Project and reconstruct
        z = torch.mm(x, q)  # (n, k)
        x_hat = torch.mm(z, q.T)  # (n, d)

        mse = (x - x_hat).pow(2).mean().item()
        var = x.var().item()
        fves.append(1.0 - mse / max(var, 1e-10))

    del x
    torch.cuda.empty_cache()
    return {"mean": np.mean(fves), "std": np.std(fves)}


def compute_pca_baseline(activations: torch.Tensor, k: int, device: str = "cuda",
                         n_samples: int = 10000):
    """Compute FVE using top-k PCA components as baseline."""
    n = min(n_samples, activations.shape[0])
    x = activations[:n].float().to(device)
    x_centered = x - x.mean(dim=0, keepdim=True)

    # Truncated SVD for efficiency
    U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
    # Reconstruct with top-k components
    x_hat = torch.mm(torch.mm(U[:, :k], torch.diag(S[:k])), Vh[:k, :])
    x_hat += x.mean(dim=0, keepdim=True)

    mse = (x - x_hat).pow(2).mean().item()
    var = x.var().item()
    fve = 1.0 - mse / max(var, 1e-10)

    del x, x_centered, U, S, Vh, x_hat
    torch.cuda.empty_cache()
    return fve


def find_max_activating_examples(sae, activations, token_texts, n_features=100,
                                  top_k=20, device="cuda"):
    """Find top-k activating examples for each feature."""
    sae.eval()
    batch_size = 4096
    n_samples = activations.shape[0]
    d_hidden = sae.d_hidden
    n_features = min(n_features, d_hidden)

    top_vals = torch.zeros(n_features, top_k)
    top_indices = torch.zeros(n_features, top_k, dtype=torch.long)

    for i in range(0, n_samples, batch_size):
        batch = activations[i:i + batch_size].to(device)
        with torch.no_grad():
            z = sae.encode(batch)

        z_subset = z[:, :n_features].cpu()

        for feat in range(n_features):
            feat_acts = z_subset[:, feat]
            combined_vals = torch.cat([top_vals[feat], feat_acts])
            combined_idx = torch.cat([top_indices[feat], torch.arange(i, i + len(feat_acts))])
            topk = torch.topk(combined_vals, top_k)
            top_vals[feat] = topk.values
            top_indices[feat] = combined_idx[topk.indices]

    features = []
    for feat in range(n_features):
        examples = []
        for k_idx in range(top_k):
            idx = top_indices[feat, k_idx].item()
            val = top_vals[feat, k_idx].item()
            if val > 0 and idx < len(token_texts):
                examples.append({"token_idx": idx, "activation": val, "text": token_texts[idx]})
        features.append({
            "feature_id": feat,
            "max_activation": top_vals[feat, 0].item(),
            "top_examples": examples,
        })

    return features


def compute_monosemanticity(features, device="cuda"):
    """Score monosemanticity using sentence-transformer embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("sentence-transformers not available, skipping monosemanticity scoring")
        return None

    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    scores = []

    for feat in features:
        texts = [ex["text"] for ex in feat.get("top_examples", []) if ex.get("text")]
        if len(texts) < 3:
            scores.append(0.0)
            continue
        embeddings = embed_model.encode(texts[:10])
        sims = cosine_similarity(embeddings)
        n = len(embeddings)
        upper_tri = sims[np.triu_indices(n, k=1)]
        scores.append(float(np.mean(upper_tri)))

    del embed_model
    torch.cuda.empty_cache()

    return {
        "scores": scores,
        "mean": float(np.mean(scores)) if scores else 0,
        "median": float(np.median(scores)) if scores else 0,
        "std": float(np.std(scores)) if scores else 0,
        "n_features": len(scores),
    }


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
