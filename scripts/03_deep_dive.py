#!/usr/bin/env python3
"""Deep dive: L1 sweep, monosemanticity scoring, cross-model comparison, downstream eval."""

import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, "/workspace/sae-mamba")

import torch
import json
import time
import numpy as np
from pathlib import Path
from src.sae import SparseAutoencoder
from src.train_sae import train_sae
from src.activation_cache import get_model_and_tokenizer, get_text_data, extract_residual_stream
from src.analyze import compute_feature_stats, find_max_activating_examples, build_token_context

DEVICE = "cuda"
RESULTS_DIR = Path("/workspace/sae-mamba/results")
CKPT_DIR = Path("/workspace/sae-mamba/checkpoints")

# ============================================================
# PART 1: L1 Coefficient Sweep (Pareto Frontier)
# ============================================================
print("=" * 70, flush=True)
print("PART 1: L1 coefficient sweep — sparsity vs reconstruction", flush=True)
print("=" * 70, flush=True)

# Use Mamba-130M layer 12 and Pythia-160M layer 6 (middle layers)
MODELS_FOR_SWEEP = {
    "mamba_130m": {"name": "state-spaces/mamba-130m-hf", "layer": 12},
    "pythia_160m": {"name": "EleutherAI/pythia-160m", "layer": 6},
}
L1_SWEEP = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
SAE_STEPS = 15000

sweep_results = {}

for model_key, model_info in MODELS_FOR_SWEEP.items():
    print(f"\n--- L1 sweep on {model_key} (layer {model_info['layer']}) ---", flush=True)

    model, tokenizer = get_model_and_tokenizer(model_info["name"], DEVICE)
    sequences = get_text_data(500_000, 512, tokenizer)

    layer_idx = model_info["layer"]
    print(f"Extracting activations at layer {layer_idx}...", flush=True)
    acts = extract_residual_stream(model, sequences, [layer_idx], DEVICE, batch_size=16)
    activations = acts[layer_idx]
    d_model = activations.shape[1]
    d_hidden = d_model * 4

    del model
    torch.cuda.empty_cache()

    sweep_results[model_key] = {"layer": layer_idx, "d_model": d_model, "sweeps": []}

    for l1 in L1_SWEEP:
        print(f"  L1={l1:.0e}...", flush=True)
        sae, history = train_sae(
            activations, d_hidden, l1,
            n_steps=SAE_STEPS, batch_size=4096, lr=1e-4,
            device=DEVICE, save_path=str(CKPT_DIR / f"{model_key}_sweep_l1_{l1:.0e}.pt"),
            log_interval=5000,
        )
        stats = compute_feature_stats(sae, activations, DEVICE)
        sweep_results[model_key]["sweeps"].append({
            "l1": l1,
            "l0": stats["l0_mean"],
            "l0_median": stats["l0_median"],
            "recon_loss": stats["avg_recon_loss"],
            "dead_frac": stats["dead_frac"],
            "alive_features": stats["alive_features"],
        })
        print(f"    L0={stats['l0_mean']:.1f}, recon={stats['avg_recon_loss']:.6f}, "
              f"dead={stats['dead_frac']:.1%}", flush=True)
        del sae
        torch.cuda.empty_cache()

    del activations
    torch.cuda.empty_cache()

with open(RESULTS_DIR / "l1_sweep.json", "w") as f:
    json.dump(sweep_results, f, indent=2)
print("Saved L1 sweep results", flush=True)

# ============================================================
# PART 2: Automated Monosemanticity Scoring
# ============================================================
print("\n" + "=" * 70, flush=True)
print("PART 2: Automated monosemanticity scoring", flush=True)
print("=" * 70, flush=True)

try:
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    HAS_EMBED = True
except Exception as e:
    print(f"Could not load sentence transformer: {e}. Using fallback.", flush=True)
    HAS_EMBED = False

mono_results = {}

# Load existing SAE results for max-activating examples
for model_key in ["mamba_130m", "mamba_370m", "pythia_160m"]:
    results_path = RESULTS_DIR / f"{model_key}_results.json"
    if not results_path.exists():
        continue

    with open(results_path) as f:
        model_data = json.load(f)

    mono_results[model_key] = {}

    for sae_key, sae_data in model_data["saes"].items():
        features = sae_data.get("top_features", [])
        if not features:
            continue

        mono_scores = []
        for feat in features:
            examples = feat.get("top_examples", [])
            texts = [ex["text"] for ex in examples if ex.get("text")]

            if len(texts) < 3:
                mono_scores.append(0.0)
                continue

            if HAS_EMBED:
                embeddings = embed_model.encode(texts[:10])
                # Average pairwise cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                sims = cosine_similarity(embeddings)
                n = len(embeddings)
                if n > 1:
                    upper_tri = sims[np.triu_indices(n, k=1)]
                    mono_scores.append(float(np.mean(upper_tri)))
                else:
                    mono_scores.append(0.0)
            else:
                # Fallback: character-level overlap
                from collections import Counter
                all_words = []
                for t in texts[:10]:
                    all_words.extend(t.lower().split())
                word_counts = Counter(all_words)
                if word_counts:
                    top_word_frac = word_counts.most_common(1)[0][1] / len(all_words)
                    mono_scores.append(top_word_frac)
                else:
                    mono_scores.append(0.0)

        mono_results[model_key][sae_key] = {
            "scores": mono_scores,
            "mean": float(np.mean(mono_scores)) if mono_scores else 0,
            "median": float(np.median(mono_scores)) if mono_scores else 0,
            "std": float(np.std(mono_scores)) if mono_scores else 0,
            "n_features": len(mono_scores),
        }
        print(f"  {sae_key}: mean_mono={mono_results[model_key][sae_key]['mean']:.3f} "
              f"(n={len(mono_scores)} features)", flush=True)

with open(RESULTS_DIR / "monosemanticity.json", "w") as f:
    json.dump(mono_results, f, indent=2)
print("Saved monosemanticity scores", flush=True)

if HAS_EMBED:
    del embed_model
    torch.cuda.empty_cache()

# ============================================================
# PART 3: Cross-Model Feature Comparison
# ============================================================
print("\n" + "=" * 70, flush=True)
print("PART 3: Cross-model feature comparison", flush=True)
print("=" * 70, flush=True)

# Compare features between Mamba and Pythia by max-activating text overlap
cross_model = {}

# Load features for middle layers
model_features = {}
for model_key in ["mamba_130m", "pythia_160m"]:
    results_path = RESULTS_DIR / f"{model_key}_results.json"
    if not results_path.exists():
        continue
    with open(results_path) as f:
        data = json.load(f)

    for sae_key, sae_data in data["saes"].items():
        # Use middle layer SAE
        if "L12" in sae_key or "L6" in sae_key:
            features = sae_data.get("top_features", [])
            model_features[model_key] = {
                "sae_key": sae_key,
                "features": features,
            }

if "mamba_130m" in model_features and "pythia_160m" in model_features:
    mamba_feats = model_features["mamba_130m"]["features"]
    pythia_feats = model_features["pythia_160m"]["features"]

    # For each Mamba feature, find most similar Pythia feature by text overlap
    matches = []
    for mi, mf in enumerate(mamba_feats[:20]):
        mamba_texts = set(ex.get("text", "")[:50] for ex in mf.get("top_examples", []))
        best_score = 0
        best_pi = -1
        for pi, pf in enumerate(pythia_feats[:20]):
            pythia_texts = set(ex.get("text", "")[:50] for ex in pf.get("top_examples", []))
            overlap = len(mamba_texts & pythia_texts)
            if overlap > best_score:
                best_score = overlap
                best_pi = pi

        matches.append({
            "mamba_feature": mi,
            "pythia_feature": best_pi,
            "overlap_count": best_score,
            "mamba_examples": [ex.get("text", "")[:100] for ex in mf.get("top_examples", [])[:3]],
            "pythia_examples": [ex.get("text", "")[:100] for ex in (pythia_feats[best_pi].get("top_examples", [])[:3] if best_pi >= 0 else [])],
        })

    cross_model = {
        "mamba_sae": model_features["mamba_130m"]["sae_key"],
        "pythia_sae": model_features["pythia_160m"]["sae_key"],
        "matches": matches,
        "avg_overlap": np.mean([m["overlap_count"] for m in matches]),
    }
    print(f"Average feature overlap: {cross_model['avg_overlap']:.2f} shared max-activating texts", flush=True)

with open(RESULTS_DIR / "cross_model.json", "w") as f:
    json.dump(cross_model, f, indent=2, default=str)
print("Saved cross-model comparison", flush=True)

# ============================================================
# PART 4: Downstream Reconstruction Evaluation
# ============================================================
print("\n" + "=" * 70, flush=True)
print("PART 4: Downstream reconstruction evaluation", flush=True)
print("=" * 70, flush=True)

downstream_results = {}

for model_key, model_info in [
    ("mamba_130m", {"name": "state-spaces/mamba-130m-hf", "layer": 12, "d_model": 768}),
    ("pythia_160m", {"name": "EleutherAI/pythia-160m", "layer": 6, "d_model": 768}),
]:
    print(f"\n--- Downstream eval: {model_key} ---", flush=True)

    model, tokenizer = get_model_and_tokenizer(model_info["name"], DEVICE)
    sequences = get_text_data(100_000, 512, tokenizer)
    layer_idx = model_info["layer"]

    # Compute baseline perplexity
    print("  Computing baseline perplexity...", flush=True)
    total_loss = 0
    n_tokens = 0
    for i in range(0, len(sequences), 8):
        batch = sequences[i:i+8].to(DEVICE)
        with torch.no_grad():
            outputs = model(batch, labels=batch)
            total_loss += outputs.loss.item() * batch.numel()
            n_tokens += batch.numel()
    baseline_ppl = np.exp(total_loss / n_tokens)
    print(f"  Baseline perplexity: {baseline_ppl:.2f}", flush=True)

    # Load the best SAE for this model/layer
    sae_key = f"{model_key}_residual_L{layer_idx}"
    sae_path = CKPT_DIR / f"{sae_key}.pt"
    if not sae_path.exists():
        print(f"  SAE not found at {sae_path}, skipping", flush=True)
        del model; torch.cuda.empty_cache()
        continue

    d_hidden = model_info["d_model"] * 4
    sae = SparseAutoencoder(model_info["d_model"], d_hidden, l1_coeff=1e-3).to(DEVICE)
    sae.load_state_dict(torch.load(sae_path, map_location=DEVICE, weights_only=True))
    sae.eval()

    # Compute perplexity with SAE-reconstructed activations
    print("  Computing perplexity with SAE reconstruction...", flush=True)
    total_loss_sae = 0
    n_tokens_sae = 0

    # Hook to replace activations with SAE reconstruction
    def make_sae_hook(sae_model, target_layer):
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            batch_size, seq_len, d = hidden_states.shape
            flat = hidden_states.reshape(-1, d)
            with torch.no_grad():
                reconstructed, _, _, _ = sae_model(flat.float())
            hidden_states_new = reconstructed.half().reshape(batch_size, seq_len, d)
            if isinstance(output, tuple):
                return (hidden_states_new,) + output[1:]
            return hidden_states_new
        return hook_fn

    # Find the right module to hook
    if "mamba" in model_key:
        target_module = model.backbone.layers[layer_idx]
    else:
        target_module = model.gpt_neox.layers[layer_idx]

    hook = target_module.register_forward_hook(make_sae_hook(sae, layer_idx))

    for i in range(0, min(len(sequences), 50), 8):  # Smaller sample for speed
        batch = sequences[i:i+8].to(DEVICE)
        with torch.no_grad():
            outputs = model(batch, labels=batch)
            total_loss_sae += outputs.loss.item() * batch.numel()
            n_tokens_sae += batch.numel()

    hook.remove()
    sae_ppl = np.exp(total_loss_sae / n_tokens_sae)
    ppl_increase = sae_ppl / baseline_ppl

    downstream_results[model_key] = {
        "baseline_ppl": baseline_ppl,
        "sae_ppl": sae_ppl,
        "ppl_increase_ratio": ppl_increase,
        "layer": layer_idx,
    }
    print(f"  SAE perplexity: {sae_ppl:.2f} (ratio: {ppl_increase:.2f}x)", flush=True)

    del model, sae
    torch.cuda.empty_cache()

with open(RESULTS_DIR / "downstream.json", "w") as f:
    json.dump(downstream_results, f, indent=2)
print("Saved downstream results", flush=True)

print("\n" + "=" * 70, flush=True)
print("All SAE deep dive analyses complete!", flush=True)
