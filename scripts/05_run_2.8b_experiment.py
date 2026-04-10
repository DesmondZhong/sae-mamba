#!/usr/bin/env python3
"""
Full 2.8B-scale SAE experiment: Mamba-1 vs Mamba-2 vs Pythia.

Three-way comparison with:
- TopK SAE (fixed K for fair comparison)
- Layer sweep (every 4th layer)
- Expansion sweep (8x, 16x, 32x)
- CKA cross-model comparison
- Random/PCA baselines
- Monosemanticity scoring
- Downstream perplexity evaluation

Designed for multi-GPU (8x L40S) with streaming extraction.
"""

import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.multiprocessing as mp
import json
import time
import gc
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.sae import create_sae, TopKSAE
from src.activation_cache import (get_model_and_tokenizer, get_text_data,
                                   extract_residual_stream)
from src.train_sae import train_sae
from src.analyze import (compute_feature_stats, compute_cka_batched,
                          compute_random_baseline, compute_pca_baseline,
                          find_max_activating_examples, compute_monosemanticity,
                          build_token_context)

# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("/root/sae-mamba/results_2.8b")
CKPT_DIR = Path("/root/sae-mamba/checkpoints_2.8b")
ACTS_DIR = Path("/root/sae-mamba/activations_2.8b")
RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)
ACTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "mamba1_2.8b": {
        "name": "state-spaces/mamba-2.8b-hf",
        "type": "mamba1",
        "n_layers": 64,
        "d_model": 2560,
    },
    "mamba2_2.7b": {
        "name": "state-spaces/mamba2-2.7b",
        "type": "mamba2",
        "n_layers": 64,
        "d_model": 2560,
    },
    "pythia_2.8b": {
        "name": "EleutherAI/pythia-2.8b",
        "type": "transformer",
        "n_layers": 32,
        "d_model": 2560,
    },
}

# Token budget
N_TOKENS = 10_000_000      # 10M tokens for activation extraction
SEQ_LEN = 512
DATASET = "pile"

# SAE config
SAE_TYPE = "topk"
K_VALUES = [32, 64, 128]   # TopK values to sweep
DEFAULT_K = 64
EXPANSION_RATIOS = [8, 16, 32]   # d_hidden = ratio * d_model
DEFAULT_EXPANSION = 16
SAE_STEPS = 30000
SAE_BATCH = 4096
SAE_LR = 3e-4

# Layer selection: every 4th layer
def get_layer_indices(n_layers):
    """Select layers to analyze: every 4th layer."""
    return list(range(0, n_layers, max(1, n_layers // 8)))


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {path}")


# ============================================================
# Phase 1: Extract activations from all models
# ============================================================

def extract_model_activations(model_key, model_info, gpu_id=0):
    """Extract activations from one model on one GPU."""
    device = f"cuda:{gpu_id}"
    print(f"\n{'='*70}")
    print(f"[GPU {gpu_id}] Extracting activations: {model_key}")
    print(f"{'='*70}")

    model, tokenizer = get_model_and_tokenizer(model_info["name"], device)

    # Prepare data
    sequences = get_text_data(N_TOKENS, SEQ_LEN, tokenizer, dataset_name=DATASET)

    # Get layer indices
    layer_indices = get_layer_indices(model_info["n_layers"])
    print(f"[{model_key}] Layers to extract: {layer_indices} ({len(layer_indices)} layers)")

    # Extract in chunks to manage memory
    chunk_size = min(len(sequences), 2000)  # ~1M tokens per chunk
    all_activations = {layer: [] for layer in layer_indices}

    for chunk_start in range(0, len(sequences), chunk_size):
        chunk_seqs = sequences[chunk_start:chunk_start + chunk_size]
        print(f"  [{model_key}] Chunk {chunk_start//chunk_size + 1}/"
              f"{(len(sequences) + chunk_size - 1)//chunk_size} "
              f"({chunk_seqs.shape[0]} sequences)")

        acts = extract_residual_stream(model, chunk_seqs, layer_indices,
                                       device, batch_size=2)
        for layer in layer_indices:
            all_activations[layer].append(acts[layer])

        del acts
        torch.cuda.empty_cache()

    # Concatenate and save
    model_acts_dir = ACTS_DIR / model_key
    model_acts_dir.mkdir(exist_ok=True)

    for layer in layer_indices:
        combined = torch.cat(all_activations[layer], dim=0)
        save_path = model_acts_dir / f"layer_{layer}.pt"
        torch.save(combined, save_path)
        print(f"  [{model_key}] Layer {layer}: {combined.shape} → {save_path}")

    # Save shared sequences for CKA (first 1000 sequences)
    shared_seqs_path = ACTS_DIR / f"{model_key}_sequences.pt"
    torch.save(sequences[:1000], shared_seqs_path)

    # Save token contexts for feature browsing
    token_texts = build_token_context(sequences[:200], tokenizer, SEQ_LEN, context_window=8)
    texts_path = ACTS_DIR / f"{model_key}_token_texts.json"
    with open(texts_path, "w") as f:
        json.dump(token_texts[:100000], f)

    del model, tokenizer, sequences, all_activations
    gc.collect()
    torch.cuda.empty_cache()

    return model_key, layer_indices


# ============================================================
# Phase 2: Train SAEs
# ============================================================

def train_single_sae(model_key, layer_idx, expansion_ratio, k, gpu_id=0,
                     sae_type=SAE_TYPE, n_steps=SAE_STEPS):
    """Train a single SAE on pre-extracted activations."""
    device = f"cuda:{gpu_id}"
    d_model = MODELS[model_key]["d_model"]
    d_hidden = d_model * expansion_ratio

    run_key = f"{model_key}_L{layer_idx}_x{expansion_ratio}_k{k}"
    ckpt_path = str(CKPT_DIR / f"{run_key}.pt")

    # Skip if already trained
    if Path(ckpt_path).exists():
        print(f"  [Skip] {run_key} already exists")
        return run_key, ckpt_path

    # Load activations
    acts_path = ACTS_DIR / model_key / f"layer_{layer_idx}.pt"
    if not acts_path.exists():
        print(f"  [Skip] No activations for {model_key} layer {layer_idx}")
        return run_key, None

    print(f"\n--- Training SAE: {run_key} (d_hidden={d_hidden}, K={k}) on GPU {gpu_id} ---")
    activations = torch.load(acts_path, map_location="cpu", weights_only=True)

    sae, history, summary = train_sae(
        activations, d_hidden, sae_type=sae_type, k=k,
        n_steps=n_steps, batch_size=SAE_BATCH, lr=SAE_LR,
        device=device, save_path=ckpt_path,
    )

    # Compute stats
    stats = compute_feature_stats(sae, activations, device)
    stats.update(summary)
    stats["model_key"] = model_key
    stats["layer"] = layer_idx
    stats["expansion_ratio"] = expansion_ratio
    stats["k"] = k
    stats["sae_type"] = sae_type
    stats["run_key"] = run_key

    stats_path = RESULTS_DIR / f"{run_key}_stats.json"
    save_json(stats, stats_path)

    del sae, activations
    gc.collect()
    torch.cuda.empty_cache()
    return run_key, ckpt_path


# ============================================================
# Phase 3: Analysis
# ============================================================

def run_cka_analysis(model_pairs, layer_pairs, device="cuda:0"):
    """Compute CKA between model pairs at matching layers."""
    print(f"\n{'='*70}")
    print("Phase 3a: CKA cross-model comparison")
    print(f"{'='*70}")

    cka_results = {}

    for model_a, model_b in model_pairs:
        pair_key = f"{model_a}_vs_{model_b}"
        cka_results[pair_key] = {}

        layers_a = get_layer_indices(MODELS[model_a]["n_layers"])
        layers_b = get_layer_indices(MODELS[model_b]["n_layers"])

        # Match layers by relative depth
        n_comparisons = min(len(layers_a), len(layers_b))
        for i in range(n_comparisons):
            la = layers_a[min(i, len(layers_a)-1)]
            lb = layers_b[min(i, len(layers_b)-1)]

            # Load SAE activations on shared inputs
            sae_a_path = CKPT_DIR / f"{model_a}_L{la}_x{DEFAULT_EXPANSION}_k{DEFAULT_K}.pt"
            sae_b_path = CKPT_DIR / f"{model_b}_L{lb}_x{DEFAULT_EXPANSION}_k{DEFAULT_K}.pt"
            acts_a_path = ACTS_DIR / model_a / f"layer_{la}.pt"
            acts_b_path = ACTS_DIR / model_b / f"layer_{lb}.pt"

            if not all(p.exists() for p in [sae_a_path, sae_b_path, acts_a_path, acts_b_path]):
                continue

            d_model = MODELS[model_a]["d_model"]
            d_hidden = d_model * DEFAULT_EXPANSION

            # Load SAEs
            sae_a = create_sae(d_model, d_hidden, sae_type=SAE_TYPE, k=DEFAULT_K).to(device)
            sae_a.load_state_dict(torch.load(sae_a_path, map_location=device, weights_only=True))
            sae_a.eval()

            sae_b = create_sae(d_model, d_hidden, sae_type=SAE_TYPE, k=DEFAULT_K).to(device)
            sae_b.load_state_dict(torch.load(sae_b_path, map_location=device, weights_only=True))
            sae_b.eval()

            # Get SAE activations on raw model activations
            raw_a = torch.load(acts_a_path, map_location="cpu", weights_only=True)
            raw_b = torch.load(acts_b_path, map_location="cpu", weights_only=True)

            n = min(10000, raw_a.shape[0], raw_b.shape[0])
            with torch.no_grad():
                z_a = sae_a.encode(raw_a[:n].to(device)).cpu()
                z_b = sae_b.encode(raw_b[:n].to(device)).cpu()

            # Also compute CKA on raw activations (pre-SAE)
            cka_raw = compute_cka_batched(raw_a[:n], raw_b[:n], device=device)
            cka_sae = compute_cka_batched(z_a, z_b, device=device)

            depth_a = la / max(MODELS[model_a]["n_layers"] - 1, 1)
            depth_b = lb / max(MODELS[model_b]["n_layers"] - 1, 1)

            cka_results[pair_key][f"depth_{depth_a:.2f}"] = {
                "layer_a": la, "layer_b": lb,
                "depth_a": depth_a, "depth_b": depth_b,
                "cka_raw": cka_raw,
                "cka_sae": cka_sae,
            }
            print(f"  {pair_key} depth={depth_a:.2f}: CKA_raw={cka_raw:.4f}, CKA_sae={cka_sae:.4f}")

            del sae_a, sae_b, raw_a, raw_b, z_a, z_b
            torch.cuda.empty_cache()

    save_json(cka_results, RESULTS_DIR / "cka_results.json")
    return cka_results


def run_baselines(device="cuda:0"):
    """Compute random direction and PCA baselines."""
    print(f"\n{'='*70}")
    print("Phase 3b: Random and PCA baselines")
    print(f"{'='*70}")

    baseline_results = {}

    for model_key in MODELS:
        baseline_results[model_key] = {}
        layers = get_layer_indices(MODELS[model_key]["n_layers"])

        for layer in layers:
            acts_path = ACTS_DIR / model_key / f"layer_{layer}.pt"
            if not acts_path.exists():
                continue

            activations = torch.load(acts_path, map_location="cpu", weights_only=True)
            k = DEFAULT_K

            # Random baseline
            rand_result = compute_random_baseline(activations, k=k, device=device)

            # PCA baseline
            pca_fve = compute_pca_baseline(activations, k=k, device=device)

            # SAE FVE (load from stats if available)
            sae_stats_path = RESULTS_DIR / f"{model_key}_L{layer}_x{DEFAULT_EXPANSION}_k{DEFAULT_K}_stats.json"
            sae_fve = None
            if sae_stats_path.exists():
                with open(sae_stats_path) as f:
                    sae_fve = json.load(f).get("fve")

            depth = layer / max(MODELS[model_key]["n_layers"] - 1, 1)
            baseline_results[model_key][f"layer_{layer}"] = {
                "layer": layer,
                "depth": depth,
                "random_fve_mean": rand_result["mean"],
                "random_fve_std": rand_result["std"],
                "pca_fve": pca_fve,
                "sae_fve": sae_fve,
            }
            print(f"  {model_key} L{layer}: random={rand_result['mean']:.4f}, "
                  f"PCA={pca_fve:.4f}, SAE={sae_fve}")

            del activations
            torch.cuda.empty_cache()

    save_json(baseline_results, RESULTS_DIR / "baselines.json")
    return baseline_results


def run_downstream_eval(device="cuda:0"):
    """Evaluate perplexity impact of SAE reconstruction."""
    print(f"\n{'='*70}")
    print("Phase 3c: Downstream perplexity evaluation")
    print(f"{'='*70}")

    downstream_results = {}

    for model_key, model_info in MODELS.items():
        print(f"\n--- Downstream eval: {model_key} ---")

        model, tokenizer = get_model_and_tokenizer(model_info["name"], device)
        sequences = get_text_data(500_000, SEQ_LEN, tokenizer, dataset_name=DATASET)

        # Baseline perplexity
        total_loss, n_tokens = 0, 0
        for i in range(0, min(len(sequences), 100), 4):
            batch = sequences[i:i+4].to(device)
            with torch.no_grad():
                outputs = model(batch, labels=batch)
                total_loss += outputs.loss.item() * batch.numel()
                n_tokens += batch.numel()
        baseline_ppl = np.exp(total_loss / n_tokens)
        print(f"  Baseline PPL: {baseline_ppl:.2f}")

        # Test SAE reconstruction at middle layer
        mid_layer = model_info["n_layers"] // 2
        sae_path = CKPT_DIR / f"{model_key}_L{mid_layer}_x{DEFAULT_EXPANSION}_k{DEFAULT_K}.pt"

        if sae_path.exists():
            d_model = model_info["d_model"]
            d_hidden = d_model * DEFAULT_EXPANSION
            sae = create_sae(d_model, d_hidden, sae_type=SAE_TYPE, k=DEFAULT_K).to(device)
            sae.load_state_dict(torch.load(sae_path, map_location=device, weights_only=True))
            sae.eval()

            def make_hook(sae_model):
                def hook_fn(module, input, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    b, s, d = hs.shape
                    flat = hs.reshape(-1, d)
                    with torch.no_grad():
                        recon, _, _, _ = sae_model(flat.float())
                    hs_new = recon.half().reshape(b, s, d)
                    if isinstance(output, tuple):
                        return (hs_new,) + output[1:]
                    return hs_new
                return hook_fn

            # Find correct module to hook
            if "mamba" in model_key and hasattr(model, 'backbone'):
                target = model.backbone.layers[mid_layer]
            elif hasattr(model, 'gpt_neox'):
                target = model.gpt_neox.layers[mid_layer]
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                target = model.model.layers[mid_layer]
            else:
                print(f"  Cannot find layers for {model_key}, skipping")
                del model, tokenizer
                torch.cuda.empty_cache()
                continue

            hook = target.register_forward_hook(make_hook(sae))
            total_loss_sae, n_tokens_sae = 0, 0
            for i in range(0, min(len(sequences), 100), 4):
                batch = sequences[i:i+4].to(device)
                with torch.no_grad():
                    outputs = model(batch, labels=batch)
                    total_loss_sae += outputs.loss.item() * batch.numel()
                    n_tokens_sae += batch.numel()
            hook.remove()
            sae_ppl = np.exp(total_loss_sae / n_tokens_sae)

            downstream_results[model_key] = {
                "baseline_ppl": baseline_ppl,
                "sae_ppl": sae_ppl,
                "ppl_ratio": sae_ppl / baseline_ppl,
                "layer": mid_layer,
                "expansion": DEFAULT_EXPANSION,
                "k": DEFAULT_K,
            }
            print(f"  SAE PPL: {sae_ppl:.2f} (ratio: {sae_ppl/baseline_ppl:.4f}x)")
            del sae
        else:
            downstream_results[model_key] = {
                "baseline_ppl": baseline_ppl,
                "sae_ppl": None,
                "layer": mid_layer,
            }

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    save_json(downstream_results, RESULTS_DIR / "downstream.json")
    return downstream_results


def run_monosemanticity_analysis(device="cuda:0"):
    """Score monosemanticity for top features."""
    print(f"\n{'='*70}")
    print("Phase 3d: Monosemanticity scoring")
    print(f"{'='*70}")

    mono_results = {}

    for model_key in MODELS:
        mono_results[model_key] = {}
        layers = get_layer_indices(MODELS[model_key]["n_layers"])
        mid_layer = layers[len(layers) // 2]

        # Load token texts
        texts_path = ACTS_DIR / f"{model_key}_token_texts.json"
        if not texts_path.exists():
            continue
        with open(texts_path) as f:
            token_texts = json.load(f)

        run_key = f"{model_key}_L{mid_layer}_x{DEFAULT_EXPANSION}_k{DEFAULT_K}"
        sae_path = CKPT_DIR / f"{run_key}.pt"
        acts_path = ACTS_DIR / model_key / f"layer_{mid_layer}.pt"

        if not sae_path.exists() or not acts_path.exists():
            continue

        d_model = MODELS[model_key]["d_model"]
        d_hidden = d_model * DEFAULT_EXPANSION
        sae = create_sae(d_model, d_hidden, sae_type=SAE_TYPE, k=DEFAULT_K).to(device)
        sae.load_state_dict(torch.load(sae_path, map_location=device, weights_only=True))
        sae.eval()

        activations = torch.load(acts_path, map_location="cpu", weights_only=True)
        features = find_max_activating_examples(sae, activations, token_texts,
                                                 n_features=50, top_k=20, device=device)

        mono = compute_monosemanticity(features, device=device)
        if mono:
            mono_results[model_key][run_key] = mono
            print(f"  {run_key}: mean_mono={mono['mean']:.3f}")

        # Save features for web visualization
        save_json(features[:30], RESULTS_DIR / f"{run_key}_features.json")

        del sae, activations
        torch.cuda.empty_cache()

    save_json(mono_results, RESULTS_DIR / "monosemanticity.json")
    return mono_results


# ============================================================
# Main orchestration
# ============================================================

def main():
    start_time = time.time()
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Tokens: {N_TOKENS:,}, Dataset: {DATASET}")
    print(f"SAE type: {SAE_TYPE}, K: {DEFAULT_K}, Expansion: {DEFAULT_EXPANSION}")

    # ========================================
    # Phase 1: Extract activations (1 model per GPU pair)
    # ========================================
    print(f"\n{'#'*70}")
    print("# PHASE 1: ACTIVATION EXTRACTION")
    print(f"{'#'*70}")

    model_layers = {}
    model_list = list(MODELS.items())

    # Run extraction sequentially per model (each model needs ~12GB+ VRAM)
    for idx, (model_key, model_info) in enumerate(model_list):
        gpu_id = idx % n_gpus
        key, layers = extract_model_activations(model_key, model_info, gpu_id=gpu_id)
        model_layers[key] = layers

    phase1_time = time.time() - start_time
    print(f"\nPhase 1 complete: {phase1_time/60:.1f} min")

    # ========================================
    # Phase 2: Train SAEs (parallel across GPUs)
    # ========================================
    print(f"\n{'#'*70}")
    print("# PHASE 2: SAE TRAINING")
    print(f"{'#'*70}")

    # Priority 1: Layer sweep at default expansion/k for all models
    training_jobs = []
    for model_key in MODELS:
        layers = model_layers.get(model_key, [])
        for layer in layers:
            training_jobs.append((model_key, layer, DEFAULT_EXPANSION, DEFAULT_K))

    # Priority 2: Expansion sweep at middle layer
    for model_key, model_info in MODELS.items():
        mid_layer = model_info["n_layers"] // 2
        for exp in EXPANSION_RATIOS:
            if exp != DEFAULT_EXPANSION:  # Already included above
                training_jobs.append((model_key, mid_layer, exp, DEFAULT_K))

    # Priority 3: K sweep at middle layer
    for model_key, model_info in MODELS.items():
        mid_layer = model_info["n_layers"] // 2
        for k_val in K_VALUES:
            if k_val != DEFAULT_K:
                training_jobs.append((model_key, mid_layer, DEFAULT_EXPANSION, k_val))

    print(f"Total SAE training jobs: {len(training_jobs)}")

    # Train sequentially but on different GPUs (round-robin)
    all_trained = {}
    for job_idx, (model_key, layer, exp, k) in enumerate(training_jobs):
        gpu_id = job_idx % n_gpus
        run_key, ckpt_path = train_single_sae(
            model_key, layer, exp, k, gpu_id=gpu_id
        )
        all_trained[run_key] = ckpt_path

    phase2_time = time.time() - start_time - phase1_time
    print(f"\nPhase 2 complete: {phase2_time/60:.1f} min")

    # ========================================
    # Phase 3: Analysis
    # ========================================
    print(f"\n{'#'*70}")
    print("# PHASE 3: ANALYSIS")
    print(f"{'#'*70}")

    # 3a: CKA
    model_pairs = [
        ("mamba1_2.8b", "pythia_2.8b"),
        ("mamba2_2.7b", "pythia_2.8b"),
        ("mamba1_2.8b", "mamba2_2.7b"),
    ]
    cka_results = run_cka_analysis(model_pairs, None, device="cuda:0")

    # 3b: Baselines
    baseline_results = run_baselines(device="cuda:1" if n_gpus > 1 else "cuda:0")

    # 3c: Downstream eval
    downstream_results = run_downstream_eval(device="cuda:0")

    # 3d: Monosemanticity
    mono_results = run_monosemanticity_analysis(
        device="cuda:1" if n_gpus > 1 else "cuda:0")

    # ========================================
    # Compile all results
    # ========================================
    print(f"\n{'#'*70}")
    print("# COMPILING RESULTS")
    print(f"{'#'*70}")

    # Aggregate all SAE stats
    all_stats = {}
    for stats_file in RESULTS_DIR.glob("*_stats.json"):
        with open(stats_file) as f:
            data = json.load(f)
        all_stats[data.get("run_key", stats_file.stem)] = data

    # Build comprehensive results
    comprehensive = {
        "config": {
            "models": {k: v["name"] for k, v in MODELS.items()},
            "n_tokens": N_TOKENS,
            "sae_type": SAE_TYPE,
            "default_k": DEFAULT_K,
            "default_expansion": DEFAULT_EXPANSION,
            "k_values": K_VALUES,
            "expansion_ratios": EXPANSION_RATIOS,
            "n_steps": SAE_STEPS,
        },
        "sae_stats": all_stats,
        "cka": cka_results,
        "baselines": baseline_results,
        "downstream": downstream_results,
        "monosemanticity": mono_results,
        "timing": {
            "phase1_extraction_min": phase1_time / 60,
            "phase2_training_min": phase2_time / 60,
            "total_min": (time.time() - start_time) / 60,
        },
    }

    save_json(comprehensive, RESULTS_DIR / "comprehensive_results.json")

    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE! Total time: {total_time:.1f} min")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
