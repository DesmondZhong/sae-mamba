#!/usr/bin/env python3
"""Full SAE experiment: extract activations, train SAEs, analyze features."""

import sys
sys.path.insert(0, "/workspace/sae-mamba")

import torch
import json
import time
from pathlib import Path
from src.activation_cache import (get_model_and_tokenizer, get_text_data,
                                   extract_residual_stream, extract_post_ssm)
from src.train_sae import train_sae
from src.analyze import find_max_activating_examples, compute_feature_stats, build_token_context

DEVICE = "cuda"
RESULTS_DIR = Path("/workspace/sae-mamba/results")
CKPT_DIR = Path("/workspace/sae-mamba/checkpoints")
RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

# Models to analyze
MODELS = {
    "mamba_130m": "state-spaces/mamba-130m-hf",
    "mamba_370m": "state-spaces/mamba-370m-hf",
    "pythia_160m": "EleutherAI/pythia-160m",
}

# SAE config
N_TOKENS = 1_000_000  # 1M tokens
SEQ_LEN = 512
SAE_EXPANSION = 4  # d_hidden = 4 * d_model
L1_COEFFS = [5e-4, 1e-3, 3e-3]
SAE_STEPS = 25000
SAE_BATCH = 4096

all_results = {}
experiment_start = time.time()

for model_key, model_name in MODELS.items():
    print(f"\n{'='*70}")
    print(f"Processing: {model_key} ({model_name})")
    print(f"{'='*70}")

    model, tokenizer = get_model_and_tokenizer(model_name, DEVICE)
    sequences = get_text_data(N_TOKENS, SEQ_LEN, tokenizer)

    # Determine model dimensions and layer count
    if "mamba" in model_key:
        config = model.config
        d_model = config.hidden_size
        n_layers = config.num_hidden_layers
    else:
        config = model.config
        d_model = config.hidden_size
        n_layers = config.num_hidden_layers

    print(f"d_model={d_model}, n_layers={n_layers}")

    # Select layers: early, middle, late
    layer_indices = [0, n_layers // 2, n_layers - 1]
    print(f"Analyzing layers: {layer_indices}")

    # Build token context for max-activating examples
    print("Building token contexts...")
    token_texts = build_token_context(sequences[:200], tokenizer, SEQ_LEN, context_window=8)

    # Extract residual stream activations
    print("\n--- Extracting residual stream activations ---")
    residual_acts = extract_residual_stream(model, sequences, layer_indices,
                                            DEVICE, batch_size=16)

    # Extract post-SSM activations (Mamba only)
    post_ssm_acts = {}
    if "mamba" in model_key:
        print("\n--- Extracting post-SSM activations ---")
        try:
            post_ssm_acts = extract_post_ssm(model, sequences, layer_indices,
                                             DEVICE, batch_size=16)
        except Exception as e:
            print(f"Post-SSM extraction failed: {e}")
            print("Continuing with residual stream only.")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Train SAEs
    model_results = {"model": model_key, "d_model": d_model, "n_layers": n_layers,
                     "layer_indices": layer_indices, "saes": {}}

    extraction_points = [("residual", residual_acts)]
    if post_ssm_acts:
        extraction_points.append(("post_ssm", post_ssm_acts))

    for ext_name, ext_acts in extraction_points:
        for layer_idx in layer_indices:
            acts = ext_acts.get(layer_idx)
            if acts is None or len(acts) == 0:
                continue

            d_hidden = d_model * SAE_EXPANSION

            # Train with best L1 coefficient (use middle one for speed)
            l1_coeff = L1_COEFFS[1]  # 1e-3
            run_key = f"{model_key}_{ext_name}_L{layer_idx}"
            print(f"\n--- Training SAE: {run_key} (d_hidden={d_hidden}, l1={l1_coeff}) ---")

            sae, history = train_sae(
                acts, d_hidden, l1_coeff,
                n_steps=SAE_STEPS, batch_size=SAE_BATCH, lr=1e-4,
                device=DEVICE,
                save_path=str(CKPT_DIR / f"{run_key}.pt"),
            )

            # Analyze features
            print(f"  Computing feature stats...")
            stats = compute_feature_stats(sae, acts, DEVICE)
            print(f"  L0={stats['l0_mean']:.1f}, dead={stats['dead_features']}/{stats['d_hidden']} "
                  f"({stats['dead_frac']:.1%}), recon={stats['avg_recon_loss']:.6f}")

            print(f"  Finding max-activating examples...")
            features = find_max_activating_examples(sae, acts, token_texts,
                                                     n_features=50, top_k=10, device=DEVICE)

            model_results["saes"][run_key] = {
                "extraction": ext_name,
                "layer": layer_idx,
                "l1_coeff": l1_coeff,
                "d_hidden": d_hidden,
                "stats": stats,
                "top_features": features[:20],  # save top 20 for web
                "history": history,
            }

            del sae
            torch.cuda.empty_cache()

    all_results[model_key] = model_results

    # Save intermediate results
    with open(RESULTS_DIR / f"{model_key}_results.json", "w") as f:
        json.dump(model_results, f, indent=2, default=str)
    print(f"\nSaved results for {model_key}")

    elapsed = time.time() - experiment_start
    print(f"Total elapsed: {elapsed/60:.1f} min")

# Save combined results
with open(RESULTS_DIR / "all_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"All experiments complete! Total time: {(time.time() - experiment_start)/60:.1f} min")
