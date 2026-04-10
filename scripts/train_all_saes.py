#!/usr/bin/env python3
"""Train all SAEs across models/layers/configs, distributing across GPUs.
Usage: python train_all_saes.py
"""

import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import time
import gc
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

MODELS = {
    "mamba1_2.8b": {"n_layers": 64, "d_model": 2560},
    "mamba2_2.7b": {"n_layers": 64, "d_model": 2560},
    "pythia_2.8b": {"n_layers": 32, "d_model": 2560},
}

ACTS_DIR = Path("/root/sae-mamba/activations_2.8b")
CKPT_DIR = Path("/root/sae-mamba/checkpoints_2.8b")
RESULTS_DIR = Path("/root/sae-mamba/results_2.8b")
CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SAE_TYPE = "topk"
DEFAULT_K = 64
DEFAULT_EXPANSION = 16
SAE_STEPS = 30000
SAE_BATCH = 4096
SAE_LR = 3e-4
N_GPUS = torch.cuda.device_count()


def get_layer_indices(n_layers):
    return list(range(0, n_layers, max(1, n_layers // 8)))


def train_one_sae(args):
    """Train a single SAE. Called in subprocess."""
    model_key, layer_idx, expansion, k, gpu_id = args

    import torch
    import sys
    sys.path.insert(0, "/root/sae-mamba")
    from src.train_sae import train_sae
    from src.analyze import compute_feature_stats

    device = f"cuda:{gpu_id}"
    d_model = MODELS[model_key]["d_model"]
    d_hidden = d_model * expansion
    run_key = f"{model_key}_L{layer_idx}_x{expansion}_k{k}"
    ckpt_path = str(CKPT_DIR / f"{run_key}.pt")

    # Skip if done
    if Path(ckpt_path).exists():
        print(f"[Skip] {run_key}", flush=True)
        return run_key, True

    acts_path = ACTS_DIR / model_key / f"layer_{layer_idx}.pt"
    if not acts_path.exists():
        print(f"[Skip] No activations: {acts_path}", flush=True)
        return run_key, False

    print(f"[GPU {gpu_id}] Training {run_key} (d_hidden={d_hidden}, K={k})", flush=True)
    activations = torch.load(acts_path, map_location="cpu", weights_only=True)

    sae, history, summary = train_sae(
        activations, d_hidden, sae_type=SAE_TYPE, k=k,
        n_steps=SAE_STEPS, batch_size=SAE_BATCH, lr=SAE_LR,
        device=device, save_path=ckpt_path,
    )

    stats = compute_feature_stats(sae, activations, device)
    stats.update(summary)
    stats["model_key"] = model_key
    stats["layer"] = layer_idx
    stats["expansion_ratio"] = expansion
    stats["k"] = k
    stats["sae_type"] = SAE_TYPE
    stats["run_key"] = run_key

    with open(RESULTS_DIR / f"{run_key}_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    del sae, activations
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Done] {run_key}: FVE={summary['final_fve']:.4f}, L0={summary['final_l0']:.1f}", flush=True)
    return run_key, True


def main():
    start = time.time()

    # Wait for all extractions to finish
    print("Waiting for activation extraction to complete...")
    for model_key in MODELS:
        done_marker = ACTS_DIR / model_key / "DONE"
        while not done_marker.exists():
            time.sleep(10)
        print(f"  {model_key}: extraction complete")

    # Build job list
    jobs = []

    # Priority 1: Layer sweep at default expansion/k
    for model_key, info in MODELS.items():
        for layer in get_layer_indices(info["n_layers"]):
            jobs.append((model_key, layer, DEFAULT_EXPANSION, DEFAULT_K))

    # Priority 2: Expansion sweep at middle layer
    for model_key, info in MODELS.items():
        mid = info["n_layers"] // 2
        for exp in [8, 32]:
            jobs.append((model_key, mid, exp, DEFAULT_K))

    # Priority 3: K sweep at middle layer
    for model_key, info in MODELS.items():
        mid = info["n_layers"] // 2
        for k in [32, 128]:
            jobs.append((model_key, mid, DEFAULT_EXPANSION, k))

    print(f"\nTotal training jobs: {len(jobs)}")
    print(f"Using {N_GPUS} GPUs")

    # Assign GPUs round-robin and run in parallel
    gpu_jobs = [(m, l, e, k, i % N_GPUS) for i, (m, l, e, k) in enumerate(jobs)]

    # Use ProcessPoolExecutor — one process per GPU to avoid contention
    # But we can run multiple sequential jobs per GPU
    completed = 0
    failed = 0

    # Group jobs by GPU
    from collections import defaultdict
    by_gpu = defaultdict(list)
    for m, l, e, k, g in gpu_jobs:
        by_gpu[g].append((m, l, e, k, g))

    # Run each GPU's jobs sequentially in a separate process
    with ProcessPoolExecutor(max_workers=N_GPUS) as executor:
        futures = []
        for gpu_id, gpu_job_list in by_gpu.items():
            for job in gpu_job_list:
                futures.append(executor.submit(train_one_sae, job))

        for future in as_completed(futures):
            try:
                run_key, success = future.result()
                if success:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error: {e}", flush=True)
                failed += 1

    elapsed = time.time() - start
    print(f"\nTraining complete! {completed} succeeded, {failed} failed. {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
