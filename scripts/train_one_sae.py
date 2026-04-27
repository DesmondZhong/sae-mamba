#!/usr/bin/env python3
"""Train a single SAE. Designed to be launched in parallel across GPUs.
Usage: CUDA_VISIBLE_DEVICES=X python train_one_sae.py <model_key> <layer> <expansion> <k>
"""

import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import time
import gc
from pathlib import Path

from src.train_sae import train_sae
from src.analyze import compute_feature_stats

MODELS = {
    "mamba1_2.8b": {"d_model": 2560},
    "mamba2_2.7b": {"d_model": 2560},
    "pythia_2.8b": {"d_model": 2560},
}

STORAGE = Path("/path/to/storage")
ACTS_DIR = STORAGE / "activations"
CKPT_DIR = STORAGE / "checkpoints"
RESULTS_DIR = STORAGE / "results"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAE_TYPE = "topk"
SAE_STEPS = 30000
SAE_BATCH = 4096
SAE_LR = 3e-4


def main():
    model_key = sys.argv[1]
    layer_idx = int(sys.argv[2])
    expansion = int(sys.argv[3])
    k = int(sys.argv[4])
    device = "cuda:0"

    d_model = MODELS[model_key]["d_model"]
    d_hidden = d_model * expansion
    run_key = f"{model_key}_L{layer_idx}_x{expansion}_k{k}"
    ckpt_path = str(CKPT_DIR / f"{run_key}.pt")

    # Skip if done
    if Path(ckpt_path).exists():
        print(f"[Skip] {run_key}")
        return

    acts_path = ACTS_DIR / model_key / f"layer_{layer_idx}.pt"
    if not acts_path.exists():
        print(f"[Skip] No activations: {acts_path}")
        return

    print(f"[Train] {run_key} (d_hidden={d_hidden}, K={k})")
    start = time.time()

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

    elapsed = time.time() - start
    print(f"[Done] {run_key}: FVE={summary['final_fve']:.4f} L0={summary['final_l0']:.1f} "
          f"dead={summary['final_dead_features']} ({elapsed/60:.1f}min)")

    del sae, activations
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
