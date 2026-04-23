#!/usr/bin/env python3
"""Extract activations from a single model, 1 layer at a time, save to NAS.
Usage: CUDA_VISIBLE_DEVICES=X python extract_model.py <model_key>
"""

import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import time
import gc
from pathlib import Path

from src.activation_cache import get_model_and_tokenizer, get_text_data, extract_residual_stream
from src.analyze import build_token_context

MODELS = {
    "mamba1_2.8b": {"name": "state-spaces/mamba-2.8b-hf", "n_layers": 64, "d_model": 2560},
    "mamba2_2.7b": {"name": "state-spaces/mamba2-2.7b", "n_layers": 64, "d_model": 2560},
    "pythia_2.8b": {"name": "EleutherAI/pythia-2.8b", "n_layers": 32, "d_model": 2560},
}

N_TOKENS = 10_000_000
SEQ_LEN = 512
DATASET = "pile"
STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/mnt/storage/desmond/excuse"))
ACTS_DIR = STORAGE / "activations"


def get_layer_indices(n_layers):
    return list(range(0, n_layers, max(1, n_layers // 8)))


def main():
    model_key = sys.argv[1]
    device = "cuda:0"  # caller sets CUDA_VISIBLE_DEVICES

    model_info = MODELS[model_key]
    model_dir = ACTS_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    if (model_dir / "DONE").exists():
        print(f"[{model_key}] Already done, skipping")
        return

    print(f"[{model_key}] Starting extraction")
    start = time.time()

    model, tokenizer = get_model_and_tokenizer(model_info["name"], device)
    sequences = get_text_data(N_TOKENS, SEQ_LEN, tokenizer, dataset_name=DATASET)

    # Save token texts and shared sequences
    token_texts = build_token_context(sequences[:200], tokenizer, SEQ_LEN, context_window=8)
    with open(ACTS_DIR / f"{model_key}_token_texts.json", "w") as f:
        json.dump(token_texts[:100000], f)
    torch.save(sequences[:1000], ACTS_DIR / f"{model_key}_sequences.pt")

    layer_indices = get_layer_indices(model_info["n_layers"])
    layers_env = os.environ.get("LAYERS")
    if layers_env:
        layer_indices = [int(x) for x in layers_env.split(",") if x.strip()]
    batch_size = 8 if "mamba2" not in model_key else 2
    batch_env = os.environ.get("BATCH_SIZE")
    if batch_env:
        batch_size = int(batch_env)
    print(f"[{model_key}] Layers: {layer_indices}, batch_size={batch_size}")

    # Single forward pass, hook all requested layers at once — 9x faster than per-layer.
    pending = [l for l in layer_indices
               if not (model_dir / f"layer_{l}.pt").exists()]
    for l in layer_indices:
        if l not in pending:
            print(f"  [{model_key}] Layer {l}: exists, skip")

    if pending:
        print(f"  [{model_key}] Extracting {len(pending)} layers in one forward: {pending}")
        acts = extract_residual_stream(model, sequences, pending,
                                       device, batch_size=batch_size)
        for layer_idx in list(acts.keys()):
            save_path = model_dir / f"layer_{layer_idx}.pt"
            torch.save(acts[layer_idx], save_path)
            size_gb = acts[layer_idx].numel() * 4 / 1e9
            print(f"  [{model_key}] Layer {layer_idx}: {acts[layer_idx].shape} "
                  f"({size_gb:.1f}GB) saved")
            del acts[layer_idx]
            gc.collect()
        del acts
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - start
    print(f"[{model_key}] DONE in {elapsed/60:.1f} min")
    (model_dir / "DONE").touch()

    del model, tokenizer, sequences
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
