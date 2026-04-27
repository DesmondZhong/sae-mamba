#!/usr/bin/env python3
"""Downstream perplexity eval: compare baseline PPL vs SAE-reconstructed PPL.
Uses normalized SAEs and proper normalization in the hook.
"""
import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np
from pathlib import Path

from src.sae import create_sae
from src.activation_cache import get_model_and_tokenizer, get_text_data, _get_layers

STORAGE = Path("/path/to/storage")
ACTS_DIR = STORAGE / "activations"
CKPT_DIR = STORAGE / "checkpoints_normed"
RESULTS = STORAGE / "results"

MODELS = {
    "mamba1_2.8b": {"name": "state-spaces/mamba-2.8b-hf", "n_layers": 64, "d_model": 2560, "mid": 32},
    "pythia_2.8b": {"name": "EleutherAI/pythia-2.8b", "n_layers": 32, "d_model": 2560, "mid": 16},
}

DEVICE = "cuda:0"
N_EVAL_SEQS = 50  # 50 sequences * 256 tokens = 12,800 eval tokens
SEQ_LEN = 256


def load_norm_params(model_key, layer):
    """Compute normalization params from cached activations."""
    path = ACTS_DIR / model_key / f"layer_{layer}.pt"
    t = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    sample = t[:10000].clone().float()
    mean = sample.mean(dim=0).to(DEVICE)
    std = sample.std(dim=0).clamp(min=1e-6).to(DEVICE)
    del t, sample
    return mean, std


def evaluate_ppl(model, sequences, hook=None, batch_size=4):
    total_loss, n_tokens = 0.0, 0
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            out = model(batch, labels=batch)
        # loss is mean over tokens; multiply by num tokens
        n = batch.numel() - batch.shape[0]  # exclude shifted token
        total_loss += out.loss.item() * n
        n_tokens += n
    return float(np.exp(total_loss / n_tokens))


results = {}

for model_key, info in MODELS.items():
    print(f"\n=== {model_key} ===")
    model, tokenizer = get_model_and_tokenizer(info["name"], DEVICE)
    sequences = get_text_data(N_EVAL_SEQS * SEQ_LEN * 2, SEQ_LEN, tokenizer, dataset_name="pile")
    sequences = sequences[:N_EVAL_SEQS]

    print(f"  Computing baseline PPL on {len(sequences)} sequences...")
    baseline_ppl = evaluate_ppl(model, sequences)
    print(f"  Baseline PPL: {baseline_ppl:.3f}")

    # Load SAE
    layer = info["mid"]
    d_model = info["d_model"]
    d_hidden = d_model * 16
    run_key = f"{model_key}_L{layer}_x16_k64_normed"
    sae_path = CKPT_DIR / f"{run_key}.pt"

    sae = create_sae(d_model, d_hidden, sae_type="topk", k=64).to(DEVICE)
    sae.load_state_dict(torch.load(sae_path, map_location=DEVICE, weights_only=True))
    sae.eval()

    act_mean, act_std = load_norm_params(model_key, layer)
    print(f"  Loaded SAE for layer {layer}, computing SAE PPL...")

    # Hook: replace residual stream with SAE reconstruction
    def make_hook():
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                h = out[0].float()
            else:
                h = out.float()
            b, s, d = h.shape
            h_flat = h.reshape(-1, d)
            h_normed = (h_flat - act_mean) / act_std
            with torch.no_grad():
                z = sae.encode(h_normed)
                h_recon = sae.decode(z)
            h_unnormed = h_recon * act_std + act_mean
            h_new = h_unnormed.half().reshape(b, s, d)
            if isinstance(out, tuple):
                return (h_new,) + out[1:]
            return h_new
        return hook_fn

    layers = _get_layers(model)
    hook = layers[layer].register_forward_hook(make_hook())
    sae_ppl = evaluate_ppl(model, sequences)
    hook.remove()

    print(f"  SAE PPL: {sae_ppl:.3f}  (ratio: {sae_ppl/baseline_ppl:.4f}x, delta: {sae_ppl-baseline_ppl:+.3f})")

    # Also test ablation: replace with mean activation (a baseline for "how much does this layer matter")
    def mean_hook():
        def fn(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
            else:
                h = out
            mean_val = (act_mean.unsqueeze(0).unsqueeze(0) * 0 + act_mean.unsqueeze(0).unsqueeze(0)).half()
            new = mean_val.expand_as(h)
            if isinstance(out, tuple):
                return (new,) + out[1:]
            return new
        return fn

    hook = layers[layer].register_forward_hook(mean_hook())
    mean_ppl = evaluate_ppl(model, sequences)
    hook.remove()
    print(f"  Mean-replaced PPL: {mean_ppl:.3f}  (replacing layer with constant mean)")

    results[model_key] = {
        "layer": layer,
        "baseline_ppl": baseline_ppl,
        "sae_ppl": sae_ppl,
        "ppl_ratio": sae_ppl / baseline_ppl,
        "ppl_delta": sae_ppl - baseline_ppl,
        "mean_replaced_ppl": mean_ppl,
        "n_eval_tokens": int(N_EVAL_SEQS * SEQ_LEN),
    }

    del model, sae
    torch.cuda.empty_cache()

with open(RESULTS / "downstream.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {RESULTS}/downstream.json")
