#!/usr/bin/env python3
"""
Steering demo: clamp individual SAE features and generate text.
Shows side-by-side baseline vs clamped generations.
"""
import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from pathlib import Path

from src.sae import create_sae
from src.activation_cache import get_model_and_tokenizer, _get_layers

STORAGE = Path("/path/to/storage")
ACTS_DIR = STORAGE / "activations"
CKPT_DIR = STORAGE / "checkpoints_normed"
RESULTS = STORAGE / "results_steering_demo"
RESULTS.mkdir(exist_ok=True)

MODELS = {
    "mamba1_2.8b": {
        "name": "state-spaces/mamba-2.8b-hf",
        "layer": 32, "device": "cuda:2",
    },
    "pythia_2.8b": {
        "name": "EleutherAI/pythia-2.8b",
        "layer": 16, "device": "cuda:3",
    },
}

PROMPTS = [
    "The weather today is",
    "My favorite hobby is",
    "Once upon a time, there was",
    "The most important thing in life is",
    "Scientists have recently discovered",
    "She walked into the room and",
    "The recipe for happiness includes",
    "In the year 2050, people will",
]

# Top features per model — picked from feature browser
# We'll auto-pick the top 3 by max_activation that have at least 5 examples
def pick_features(model_key, layer, n=3):
    f = STORAGE / f"results/{model_key}_L{layer}_x16_k64_normed_features.json"
    with open(f) as fh:
        feats = json.load(fh)
    # Filter: at least 5 examples and reasonable activation
    valid = [f for f in feats if len(f.get('top_examples', [])) >= 5 and f['max_activation'] > 5]
    valid.sort(key=lambda f: f['max_activation'], reverse=True)
    return [(f['feature_id'], f['max_activation'], f.get('top_examples', [])[:3]) for f in valid[:n]]


def load_norm(model_key, layer, device):
    """Compute mean/std from a 10K subsample."""
    p = ACTS_DIR / model_key / f"layer_{layer}.pt"
    t = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
    s = t[:10000].clone().float()
    mean = s.mean(dim=0).to(device)
    std = s.std(dim=0).clamp(min=1e-6).to(device)
    del t, s
    return mean, std


@torch.no_grad()
def generate(model, tokenizer, prompt, n_tokens=50, hook=None, target_layer=None):
    layers = _get_layers(model)
    h = None
    if hook is not None and target_layer is not None:
        h = layers[target_layer].register_forward_hook(hook)
    try:
        inp = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
        out = model.generate(
            inp, max_new_tokens=n_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
    finally:
        if h is not None:
            h.remove()
    return text


def make_clamp_hook(sae, mean, std, feat_idx, clamp_value):
    """Hook that encodes activations through SAE, clamps a feature, and decodes back."""
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
        else:
            h = out
        h_f = h.float()
        b, s, d = h_f.shape
        h_flat = h_f.reshape(-1, d)
        h_normed = (h_flat - mean) / std
        z = sae.encode(h_normed)
        # Clamp the chosen feature
        z[:, feat_idx] = clamp_value
        h_recon = sae.decode(z)
        h_unnormed = h_recon * std + mean
        h_new = h_unnormed.to(h.dtype).reshape(b, s, d)
        if isinstance(out, tuple):
            return (h_new,) + out[1:]
        return h_new
    return hook_fn


def main():
    results = {}
    for model_key, info in MODELS.items():
        layer = info["layer"]
        device = info["device"]
        print(f"\n{'='*60}")
        print(f"{model_key} L{layer} on {device}")
        print(f"{'='*60}")

        # Pick features
        features = pick_features(model_key, layer, n=3)
        print(f"Selected features: {[f[0] for f in features]}")

        # Load model + SAE
        model, tokenizer = get_model_and_tokenizer(info["name"], device)
        d_model = 2560
        d_hidden = d_model * 16
        sae = create_sae(d_model, d_hidden, sae_type="topk", k=64).to(device)
        sae.load_state_dict(torch.load(
            CKPT_DIR / f"{model_key}_L{layer}_x16_k64_normed.pt",
            map_location=device, weights_only=True))
        sae.eval()

        mean, std = load_norm(model_key, layer, device)

        model_results = {
            "layer": layer,
            "features": [],
        }

        for feat_idx, max_act, examples in features:
            print(f"\nFeature {feat_idx} (max act: {max_act:.2f})")
            print(f"  Top examples:")
            for ex in examples:
                print(f"    [{ex['activation']:.2f}] {ex['text'][:100]}")

            feat_data = {
                "feature_id": int(feat_idx),
                "max_activation": float(max_act),
                "top_examples": [{"text": ex["text"][:200], "activation": float(ex["activation"])}
                                 for ex in examples],
                "prompts": [],
            }

            # Pick clamp values: 0 (ablate), 2x max, 5x max
            clamp_values = [0.0, max_act * 2, max_act * 5]

            for prompt in PROMPTS:
                row = {"prompt": prompt}
                # Baseline
                row["baseline"] = generate(model, tokenizer, prompt)

                # Clamped versions
                for cv in clamp_values:
                    hook = make_clamp_hook(sae, mean, std, feat_idx, cv)
                    text = generate(model, tokenizer, prompt, hook=hook, target_layer=layer)
                    label = "ablate" if cv == 0 else f"clamp_{cv:.0f}"
                    row[label] = text

                feat_data["prompts"].append(row)
                print(f"  Prompt: {prompt!r}")
                print(f"    baseline: {row['baseline'][:80]}")
                print(f"    clamp_high: {row[f'clamp_{clamp_values[2]:.0f}'][:80]}")

            model_results["features"].append(feat_data)

        results[model_key] = model_results
        del model, sae
        torch.cuda.empty_cache()

    out = RESULTS / "steering_demo.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
