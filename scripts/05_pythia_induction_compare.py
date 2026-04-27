#!/usr/bin/env python3
"""Phase 4b: Induction-circuit localization on Pythia-2.8B for comparison.

Mirrors scripts/04_induction_circuit.py but patches Pythia (GPT-NeoX) internals:
  - attention_output     (output of the whole self-attention block)
  - mlp_output           (output of the MLP block)
  - attention_qkv        (fused query/key/value projection output)
  - mlp_dense_h_to_4h    (first MLP linear output)

Expected: induction signal concentrates on `attention_output` (the induction-head
hypothesis from Olsson et al. 2022). This serves as a control/sanity check for
the Mamba-1 localization.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.activation_cache import get_model_and_tokenizer
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/path/to/storage"))
CKPT_DIR = STORAGE / "checkpoints_normed"
ACTS_DIR = STORAGE / "activations"
RESULTS_DIR = STORAGE / "results_phase4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_KEY = "pythia_2.8b"
MODEL_NAME = "EleutherAI/pythia-2.8b"
D_MODEL = 2560
MID_LAYER = 16  # matched relative depth to Mamba L32 (half-depth)

SAE_EXPANSION = 16
SAE_K = 64


def get_pythia_layer(model, layer_idx):
    return model.gpt_neox.layers[layer_idx]


# Pythia submodules we can hook. Values are (getter, kind).
# kind="output": hook forward, replace output.
# kind="input":  hook forward_pre, replace input tuple[0].
PYTHIA_COMPONENTS = {
    "attention_output": (lambda layer: layer.attention, "output_tuple"),
    "attention_qkv": (lambda layer: layer.attention.query_key_value, "output"),
    "mlp_output": (lambda layer: layer.mlp, "output"),
    "mlp_dense_h_to_4h": (lambda layer: layer.mlp.dense_h_to_4h, "output"),
}


def load_sae_and_norm(layer: int, device: str):
    d_hidden = D_MODEL * SAE_EXPANSION
    run_key = f"{MODEL_KEY}_L{layer}_x{SAE_EXPANSION}_k{SAE_K}_normed"
    ckpt_path = CKPT_DIR / f"{run_key}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    sae = create_sae(D_MODEL, d_hidden, sae_type="topk", k=SAE_K).to(device)
    sae.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    sae.eval()

    acts_path = ACTS_DIR / MODEL_KEY / f"layer_{layer}.pt"
    t = torch.load(acts_path, map_location="cpu", weights_only=True, mmap=True)
    sample = t[:10000].clone().float()
    act_mean = sample.mean(dim=0).to(device)
    act_std = sample.std(dim=0).clamp(min=1e-6).to(device)
    del t, sample
    return sae, act_mean, act_std


def make_induction_pair(vocab_size, n_pairs=32, prefix_len=40, pattern_len=5,
                        mid_len=40, device="cuda", seed=42):
    rng = np.random.default_rng(seed)
    clean_seqs, corr_seqs = [], []
    for _ in range(n_pairs):
        prefix = rng.integers(10, vocab_size - 10, size=prefix_len).tolist()
        pattern = rng.integers(10, vocab_size - 10, size=pattern_len).tolist()
        mid = rng.integers(10, vocab_size - 10, size=mid_len).tolist()
        alt_pattern = rng.integers(10, vocab_size - 10, size=pattern_len).tolist()
        clean_seqs.append(prefix + pattern + mid + pattern)
        corr_seqs.append(prefix + pattern + mid + alt_pattern)
    clean = torch.tensor(clean_seqs, dtype=torch.long, device=device)
    corr = torch.tensor(corr_seqs, dtype=torch.long, device=device)
    second_start = prefix_len + pattern_len + mid_len
    ind_pos = list(range(second_start, second_start + pattern_len))
    return clean, corr, ind_pos


class PythiaResidualCapture:
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.captured = None
        self._hook = None

    def __enter__(self):
        layer = get_pythia_layer(self.model, self.layer_idx)

        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            self.captured = out.detach().clone()
        self._hook = layer.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self._hook.remove()


class PythiaComponentCapture:
    def __init__(self, model, sites):
        self.model = model
        self.sites = sites
        self.captured = {}
        self._hooks = []

    def __enter__(self):
        for layer_idx, component in self.sites:
            getter, kind = PYTHIA_COMPONENTS[component]
            mod = getter(get_pythia_layer(self.model, layer_idx))
            key = (layer_idx, component)

            def make_hook(k, kd):
                def hook(module, inputs, output):
                    if kd == "output_tuple":
                        # Attention returns (output, present, ...); index 0 is the hidden state.
                        val = output[0] if isinstance(output, tuple) else output
                    else:
                        val = output
                    self.captured[k] = val.detach().clone()
                return hook

            self._hooks.append(mod.register_forward_hook(make_hook(key, kind)))
        return self

    def __exit__(self, *a):
        for h in self._hooks:
            h.remove()


class PythiaComponentPatcher:
    def __init__(self, model, patches):
        self.model = model
        self.patches = patches
        self._hooks = []

    def __enter__(self):
        for (layer_idx, component), replacement in self.patches.items():
            getter, kind = PYTHIA_COMPONENTS[component]
            mod = getter(get_pythia_layer(self.model, layer_idx))

            def make_hook(r, kd):
                def hook(module, inputs, output):
                    rep = r.to(device=output[0].device if isinstance(output, tuple) else output.device,
                              dtype=output[0].dtype if isinstance(output, tuple) else output.dtype)
                    if kd == "output_tuple" and isinstance(output, tuple):
                        return (rep,) + output[1:]
                    return rep
                return hook

            self._hooks.append(mod.register_forward_hook(make_hook(replacement, kind)))
        return self

    def __exit__(self, *a):
        for h in self._hooks:
            h.remove()


@torch.no_grad()
def encode_residual(model, tokens, sae, act_mean, act_std, layer, positions):
    with PythiaResidualCapture(model, layer) as cap:
        model(tokens)
    h = cap.captured.float()
    B, L, D = h.shape
    h_flat = h.reshape(-1, D)
    h_normed = (h_flat - act_mean) / act_std
    z = sae.encode(h_normed).reshape(B, L, -1)
    return z[:, positions, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--layers", nargs="+", type=int,
        default=[2, 4, 6, 8, 10, 12, 14, 15, 16],
    )
    parser.add_argument("--top_features", type=int, default=10)
    parser.add_argument("--n_pairs", type=int, default=32)
    args = parser.parse_args()

    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)

    print(f"Loading Pythia L{MID_LAYER} SAE...")
    sae, act_mean, act_std = load_sae_and_norm(MID_LAYER, device)

    # Identify induction features.
    print("\n=== Identifying Pythia induction features ===")
    clean, corr, ind_pos = make_induction_pair(
        tokenizer.vocab_size, n_pairs=64, device=device,
    )
    z_clean = encode_residual(model, clean, sae, act_mean, act_std, MID_LAYER, ind_pos)
    z_corr = encode_residual(model, corr, sae, act_mean, act_std, MID_LAYER, ind_pos)
    mean_clean = z_clean.reshape(-1, z_clean.shape[-1]).mean(dim=0)
    mean_corr = z_corr.reshape(-1, z_corr.shape[-1]).mean(dim=0)
    induction_score = mean_clean - mean_corr
    top_vals, top_idx = torch.topk(induction_score, args.top_features)
    top_feats = top_idx.cpu().tolist()
    print(f"  Top {len(top_feats)} induction features: {top_feats}")

    with open(RESULTS_DIR / "pythia_induction_features.json", "w") as f:
        json.dump({
            "feature": top_feats,
            "score": top_vals.cpu().tolist(),
            "mean_clean": mean_clean[top_idx].cpu().tolist(),
            "mean_corr": mean_corr[top_idx].cpu().tolist(),
        }, f, indent=2, default=str)

    # Patching sweep.
    clean, corr, ind_pos = make_induction_pair(
        tokenizer.vocab_size, n_pairs=args.n_pairs, device=device,
    )
    target_idx = torch.tensor(top_feats, device=device, dtype=torch.long)

    def mean_target_act(tokens, patches=None):
        if patches is not None:
            with PythiaComponentPatcher(model, patches):
                z = encode_residual(model, tokens, sae, act_mean, act_std, MID_LAYER, ind_pos)
        else:
            z = encode_residual(model, tokens, sae, act_mean, act_std, MID_LAYER, ind_pos)
        return z[:, :, target_idx].mean().item()

    baseline_act = mean_target_act(clean)
    corrupted_act = mean_target_act(corr)
    total_effect = baseline_act - corrupted_act

    per_site = []
    print("\n=== Pythia component patching sweep ===")
    for layer_idx in tqdm(args.layers, desc="Layer sweep"):
        for component in PYTHIA_COMPONENTS:
            site = (layer_idx, component)
            with PythiaComponentCapture(model, sites=[site]) as cap:
                with torch.no_grad():
                    model(corr)
            corrupted_internal = cap.captured[site]

            patched_act = mean_target_act(clean, patches={site: corrupted_internal})
            patch_damage = (
                1.0 - (patched_act - corrupted_act) / total_effect
                if abs(total_effect) > 1e-8 else 0.0
            )
            per_site.append({
                "layer": layer_idx,
                "component": component,
                "baseline_act": baseline_act,
                "corrupted_act": corrupted_act,
                "patched_act": patched_act,
                "patch_damage": patch_damage,
            })
            print(f"  L{layer_idx:2d} {component:<20s} "
                  f"patch_damage={patch_damage:+.3f}", flush=True)

    results = {
        "baseline_act": baseline_act,
        "corrupted_act": corrupted_act,
        "total_effect": total_effect,
        "target_features": top_feats,
        "per_site": per_site,
    }
    with open(RESULTS_DIR / "pythia_patching_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    sites_sorted = sorted(per_site, key=lambda x: x["patch_damage"], reverse=True)
    print("\n=== Pythia ranked sites ===")
    for s in sites_sorted[:15]:
        print(f"  L{s['layer']:2d} {s['component']:<20s}  patch_damage={s['patch_damage']:+.3f}")


if __name__ == "__main__":
    main()
