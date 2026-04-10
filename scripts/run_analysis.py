#!/usr/bin/env python3
"""Run all analysis: CKA, baselines, downstream eval, monosemanticity, compile results."""

import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import time
import gc
import numpy as np
from pathlib import Path

from src.sae import create_sae
from src.activation_cache import get_model_and_tokenizer, get_text_data
from src.analyze import (compute_feature_stats, compute_cka_batched,
                          compute_random_baseline, compute_pca_baseline,
                          find_max_activating_examples, compute_monosemanticity)

MODELS = {
    "mamba1_2.8b": {"name": "state-spaces/mamba-2.8b-hf", "n_layers": 64, "d_model": 2560},
    "mamba2_2.7b": {"name": "state-spaces/mamba2-2.7b", "n_layers": 64, "d_model": 2560},
    "pythia_2.8b": {"name": "EleutherAI/pythia-2.8b", "n_layers": 32, "d_model": 2560},
}

STORAGE = Path("/mnt/storage/desmond/excuse")
ACTS_DIR = STORAGE / "activations"
CKPT_DIR = STORAGE / "checkpoints"
RESULTS_DIR = STORAGE / "results"
DEFAULT_K = 64
DEFAULT_EXPANSION = 16
SAE_TYPE = "topk"
SEQ_LEN = 512
DATASET = "pile"


def get_layer_indices(n_layers):
    return list(range(0, n_layers, max(1, n_layers // 8)))


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {path}")


def run_cka(device="cuda:0"):
    print(f"\n{'='*60}")
    print("CKA Analysis")
    print(f"{'='*60}")

    model_pairs = [
        ("mamba1_2.8b", "pythia_2.8b"),
        ("mamba2_2.7b", "pythia_2.8b"),
        ("mamba1_2.8b", "mamba2_2.7b"),
    ]

    cka_results = {}

    for model_a, model_b in model_pairs:
        pair_key = f"{model_a}_vs_{model_b}"
        cka_results[pair_key] = {}

        layers_a = get_layer_indices(MODELS[model_a]["n_layers"])
        layers_b = get_layer_indices(MODELS[model_b]["n_layers"])
        n_compare = min(len(layers_a), len(layers_b))

        for i in range(n_compare):
            la = layers_a[i] if i < len(layers_a) else layers_a[-1]
            lb = layers_b[i] if i < len(layers_b) else layers_b[-1]

            acts_a_path = ACTS_DIR / model_a / f"layer_{la}.pt"
            acts_b_path = ACTS_DIR / model_b / f"layer_{lb}.pt"
            sae_a_path = CKPT_DIR / f"{model_a}_L{la}_x{DEFAULT_EXPANSION}_k{DEFAULT_K}.pt"
            sae_b_path = CKPT_DIR / f"{model_b}_L{lb}_x{DEFAULT_EXPANSION}_k{DEFAULT_K}.pt"

            if not all(p.exists() for p in [acts_a_path, acts_b_path]):
                continue

            raw_a = torch.load(acts_a_path, map_location="cpu", weights_only=True)
            raw_b = torch.load(acts_b_path, map_location="cpu", weights_only=True)
            n = min(10000, raw_a.shape[0], raw_b.shape[0])

            # CKA on raw activations
            cka_raw = compute_cka_batched(raw_a[:n], raw_b[:n], device=device)

            # CKA on SAE features (if SAEs exist)
            cka_sae = None
            if sae_a_path.exists() and sae_b_path.exists():
                d_model = MODELS[model_a]["d_model"]
                d_hidden = d_model * DEFAULT_EXPANSION

                sae_a = create_sae(d_model, d_hidden, sae_type=SAE_TYPE, k=DEFAULT_K).to(device)
                sae_a.load_state_dict(torch.load(sae_a_path, map_location=device, weights_only=True))
                sae_a.eval()

                sae_b = create_sae(d_model, d_hidden, sae_type=SAE_TYPE, k=DEFAULT_K).to(device)
                sae_b.load_state_dict(torch.load(sae_b_path, map_location=device, weights_only=True))
                sae_b.eval()

                with torch.no_grad():
                    z_a = sae_a.encode(raw_a[:n].to(device)).cpu()
                    z_b = sae_b.encode(raw_b[:n].to(device)).cpu()

                cka_sae = compute_cka_batched(z_a, z_b, device=device)
                del sae_a, sae_b, z_a, z_b
                torch.cuda.empty_cache()

            depth_a = la / max(MODELS[model_a]["n_layers"] - 1, 1)
            depth_b = lb / max(MODELS[model_b]["n_layers"] - 1, 1)

            cka_results[pair_key][f"depth_{depth_a:.2f}"] = {
                "layer_a": la, "layer_b": lb,
                "depth_a": depth_a, "depth_b": depth_b,
                "cka_raw": cka_raw,
                "cka_sae": cka_sae,
            }
            print(f"  {pair_key} depth={depth_a:.2f}: raw={cka_raw:.4f}, sae={cka_sae}")

            del raw_a, raw_b
            torch.cuda.empty_cache()

    save_json(cka_results, RESULTS_DIR / "cka_results.json")
    return cka_results


def run_baselines(device="cuda:1"):
    print(f"\n{'='*60}")
    print("Baselines (Random + PCA)")
    print(f"{'='*60}")

    results = {}
    for model_key, info in MODELS.items():
        results[model_key] = {}
        for layer in get_layer_indices(info["n_layers"]):
            acts_path = ACTS_DIR / model_key / f"layer_{layer}.pt"
            if not acts_path.exists():
                continue

            acts = torch.load(acts_path, map_location="cpu", weights_only=True)
            rand = compute_random_baseline(acts, k=DEFAULT_K, device=device)
            pca = compute_pca_baseline(acts, k=DEFAULT_K, device=device)

            # Load SAE FVE if available
            stats_path = RESULTS_DIR / f"{model_key}_L{layer}_x{DEFAULT_EXPANSION}_k{DEFAULT_K}_stats.json"
            sae_fve = None
            if stats_path.exists():
                with open(stats_path) as f:
                    sae_fve = json.load(f).get("fve")

            depth = layer / max(info["n_layers"] - 1, 1)
            results[model_key][f"layer_{layer}"] = {
                "layer": layer, "depth": depth,
                "random_fve_mean": rand["mean"], "random_fve_std": rand["std"],
                "pca_fve": pca, "sae_fve": sae_fve,
            }
            print(f"  {model_key} L{layer}: rand={rand['mean']:.4f} PCA={pca:.4f} SAE={sae_fve}")
            del acts; torch.cuda.empty_cache()

    save_json(results, RESULTS_DIR / "baselines.json")
    return results


def run_downstream(device="cuda:2"):
    print(f"\n{'='*60}")
    print("Downstream Perplexity Evaluation")
    print(f"{'='*60}")

    results = {}
    for model_key, info in MODELS.items():
        print(f"\n--- {model_key} ---")
        model, tokenizer = get_model_and_tokenizer(info["name"], device)
        sequences = get_text_data(500_000, SEQ_LEN, tokenizer, dataset_name=DATASET)

        # Baseline PPL
        total_loss, n_tok = 0, 0
        for i in range(0, min(len(sequences), 100), 4):
            batch = sequences[i:i+4].to(device)
            with torch.no_grad():
                out = model(batch, labels=batch)
                total_loss += out.loss.item() * batch.numel()
                n_tok += batch.numel()
        baseline_ppl = np.exp(total_loss / n_tok)
        print(f"  Baseline PPL: {baseline_ppl:.2f}")

        # Find layers module
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
            model_layers = model.backbone.layers
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            model_layers = model.gpt_neox.layers
        else:
            print(f"  Cannot find layers, skipping")
            del model, tokenizer; torch.cuda.empty_cache()
            continue

        mid = info["n_layers"] // 2
        sae_path = CKPT_DIR / f"{model_key}_L{mid}_x{DEFAULT_EXPANSION}_k{DEFAULT_K}.pt"
        if sae_path.exists():
            d_hidden = info["d_model"] * DEFAULT_EXPANSION
            sae = create_sae(info["d_model"], d_hidden, SAE_TYPE, k=DEFAULT_K).to(device)
            sae.load_state_dict(torch.load(sae_path, map_location=device, weights_only=True))
            sae.eval()

            def make_hook(sae_m):
                def fn(mod, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    b, s, d = hs.shape
                    with torch.no_grad():
                        recon, _, _, _ = sae_m(hs.reshape(-1, d).float())
                    new = recon.half().reshape(b, s, d)
                    return (new,) + out[1:] if isinstance(out, tuple) else new
                return fn

            hook = model_layers[mid].register_forward_hook(make_hook(sae))
            tl, nt = 0, 0
            for i in range(0, min(len(sequences), 100), 4):
                batch = sequences[i:i+4].to(device)
                with torch.no_grad():
                    out = model(batch, labels=batch)
                    tl += out.loss.item() * batch.numel()
                    nt += batch.numel()
            hook.remove()
            sae_ppl = np.exp(tl / nt)
            results[model_key] = {
                "baseline_ppl": baseline_ppl, "sae_ppl": sae_ppl,
                "ppl_ratio": sae_ppl / baseline_ppl, "layer": mid,
            }
            print(f"  SAE PPL: {sae_ppl:.2f} (ratio: {sae_ppl/baseline_ppl:.4f}x)")
            del sae
        else:
            results[model_key] = {"baseline_ppl": baseline_ppl, "sae_ppl": None, "layer": mid}

        del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

    save_json(results, RESULTS_DIR / "downstream.json")
    return results


def run_features_and_mono(device="cuda:3"):
    print(f"\n{'='*60}")
    print("Feature Analysis & Monosemanticity")
    print(f"{'='*60}")

    mono_results = {}
    for model_key, info in MODELS.items():
        mono_results[model_key] = {}
        mid = info["n_layers"] // 2
        run_key = f"{model_key}_L{mid}_x{DEFAULT_EXPANSION}_k{DEFAULT_K}"

        texts_path = ACTS_DIR / f"{model_key}_token_texts.json"
        sae_path = CKPT_DIR / f"{run_key}.pt"
        acts_path = ACTS_DIR / model_key / f"layer_{mid}.pt"

        if not all(p.exists() for p in [texts_path, sae_path, acts_path]):
            print(f"  {run_key}: missing files, skip")
            continue

        with open(texts_path) as f:
            token_texts = json.load(f)

        d_hidden = info["d_model"] * DEFAULT_EXPANSION
        sae = create_sae(info["d_model"], d_hidden, SAE_TYPE, k=DEFAULT_K).to(device)
        sae.load_state_dict(torch.load(sae_path, map_location=device, weights_only=True))
        sae.eval()

        acts = torch.load(acts_path, map_location="cpu", weights_only=True)
        features = find_max_activating_examples(sae, acts, token_texts,
                                                 n_features=50, top_k=20, device=device)
        save_json(features[:30], RESULTS_DIR / f"{run_key}_features.json")

        mono = compute_monosemanticity(features, device=device)
        if mono:
            mono_results[model_key][run_key] = mono
            print(f"  {run_key}: mono={mono['mean']:.3f}")

        del sae, acts; torch.cuda.empty_cache()

    save_json(mono_results, RESULTS_DIR / "monosemanticity.json")
    return mono_results


def compile_results():
    print(f"\n{'='*60}")
    print("Compiling all results")
    print(f"{'='*60}")

    all_stats = {}
    for f in RESULTS_DIR.glob("*_stats.json"):
        with open(f) as fh:
            data = json.load(fh)
            all_stats[data.get("run_key", f.stem)] = data

    results = {"sae_stats": all_stats}
    for name in ["cka_results", "baselines", "downstream", "monosemanticity"]:
        p = RESULTS_DIR / f"{name}.json"
        if p.exists():
            with open(p) as f:
                results[name.replace("_results", "")] = json.load(f)

    results["config"] = {
        "models": {k: v["name"] for k, v in MODELS.items()},
        "sae_type": SAE_TYPE, "default_k": DEFAULT_K,
        "default_expansion": DEFAULT_EXPANSION,
    }

    save_json(results, RESULTS_DIR / "comprehensive_results.json")
    print(f"Compiled {len(all_stats)} SAE results")


def main():
    start = time.time()

    # Run analyses (some can use different GPUs)
    run_cka(device="cuda:0")
    run_baselines(device="cuda:1")
    run_downstream(device="cuda:2")
    run_features_and_mono(device="cuda:3")
    compile_results()

    print(f"\nAll analysis complete! {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
