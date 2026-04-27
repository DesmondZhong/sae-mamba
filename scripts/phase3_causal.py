#!/usr/bin/env python3
"""
Phase 3: Causal experiments on SAE features.

Experiment A: Feature Steering
  - For each model, ablate top SAE features one at a time
  - Measure KL divergence of output logits
  - Compare effect magnitude across architectures
  - Prediction: Pythia features are higher-leverage (fewer effective features)

Experiment B: Induction Feature Detection
  - Feed [A B ... A B] repeated sequences
  - Find SAE features that fire specifically on the second occurrence of B
  - Compare induction-like features across architectures
"""

import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.sae import create_sae
from src.activation_cache import get_model_and_tokenizer, _get_layers

STORAGE = Path("/path/to/storage")
ACTS_DIR = STORAGE / "activations"
CKPT_DIR = STORAGE / "checkpoints_normed"
PHASE3_DIR = STORAGE / "results_phase3"
PHASE3_DIR.mkdir(exist_ok=True)

MODELS = {
    "mamba1_2.8b": {
        "name": "state-spaces/mamba-2.8b-hf",
        "n_layers": 64, "d_model": 2560, "mid_layer": 32,
    },
    "pythia_2.8b": {
        "name": "EleutherAI/pythia-2.8b",
        "n_layers": 32, "d_model": 2560, "mid_layer": 16,
    },
}


def load_sae_and_norm(model_key, layer, expansion=16, k=64, device="cuda"):
    """Load normalized SAE + compute normalization params from activations."""
    d_model = MODELS[model_key]["d_model"]
    d_hidden = d_model * expansion
    run_key = f"{model_key}_L{layer}_x{expansion}_k{k}_normed"
    path = CKPT_DIR / f"{run_key}.pt"
    if not path.exists():
        return None, None, None

    sae = create_sae(d_model, d_hidden, sae_type="topk", k=k).to(device)
    sae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    sae.eval()

    # Load norm params from activation subsample
    acts_path = ACTS_DIR / model_key / f"layer_{layer}.pt"
    t = torch.load(acts_path, map_location="cpu", weights_only=True, mmap=True)
    sample = t[:10000].clone().float()
    act_mean = sample.mean(dim=0).to(device)
    act_std = sample.std(dim=0).clamp(min=1e-6).to(device)
    del t, sample

    return sae, act_mean, act_std


# ============================================================
# Experiment A: Feature Steering
# ============================================================

def experiment_a_steering(device="cuda:0"):
    """Ablate individual SAE features and measure causal effect."""
    print("=" * 60)
    print("Experiment A: Feature Steering (Causal Ablation)")
    print("=" * 60)

    results = {}

    for model_key, info in MODELS.items():
        print(f"\n--- {model_key} ---")
        model, tokenizer = get_model_and_tokenizer(info["name"], device)

        mid_layer = info["mid_layer"]
        sae, act_mean, act_std = load_sae_and_norm(model_key, mid_layer, device=device)
        if sae is None:
            print(f"  No SAE for {model_key} L{mid_layer}")
            del model; torch.cuda.empty_cache()
            continue

        # Prepare test sequences
        from datasets import load_dataset
        dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        test_texts = []
        for ex in dataset:
            if ex["text"] and len(ex["text"].strip()) > 100:
                test_texts.append(ex["text"][:500])
            if len(test_texts) >= 20:
                break

        test_tokens = []
        for text in test_texts:
            toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            test_tokens.append(toks.to(device))

        # Get baseline logits
        model_layers = _get_layers(model)
        backend = getattr(model, '_model_backend', 'transformers')

        print(f"  Computing baseline logits...")
        baseline_logits = []
        for toks in test_tokens:
            with torch.no_grad():
                out = model(toks)
                baseline_logits.append(out.logits[:, -1, :].float())  # last token logits

        # Find top-k most active features on test data
        print(f"  Finding top features...")
        all_z = []
        hook_output = {}

        def capture_hook(module, inp, out):
            if backend == "mamba_ssm" and isinstance(out, tuple) and len(out) == 2:
                # mamba_ssm Block: (mixer_output, residual_before_block) — sum is residual stream.
                hook_output["h"] = out[0] + out[1]
            elif isinstance(out, tuple):
                hook_output["h"] = out[0]
            else:
                hook_output["h"] = out

        hook = model_layers[mid_layer].register_forward_hook(capture_hook)
        for toks in test_tokens:
            with torch.no_grad():
                model(toks)
            h = hook_output["h"].float()
            # Flatten to (total_tokens, d_model)
            h_flat = h.reshape(-1, h.shape[-1])
            # Normalize
            h_normed = (h_flat - act_mean) / act_std
            z = sae.encode(h_normed)
            # Mean activation per feature across all tokens
            all_z.append(z.mean(dim=0))
        hook.remove()

        mean_act = torch.stack(all_z).mean(dim=0)  # (d_hidden,)
        # Top 100 most active features
        top_features = torch.topk(mean_act, 100).indices.cpu().tolist()

        # Ablate each top feature and measure KL divergence
        print(f"  Ablating top 100 features...")
        kl_divs = []

        for feat_idx in tqdm(top_features, desc="Ablating"):
            def ablation_hook(module, inp, out, feat=feat_idx):
                if backend == "mamba_ssm" and isinstance(out, tuple) and len(out) == 2:
                    # Residual stream = mixer_output + residual. Modify so ablated R propagates.
                    h = (out[0] + out[1]).float()
                elif isinstance(out, tuple):
                    h = out[0].float()
                else:
                    h = out.float()

                b, s, d = h.shape
                h_flat = h.reshape(-1, d)
                h_normed = (h_flat - act_mean) / act_std
                z = sae.encode(h_normed)

                # Zero out this feature
                z[:, feat] = 0.0

                # Reconstruct
                h_recon = sae.decode(z)
                # Unnormalize
                h_unnormed = h_recon * act_std + act_mean
                h_new = h_unnormed.reshape(b, s, d).to(out[0].dtype if isinstance(out, tuple) else out.dtype)

                if backend == "mamba_ssm" and isinstance(out, tuple) and len(out) == 2:
                    # Preserve output[1] (residual_before_block); rewrite output[0] so sum = h_new.
                    return (h_new - out[1], out[1])
                if isinstance(out, tuple):
                    return (h_new,) + out[1:]
                return h_new

            hook = model_layers[mid_layer].register_forward_hook(ablation_hook)

            feat_kls = []
            for i, toks in enumerate(test_tokens):
                with torch.no_grad():
                    out = model(toks)
                    ablated_logits = out.logits[:, -1, :].float()

                # KL divergence
                baseline_probs = F.softmax(baseline_logits[i], dim=-1)
                ablated_log_probs = F.log_softmax(ablated_logits, dim=-1)
                kl = F.kl_div(ablated_log_probs, baseline_probs, reduction="batchmean").item()
                feat_kls.append(kl)

            hook.remove()
            kl_divs.append({
                "feature": feat_idx,
                "mean_kl": float(np.mean(feat_kls)),
                "max_kl": float(np.max(feat_kls)),
                "mean_activation": float(mean_act[feat_idx].item()),
            })

        # Sort by KL divergence
        kl_divs.sort(key=lambda x: x["mean_kl"], reverse=True)

        # Summary stats
        all_kls = [x["mean_kl"] for x in kl_divs]
        results[model_key] = {
            "layer": mid_layer,
            "n_features_tested": len(kl_divs),
            "mean_kl": float(np.mean(all_kls)),
            "median_kl": float(np.median(all_kls)),
            "max_kl": float(np.max(all_kls)),
            "std_kl": float(np.std(all_kls)),
            "top10_mean_kl": float(np.mean(all_kls[:10])),
            "top10_features": kl_divs[:10],
            "kl_distribution": {
                "above_0.1": int(sum(1 for x in all_kls if x > 0.1)),
                "above_0.01": int(sum(1 for x in all_kls if x > 0.01)),
                "above_0.001": int(sum(1 for x in all_kls if x > 0.001)),
            },
            "all_kls": kl_divs,
        }

        print(f"  {model_key}: mean_KL={np.mean(all_kls):.4f}, "
              f"max_KL={np.max(all_kls):.4f}, "
              f"top10_mean={np.mean(all_kls[:10]):.4f}")
        print(f"  KL>0.1: {results[model_key]['kl_distribution']['above_0.1']}, "
              f"KL>0.01: {results[model_key]['kl_distribution']['above_0.01']}")

        del model, sae
        torch.cuda.empty_cache()

    with open(PHASE3_DIR / "steering.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved steering.json")
    return results


# ============================================================
# Experiment B: Induction Feature Detection
# ============================================================

def experiment_b_induction(device="cuda:0"):
    """Find features that detect repeated patterns (induction-like behavior)."""
    print("\n" + "=" * 60)
    print("Experiment B: Induction Feature Detection")
    print("=" * 60)

    results = {}

    for model_key, info in MODELS.items():
        print(f"\n--- {model_key} ---")
        model, tokenizer = get_model_and_tokenizer(info["name"], device)

        mid_layer = info["mid_layer"]
        sae, act_mean, act_std = load_sae_and_norm(model_key, mid_layer, device=device)
        if sae is None:
            del model; torch.cuda.empty_cache()
            continue

        model_layers = _get_layers(model)
        backend = getattr(model, '_model_backend', 'transformers')

        # Create repeated-pattern sequences: [random A B C D ... A B C D]
        # Induction features should fire on the SECOND occurrence
        n_trials = 20
        pattern_len = 10
        n_repeats = 3

        induction_scores = torch.zeros(sae.d_hidden)
        control_scores = torch.zeros(sae.d_hidden)
        count = 0

        for trial in range(n_trials):
            # Random pattern
            pattern = torch.randint(100, 5000, (pattern_len,))
            # Repeated sequence: [prefix] [pattern] [pattern] [pattern]
            prefix = torch.randint(100, 5000, (20,))
            repeated_seq = torch.cat([prefix] + [pattern] * n_repeats).unsqueeze(0).to(device)

            # Random (non-repeated) control
            control_seq = torch.randint(100, 5000, (repeated_seq.shape[1],)).unsqueeze(0).to(device)

            hook_output = {}
            def capture(module, inp, out):
                if isinstance(out, tuple):
                    hook_output["h"] = out[0].float()
                else:
                    hook_output["h"] = out.float()

            hook = model_layers[mid_layer].register_forward_hook(capture)

            # Repeated sequence
            with torch.no_grad():
                model(repeated_seq)
            h_rep = hook_output["h"]  # (1, seq_len, d_model)

            # Control sequence
            with torch.no_grad():
                model(control_seq)
            h_ctrl = hook_output["h"]

            hook.remove()

            # Encode with SAE
            with torch.no_grad():
                h_rep_normed = (h_rep.reshape(-1, h_rep.shape[-1]) - act_mean) / act_std
                z_rep = sae.encode(h_rep_normed)  # (seq_len, d_hidden)

                h_ctrl_normed = (h_ctrl.reshape(-1, h_ctrl.shape[-1]) - act_mean) / act_std
                z_ctrl = sae.encode(h_ctrl_normed)

            # Positions of 2nd and 3rd pattern occurrence
            first_pattern_start = 20  # after prefix
            second_pattern_start = 20 + pattern_len
            third_pattern_start = 20 + 2 * pattern_len

            # Feature activation at 2nd/3rd pattern positions (induction territory)
            induction_pos = list(range(second_pattern_start, second_pattern_start + pattern_len)) + \
                           list(range(third_pattern_start, third_pattern_start + pattern_len))
            first_pos = list(range(first_pattern_start, first_pattern_start + pattern_len))

            if max(induction_pos) < z_rep.shape[0] and max(first_pos) < z_rep.shape[0]:
                # Induction score: activation at repeated positions minus first occurrence
                z_induction = z_rep[induction_pos].mean(dim=0)
                z_first = z_rep[first_pos].mean(dim=0)
                z_control = z_ctrl[induction_pos].mean(dim=0) if max(induction_pos) < z_ctrl.shape[0] else z_ctrl.mean(dim=0)

                # Features that fire MORE on repeated than first occurrence
                induction_scores += (z_induction - z_first).cpu()
                control_scores += z_control.cpu()
                count += 1

        if count > 0:
            induction_scores /= count
            control_scores /= count

        # Find features with highest induction score (fire more on repeats)
        top_induction = torch.topk(induction_scores, 50)
        induction_features = []
        for idx, (score, feat) in enumerate(zip(top_induction.values, top_induction.indices)):
            induction_features.append({
                "feature": int(feat),
                "induction_score": float(score),
                "control_activation": float(control_scores[feat]),
                "ratio": float(score / (control_scores[feat] + 1e-8)),
            })

        # Count features with strong induction behavior
        strong_induction = (induction_scores > induction_scores.std() * 2).sum().item()

        results[model_key] = {
            "layer": mid_layer,
            "n_trials": n_trials,
            "n_strong_induction": strong_induction,
            "mean_induction_score": float(induction_scores.mean()),
            "max_induction_score": float(induction_scores.max()),
            "std_induction_score": float(induction_scores.std()),
            "top_induction_features": induction_features[:20],
        }

        print(f"  {model_key}: {strong_induction} strong induction features, "
              f"max_score={induction_scores.max():.4f}")

        del model, sae
        torch.cuda.empty_cache()

    with open(PHASE3_DIR / "induction.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved induction.json")
    return results


def main():
    print("=" * 60)
    print("PHASE 3: CAUSAL EXPERIMENTS")
    print("=" * 60)

    # Run on separate GPUs
    experiment_a_steering("cuda:0")
    experiment_b_induction("cuda:0")

    print("\n" + "=" * 60)
    print("ALL PHASE 3 EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
