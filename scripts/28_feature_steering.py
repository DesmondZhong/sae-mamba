#!/usr/bin/env python3
"""Feature steering: clamp the top internal-SAE induction feature high at L30
during generation. Does the model produce more repetition?

Implementation:
  During forward, hook x_proj INPUT at L30. Project current d_inner (5120) input
  through the internal SAE's encoder to get z (40,960). Modify z at the target
  feature: set it to `clamp_value * 5` × its pre-clamp typical magnitude.
  Decode: reconstruct x = z @ W_dec + b. Replace the x_proj input with this
  modified reconstruction (or add delta = modified - original).

Measure:
  - Generation perplexity on a held-out prompt with vs without steering
  - Repetition rate (fraction of 4-grams that repeat) on generated text
  - Induction-feature activation at L32 on the generated text (should be elevated)

Output: $STORAGE/results_phase4/feature_steering.json
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.activation_cache import get_model_and_tokenizer
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
CKPT_DIR = STORAGE / "checkpoints_normed"
RESULTS_DIR = STORAGE / "results_phase4"

MODEL_NAME = "state-spaces/mamba-2.8b-hf"
MODEL_KEY = "mamba1_2.8b"
L30 = 30
D_IN = 5120
INTERNAL_EXP = 8
K = 64


def load_internal_sae(device):
    d_hidden = D_IN * INTERNAL_EXP
    run_key = f"{MODEL_KEY}_L{L30}_xprojin_x{INTERNAL_EXP}_k{K}_normed"
    sae = create_sae(D_IN, d_hidden, sae_type="topk", k=K).to(device)
    sae.load_state_dict(torch.load(CKPT_DIR / f"{run_key}.pt",
                                    map_location=device, weights_only=True))
    sae.eval()
    np_params = torch.load(CKPT_DIR / f"{run_key}_normparams.pt",
                            map_location=device, weights_only=True)
    return sae, np_params["act_mean"].to(device), np_params["act_std"].to(device)


def generate_with_steering(model, tokenizer, prompt, max_new_tokens=80,
                            steer_feature=None, steer_strength=0.0,
                            sae=None, act_mean=None, act_std=None,
                            device="cuda:0", seed=0):
    """Generate autoregressively. If steer_feature/strength are set, at each
    forward pass we modify L30 x_proj input by adding a contribution from the
    steer feature's decoder direction, scaled by strength."""
    torch.manual_seed(seed)
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    active_hooks = []
    if steer_feature is not None and steer_strength != 0.0 and sae is not None:
        # Get the steer feature's pre-decoder direction (its column in W_dec)
        # In our TopK SAE: sae.decoder is a Linear(d_hidden, d_input). W_dec has
        # shape (d_input, d_hidden), column [f] is the feature's reconstruction direction.
        feat_dir = sae.decoder.weight.data[:, steer_feature]  # (d_input,)
        # Rescale to act_std (denormalize) to match the x_proj INPUT space
        # The SAE was trained on normalized inputs. The decoder produces outputs
        # in the NORMALIZED space. To add to the un-normalized x_proj input, we
        # need to denormalize: raw_delta = feat_dir * act_std (act_mean doesn't shift direction).
        raw_dir = feat_dir * act_std.squeeze(0)   # (d_input,)
        magnitude = float(raw_dir.norm().item())
        print(f"  steering feature {steer_feature}: raw-direction norm = {magnitude:.3f}")

        def steer_hook(mod, ins):
            # ins[0] shape: (B, L, d_inner=5120)
            x = ins[0].clone()
            # Add steer_strength * raw_dir to every position
            x = x + steer_strength * raw_dir.to(x.dtype).view(1, 1, -1)
            return (x,) + ins[1:] if len(ins) > 1 else (x,)
        h = model.backbone.layers[L30].mixer.x_proj.register_forward_pre_hook(steer_hook)
        active_hooks.append(h)

    # Greedy decode one token at a time (so we don't need caching complexity)
    generated = ids.clone()
    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(generated)
        next_logits = out.logits[:, -1, :]
        next_id = next_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)

    for h in active_hooks:
        h.remove()

    return generated, tokenizer.decode(generated[0])


def repetition_rate(token_ids, n=4):
    """Fraction of tokens that complete a repeated n-gram."""
    ids = token_ids.tolist()
    L = len(ids)
    if L < n + 1:
        return 0.0
    count_repeat = 0
    for i in range(n, L):
        ng = tuple(ids[i - n + 1:i + 1])
        # Check if this n-gram occurred earlier
        for j in range(i - 1, n - 2, -1):
            if tuple(ids[j - n + 1:j + 1]) == ng:
                count_repeat += 1
                break
    return count_repeat / (L - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--steer_feature", type=int, default=33108)
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    sae_int, int_mean, int_std = load_internal_sae(device)

    prompts = [
        # Explicit induction prompt: bigram "Alice loves Bob" should repeat
        "Alice loves Bob. Bob loves Alice. Alice loves Bob. Bob loves",
        # Natural repetitive prompt
        "Python is a programming language that",
        # Pattern-repeat prompt (Olsson-style)
        "The code repeats every 4 tokens: foo bar baz qux foo bar baz qux foo bar baz qux foo",
    ]

    all_results = {"steer_feature": args.steer_feature, "per_prompt": []}
    for prompt in prompts:
        prompt_result = {"prompt": prompt, "generations": []}
        print(f"\n=== Prompt: '{prompt}' ===")
        for strength in args.strengths:
            gen_ids, gen_text = generate_with_steering(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                steer_feature=args.steer_feature,
                steer_strength=strength,
                sae=sae_int, act_mean=int_mean, act_std=int_std,
                device=device,
            )
            rep = repetition_rate(gen_ids[0, -args.max_new_tokens:])
            preview = gen_text[len(prompt):][:200].replace("\n", " ")
            print(f"  strength={strength:4.1f}  rep_rate={rep:.3f}  text: {preview}")
            prompt_result["generations"].append({
                "strength": strength,
                "repetition_rate": rep,
                "continuation": gen_text[len(prompt):],
            })
        all_results["per_prompt"].append(prompt_result)

    out_path = RESULTS_DIR / "feature_steering.json"
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
