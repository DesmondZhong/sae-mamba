#!/usr/bin/env python3
"""Max-activating natural-text examples for the L30 x_proj internal SAE
induction features.

Analog of scripts/19_induction_feature_examples.py but reads features from
the internal SAE (trained on x_proj INPUT at L30) rather than the L32
residual-stream SAE.
"""
import argparse, json, os, sys, heapq
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.activation_cache import get_model_and_tokenizer
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
CKPT_DIR = STORAGE / "checkpoints_normed"
RESULTS_DIR = STORAGE / "results_phase4"

MODEL_NAME = "state-spaces/mamba-2.8b-hf"
MODEL_KEY = "mamba1_2.8b"
LAYER = 30
D_IN = 5120
EXPANSION = 8
K = 64
SEQ_LEN = 256
CONTEXT_WINDOW = 24


def load_internal_sae(device):
    d_hidden = D_IN * EXPANSION
    run_key = f"{MODEL_KEY}_L{LAYER}_xprojin_x{EXPANSION}_k{K}_normed"
    sae = create_sae(D_IN, d_hidden, sae_type="topk", k=K).to(device)
    sae.load_state_dict(torch.load(CKPT_DIR / f"{run_key}.pt",
                                    map_location=device, weights_only=True))
    sae.eval()
    np_params = torch.load(CKPT_DIR / f"{run_key}_normparams.pt",
                            map_location=device, weights_only=True)
    return sae, np_params["act_mean"].to(device), np_params["act_std"].to(device)


def capture_xproj_input(model, tokens, layer):
    captured = {}
    def pre_hook(mod, ins):
        captured["x"] = ins[0].detach()
    h = model.backbone.layers[layer].mixer.x_proj.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["x"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=1500)
    ap.add_argument("--top_feats", type=int, default=10)
    ap.add_argument("--examples_per_feature", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device

    ind = json.load(open(RESULTS_DIR / "xproj_internal_sae_induction_features.json"))
    top_feats = ind["top_features"]["feature"][:args.top_feats]
    print(f"Feature set: {top_feats}")

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    sae, act_mean, act_std = load_internal_sae(device)

    top_heap = {f: [] for f in top_feats}

    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    it = iter(ds)
    pbar = tqdm(desc="docs", total=args.n_docs)
    n_processed = 0

    while n_processed < args.n_docs:
        try:
            ex = next(it)
        except StopIteration:
            break
        text = ex.get("text", "")
        if not text or len(text) < 400:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=SEQ_LEN)["input_ids"][0]
        if ids.shape[0] < 64:
            continue

        xp_in = capture_xproj_input(model, ids.unsqueeze(0).to(device), LAYER)
        normed = (xp_in.float() - act_mean) / act_std
        with torch.no_grad():
            _, z, *_ = sae(normed)
        z = z[0].detach().cpu().numpy()

        for f in top_feats:
            col = z[:, f]
            valid = np.arange(CONTEXT_WINDOW, len(col))
            if len(valid) == 0:
                continue
            top3_pos = valid[np.argsort(col[valid])[-3:]]
            for p in top3_pos:
                act = float(col[p])
                if act < 1e-5:
                    continue
                start = max(0, p - CONTEXT_WINDOW)
                end = min(len(col), p + 4)
                ctx_ids = ids[start:end].tolist()
                ctx = tokenizer.decode(ctx_ids).replace("\n", " ")
                tok = tokenizer.decode([ids[p].item()])
                entry = (act, n_processed, int(p), int(ids[p].item()), tok, ctx)
                if len(top_heap[f]) < args.examples_per_feature:
                    heapq.heappush(top_heap[f], entry)
                elif act > top_heap[f][0][0]:
                    heapq.heapreplace(top_heap[f], entry)

        n_processed += 1
        pbar.update(1)
    pbar.close()

    result = {"layer": LAYER, "site": "x_proj input", "features": []}
    for f in top_feats:
        top_k = sorted(top_heap[f], reverse=True)
        result["features"].append({
            "feature": f,
            "examples": [
                {"activation": a, "doc_idx": d, "position": p,
                 "token_id": tid, "max_token": tok, "context": ctx}
                for (a, d, p, tid, tok, ctx) in top_k
            ],
        })

    out_path = RESULTS_DIR / "xproj_internal_sae_feature_examples.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")
    print("\n=== Top-activating examples for each internal feature ===")
    for entry in result["features"]:
        print(f"\n--- feat {entry['feature']} ---")
        for ex in entry["examples"][:3]:
            print(f"  act={ex['activation']:.3f}  |{ex['context']} → [{ex['max_token']!r}]")


if __name__ == "__main__":
    main()
