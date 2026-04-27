#!/usr/bin/env python3
"""Dump max-activating natural-text examples for the top induction features.

For each of the top-10 Mamba-1 L32 SAE induction features, scan Pile text,
find the top-5 positions where the feature fires most strongly, and record the
text context around each. This is the standard "feature browser" dump for
interpretability.

Output: $STORAGE/results_phase4/induction_feature_examples.json
"""
import argparse, json, os, sys
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

MODEL_NAME = "state-spaces/mamba-2.8b-hf"
MODEL_KEY = "mamba1_2.8b"
D_MODEL = 2560
MID_LAYER = 32
SAE_EXPANSION = 16
SAE_K = 64
SEQ_LEN = 256
CONTEXT_WINDOW = 24


def load_sae_and_norm(device):
    d_hidden = D_MODEL * SAE_EXPANSION
    run_key = f"{MODEL_KEY}_L{MID_LAYER}_x{SAE_EXPANSION}_k{SAE_K}_normed"
    sae = create_sae(D_MODEL, d_hidden, sae_type="topk", k=SAE_K).to(device)
    sae.load_state_dict(torch.load(CKPT_DIR / f"{run_key}.pt",
                                   map_location=device, weights_only=True))
    sae.eval()
    acts_path = ACTS_DIR / MODEL_KEY / f"layer_{MID_LAYER}.pt"
    t = torch.load(acts_path, map_location="cpu", weights_only=True, mmap=True)
    sample = t[:10000].clone().float()
    act_mean = sample.mean(dim=0).to(device)
    act_std = sample.std(dim=0).clamp(min=1e-6).to(device)
    del t, sample
    return sae, act_mean, act_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=1500)
    ap.add_argument("--top_feats", type=int, default=10)
    ap.add_argument("--examples_per_feature", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device

    ind = json.load(open(RESULTS_DIR / "induction_features.json"))
    top_feats = ind["feature"][:args.top_feats]
    print(f"Feature set: {top_feats}")

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    sae, act_mean, act_std = load_sae_and_norm(device)

    # Per-feature running top-K: (activation, doc_idx, token_idx, tokens, text)
    import heapq
    top_heap = {f: [] for f in top_feats}  # min-heap per feature

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

        # Forward + encode
        captured = {}
        def hook(mod, ins, out):
            captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()
        h = model.backbone.layers[MID_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            model(ids.unsqueeze(0).to(device))
        h.remove()

        normed = (captured["r"].float() - act_mean) / act_std
        _, z, *_ = sae(normed)
        z = z[0].detach().cpu().numpy()  # (L, d_hidden)

        # For each target feature, find top positions in this doc
        for f in top_feats:
            col = z[:, f]
            # Find top 3 positions in this doc (avoid positions < CONTEXT_WINDOW)
            valid = np.arange(CONTEXT_WINDOW, len(col))
            if len(valid) == 0:
                continue
            # Top-3 per doc
            top3_pos = valid[np.argsort(col[valid])[-3:]]
            for p in top3_pos:
                act = float(col[p])
                if act < 1e-5:
                    continue
                start = max(0, p - CONTEXT_WINDOW)
                end = min(len(col), p + 4)
                context_ids = ids[start:end].tolist()
                context_text = tokenizer.decode(context_ids)
                token = tokenizer.decode([ids[p].item()])
                entry = (act, n_processed, int(p), int(ids[p].item()),
                         token, context_text.replace("\n", " "))
                if len(top_heap[f]) < args.examples_per_feature:
                    heapq.heappush(top_heap[f], entry)
                elif act > top_heap[f][0][0]:
                    heapq.heapreplace(top_heap[f], entry)

        n_processed += 1
        pbar.update(1)
    pbar.close()

    # Convert heaps → sorted lists
    result = {"features": []}
    for f in top_feats:
        top_k = sorted(top_heap[f], reverse=True)
        result["features"].append({
            "feature": f,
            "examples": [
                {
                    "activation": act,
                    "doc_idx": d,
                    "position": p,
                    "token_id": tid,
                    "max_token": tok,
                    "context": ctx,
                }
                for (act, d, p, tid, tok, ctx) in top_k
            ],
        })

    out_path = RESULTS_DIR / "induction_feature_examples.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"Wrote {out_path}")

    # Print summary
    print("\n=== Top-activating examples for each induction feature ===")
    for entry in result["features"]:
        print(f"\n--- feat {entry['feature']} ---")
        for ex in entry["examples"][:3]:
            arrow = " → HERE:"
            print(f"  act={ex['activation']:.3f}  |{ex['context']}{arrow} [{ex['max_token']!r}]")


if __name__ == "__main__":
    main()
