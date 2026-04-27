#!/usr/bin/env python3
"""Extract x_proj INPUT and OUTPUT activations at a specified Mamba-1 layer.

x_proj is the selective-scan parameter generator in MambaMixer. Its input is
the post-conv d_inner (5120-dim) representation; its output is the concat of
(dt_pre, B, C) of shape (dt_rank + 2*state_size) = 192-dim. Hooking both lets
us later train an internal SAE on the pre-scan representation (input) or the
selective-scan parameters themselves (output).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/07_extract_xproj.py \
        --layer 30 --n_tokens 10_000_000
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.activation_cache import get_model_and_tokenizer, get_text_data

MODEL_NAME = "state-spaces/mamba-2.8b-hf"
SEQ_LEN = 512
DATASET = "pile"

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/path/to/storage"))
ACTS_DIR = STORAGE / "activations" / "mamba1_2.8b"
ACTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--n_tokens", type=int, default=10_000_000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    in_path = ACTS_DIR / f"layer_{args.layer}_xproj_in.fp16.npy"
    out_path = ACTS_DIR / f"layer_{args.layer}_xproj_out.fp16.npy"
    meta_path = ACTS_DIR / f"layer_{args.layer}_xproj_meta.json"

    if in_path.exists() and out_path.exists():
        print(f"[skip] {in_path} and {out_path} already exist")
        return

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, args.device)
    mixer = model.backbone.layers[args.layer].mixer
    d_in = mixer.x_proj.in_features
    d_out = mixer.x_proj.out_features
    print(f"L{args.layer} x_proj: in_dim={d_in}, out_dim={d_out}")

    # Tokenize
    sequences = get_text_data(args.n_tokens, SEQ_LEN, tokenizer, dataset_name=DATASET)
    n_seqs = sequences.shape[0]
    total_tokens = n_seqs * SEQ_LEN
    print(f"  {n_seqs} seqs × {SEQ_LEN} tokens = {total_tokens:,} tokens")

    # Pre-allocate memmaps on disk (fp16 to halve footprint)
    print(f"Pre-allocating {in_path} ({total_tokens * d_in * 2 / 1e9:.1f} GB)...")
    mmap_in = np.memmap(in_path, dtype="float16", mode="w+", shape=(total_tokens, d_in))
    print(f"Pre-allocating {out_path} ({total_tokens * d_out * 2 / 1e9:.1f} GB)...")
    mmap_out = np.memmap(out_path, dtype="float16", mode="w+", shape=(total_tokens, d_out))

    cursor = {"v": 0}

    def pre_hook(mod, inputs):
        flat = inputs[0].reshape(-1, d_in).to(torch.float16).cpu().numpy()
        n = flat.shape[0]
        mmap_in[cursor["v"]:cursor["v"] + n] = flat

    def post_hook(mod, inputs, output):
        flat = output.reshape(-1, d_out).to(torch.float16).cpu().numpy()
        n = flat.shape[0]
        mmap_out[cursor["v"]:cursor["v"] + n] = flat
        cursor["v"] += n  # advance cursor AFTER both hooks run

    h1 = mixer.x_proj.register_forward_pre_hook(pre_hook)
    h2 = mixer.x_proj.register_forward_hook(post_hook)

    n_batches = (n_seqs + args.batch_size - 1) // args.batch_size
    start = time.time()
    for i in tqdm(range(n_batches), desc=f"Forward L{args.layer}"):
        batch = sequences[i * args.batch_size:(i + 1) * args.batch_size].to(args.device)
        with torch.no_grad():
            model(batch)
        torch.cuda.empty_cache()

    h1.remove(); h2.remove()
    mmap_in.flush(); mmap_out.flush()
    elapsed = time.time() - start
    print(f"Forward done in {elapsed/60:.1f} min. Cursor at {cursor['v']:,} / {total_tokens:,}")

    import json
    json.dump({
        "layer": args.layer,
        "model": MODEL_NAME,
        "n_tokens_written": cursor["v"],
        "d_in": d_in,
        "d_out": d_out,
        "seq_len": SEQ_LEN,
        "dtype": "float16",
        "in_shape": [cursor["v"], d_in],
        "out_shape": [cursor["v"], d_out],
    }, open(meta_path, "w"), indent=2)
    print(f"Wrote {in_path}, {out_path}, {meta_path}")


if __name__ == "__main__":
    main()
