#!/usr/bin/env python3
"""Round-robin launcher for Mamba-2 SAE training across multiple GPUs.

Each GPU worker runs a sequential queue of (layer, expansion, k) jobs.
Respects SAE_MAMBA_STORAGE. Skips any SAE whose checkpoint already exists.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL_KEY = "mamba2_2.7b"

# The canonical 12-SAE Mamba-2 sweep.
JOBS = [
    # Layer sweep @ x16 k64 (8 layers)
    (0, 16, 64),
    (8, 16, 64),
    (16, 16, 64),
    (24, 16, 64),
    (32, 16, 64),
    (40, 16, 64),
    (48, 16, 64),
    (56, 16, 64),
    # K sweep at middle layer L32 (2)
    (32, 16, 32),
    (32, 16, 128),
    # Expansion sweep at middle layer L32 (2)
    (32, 8, 64),
    (32, 32, 64),
]


def run_worker(gpu_id: int, queue: list, log_dir: Path, storage: Path) -> None:
    """Run a sequential queue of SAE training jobs on a single GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["SAE_MAMBA_STORAGE"] = str(storage)
    for layer, exp, k in queue:
        ckpt = storage / "checkpoints_normed" / f"{MODEL_KEY}_L{layer}_x{exp}_k{k}_normed.pt"
        log_file = log_dir / f"train_{MODEL_KEY}_L{layer}_x{exp}_k{k}_gpu{gpu_id}.log"
        if ckpt.exists():
            print(f"[gpu {gpu_id}] skip L{layer} x{exp} k{k} (exists)", flush=True)
            continue
        print(f"[gpu {gpu_id}] train L{layer} x{exp} k{k} (log: {log_file})", flush=True)
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                [sys.executable, "scripts/train_one_sae_normed.py",
                 MODEL_KEY, str(layer), str(exp), str(k)],
                cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT,
            )
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        print(f"[gpu {gpu_id}] done L{layer} x{exp} k{k}: {status}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="GPU indices to use.")
    parser.add_argument("--storage", default=os.environ.get("SAE_MAMBA_STORAGE",
                                                            "/workspace/excuse"))
    parser.add_argument("--log_dir", default="/workspace/logs")
    args = parser.parse_args()

    storage = Path(args.storage)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Round-robin across GPUs
    queues: dict = {g: [] for g in args.gpus}
    for i, job in enumerate(JOBS):
        queues[args.gpus[i % len(args.gpus)]].append(job)

    for g, q in queues.items():
        print(f"[launcher] GPU {g}: {len(q)} jobs  ({', '.join(f'L{l}x{e}k{k}' for l,e,k in q)})",
              flush=True)

    # Spawn one child process per GPU
    import multiprocessing as mp
    procs = []
    for g in args.gpus:
        p = mp.Process(target=run_worker, args=(g, queues[g], log_dir, storage))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    print("[launcher] all workers finished.", flush=True)


if __name__ == "__main__":
    main()
