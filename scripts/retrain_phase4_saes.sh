#!/usr/bin/env bash
# Fallback: retrain Mamba-1 L32 and Pythia L16 SAEs on x16 k64, in case
# the essential checkpoint tarball never finishes transfer. These are needed for phase 4.
# Runs 2 jobs in parallel (one per GPU), each ~60 min on an H100.
# Usage: bash scripts/retrain_phase4_saes.sh <gpu_for_mamba1> <gpu_for_pythia>
set -u
cd "$(dirname "$0")/.."
: "${SAE_MAMBA_STORAGE:=/path/to/storage}"
export SAE_MAMBA_STORAGE
: "${HF_HOME:=/workspace/hf_cache}"
export HF_HOME

GPU_M1="${1:-0}"
GPU_PY="${2:-1}"

echo "=== phase-4 SAE retrain: $(date) ==="
echo "mamba1_2.8b L32 x16 k64 on GPU $GPU_M1"
echo "pythia_2.8b  L16 x16 k64 on GPU $GPU_PY"

CUDA_VISIBLE_DEVICES=$GPU_M1 \
    python scripts/train_one_sae_normed.py mamba1_2.8b 32 16 64 \
    > /workspace/logs/retrain_mamba1_L32.log 2>&1 &
M1_PID=$!
CUDA_VISIBLE_DEVICES=$GPU_PY \
    python scripts/train_one_sae_normed.py pythia_2.8b 16 16 64 \
    > /workspace/logs/retrain_pythia_L16.log 2>&1 &
PY_PID=$!

echo "mamba1 pid=$M1_PID  pythia pid=$PY_PID"
wait $M1_PID; M1_RC=$?
wait $PY_PID; PY_RC=$?
echo "=== retrain done: mamba1 rc=$M1_RC  pythia rc=$PY_RC ($(date)) ==="
