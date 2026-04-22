#!/usr/bin/env bash
# Wait until a specified checkpoint file exists, then train a target SAE on a specified GPU.
# Usage: bash retrain_one_when_free.sh <wait_ckpt_path> <gpu> <model> <layer> <expansion> <k> <log_file>
set -u
WAIT_CKPT="$1"
GPU="$2"
MODEL="$3"
LAYER="$4"
EXP="$5"
K="$6"
LOG="$7"

cd "$(dirname "$0")/.."
: "${SAE_MAMBA_STORAGE:=/workspace/excuse}"
export SAE_MAMBA_STORAGE
: "${HF_HOME:=/workspace/hf_cache}"
export HF_HOME

echo "[watch] waiting for $WAIT_CKPT  ($(date))" >> "$LOG"
until [ -f "$WAIT_CKPT" ]; do sleep 10; done
echo "[watch] $WAIT_CKPT appeared; short settle then launch" >> "$LOG"
sleep 15  # let the parent Mamba-2 worker tear down / release GPU
echo "[launch] CUDA_VISIBLE_DEVICES=$GPU train $MODEL L$LAYER x$EXP k$K  ($(date))" >> "$LOG"
CUDA_VISIBLE_DEVICES=$GPU python scripts/train_one_sae_normed.py "$MODEL" "$LAYER" "$EXP" "$K" >> "$LOG" 2>&1
echo "[done] rc=$?  $(date)" >> "$LOG"
