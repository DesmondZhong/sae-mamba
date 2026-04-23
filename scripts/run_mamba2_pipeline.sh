#!/usr/bin/env bash
# End-to-end Mamba-2 SAE pipeline:
#   1. Extract activations at layers 0,8,16,24,32,40,48,56,63 on 1 GPU (single forward).
#   2. Train 9 SAEs in parallel across all GPUs.
# Usage: bash scripts/run_mamba2_pipeline.sh

set -u
cd "$(dirname "$0")/.."

: "${SAE_MAMBA_STORAGE:=/workspace/excuse}"
export SAE_MAMBA_STORAGE
: "${HF_HOME:=/workspace/hf_cache}"
export HF_HOME
: "${HF_HUB_CACHE:=/workspace/hf_cache}"
export HF_HUB_CACHE
: "${TRANSFORMERS_CACHE:=/workspace/hf_cache}"
export TRANSFORMERS_CACHE

LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"

MODEL_KEY="${MODEL_KEY:-mamba2_2.7b}"
LAYERS="${LAYERS:-0,8,16,24,32,40,48,56,63}"

echo "=== Mamba-2 pipeline start: $(date) ==="
echo "SAE_MAMBA_STORAGE=$SAE_MAMBA_STORAGE"
echo "HF cache=$HF_HOME"
echo "Log dir=$LOG_DIR"
echo

# ---- Phase 1: extraction (single GPU, all 9 layers in one forward) ----
EXTRACT_LOG="$LOG_DIR/extract_${MODEL_KEY}.log"
echo "[phase1] Extracting activations for $MODEL_KEY layers [$LAYERS] (log: $EXTRACT_LOG)"
CUDA_VISIBLE_DEVICES=0 LAYERS="$LAYERS" \
    python scripts/extract_model.py "$MODEL_KEY" > "$EXTRACT_LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "[phase1] FAILED extraction (exit $rc). See $EXTRACT_LOG"
    exit $rc
fi
echo "[phase1] Extraction done."
echo

# ---- Phase 2: training (parallel across N_GPUS, round-robin layers) ----
echo "[phase2] Launching SAE training across all GPUs..."
LAYERS_SPACE="${LAYERS//,/ }"
LAYERS="$LAYERS_SPACE" LOG_DIR="$LOG_DIR" \
    bash scripts/run_mamba2_saes.sh
rc=$?
echo "[phase2] Training exit code: $rc"
echo "=== Pipeline done: $(date) ==="
