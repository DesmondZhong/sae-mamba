#!/usr/bin/env bash
# Safety-net: produce pythia_2.8b L16 activations + SAE on a single GPU.
# Feeds scripts/05_pythia_induction_compare.py if the tar doesn't ship it.
# Usage: GPU=1 bash scripts/run_pythia_l16_safety_net.sh

set -u
cd "$(dirname "$0")/.."

: "${GPU:=1}"
: "${SAE_MAMBA_STORAGE:=/path/to/storage}"
: "${HF_HOME:=/workspace/hf_cache}"
: "${LOG_DIR:=/workspace/logs}"
export SAE_MAMBA_STORAGE HF_HOME HF_HUB_CACHE="$HF_HOME" TRANSFORMERS_CACHE="$HF_HOME"

MODEL_KEY="pythia_2.8b"
LAYER=16
EXPANSION=16
K=64

extract_log="$LOG_DIR/extract_${MODEL_KEY}.log"
train_log="$LOG_DIR/train_${MODEL_KEY}_L${LAYER}_x${EXPANSION}_k${K}.log"

acts_path="$SAE_MAMBA_STORAGE/activations/$MODEL_KEY/layer_${LAYER}.pt"
ckpt_path="$SAE_MAMBA_STORAGE/checkpoints_normed/${MODEL_KEY}_L${LAYER}_x${EXPANSION}_k${K}_normed.pt"

echo "=== Pythia L16 safety net start: $(date) (GPU $GPU) ==="

if [ -f "$acts_path" ]; then
    echo "[extract] Skipping — $acts_path already exists."
else
    echo "[extract] Extracting Pythia L${LAYER} (log: $extract_log)"
    CUDA_VISIBLE_DEVICES=$GPU LAYERS="$LAYER" BATCH_SIZE=8 TQDM_DISABLE=1 \
        python scripts/extract_model.py "$MODEL_KEY" > "$extract_log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[extract] FAILED (exit $rc). See $extract_log"
        exit $rc
    fi
fi

if [ -f "$ckpt_path" ]; then
    echo "[train] Skipping — $ckpt_path already exists."
else
    echo "[train] Training Pythia L${LAYER} SAE x${EXPANSION} k${K} (log: $train_log)"
    CUDA_VISIBLE_DEVICES=$GPU \
        python scripts/train_one_sae_normed.py "$MODEL_KEY" "$LAYER" "$EXPANSION" "$K" \
        > "$train_log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[train] FAILED (exit $rc). See $train_log"
        exit $rc
    fi
fi

echo "=== Pythia L16 safety net done: $(date) ==="
