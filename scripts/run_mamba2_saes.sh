#!/usr/bin/env bash
# Run Mamba-2 SAE training across all available GPUs.
# Each GPU trains a sequential queue of SAEs. Round-robin assignment.
# Usage: bash scripts/run_mamba2_saes.sh

set -u
cd "$(dirname "$0")/.."

: "${SAE_MAMBA_STORAGE:?SAE_MAMBA_STORAGE must be set}"
MODEL_KEY="${MODEL_KEY:-mamba2_2.7b}"
EXPANSION="${EXPANSION:-16}"
K="${K:-64}"
LAYERS="${LAYERS:-0 8 16 24 32 40 48 56 63}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
N_GPUS="${N_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

mkdir -p "$LOG_DIR"

echo "[launcher] model=$MODEL_KEY expansion=$EXPANSION k=$K n_gpus=$N_GPUS"
echo "[launcher] layers: $LAYERS"
echo "[launcher] storage: $SAE_MAMBA_STORAGE"

# Build per-GPU queues (round-robin)
declare -A GPU_QUEUE
i=0
for layer in $LAYERS; do
    gpu=$(( i % N_GPUS ))
    GPU_QUEUE[$gpu]="${GPU_QUEUE[$gpu]:-} $layer"
    i=$(( i + 1 ))
done

# Launch one worker per GPU; each worker processes its queue sequentially
pids=()
for gpu in $(seq 0 $(( N_GPUS - 1 ))); do
    queue="${GPU_QUEUE[$gpu]:-}"
    [ -z "$queue" ] && continue
    (
        for layer in $queue; do
            log_file="$LOG_DIR/train_${MODEL_KEY}_L${layer}_x${EXPANSION}_k${K}_gpu${gpu}.log"
            echo "[gpu $gpu] Training layer $layer (log: $log_file)"
            CUDA_VISIBLE_DEVICES="$gpu" \
                python scripts/train_one_sae_normed.py \
                "$MODEL_KEY" "$layer" "$EXPANSION" "$K" \
                > "$log_file" 2>&1
            rc=$?
            if [ $rc -ne 0 ]; then
                echo "[gpu $gpu] FAILED layer $layer (exit $rc), see $log_file"
            else
                echo "[gpu $gpu] Done layer $layer"
            fi
        done
    ) &
    pids+=($!)
done

# Wait for all workers
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "[launcher] All workers finished."
