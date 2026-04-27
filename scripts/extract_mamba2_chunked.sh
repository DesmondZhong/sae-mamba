#!/usr/bin/env bash
# Chunked Mamba-2 extraction — 3 passes of 3 layers each to stay under the
# container's memory limit (full 9-layer extraction OOM'd at the end-of-loop
# torch.cat). Each chunk peaks at ~410 GB CPU RAM (3× 102 GB list + 102 GB cat).
# Usage: bash scripts/extract_mamba2_chunked.sh
set -u
cd "$(dirname "$0")/.."
: "${SAE_MAMBA_STORAGE:=/path/to/storage}"
export SAE_MAMBA_STORAGE
: "${HF_HOME:=/workspace/hf_cache}"
export HF_HOME
: "${HF_HUB_CACHE:=/workspace/hf_cache}"
export HF_HUB_CACHE
: "${TRANSFORMERS_CACHE:=/workspace/hf_cache}"
export TRANSFORMERS_CACHE

LOG="/workspace/logs/extract_mamba2_2.7b.log"
DONE_FILE="$SAE_MAMBA_STORAGE/activations/mamba2_2.7b/DONE"

CHUNKS=(
  "0,8,16"
  "24,32,40"
  "48,56,63"
)

echo "=== chunked mamba2 extraction start: $(date) ===" >> "$LOG"
for CHUNK in "${CHUNKS[@]}"; do
  echo "---- chunk: $CHUNK  $(date) ----" >> "$LOG"
  # Clear DONE so extract_model.py proceeds with this chunk
  rm -f "$DONE_FILE"
  CUDA_VISIBLE_DEVICES=0 LAYERS="$CHUNK" \
      python scripts/extract_model.py mamba2_2.7b >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAILED chunk $CHUNK (exit $rc)" >> "$LOG"
    exit $rc
  fi
done
# Leave a DONE so the pipeline skips re-extract on any re-run
touch "$DONE_FILE"
echo "=== chunked mamba2 extraction done: $(date) ===" >> "$LOG"
