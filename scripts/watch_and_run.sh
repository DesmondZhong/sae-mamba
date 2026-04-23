#!/usr/bin/env bash
# Wait until a checkpoint exists, then run an arbitrary command.
# Usage: bash watch_and_run.sh <wait_ckpt> <log_file> -- <cmd...>
set -u
WAIT_CKPT="$1"; shift
LOG="$1"; shift
# drop optional --
if [ "$1" = "--" ]; then shift; fi

cd "$(dirname "$0")/.."
: "${SAE_MAMBA_STORAGE:=/workspace/excuse}"
export SAE_MAMBA_STORAGE
: "${HF_HOME:=/workspace/hf_cache}"
export HF_HOME

echo "[watch] waiting for $WAIT_CKPT  ($(date))" >> "$LOG"
until [ -f "$WAIT_CKPT" ]; do sleep 10; done
echo "[watch] $WAIT_CKPT appeared; short settle then run" >> "$LOG"
sleep 20  # let GPU release cleanly from upstream worker
echo "[run] $*  ($(date))" >> "$LOG"
"$@" >> "$LOG" 2>&1
echo "[done] rc=$?  $(date)" >> "$LOG"
