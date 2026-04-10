#!/bin/bash
set -e
cd /root/sae-mamba
STORAGE="/mnt/storage/desmond/excuse"
LOGDIR="$STORAGE/logs"

echo "$(date) auto_complete started"

# ============================================
# STEP 1: Wait for all current training to finish
# ============================================
while ps aux | grep "train_one_sae_normed" | grep -v grep > /dev/null 2>&1; do
    RUNNING=$(ps aux | grep "train_one_sae_normed" | grep -v grep | wc -l)
    CKPTS=$(ls $STORAGE/checkpoints_normed/*.pt 2>/dev/null | wc -l)
    echo "  $(date +%H:%M) waiting: $RUNNING processes, $CKPTS/24 checkpoints"
    sleep 120
done
echo "$(date) All current training done. Checkpoints: $(ls $STORAGE/checkpoints_normed/*.pt 2>/dev/null | wc -l)/24"

# ============================================
# STEP 2: Check for missing/failed jobs, rerun
# ============================================
echo ""
echo "$(date) Checking for missing checkpoints..."

MISSING_JOBS=()

# Layer sweep
for M in mamba1_2.8b pythia_2.8b; do
    if [ "$M" = "pythia_2.8b" ]; then LAYERS="0 4 8 12 16 20 24 28"; else LAYERS="0 8 16 24 32 40 48 56"; fi
    for L in $LAYERS; do
        [ ! -f "$STORAGE/checkpoints_normed/${M}_L${L}_x16_k64_normed.pt" ] && MISSING_JOBS+=("$M $L 16 64")
    done
done

# K sweep
for M_L in "mamba1_2.8b 32" "pythia_2.8b 16"; do
    read -r M L <<< "$M_L"
    for K in 32 128; do
        [ ! -f "$STORAGE/checkpoints_normed/${M}_L${L}_x16_k${K}_normed.pt" ] && MISSING_JOBS+=("$M $L 16 $K")
    done
done

# Expansion sweep
for M_L in "mamba1_2.8b 32" "pythia_2.8b 16"; do
    read -r M L <<< "$M_L"
    for E in 8 32; do
        [ ! -f "$STORAGE/checkpoints_normed/${M}_L${L}_x${E}_k64_normed.pt" ] && MISSING_JOBS+=("$M $L $E 64")
    done
done

TOTAL_MISSING=${#MISSING_JOBS[@]}
echo "Missing: $TOTAL_MISSING jobs"

if [ $TOTAL_MISSING -gt 0 ]; then
    IDX=0
    while [ $IDX -lt $TOTAL_MISSING ]; do
        PIDS=()
        for SLOT in $(seq 0 3); do
            JOB_IDX=$((IDX + SLOT))
            [ $JOB_IDX -ge $TOTAL_MISSING ] && break
            read -r M L E K <<< "${MISSING_JOBS[$JOB_IDX]}"
            RUNKEY="${M}_L${L}_x${E}_k${K}_normed"
            [ -f "$STORAGE/checkpoints_normed/${RUNKEY}.pt" ] && continue
            echo "  [GPU $SLOT] $RUNKEY"
            CUDA_VISIBLE_DEVICES=$SLOT python3 scripts/train_one_sae_normed.py $M $L $E $K \
                > "$LOGDIR/train_${RUNKEY}.log" 2>&1 &
            PIDS+=($!)
        done
        for P in "${PIDS[@]}"; do wait $P 2>/dev/null || true; done
        IDX=$((IDX + 4))
        echo "  Rerun batch done. Checkpoints: $(ls $STORAGE/checkpoints_normed/*.pt 2>/dev/null | wc -l)/24"
    done
fi

# Second pass: verify all exist
STILL_MISSING=0
for M in mamba1_2.8b pythia_2.8b; do
    if [ "$M" = "pythia_2.8b" ]; then LAYERS="0 4 8 12 16 20 24 28"; else LAYERS="0 8 16 24 32 40 48 56"; fi
    for L in $LAYERS; do
        [ ! -f "$STORAGE/checkpoints_normed/${M}_L${L}_x16_k64_normed.pt" ] && echo "  STILL MISSING: ${M}_L${L}" && STILL_MISSING=$((STILL_MISSING+1))
    done
done
echo "$(date) Verification: $STILL_MISSING still missing"

echo ""
echo "$(date) STEP 2 COMPLETE: $(ls $STORAGE/checkpoints_normed/*.pt 2>/dev/null | wc -l) normed checkpoints"

# ============================================
# STEP 3: Print final results table
# ============================================
echo ""
echo "=== FINAL NORMALIZED RESULTS ==="
for f in $STORAGE/results_normed/*_stats.json; do
    python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
    print(f\"{d['model_key']:>15} L{d['layer']:<3} x{d.get('expansion_ratio','?'):<3} K={d.get('k','?'):<4} FVE={d.get('fve',d.get('final_fve',0)):.4f}  dead={d.get('dead_features',d.get('final_dead_features','?'))}\")
" 2>/dev/null
done | sort

echo ""
echo "=== $(date) AUTO_COMPLETE FINISHED ==="
