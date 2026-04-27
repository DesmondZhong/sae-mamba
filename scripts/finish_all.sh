#!/bin/bash
# Finish all remaining training jobs after current Pythia batch + Mamba-2 re-extraction
# Then run analysis and build viz
set -e
cd /root/sae-mamba

STORAGE="/path/to/storage"
LOGDIR="$STORAGE/logs"

echo "=== $(date) Waiting for current jobs to finish ==="

# Wait for all current training + extraction to finish
while ps aux | grep -E "train_one_sae|extract_model" | grep -v grep > /dev/null 2>&1; do
    TRAINING=$(ps aux | grep "train_one_sae" | grep -v grep | wc -l)
    EXTRACTING=$(ps aux | grep "extract_model" | grep -v grep | wc -l)
    CKPTS=$(ls $STORAGE/checkpoints/*.pt 2>/dev/null | wc -l)
    echo "  $(date +%H:%M:%S) training=$TRAINING extracting=$EXTRACTING checkpoints=$CKPTS/36"
    sleep 60
done

echo "=== $(date) Current batch done. Checkpoints: $(ls $STORAGE/checkpoints/*.pt 2>/dev/null | wc -l)/36 ==="

# Now train remaining Pythia layers (20, 24, 28) + all Mamba-2 layers + sweeps
# Collect all remaining jobs
JOBS=()

# Pythia remaining layers
for L in 20 24 28; do
    [ ! -f "$STORAGE/checkpoints/pythia_2.8b_L${L}_x16_k64.pt" ] && JOBS+=("pythia_2.8b $L 16 64")
done

# All Mamba-2 layer sweep
for L in 0 8 16 24 32 40 48 56; do
    [ ! -f "$STORAGE/checkpoints/mamba2_2.7b_L${L}_x16_k64.pt" ] && JOBS+=("mamba2_2.7b $L 16 64")
done

# Expansion sweeps (all 3 models)
for MODEL in mamba1_2.8b mamba2_2.7b pythia_2.8b; do
    if [ "$MODEL" = "pythia_2.8b" ]; then MID=16; else MID=32; fi
    for EXP in 8 32; do
        [ ! -f "$STORAGE/checkpoints/${MODEL}_L${MID}_x${EXP}_k64.pt" ] && JOBS+=("$MODEL $MID $EXP 64")
    done
done

# K sweeps (all 3 models)
for MODEL in mamba1_2.8b mamba2_2.7b pythia_2.8b; do
    if [ "$MODEL" = "pythia_2.8b" ]; then MID=16; else MID=32; fi
    for K in 32 128; do
        [ ! -f "$STORAGE/checkpoints/${MODEL}_L${MID}_x16_k${K}.pt" ] && JOBS+=("$MODEL $MID 16 $K")
    done
done

TOTAL=${#JOBS[@]}
echo "=== Remaining jobs: $TOTAL ==="
BATCH_SIZE=6

IDX=0
while [ $IDX -lt $TOTAL ]; do
    PIDS=()
    for SLOT in $(seq 0 $((BATCH_SIZE - 1))); do
        JOB_IDX=$((IDX + SLOT))
        if [ $JOB_IDX -ge $TOTAL ]; then break; fi
        JOB="${JOBS[$JOB_IDX]}"
        read -r M L E K <<< "$JOB"
        RUNKEY="${M}_L${L}_x${E}_k${K}"
        if [ -f "$STORAGE/checkpoints/${RUNKEY}.pt" ]; then
            echo "  [Skip] $RUNKEY"
            continue
        fi
        GPU=$SLOT
        echo "  [GPU $GPU] $RUNKEY"
        CUDA_VISIBLE_DEVICES=$GPU python3 scripts/train_one_sae.py $M $L $E $K \
            > "$LOGDIR/train_${RUNKEY}.log" 2>&1 &
        PIDS+=($!)
    done
    for P in "${PIDS[@]}"; do wait $P 2>/dev/null || true; done
    IDX=$((IDX + BATCH_SIZE))
    echo "  Batch done ($IDX/$TOTAL dispatched). Checkpoints: $(ls $STORAGE/checkpoints/*.pt 2>/dev/null | wc -l)"
done

echo ""
echo "=== $(date) ALL TRAINING COMPLETE ==="
echo "Checkpoints: $(ls $STORAGE/checkpoints/*.pt 2>/dev/null | wc -l)"

echo ""
echo "=== PHASE 3: ANALYSIS ==="
python3 scripts/run_analysis.py > "$LOGDIR/analysis.log" 2>&1
echo "[DONE] Analysis"

echo ""
echo "=== PHASE 4: VISUALIZATION ==="
python3 scripts/06_build_2.8b_web.py > "$LOGDIR/web.log" 2>&1
echo "[DONE] Visualization"

echo ""
echo "=== $(date) ALL COMPLETE! ==="
echo "Results: $STORAGE/results/"
echo "Web: /root/sae-mamba/web/index_2.8b.html"
