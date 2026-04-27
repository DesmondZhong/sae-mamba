#!/bin/bash
set -e
cd /root/sae-mamba

STORAGE="/path/to/storage"
LOGDIR="$STORAGE/logs"
mkdir -p "$LOGDIR"

echo "========================================================"
echo "PHASE 1: PARALLEL EXTRACTION (3 models × 3 GPUs)"
echo "========================================================"

# Each model extracts 1 layer at a time (~102GB RAM), 3 models = ~306GB of 755GB
CUDA_VISIBLE_DEVICES=0 python3 scripts/extract_model.py mamba1_2.8b > "$LOGDIR/extract_mamba1.log" 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=1 python3 scripts/extract_model.py mamba2_2.7b > "$LOGDIR/extract_mamba2.log" 2>&1 &
PID2=$!
CUDA_VISIBLE_DEVICES=2 python3 scripts/extract_model.py pythia_2.8b > "$LOGDIR/extract_pythia.log" 2>&1 &
PID3=$!

echo "Extraction PIDs: mamba1=$PID1, mamba2=$PID2, pythia=$PID3"
wait $PID1 && echo "[DONE] Mamba-1" || echo "[FAIL] Mamba-1"
wait $PID2 && echo "[DONE] Mamba-2" || echo "[FAIL] Mamba-2"
wait $PID3 && echo "[DONE] Pythia" || echo "[FAIL] Pythia"

echo ""
echo "========================================================"
echo "PHASE 2: PARALLEL SAE TRAINING (8 GPUs)"
echo "========================================================"

# Build job list: layer sweep + expansion sweep + K sweep
# 3 models × 8 layers = 24 layer-sweep jobs
# 3 models × 2 extra expansions = 6 expansion-sweep jobs
# 3 models × 2 extra K values = 6 K-sweep jobs
# Total: 36 jobs across 8 GPUs

JOBS=()

# Layer sweep at default expansion=16, k=64
for MODEL in mamba1_2.8b mamba2_2.7b pythia_2.8b; do
    if [ "$MODEL" = "pythia_2.8b" ]; then
        LAYERS="0 4 8 12 16 20 24 28"
    else
        LAYERS="0 8 16 24 32 40 48 56"
    fi
    for L in $LAYERS; do
        JOBS+=("$MODEL $L 16 64")
    done
done

# Expansion sweep at middle layer
for MODEL in mamba1_2.8b mamba2_2.7b pythia_2.8b; do
    if [ "$MODEL" = "pythia_2.8b" ]; then MID=16; else MID=32; fi
    for EXP in 8 32; do
        JOBS+=("$MODEL $MID $EXP 64")
    done
done

# K sweep at middle layer
for MODEL in mamba1_2.8b mamba2_2.7b pythia_2.8b; do
    if [ "$MODEL" = "pythia_2.8b" ]; then MID=16; else MID=32; fi
    for K in 32 128; do
        JOBS+=("$MODEL $MID 16 $K")
    done
done

TOTAL=${#JOBS[@]}
echo "Total SAE training jobs: $TOTAL"

# Launch 8 jobs at a time (one per GPU), wait for batch, launch next batch
IDX=0
while [ $IDX -lt $TOTAL ]; do
    PIDS=()
    BATCH_END=$(( IDX + 8 ))
    if [ $BATCH_END -gt $TOTAL ]; then BATCH_END=$TOTAL; fi

    for GPU in $(seq 0 7); do
        JOB_IDX=$(( IDX + GPU ))
        if [ $JOB_IDX -ge $TOTAL ]; then break; fi

        JOB="${JOBS[$JOB_IDX]}"
        read -r M L E K <<< "$JOB"
        RUNKEY="${M}_L${L}_x${E}_k${K}"
        echo "  [GPU $GPU] $RUNKEY"
        CUDA_VISIBLE_DEVICES=$GPU python3 scripts/train_one_sae.py $M $L $E $K \
            > "$LOGDIR/train_${RUNKEY}.log" 2>&1 &
        PIDS+=($!)
    done

    # Wait for this batch
    for P in "${PIDS[@]}"; do
        wait $P 2>/dev/null
    done

    IDX=$BATCH_END
    echo "  Batch complete ($IDX/$TOTAL)"
done

echo ""
echo "========================================================"
echo "PHASE 3: ANALYSIS (4 GPUs in parallel)"
echo "========================================================"
python3 scripts/run_analysis.py > "$LOGDIR/analysis.log" 2>&1
echo "[DONE] Analysis"

echo ""
echo "========================================================"
echo "PHASE 4: BUILD VISUALIZATION"
echo "========================================================"
python3 scripts/06_build_2.8b_web.py > "$LOGDIR/web.log" 2>&1
echo "[DONE] Visualization"

echo ""
echo "========================================================"
echo "ALL COMPLETE!"
echo "Results: $STORAGE/results/"
echo "Web: web/index_2.8b.html"
echo "========================================================"
