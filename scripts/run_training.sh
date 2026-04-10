#!/bin/bash
# Train all SAEs: 4 jobs at a time to fit in RAM (4 × 96GB = 384GB of 755GB)
set -e
cd /root/sae-mamba

LOGDIR="/mnt/storage/desmond/excuse/logs"
mkdir -p "$LOGDIR"

# Build job list
JOBS=()

# Layer sweep: 3 models × 8 layers = 24 jobs
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

# Expansion sweep at middle layer: 3 models × 2 = 6 jobs
for MODEL in mamba1_2.8b mamba2_2.7b pythia_2.8b; do
    if [ "$MODEL" = "pythia_2.8b" ]; then MID=16; else MID=32; fi
    for EXP in 8 32; do
        JOBS+=("$MODEL $MID $EXP 64")
    done
done

# K sweep at middle layer: 3 models × 2 = 6 jobs
for MODEL in mamba1_2.8b mamba2_2.7b pythia_2.8b; do
    if [ "$MODEL" = "pythia_2.8b" ]; then MID=16; else MID=32; fi
    for K in 32 128; do
        JOBS+=("$MODEL $MID 16 $K")
    done
done

TOTAL=${#JOBS[@]}
echo "Total training jobs: $TOTAL"
echo "Running 6 at a time (6 × 100GB ≈ 600GB of 755GB RAM)"

IDX=0
BATCH_SIZE=6

while [ $IDX -lt $TOTAL ]; do
    PIDS=()
    for SLOT in $(seq 0 $((BATCH_SIZE - 1))); do
        JOB_IDX=$((IDX + SLOT))
        if [ $JOB_IDX -ge $TOTAL ]; then break; fi

        JOB="${JOBS[$JOB_IDX]}"
        read -r M L E K <<< "$JOB"
        RUNKEY="${M}_L${L}_x${E}_k${K}"

        # Skip if checkpoint already exists
        if [ -f "/mnt/storage/desmond/excuse/checkpoints/${RUNKEY}.pt" ]; then
            echo "  [Skip] $RUNKEY"
            continue
        fi

        GPU=$SLOT
        echo "  [GPU $GPU] $RUNKEY"
        CUDA_VISIBLE_DEVICES=$GPU python3 scripts/train_one_sae.py $M $L $E $K \
            > "$LOGDIR/train_${RUNKEY}.log" 2>&1 &
        PIDS+=($!)
    done

    # Wait for this batch
    for P in "${PIDS[@]}"; do
        wait $P 2>/dev/null || true
    done

    IDX=$((IDX + BATCH_SIZE))
    echo "Batch complete ($IDX/$TOTAL jobs dispatched)"
done

echo ""
echo "ALL SAE TRAINING COMPLETE!"
echo "Checkpoints: $(ls /mnt/storage/desmond/excuse/checkpoints/*.pt 2>/dev/null | wc -l)"
