#!/bin/bash
# Re-extract Mamba-2 one layer at a time (alone, no other jobs competing for RAM)
# Then train all Mamba-2 SAEs
set -e
cd /root/sae-mamba

STORAGE="/mnt/storage/desmond/excuse"
LOGDIR="$STORAGE/logs"

echo "=== $(date) Re-extracting Mamba-2 with fixed hooks ==="

# Wait until no other training/extraction jobs are running
while ps aux | grep -E "train_one_sae|extract_model" | grep -v grep > /dev/null 2>&1; do
    echo "  Waiting for other jobs to finish..."
    sleep 60
done

echo "=== $(date) Starting Mamba-2 extraction (1 layer at a time) ==="
CUDA_VISIBLE_DEVICES=0 python3 scripts/extract_model.py mamba2_2.7b \
    > "$LOGDIR/extract_mamba2_final.log" 2>&1

echo "=== $(date) Mamba-2 extraction done ==="
echo "Layers: $(ls $STORAGE/activations/mamba2_2.7b/layer_*.pt 2>/dev/null | wc -l)"

# Verify activation norms look reasonable
python3 -c "
import torch
for layer in [0, 32, 56]:
    t = torch.load('$STORAGE/activations/mamba2_2.7b/layer_' + str(layer) + '.pt',
                   map_location='cpu', weights_only=True, mmap=True)
    sample = t[:1000].clone()
    print(f'L{layer}: norm={sample.norm(dim=-1).mean():.2f}, std={sample.std():.4f}')
    del t, sample
"

# Train all Mamba-2 SAEs (6 at a time)
echo ""
echo "=== $(date) Training Mamba-2 SAEs ==="

JOBS=()
for L in 0 8 16 24 32 40 48 56; do
    [ ! -f "$STORAGE/checkpoints/mamba2_2.7b_L${L}_x16_k64.pt" ] && JOBS+=("mamba2_2.7b $L 16 64")
done
# Expansion sweep
for EXP in 8 32; do
    [ ! -f "$STORAGE/checkpoints/mamba2_2.7b_L32_x${EXP}_k64.pt" ] && JOBS+=("mamba2_2.7b 32 $EXP 64")
done
# K sweep
for K in 32 128; do
    [ ! -f "$STORAGE/checkpoints/mamba2_2.7b_L32_x16_k${K}.pt" ] && JOBS+=("mamba2_2.7b 32 16 $K")
done

TOTAL=${#JOBS[@]}
echo "Mamba-2 jobs: $TOTAL"

IDX=0
while [ $IDX -lt $TOTAL ]; do
    PIDS=()
    for SLOT in $(seq 0 5); do
        JOB_IDX=$((IDX + SLOT))
        [ $JOB_IDX -ge $TOTAL ] && break
        JOB="${JOBS[$JOB_IDX]}"
        read -r M L E K <<< "$JOB"
        RUNKEY="${M}_L${L}_x${E}_k${K}"
        [ -f "$STORAGE/checkpoints/${RUNKEY}.pt" ] && continue
        echo "  [GPU $SLOT] $RUNKEY"
        CUDA_VISIBLE_DEVICES=$SLOT python3 scripts/train_one_sae.py $M $L $E $K \
            > "$LOGDIR/train_${RUNKEY}.log" 2>&1 &
        PIDS+=($!)
    done
    for P in "${PIDS[@]}"; do wait $P 2>/dev/null || true; done
    IDX=$((IDX + 6))
    echo "  Batch done. Checkpoints: $(ls $STORAGE/checkpoints/mamba2*.pt 2>/dev/null | wc -l)"
done

echo "=== $(date) Mamba-2 COMPLETE ==="
