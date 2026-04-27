#!/bin/bash
# Master automation: wait for training → validate → phase 2 analysis → rebuild viz
set -e
cd /root/sae-mamba
STORAGE="/path/to/storage"
LOGDIR="$STORAGE/logs"

echo "$(date) === MASTER AUTO START ==="

# ============================================
# Wait for auto_complete.sh to finish
# ============================================
echo "$(date) Waiting for auto_complete to finish..."
while ps aux | grep "auto_complete.sh" | grep -v grep > /dev/null 2>&1; do
    sleep 120
done
echo "$(date) auto_complete done"

# ============================================
# Wait for any remaining manual training jobs
# ============================================
while ps aux | grep "train_one_sae_normed" | grep -v grep > /dev/null 2>&1; do
    RUNNING=$(ps aux | grep "train_one_sae_normed" | grep -v grep | wc -l)
    echo "  $(date +%H:%M) $RUNNING training processes still running"
    sleep 120
done
echo "$(date) All training done"

# ============================================
# Check and rerun any missing normalized SAEs
# ============================================
echo ""
echo "$(date) === VALIDATION ==="
MISSING=0
for M in mamba1_2.8b pythia_2.8b; do
    if [ "$M" = "pythia_2.8b" ]; then LAYERS="0 4 8 12 16 20 24 28"; else LAYERS="0 8 16 24 32 40 48 56"; fi
    for L in $LAYERS; do
        if [ ! -f "$STORAGE/checkpoints_normed/${M}_L${L}_x16_k64_normed.pt" ]; then
            echo "  MISSING: ${M}_L${L}_x16_k64 — retraining..."
            CUDA_VISIBLE_DEVICES=0 python3 scripts/train_one_sae_normed.py $M $L 16 64 \
                > "$LOGDIR/train_${M}_L${L}_x16_k64_normed_retry.log" 2>&1
            MISSING=$((MISSING+1))
        fi
    done
done

# Sweep jobs
for M_L in "mamba1_2.8b 32" "pythia_2.8b 16"; do
    read -r M L <<< "$M_L"
    for K in 32 128; do
        if [ ! -f "$STORAGE/checkpoints_normed/${M}_L${L}_x16_k${K}_normed.pt" ]; then
            echo "  MISSING: ${M}_L${L}_x16_k${K} — retraining..."
            CUDA_VISIBLE_DEVICES=0 python3 scripts/train_one_sae_normed.py $M $L 16 $K \
                > "$LOGDIR/train_${M}_L${L}_x16_k${K}_normed_retry.log" 2>&1
            MISSING=$((MISSING+1))
        fi
    done
    for E in 8 32; do
        if [ ! -f "$STORAGE/checkpoints_normed/${M}_L${L}_x${E}_k64_normed.pt" ]; then
            echo "  MISSING: ${M}_L${L}_x${E}_k64 — retraining..."
            CUDA_VISIBLE_DEVICES=0 python3 scripts/train_one_sae_normed.py $M $L $E 64 \
                > "$LOGDIR/train_${M}_L${L}_x${E}_k64_normed_retry.log" 2>&1
            MISSING=$((MISSING+1))
        fi
    done
done

echo "$(date) Validation complete. Retrained $MISSING missing SAEs."
echo "Total normed checkpoints: $(ls $STORAGE/checkpoints_normed/*.pt 2>/dev/null | wc -l)"

# ============================================
# Validate results: check FVE is reasonable
# ============================================
echo ""
echo "$(date) === FVE VALIDATION ==="
python3 -c "
import json, sys
from pathlib import Path
RESULTS = Path('$STORAGE/results_normed')
bad = 0
for f in sorted(RESULTS.glob('*_stats.json')):
    with open(f) as fh:
        d = json.load(fh)
    fve = d.get('fve', d.get('final_fve', -999))
    if fve < 0:
        print(f'  BAD FVE: {d[\"run_key\"]} FVE={fve:.4f}')
        bad += 1
print(f'Total bad FVE: {bad}')
if bad > 0:
    print('WARNING: Some normalized SAEs still have negative FVE!')
"

# ============================================
# Run Phase 2 experiments
# ============================================
echo ""
echo "$(date) === PHASE 2: DEEP ANALYSIS ==="
python3 scripts/phase2_experiments.py > "$LOGDIR/phase2.log" 2>&1
echo "$(date) Phase 2 complete"

# ============================================
# Update comprehensive results and rebuild viz
# ============================================
echo ""
echo "$(date) === UPDATING RESULTS & VIZ ==="
python3 -c "
import json
from pathlib import Path

STORAGE = Path('$STORAGE')
RESULTS = STORAGE / 'results'
NORMED = STORAGE / 'results_normed'
PHASE2 = STORAGE / 'results_phase2'

# Load existing
with open(RESULTS / 'comprehensive_results.json') as f:
    comp = json.load(f)

# Update normed stats
normed = {}
for f in sorted(NORMED.glob('*_stats.json')):
    with open(f) as fh:
        d = json.load(fh)
        normed[d.get('run_key', f.stem)] = d
comp['normed_stats'] = normed

# Add phase 2 results
for name in ['feature_frequency', 'decoder_geometry', 'coactivation', 'within_model_cka', 'effective_dim']:
    p = PHASE2 / f'{name}.json'
    if p.exists():
        with open(p) as f:
            comp[name] = json.load(f)

# Update baselines and CKA
for name in ['baselines', 'cka_results']:
    p = RESULTS / f'{name}.json'
    if p.exists():
        with open(p) as f:
            comp[name.replace('_results', '')] = json.load(f)

with open(RESULTS / 'comprehensive_results.json', 'w') as f:
    json.dump(comp, f, indent=2, default=str)

print(f'Updated: {len(normed)} normed stats + phase2 analyses')
"

# Rebuild web visualization
python3 scripts/06_build_2.8b_web.py > "$LOGDIR/web_rebuild.log" 2>&1
echo "$(date) Web visualization rebuilt"

echo ""
echo "=== $(date) MASTER AUTO COMPLETE ==="
echo "Results: $STORAGE/results/"
echo "Phase 2: $STORAGE/results_phase2/"
echo "Web: /root/sae-mamba/web/index_2.8b.html"
