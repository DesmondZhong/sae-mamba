# Sparse Autoencoders on State Space Models (Mamba)

**Research question**: Do SSMs develop monosemantic features like transformers do?

SAEs have been applied to transformers, vision transformers, VLMs, and protein models — but never to state space models. We train SAEs on Mamba-130M, Mamba-370M, and Pythia-160M (transformer baseline) residual stream activations to compare feature properties across architectures.

## Design Choices

- **Models**: Mamba-130M (24 layers, d=768), Mamba-370M (48 layers, d=1024), Pythia-160M (12 layers, d=768). Chosen for comparable scale and shared training data (The Pile).
- **Extraction point**: Residual stream between blocks (analogous across architectures). Post-SSM extraction was explored but skipped due to memory constraints.
- **SAE architecture**: Encoder-decoder with ReLU + L1 sparsity (Anthropic's approach). d_hidden = 4x d_model, L1 coefficient = 1e-3.
- **Data**: 500K tokens from WikiText-103, 3 layers per model (early/middle/late).

## Setup and Run

```bash
pip install torch "transformers>=4.44,<4.50" datasets accelerate safetensors
pip install einops plotly tqdm pandas numpy scipy scikit-learn

# Run full experiment (~100 min on A40)
python scripts/01_run_experiment.py

# Build web demo
python scripts/02_build_webpage.py
```

## View Results

Open `web/index.html` in a browser. See [report.md](report.md) for analysis.

## Phase 4: Induction-Circuit Localization (Mamba-1 2.8B)

Goal: reverse-engineer *where* in the Mamba-1 block the attention-free
induction behavior is computed. We reuse the existing L32 SAE to identify
induction features, then activation-patch each internal mixer submodule
(`in_proj`, `conv1d`, `x_proj`, `dt_proj`, and `out_proj` input) across a
layer sweep. A large `patch_damage` means that submodule carries the
induction signal.

Core code:
- `src/mamba_internals.py`: capture / patcher context managers for Mamba-1 mixer internals.
- `scripts/04_induction_circuit.py`: main experiment (Mamba-1).
- `scripts/05_pythia_induction_compare.py`: matched experiment on Pythia-2.8B for comparison.

Run:
```bash
# Main Mamba-1 localization (~10 min on 1x H100)
python scripts/04_induction_circuit.py --layers 4 8 12 16 20 24 28 30 31 32

# Pythia comparison (~10 min on 1x H100)
python scripts/05_pythia_induction_compare.py --layers 2 4 6 8 10 12 14 15 16
```

Results land in `$SAE_MAMBA_STORAGE/results_phase4/`
(default: `/path/to/storage/results_phase4/`).

### Headline Phase-4 finding

**Mamba-1's induction circuit localizes to the 16-dimensional `C` matrix of `x_proj` at layer 30**. Patching this single 16-dim slice (out of 40,960 SAE features at L32, or 163,840 hidden dims at depth 30) destroys 80% of the induction-feature signal and drops next-token logit by 47.5%. See [`report_phase4.md`](report_phase4.md) for the full story — slice-level necessity (`Δ=+0.01, B=+0.00, C=+0.80`), sufficiency asymmetry (necessary but not sufficient), linear-probe representation-vs-computation distinction, cross-layer emergence (L28→L30), pattern-length and SAE-hyperparameter robustness, natural-text generalization (5.4× activation ratio, induction surviving 256+ token gaps), and a Mamba-2 comparison showing SSD has ~15× weaker induction.

Interactive dashboard: [`web/index_phase4.html`](web/index_phase4.html).

Extended Phase-4 scripts (`scripts/07_...` through `scripts/24_...`): slice patching, sufficiency test, per-position patching, pattern-length sweep, SAE hparam sweep, next-token damage, real-text validation, natural-text gap sweep, internal SAE training + feature-example dump at L28 and L30, Mamba-2 adapter, linear probe.
