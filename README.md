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
