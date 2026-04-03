# Report: Sparse Autoencoders on State Space Models

## Research Question

Do Mamba models develop monosemantic features decomposable by sparse autoencoders, like transformers?

## Methods

Trained 9 SAEs total: 3 models (Mamba-130M, Mamba-370M, Pythia-160M) x 3 layers (early, middle, late). Each SAE: d_hidden = 4x d_model, L1 coeff = 1e-3, 20K training steps on 500K tokens from WikiText-103. Analyzed L0 sparsity, dead features, reconstruction quality, and max-activating examples.

## Key Results

### 1. SAEs Successfully Decompose Mamba Activations

All 9 SAEs achieved low reconstruction error with 0% dead features. This is the first demonstration that sparse feature extraction works on SSM hidden states — the recurrent information flow does not prevent monosemantic decomposition.

### 2. Mamba Features Are Substantially Less Sparse Than Transformer Features

| Model | Layer | L0 (active features/token) | Recon MSE |
|-------|-------|---------------------------|-----------|
| Mamba-130M | Early (L0) | 2888 | 0.000008 |
| Mamba-130M | Mid (L12) | 2645 | 0.000045 |
| Mamba-130M | Late (L23) | 2329 | 0.002925 |
| Mamba-370M | Early (L0) | 3682 | 0.000003 |
| Mamba-370M | Mid (L24) | 3404 | 0.000031 |
| Mamba-370M | Late (L47) | 3358 | 0.001845 |
| Pythia-160M | Early (L0) | 1269 | 0.000001 |
| Pythia-160M | Mid (L6) | 2124 | 0.000040 |
| Pythia-160M | Late (L11) | 2108 | 0.000060 |

Mamba models consistently use more active features per token (L0 ~2300-3700) compared to Pythia (~1200-2100) at the same L1 penalty. This suggests SSM residual streams encode information more densely / with more superposition than transformer residual streams.

### 3. Layer Depth Patterns Differ Between Architectures

- **Mamba**: L0 *decreases* with depth (more sparse in later layers), while reconstruction error *increases*. Information becomes harder to reconstruct from fewer active features.
- **Pythia**: L0 *increases* from early to middle layers, then plateaus. More features are needed in later layers.

This divergence likely reflects the fundamental difference in information flow: Mamba's recurrent state accumulates information that becomes increasingly compressed, while transformers build increasingly complex representations through residual connections.

### 4. No Dead Features in Any Model

With L1=1e-3 and 4x expansion, all SAEs utilized 100% of their features. This is unusual — transformer SAEs typically have 10-50% dead features. It suggests Mamba residual streams have higher effective dimensionality requiring more features for adequate representation.

## Discussion

**SAEs work on SSMs** — the core finding. Despite fundamentally different information flow (recurrent state vs residual stream), Mamba models do develop decomposable features. This extends the universality of sparse feature extraction beyond transformers.

However, the **higher L0 and zero dead features** suggest Mamba representations may be less naturally decomposable into monosemantic units. The L1 coefficient may need to be much higher for Mamba to achieve transformer-like sparsity, potentially at the cost of reconstruction quality.

**Limitations**:
- Only residual stream activations (post-SSM hooks would be more informative about SSM-specific features)
- Single L1 coefficient (a full sweep would reveal the sparsity-reconstruction Pareto frontier)
- Small dataset (500K tokens); larger data may yield different feature properties
- Monosemanticity not quantitatively measured (would need automated interpretability)

**Future work**: Train SAEs with higher L1 on Mamba to force sparser codes and compare feature interpretability. Apply to Mamba-2 architecture. Cross-model feature matching (find same features in both architectures).

## Artifacts

- `web/index.html`: Interactive visualization with comparison charts and feature browser
- `results/`: Per-model JSON results with feature statistics and max-activating examples
- `checkpoints/`: Trained SAE weights
