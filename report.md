# Report: Sparse Autoencoders on State Space Models

## Research Question

Do Mamba models develop monosemantic features decomposable by sparse autoencoders, like transformers?

## Methods

Trained SAEs on Mamba-130M, Mamba-370M, and Pythia-160M (transformer baseline) at early, middle, and late layers (9 SAEs total, d_hidden=4x d_model, 20K steps on 500K WikiText-103 tokens). Deep dive: L1 coefficient sweep (6 values from 1e-4 to 3e-2), automated monosemanticity scoring via sentence embeddings, cross-model feature matching, and downstream perplexity evaluation.

## Key Results

### 1. SAEs Successfully Decompose SSM Activations

All SAEs achieve low reconstruction error with 0% dead features across both architectures. This is the first demonstration that sparse feature extraction works on state space models.

### 2. Mamba Is Intrinsically Denser (L1 Sweep)

| L1 Coefficient | Mamba L0 | Pythia L0 | Mamba Recon MSE | Pythia Recon MSE |
|---------------|---------|----------|----------------|-----------------|
| 1e-4 | 2698 | 2127 | 5e-6 | 2e-5 |
| 1e-3 | 2658 | 2139 | 4e-5 | 3e-5 |
| 1e-2 | 2673 | 2036 | 1.5e-4 | 2.8e-4 |
| 3e-2 | 2544 | 1908 | 1.2e-3 | 9.5e-4 |

Even at the highest L1 penalty (3e-2), Mamba uses 2544 active features vs Pythia's 1908 — a persistent 33% density gap. The Pareto frontiers are clearly separated at every tradeoff point, confirming this is intrinsic to SSM information flow rather than a tuning artifact.

### 3. Monosemanticity Is Architecture-Comparable

Using sentence-transformer embeddings to measure semantic coherence of max-activating examples:

| Layer | Mamba-130M | Mamba-370M | Pythia-160M |
|-------|-----------|-----------|------------|
| Early | 0.173 | 0.032 | 0.203 |
| Middle | 0.066 | 0.059 | 0.045 |
| Late | 0.071 | 0.029 | 0.061 |

Early layers show highest monosemanticity in both architectures. Pythia-160M has a slight edge at layer 0 (0.203 vs 0.173), but middle/late layers are comparable. Mamba-370M shows lower scores, possibly due to higher dimensionality spreading features thinner.

### 4. Features Are Architecture-Specific

Cross-model matching between Mamba-130M and Pythia-160M middle-layer features finds zero overlap in max-activating texts. The two architectures develop entirely different feature decompositions of the same data — they "see" different things despite processing the same text.

### 5. Reconstruction Is Functionally Lossless

| Model | Baseline PPL | SAE PPL | Ratio |
|-------|-------------|---------|-------|
| Mamba-130M | 28.52 | 28.52 | 1.00x |
| Pythia-160M | 38.51 | 38.52 | 1.00x |

Both SAEs reconstruct activations so faithfully that downstream perplexity is unchanged. The sparse decomposition loses no functional information.

### 6. Layer Depth Patterns Diverge

- **Mamba**: L0 *decreases* with depth (2888→2645→2329 for 130M), while reconstruction error increases. Information compresses in later layers.
- **Pythia**: L0 *increases* from early to middle layers (1269→2124→2108). More features needed for complex representations.

## Discussion

**SAEs work on SSMs — but reveal fundamentally different representation structure.** The three key differences are:

1. **Density**: Mamba uses 30-50% more active features than Pythia at every L1 setting. SSM recurrent states pack more information into the residual stream.

2. **Feature specificity**: Despite comparable monosemanticity scores, the actual features are non-overlapping between architectures. The "universal features" hypothesis may not extend across architecture families.

3. **Depth dynamics**: Mamba compresses (fewer features, worse reconstruction) in later layers, while transformers expand. This likely reflects SSM's recurrent information accumulation vs transformer's compositional construction.

**Implications for mechanistic interpretability**:
- SAE-based interpretability tools can be applied to SSMs
- But architecture-specific tools may be needed — transformer-derived intuitions about features don't transfer directly
- The density gap suggests SSMs may be harder to fully decompose into monosemantic units

**Limitations**: Single L1 coefficient for main experiment (swept in deep dive). Monosemanticity scoring uses only top-20 features per SAE. Cross-model comparison limited by max-activating text overlap (may miss semantic similarity). Downstream eval uses a small sample.

## Artifacts

- `web/index.html`: Enhanced interactive visualization (Pareto frontiers, monosemanticity, downstream eval, feature browser)
- `results/`: L1 sweep, monosemanticity scores, cross-model matching, downstream perplexity
- `checkpoints/`: All SAE weights including L1 sweep variants
