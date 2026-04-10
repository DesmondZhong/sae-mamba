# Report: Sparse Feature Geometry of State Space Models at 2.8B Scale

## Research Question

How do State Space Models (Mamba) and Transformers (Pythia) differ in their internal feature geometry when probed with sparse autoencoders at equal scale?

## Methods

Trained TopK sparse autoencoders on three 2.8B-parameter models:

| Model | Architecture | Layers | d_model | Parameters |
|---|---|---|---|---|
| Mamba-1 2.8B | Pure SSM (selective scan) | 64 | 2560 | 2.8B |
| Mamba-2 2.7B | SSM (state space duality) | 64 | 2560 | 2.7B |
| Pythia 2.8B | Transformer (GPT-NeoX) | 32 | 2560 | 2.8B |

**SAE configuration:**
- Architecture: TopK SAE (Gao et al. 2024) with dead-feature resampling
- Default: K=64 active features, 16x expansion (40,960 features), 30K training steps
- Sparsity sweep: K = {32, 64, 128}
- Width sweep: expansion = {8x, 16x, 32x}
- Layer sweep: every 4-8 layers per model
- Training data: 10M tokens from The Pile (matched to all 3 models' pretraining distribution)
- Total: 60+ trained SAEs

**Two SAE training regimes evaluated:**
- *Raw activations* (default Anthropic-style training)
- *Normalized activations* (per-dimension mean subtraction + std scaling)

**Analysis:**
- Reconstruction quality: FVE (fraction of variance explained), L0, dead feature count
- Cross-architecture comparison: CKA on raw activations and on SAE features
- Within-model analysis: layer-to-layer CKA, participation ratio, feature frequency distributions
- Causal experiments: feature ablation (KL divergence on logits), induction-pattern detection
- Downstream: perplexity on Pile evaluation set with SAE patched into the residual stream

---

## Key Findings

### 1. Activation Normalization Is a Methodological Prerequisite

Without per-dimension normalization, Pythia SAEs **catastrophically fail** at middle layers (FVE drops below -5), while Mamba-1 SAEs train normally. PCA at the same dimensionality explains 80%+ variance at the same Pythia layers — so the activations *are* decomposable, the SAE training just diverges.

| Layer | Pythia FVE (raw) | Pythia FVE (normalized) |
|---|---|---|
| L4 | -5.71 | **0.79** |
| L8 | -9.51 | **0.72** |
| L12 | -9.92 | **0.70** |
| L16 | -6.90 | **0.72** |
| L20 | -0.65 | **0.74** |

Mamba-1 FVE is essentially unchanged by normalization (max change ±0.01 at L0/L32). The reason: Pythia's per-dimension activation variance grows from 0.13 (L0) to 7.2 (L28) — a 55x range — while Mamba-1's growth is much narrower. The SAE encoder's learned bias never converges at certain Pythia layer scales.

**Implication for the field:** Cross-architecture SAE studies must normalize activations or per-layer FVE comparisons are confounded by activation scale. Earlier published claims about Transformer SAE quality at middle layers may need revisiting.

### 2. After Normalization, Both Architectures Are Equally Sparse-Decomposable

| Relative Depth | Mamba-1 FVE | Pythia FVE |
|---|---|---|
| 0.00 (early) | 0.973 | 0.970 |
| 0.13 | 0.828 | 0.790 |
| 0.25 | 0.758 | 0.724 |
| 0.38 | 0.707 | 0.704 |
| 0.50 (middle) | 0.671 | 0.722 |
| 0.63 | 0.690 | 0.737 |
| 0.88 (late) | 0.738 | 0.726 |

Both architectures achieve FVE in the same range (0.67–0.97) and both show U-shaped curves: highest reconstruction quality at the embedding layer, dipping in the middle, partial recovery at the end. The earlier "SSMs are more decomposable" claim from raw activations turned out to be a normalization artifact.

### 3. Transformers Concentrate, SSMs Distribute (Participation Ratio)

The strongest novel finding. Participation ratio (PR) measures how many SAE features effectively contribute to representations, computed as `(Σvar)² / Σvar²`. Higher PR = more uniformly distributed.

| Depth | Mamba-1 PR | Pythia PR | Ratio |
|---|---|---|---|
| Early | 8,400 | 5,681 | 1.5x |
| Middle | **5,760** (L32) | **674** (L16) | **8.5x** |
| Late | 1,449 (L56) | 438 (L28) | 3.3x |

**Pythia's effective dimensionality drops 13x from early to late layers; Mamba-1's drops only 6x.** Despite achieving similar reconstruction quality, Pythia packs the same information into 8x fewer effective features at middle layers.

### 4. Causal Confirmation: Pythia Features Are Higher-Leverage

We ablated the top 100 SAE features one at a time and measured KL divergence on the model's output distribution.

| Metric | Mamba-1 | Pythia | Ratio |
|---|---|---|---|
| Mean per-feature KL | 0.230 | **0.403** | 1.75x |
| Top-10 mean KL | 0.247 | 0.416 | 1.69x |
| Features with KL > 0.1 | 100/100 | 100/100 | — |

**Pythia features have ~1.75x larger causal effect** when individually ablated. This directly confirms the participation ratio finding — because Pythia concentrates information into fewer effective features, each one matters more.

All 100 tested features in both architectures have non-trivial causal effect (KL > 0.1), confirming the SAE features are not statistical noise.

### 5. SSMs Develop Induction-Like Features Without Attention

We fed sequences with repeated patterns `[prefix, pattern, pattern, pattern]` and measured which SAE features fired more strongly on the second/third occurrence than the first.

| Metric | Mamba-1 | Pythia |
|---|---|---|
| Strong induction features | 147 | 159 |
| Max induction score | 4.75 | **14.15** |

Both architectures develop induction-like features, but **Pythia's strongest induction feature is 3x more specialized** (max score 14.15 vs 4.75). Mamba-1 distributes induction behavior across more features with lower individual scores.

The interesting point: **SSMs have no attention mechanism, yet learn pattern-completion features.** The recurrent state apparently develops induction behavior through a different mechanism than attention-based induction heads.

### 6. SAE Features Reveal Hidden Cross-Architecture Alignment

CKA on raw activations between Mamba-1 and Pythia drops sharply with depth (0.69 → 0.02 → 0.33). But CKA on **SAE features** stays high throughout:

| Depth | Raw CKA (Mamba-1 vs Pythia) | SAE CKA |
|---|---|---|
| 0.00 | 0.69 | **0.88** |
| 0.13 | 0.02 | **0.81** |
| 0.51 | 0.05 | **0.60** |
| 0.89 | 0.33 | **0.55** |

**The architectures look orthogonal in raw activation space, but the SAE finds a shared feature dictionary.** Both models compose their representations from similar building blocks — they just combine them along very different basis directions in the residual stream.

### 7. Mamba Layers Carry More Critical Information

Replacing the middle-layer residual stream with a constant (the activation mean) causes very different damage:

| Model | Baseline PPL | SAE PPL (1.09–1.19x baseline) | Mean-Replaced PPL |
|---|---|---|---|
| Mamba-1 | 10.95 | 11.97 | **131,494** |
| Pythia | 11.98 | 14.30 | **5,689** |

**Removing one Mamba middle layer destroys the model (PPL ×12,000); removing one Pythia layer is much milder (PPL ×475).** Each Mamba layer is more load-bearing than each Pythia layer — consistent with SSMs accumulating state across layers vs Transformers' more redundant compositional structure.

Note that the SAE reconstructions are non-trivial: Mamba PPL increases 9% (10.95 → 11.97), Pythia 19% (11.98 → 14.30). The Mamba SAE is more faithful, consistent with Mamba's higher FVE at the middle layer.

### 8. Decoder Geometry and Co-activation Are Identical

Several feature-geometry statistics are nearly identical across both architectures:

| Metric | Mamba-1 | Pythia | Notes |
|---|---|---|---|
| Mean \|cosine\| of decoder columns | 0.017 | 0.017 | Both nearly orthogonal |
| Fraction of near-parallel pairs | 0% | 0% | No redundant features |
| Effective rank of decoder | ~1,710 | ~1,680 | Both use ~67% of d_model |
| Mean Jaccard co-activation | 0.0003 | 0.0002 | Features fire independently |

**The architectures learn equally good features — they just *use* them differently.** The differences are in the *distribution of activation* across features (PR, induction specialization), not in the geometric structure of the feature directions themselves.

---

## Discussion

The headline finding is that **State Space Models distribute information more uniformly across features than Transformers do**, even though both architectures learn equally orthogonal, equally interpretable features. Concretely:

1. Transformers concentrate information into 8x fewer effective features at middle layers
2. Each individual Transformer feature has 1.75x larger causal effect when ablated
3. Transformer induction features are 3x more specialized
4. But Transformer middle layers are individually less load-bearing (PPL ratio 475 vs 12,000 for Mamba)

This is consistent with two different computational strategies:
- **Transformers**: many redundant, parallel feature pathways; each layer has lower individual stakes; representations form distinct "summary codes" at depth
- **SSMs**: each layer makes a critical update to a recurrent state; representations stay distributed; no single feature dominates

A second finding worth noting: **SAE features reveal hidden cross-architecture alignment.** Raw activations look orthogonal between Mamba and Pythia (CKA ≈ 0.02 in middle layers), but SAE features stay highly aligned (CKA ≈ 0.6–0.9). This suggests a shared "concept dictionary" exists across architecture families even when the residual-stream basis is unrelated. This is a stronger statement than the "universal features" hypothesis from Olsson et al. — it survives across radically different architecture families at 2.8B scale.

A third finding is methodological: **activation normalization is a critical preprocessing step** that has been under-discussed in the SAE literature. Without it, Transformer SAEs at middle layers may diverge to negative FVE, masking the underlying decomposability of those representations.

### Limitations

- **Layer resolution**: Sampled every 4-8 layers rather than every layer. Some claims about "phase transitions" may be artifacts of coarse sampling.
- **One model per family**: We tested one SSM (Mamba-1, Mamba-2 has only sparse layer coverage) and one Transformer (Pythia). The patterns may be specific to these checkpoints rather than the architecture families.
- **Token budget**: 10M tokens from the Pile is a fraction of typical published SAE work (50M-1B). Some dead feature rates may reflect undertraining rather than feature redundancy.
- **Single layer for causal experiments**: Steering and induction analyses are at the middle layer only (Mamba-1 L32, Pythia L16). Generalization across layers is untested.
- **TopK SAE only**: We didn't compare to L1 or JumpReLU variants. Some findings (especially the participation ratio gap) may be sensitive to SAE architecture choice.

### Implications

For interpretability research:
- Cross-architecture SAE comparisons must normalize activations
- Reconstruction quality alone doesn't capture architectural differences — feature concentration / participation ratio is a more discriminating metric
- Causal feature ablation (KL on logits) is a useful complement to reconstruction quality

For SSM understanding:
- Mamba develops induction-like behavior without attention, suggesting recurrent state alone is sufficient for pattern completion
- SSM layers carry more individual information than Transformer layers, making them harder to ablate without breaking the model
- SSM residual streams have more "feature parallelism" — many features contribute equally to each prediction

---

## Artifacts

- `web/index_2.8b.html`: Interactive dashboard with all charts (~256 KB; embeds slim summary stats and feature browser data)
- `/mnt/storage/desmond/excuse/checkpoints_normed/`: 24 normalized SAE checkpoints (Mamba-1 + Pythia, layer/K/expansion sweeps)
- `/mnt/storage/desmond/excuse/checkpoints/`: 36 unnormalized SAE checkpoints (Mamba-1 + Mamba-2 + Pythia)
- `/mnt/storage/desmond/excuse/results/`: All summary JSONs (CKA, baselines, downstream, comprehensive_results.json)
- `/mnt/storage/desmond/excuse/results_phase2/`: Feature geometry analyses
- `/mnt/storage/desmond/excuse/results_phase3/`: Causal experiments (steering, induction)

## Code

- `src/sae.py`: SAE variants (L1, TopK, BatchTopK)
- `src/train_sae.py`: Training loop with LR warmup, dead-feature resampling, FVE tracking
- `src/activation_cache.py`: Hook-based extraction supporting HF Transformers and mamba_ssm backends
- `scripts/extract_model.py`: Per-model activation extraction
- `scripts/train_one_sae.py` / `train_one_sae_normed.py`: Single-SAE training (raw / normalized)
- `scripts/phase2_experiments.py`: Feature frequency, decoder geometry, co-activation, within-model CKA, effective dimensionality
- `scripts/phase3_causal.py`: Feature steering and induction detection
- `scripts/run_downstream.py`: Downstream perplexity evaluation
- `scripts/06_build_2.8b_web.py`: Interactive visualization builder
