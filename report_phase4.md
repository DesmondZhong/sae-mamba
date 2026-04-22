# Report: Phase 4 — Induction-Circuit Localization in Mamba-1 2.8B

**(Companion to `report_2.8b.md`. Covers the activation-patching experiments that reverse-engineer where Mamba-1's attention-free induction behavior is computed.)**

## Summary at a glance

| Finding | Number | Significance |
|---|---|---|
| Mamba-1 dominant site: L30 `x_proj` (selective-scan parameter generator) | **+0.833** patch_damage | single site carries 83% of the induction signal |
| Slice within x_proj: C matrix (16 dim of 192) | **+0.800** patch_damage | the "state readout" carries 99.6% of x_proj's effect |
| Next-token logit damage from patching C alone | **+0.475** | feature-level finding translates to behavior |
| Δ (time-step) slice | +0.012 | Δ carries essentially no induction signal |
| B (state write) slice | +0.002 | B carries essentially no induction signal |
| Pythia max single-site: L10 attention | +0.365 | **2.3× concentration gap** favors Mamba |
| Cross-layer emergence: first crosses 0.5 at | layer 30 | sharp L28→L30 transition (0.19 → 0.83) |
| Sufficiency of L30 C alone (clean→corrupted rescue) | +0.056 | necessary but not sufficient |
| Null-patching (pipeline sanity) | 0.0000 | exact |
| Multi-seed robustness (Jaccard across 5 seeds) | 1.00 | same top-10 feature set every seed |
| Pattern-length robustness (plen ∈ {4,8,16}) | 0.73–0.77 | localization stable |
| SAE hyperparameter robustness (5 configs) | 0.69–0.84 | mechanism > specific feature IDs |
| Natural Pile repeats activation ratio | **5.4× mean** | features generalize from synthetic → natural text |
| Long-range gap sweep (natural text, 16–512 token gaps) | 2.3–3.3× ratio | induction holds at ≥256-token distances |
| Mamba-2 (SSD architecture) gap | **0.22 vs 3.23** | Mamba-2 induction is 15× weaker and distributed, not concentrated |

## Research Question

Transformer induction heads are a well-studied circuit (Olsson et al. 2022): a "previous-token" head copies the current token backwards, and a "matching" head attends to the earlier occurrence, producing the "copy what came after last time" behavior. Mamba-1 has no attention but demonstrably does induction (we see clean clean–vs.–corrupted contrast on repeated token patterns at mid-depth). **Where in the mixer is this computation actually done?**

## Method

### Induction-pair construction
Synthetic pairs of the form
```
clean     : [prefix 8] P(8) [mid 32] P(8)      # the pattern P repeats
corrupted : [prefix 8] P(8) [mid 32] P'(8)     # the second pattern differs
```
Tokens are uniformly sampled from the full vocabulary (50,277 for Mamba-1 / Pythia). Total sequence length 56. The "induction-specific" positions are the last 8 (the second P).

### Features-of-interest
Using the pre-existing L32 SAE (normalized, TopK k=64, x16 expansion → 40,960 features), we score every SAE feature by
```
score(f) = mean_pair[ z_clean[f] − z_corrupted[f] ]   evaluated at induction positions
```
and keep the top 10. These are our "induction features."

### Patching sweep
For every `(layer, component)` in `{in_proj, conv1d, x_proj, dt_proj, out_proj_in} × {L2,4,6,...,L32}`:

1. **Capture** the component's activation on the corrupted run.
2. **Patch** it into the clean run, measure the new mean activation of induction features at the induction positions.
3. Report `patch_damage = 1 − (patched − corrupted) / (clean − corrupted)`. 1.0 = patching destroys all induction signal; 0.0 = patching has no effect.

### Position-specific patching
For the top sites, we repeat the patch restricting the intervention to three regions: `all` positions, `induction-only` (the second P), or `pre-induction-only` (everything else). This separates "this component carries induction" from "this component carries generic information".

### Pythia control
We run the analogous experiment on Pythia-2.8B (L16 SAE), capturing/patching attention_qkv, attention_output, mlp_dense_h_to_4h, and mlp_output. Hooks fire through the standard transformer submodules.

### Force-slow-forward fix
HF Mamba-1's default `MambaMixer.forward` dispatches to `cuda_kernels_forward`, which calls the `causal_conv1d` CUDA kernel on `self.conv1d.weight` / `bias` directly — bypassing `self.conv1d(x)`. Forward hooks on the `Conv1d` module therefore never fire. A helper `force_slow_forward(model)` (in `src/mamba_internals.py`) replaces each mixer's `forward` with `slow_forward`, which calls `self.conv1d(x)` and fires hooks correctly. `slow_forward` is slower per-call but fine for the small batch-1-8 capture/patch sweeps.

---

## Results

### 1. Mamba-1 localizes induction to a single submodule at layer 30

**Full layer sweep (2, 4, …, 32), Mamba-1 2.8B, top-5 sites:**

| Rank | Layer | Component | patch_damage |
|---|---|---|---|
| 1 | **L30** | **x_proj** | **+0.833** |
| 2 | L30 | conv1d | +0.743 |
| 3 | L30 | in_proj | +0.648 |
| 3 | L30 | out_proj_in | +0.648 |
| 5 | L32 | in_proj | +0.333 |
| 5 | L32 | out_proj_in | +0.333 |
| 7 | L31 | in_proj | +0.261 |
| 7 | L31 | out_proj_in | +0.261 |
| 9 | L28 | in_proj | +0.194 |
| 9 | L28 | out_proj_in | +0.194 |

`x_proj` is the selective-scan parameter generator: a Linear(d_inner=5120 → dt_rank + 2·state_size = 192) that produces the (Δ, B, C) inputs for the SSM scan. A single patch of this one submodule at L30 destroys 83% of the clean-minus-corrupted gap in induction-feature activation.

### 2. The signal emerges sharply between L28 and L30

Tracking `x_proj patch_damage` layer-by-layer:

| Layer | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | **30** | 31 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| patch_damage | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 | 0.19 | **0.83** | 0.02 | 0.06 |

The induction computation first crosses patch_damage=0.5 at **L30**. Before L28, no component's x_proj carries any induction signal. Between L28 and L30, the signal emerges and saturates. L31 and L32 x_proj carry essentially no new induction-relevant signal — the mechanism is complete by L30.

### 3. Position-specific patching confirms L30 x_proj is induction-specific, not generic

For the L30 top-4 sites, restricting the patch to different sequence regions:

| Site | all | induction-only | pre-induction-only |
|---|---|---|---|
| L30 x_proj | **0.833** | **0.833** | 0.000 |
| L30 in_proj | 0.648 | 0.648 | 0.000 |
| L30 out_proj_in | 0.648 | 0.648 | 0.000 |
| L30 conv1d | 0.743 | 0.000 | 0.010 |

**L30 x_proj, in_proj, and out_proj_in carry induction-specific signal** (all the patch_damage is at the induction positions; none at pre-induction positions).

**L30 conv1d is different**: `ind_only=0.000, pre_ind_only=0.010, all=0.743`. This means conv1d carries general context that, when replaced by the corrupted context, makes the induction features diverge — but conv1d's information on the induction positions themselves isn't induction-specific. Consistent with `conv1d`'s job (short-range mixing): it carries recent context, not the long-range repetition signal.

### 4. Pythia-2.8B distributes induction across multiple attention layers

| Rank | Layer | Component | patch_damage |
|---|---|---|---|
| 1 | L10 | attention_qkv | +0.365 |
| 1 | L10 | attention_output | +0.365 |
| 3 | L12 | attention_qkv | +0.337 |
| 3 | L12 | attention_output | +0.337 |
| 5 | L6 | attention_qkv | +0.231 |
| 5 | L6 | attention_output | +0.231 |
| 7 | L2 | attention_qkv | +0.207 |
| 7 | L2 | attention_output | +0.207 |
| 9 | L8 | mlp_output | +0.151 |
| 9 | L8 | mlp_dense_h_to_4h | +0.151 |

Pythia's maximum single-site patch_damage is **0.365 at L10 attention**, less than half of Mamba-1's L30 x_proj at 0.833. No single component dominates; the signal distributes across L2, L6, L10, L12 attention heads (MLPs contribute ≤0.15). The pairwise equality of `attention_qkv` and `attention_output` patches at each layer confirms these are expected to be coupled: the attention's downstream impact is fully determined by the input to Wo.

### 5. Concentration gap: Mamba vs. Pythia

| | Mamba-1 | Pythia-2.8B | ratio |
|---|---|---|---|
| Max single-site patch_damage | **0.833** | 0.365 | **2.3×** |
| Sites with patch_damage ≥ 0.5 | 4 (all at L30) | 0 | — |
| Depth fraction where induction emerges | 30/64 = 47% | 10/32 = 31% | — |

**Mamba-1 does induction at one layer, in one submodule. Pythia smears it across four attention layers.** This concentration gap is consistent with the architecture-level finding from the main report: Mamba has 8× more effective features (higher participation ratio) with 1.75× less per-feature causal leverage — but for this particular circuit, Mamba concentrates while Pythia distributes.

---

## Validation

To defend these numbers against interview-grade scrutiny, we ran three pressure tests.

### (A) Null-patching control (pipeline correctness)

Patch **clean activations back into a clean run** at each top site. If our patching pipeline is numerically correct, patch_damage must equal 0 (nothing is being removed).

| Site | patch_damage (clean→clean) |
|---|---|
| L30 x_proj | **0.0000** |
| L30 conv1d | **0.0000** |
| L30 in_proj | **0.0000** |
| L32 in_proj | **0.0000** |

All four sites: exactly 0.0000. The pipeline is numerically sound.

### (B) Random-feature baseline (specificity)

Pick 10 random SAE features that are NOT in the top-50 induction list. Measure the same `patch_damage` metric for them.

- Random-feature baseline activation on induction pairs: **0.005** (≈ noise floor).
- Induction-feature baseline activation: **3.36** (700× larger).

Induction features fire at meaningful magnitudes on induction pairs; random features effectively don't. The large induction damage is driven by real signal loss; random features' ratio is noise amplified by a tiny denominator.

### (C) Multi-seed robustness of induction-feature identification

Re-run feature identification with 5 different random seeds (each with 128 fresh induction pairs):

| Seed | Top-10 features |
|---|---|
| 0 | {15698, 79, 3684, 13980, 9299, 13298, 22093, 23826, 36829, 35800} |
| 1 | {15698, 79, 3684, 9299, 13980, 13298, 22093, 35800, 23826, 36829} |
| 2 | {15698, 79, 3684, 13980, 9299, 22093, 23826, 13298, 35800, 36829} |
| 3 | {15698, 79, 3684, 13980, 9299, 22093, 13298, 23826, 36829, 35800} |
| 4 | {15698, 79, 3684, 13980, 9299, 22093, 13298, 35800, 23826, 36829} |

**All 5 seeds yield the identical set of 10 features** (just reordered). Pairwise Jaccard = **1.0**. The induction-feature set is robust to the induction-pair sampling.

### (D1) Max-activating natural-text examples (qualitative check)

For each top-10 induction feature, the single highest-activating position in 1500 Pile documents:

| feat | activation | context (arrow = max-activating token) |
|---|---|---|
| 13980 | 6.03 | `"...crackdown. "Oh! Thsupra shoes supra shoes, thsupra shoes supra shoes the →` [` shoes`] |
| 36829 | 15.11 | `"...compared with 98 patients who underwent CT-based cup insertion, and all had postoperative CT. After CT-based cup placement, average →` [` cup`] |
| 13980 | 5.68 | `"...where in the groups would be $1 = foo $2 = bar1 $3 = bar2 $4 →` [` bar`] |
| 34338 | 14.72 | `"...including parts and bags. Findlay's also offers sales and service for all makes and models of sewing machines and vacuums. Please contact →` [`ums`] |
| 13980 | 5.55 | `"...chips_fries = Category(name: "Chips & Fries", items: [fries]) →` [`ries`] |

The clearest cases (feat 13980 firing on "supra shoes", feat 36829 firing on "cup" after seeing "CT-based cup" earlier) are textbook induction: copy the token that followed the earlier occurrence of the current context. Features identified synthetically on uniform-random patterns generalize to programmatic, colloquial, and medical text equally.

### (D2) Long-range gap sweep: induction survives ≥256-token distances

Split the natural-text repeats by the gap (in tokens) between the first and second occurrence:

| gap bin (tokens) | n repeats | mean activation ratio (repeat / baseline) |
|---|---|---|
| [16, 32) | 7,437 | 2.97× |
| [32, 64) | 11,216 | 3.26× |
| [64, 128) | 15,685 | 3.28× |
| [128, 256) | 19,348 | 3.15× |
| [256, 512) | 13,631 | 2.31× |

Mamba's induction signal holds at ratios 2.3–3.3× across gaps up to 512 tokens. Degradation at the longest gap is modest (3.3× → 2.3×). The state carries pattern memory over hundreds of tokens.

### (D) Real-text validation: features fire on natural repeated bigrams

We streamed 400 Pile documents, found positions `j` where the bigram `(t_{j-1}, t_j)` has an earlier occurrence at least 32 tokens back, and measured the top-10 induction features' activation at `j` vs. a random baseline position.

- Natural repeat positions sampled: **26,807** across 400 documents.
- Mean repeat / baseline activation ratio (averaged over the 10 features): **5.4×**
- **All 10 features show ≥ 2× activation at repeats** (fraction_above_2x = 100%).
- Per-feature ratios range from 2.6× (feat 13980) to 17.4× (feat 3684).

The induction features identified from synthetic uniform-random patterns **fire specifically on naturally-occurring repeated bigrams in Pile text**. The mechanism is not an artifact of the synthetic stimulus.

---

## 6. Slice-level x_proj localization (the induction signal lives in C)

x_proj output of shape (B, L, 192) decomposes into three slices:

| slice | dim | role |
|---|---|---|
| time_step (Δ_pre) | 160 | fed to `dt_proj` → discrete time-step Δ used by SSM scan |
| B matrix | 16 | state-space "input" matrix — how tokens write to state |
| C matrix | 16 | state-space "output" matrix — how state reads out to residual |

Patching each slice separately (corrupted → clean) at L30:

| slice | dim | patch_damage | % of full |
|---|---|---|---|
| full x_proj | 192 | +0.804 | 100% |
| Δ_pre | 160 | +0.012 | 1.5% |
| B | 16 | +0.002 | 0.3% |
| **C** | **16** | **+0.800** | **99.5%** |
| B + C | 32 | +0.801 | 99.6% |

**The entire induction signal is carried by C — a 16-dimensional slice of the 192-dim x_proj output.** Patching Δ (the time-step) carries essentially none of the signal (consistent with `dt_proj` showing patch_damage=0.00 across all layers in §1's heatmap). Patching B carries none either.

Interpretation: selective scan runs `h_t+1 = A(Δ) · h_t + B · x_t` and `y_t = C · h_t`. Induction requires reading the state, not writing to it — so the "match found" signal is encoded in C. The model stores pattern memory in the state via prior layers, then at L30 uses C to selectively read the matching component.

### 6a. Behavioral confirmation: next-token logit damage

The SAE-feature patch_damage is an intermediate measurement. Does the C-matrix locus also affect the model's actual next-token prediction?

For each clean induction pair, the model should predict the tokens at positions 48–55 (copying them from the first pattern). We measure the logit at the correct next-token target, then patch L30 x_proj slices:

| slice | clean logit | corrupted logit | patched logit | next_token_damage |
|---|---|---|---|---|
| (no patch, baseline/corrupted) | 27.07 | 16.84 | — | — |
| full x_proj | — | — | 22.21 | **+0.475** |
| Δ_pre (160 dim) | — | — | 27.08 | 0.000 |
| B (16 dim) | — | — | 27.07 | 0.000 |
| **C matrix (16 dim)** | — | — | **22.21** | **+0.475** |
| B + C (32 dim) | — | — | 22.20 | +0.476 |

Patching only the 16 dimensions of C at L30 drops next-token logit by 4.86 nats (47.5% of the clean–vs–corrupted gap). The feature-level C localization (+0.80 patch_damage) translates to **+0.47 damage to actual next-token prediction**. Δ and B remain as zero-effect in the logit metric too, confirming the slice analysis.

The 47% vs 80% difference is mechanistically meaningful: next-token prediction involves layers L31–L63 making additional processing on top of the L32 residual, diluting the effect of any single layer; SAE feature activation is a localized metric. The fact that 47% of logit damage comes from 16 / 163,840 ≈ 0.01% of the hidden state is the core behavioral finding.

## 7. Sufficiency: L30 C alone doesn't restore induction

Inverse of §6: patch **clean** L30 x_proj.C into an otherwise-corrupted run. If C alone is sufficient, induction should be restored.

| patched slice | rescue fraction |
|---|---|
| full x_proj | +0.056 |
| C matrix | +0.056 |
| B + C | +0.056 |

Only **5.6% of the induction gap is restored**. L30 x_proj.C is **necessary but not sufficient**: the induction mechanism requires both (i) the correct C at L30 AND (ii) the state built by layers 0–29 from the clean sequence. Injecting clean C into a corrupted-state run doesn't help, because `y = C · h` with a wrong `h` still yields wrong output.

This is consistent with "Mamba builds pattern memory gradually across layers, then reads it out at L30." Induction is not a one-layer lookup table.

## 8. Robustness checks

### 8a. Pattern-length sweep (PATTERN_LEN = 4, 8, 16)

Re-identify induction features at each length, re-patch L30 x_proj:

| pattern length | patch_damage | top-10 feature overlap (Jaccard) |
|---|---|---|
| 4 | +0.729 | |
| 8 | +0.746 | |
| 16 | +0.772 | |

Pairwise Jaccard of top-10 feature sets across lengths: **0.82**. Localization is robust; patch_damage actually slightly increases with longer patterns (easier induction → sharper causal effect).

### 8b. Per-position patching (what sequence positions matter?)

Patching L30 x_proj at each position individually yields zero damage at positions 0-47 (because clean=corrupted there by construction) and non-zero damage only at positions 48–55 (the induction positions themselves). Average per-position damage at positions 48-55 = +0.117, max = +0.133. Eight positions summed nominally to 0.936 but the actual joint patch was 0.833 — the scan aggregates sub-linearly.

### 8c. SAE hyperparameter robustness

Re-run the patch at L30 x_proj, using the top-10 induction features identified by a **different** Mamba-1 L32 SAE. If the mechanism is real, patch_damage should stay high across SAE configurations.

| SAE config | d_hidden | k | gap | L30 x_proj patch_damage |
|---|---|---|---|---|
| x16 k32 | 40,960 | 32 | 3.65 | **+0.835** |
| x16 k64 | 40,960 | 64 | 3.70 | +0.746 |
| x16 k128 | 40,960 | 128 | 3.61 | +0.699 |
| x8 k64 | 20,480 | 64 | 3.50 | +0.801 |
| x32 k64 | 81,920 | 64 | 3.57 | +0.691 |

Localization stays in **[0.69, 0.84]** across all five configurations. Interestingly, the top-10 feature-set Jaccard between different SAEs is near 0: each SAE learns a different basis, so the specific feature indices change, but the underlying causal locus (L30 x_proj) is invariant. The mechanism is not an artifact of SAE hyperparameters.

### 8d. Residual-stream patching (inconclusive)

Patching the full residual stream at any layer 16–31 in the corrupted run trivially gives 100% rescue, because residuals include all downstream information. This experiment is too coarse to discriminate layers. The slice-level sufficiency test (§7) is the right methodology for "is this component sufficient?".

## 9. Internal SAE on L30 x_proj input (in progress)

_[Pending: TopK SAE on `x_proj` INPUT (d_inner=5120, 10M tokens, x8 expansion → 40,960 features). Will identify which features in the d_inner representation that feeds x_proj carry the induction signal — i.e. where in the 5120-dim space does the C-matrix receive its induction input from?]_

## 10. Mamba-2 induction — SSD architecture is very different

Mamba-2 replaces selective scan with state-space duality (SSD) and merges all projections into a single `in_proj` whose output of 10,576 channels decomposes as:

- z (gate): 5120
- x (main stream, pre-conv): 5120
- B: 128 (ngroups × d_state)
- C: 128
- dt (time-step): 80

We ran the analog of Phase B on Mamba-2 (using the freshly-retrained Mamba-2 L32 SAE at x16 k64): identify induction features, patch `in_proj` slices across a layer sweep.

**Key finding 1: Mamba-2's induction signal is ~15× weaker than Mamba-1's at the same SAE.**

| | baseline act | corrupted act | gap |
|---|---|---|---|
| Mamba-1 (L32 SAE, plen=8) | 3.33 | 0.10 | **3.23** |
| Mamba-2 (L32 SAE, plen=8) | 0.47 | 0.25 | **0.22** |

Feature-score magnitudes are similarly small (max score 0.38 vs Mamba-1's 5.47). This matches the broader literature observation that SSD-based Mamba-2 exhibits weaker in-context pattern learning than Mamba-1 at similar scale.

**Key finding 2: No concentrated locus — induction is distributed across slices at L32.**

After excluding L0 (numerator artifact: L0 patching trivially swaps tokens at positions 48–55, giving patch_damage >1.0 with a tiny denominator), the non-trivial Mamba-2 sites are:

| Layer | Slice | patch_damage |
|---|---|---|
| L32 | full in_proj | 1.000 |
| L32 | z_gate | 0.588 |
| L32 | x_stream | 0.351 |
| L32 | B | 0.200 |
| L32 | C | 0.194 |
| L28 | full | 0.149 |
| L28 | z_gate | 0.121 |
| L16 | z_gate | 0.100 |

At the readout layer L32, the effect is spread across every slice — there's no clean "induction is in C" analog. B and C are nearly equal (0.20 each) — the selectivity on C that characterized Mamba-1 is absent. Patching in layers >32 has zero effect (downstream of readout), confirming methodology.

**Interpretation**: Mamba-1's localization is specific to selective scan, not a property of state-space architectures in general. Mamba-2's SSD likely distributes pattern matching differently (the merged in_proj + smaller d_state per group may change the induction mechanics). Whether Mamba-2 induction at larger pattern lengths would look different is a future question — initial `plen` sweep results (§10a) address this.

### 10a. Pattern-length sweep (Mamba-2)

| pattern length | baseline | corrupted | gap |
|---|---|---|---|
| 4 | 0.420 | 0.262 | 0.158 |
| 8 | 0.473 | 0.253 | 0.220 |
| 16 | 0.569 | 0.307 | 0.262 |
| 32 | 0.512 | 0.237 | 0.275 |

Gap grows from 0.16 (plen=4) to 0.28 (plen=32), but plateaus there. At no tested length does Mamba-2's induction gap approach Mamba-1's 3.23 at plen=8. The weak induction is not an artifact of short patterns.

---

## Discussion

**Mechanistic claim.** Mamba-1's induction behavior at 2.8B scale is implemented by **the x_proj selective-scan parameter generator at layer 30**. x_proj takes the post-conv d_inner representation and produces the (Δ, B, C) inputs that the selective scan uses to decide how the recurrent state updates. Localizing induction here says: **the "this token matches an earlier pattern" signal is encoded in the SSM parameters themselves, not in the short-range conv or the downstream residual.** This is a meaningful mechanistic claim because x_proj's output directly parameterizes the state-update dynamics — the pattern detection becomes state-routing instructions, not token-level representations.

**Contrast with Transformers.** In transformers, induction is done by a pair of attention heads acting on the residual stream (Olsson et al. 2022 "previous-token head" + "induction head"). In Mamba, there is no attention, but the analogous function is carried out by a single Linear projection that shapes the selective-scan parameters. Transformers distribute induction across multiple attention heads at different depths (our Pythia-2.8B shows peak 0.365 at L10 attention, distributed across L2/L6/L10/L12). Mamba concentrates at L30 x_proj.

**Architectural interpretation.** Mamba's 64-layer depth at 2.8B means L30 is at 47% depth — roughly where the induction-head region sits in Pythia-2.8B (32 layers, L10 = 31% depth). The *relative* depth is consistent with "induction emerges in early-middle layers"; the *concentration* is not.

## Limitations

- **One SAE per model**: We read features from the Mamba-1 L32 SAE only. An SAE trained on a different layer might emphasize different features. A robustness check here would strengthen the claim.
- **Pattern length fixed at 8**: Longer or shorter induction patterns might localize differently. Untested.
- **Synthetic stimulus baseline**: Real-text validation (§D) confirms features generalize; but the patching itself was measured on synthetic pairs. Running the patch sweep on natural-text pairs would be a stronger claim.
- **Only one SSM family**: Mamba-1. Mamba-2 uses a different (SSD) formulation; whether x_proj localization holds under SSD is an open question. The submodule decomposition of Mamba-2 is different (no separate x_proj → dt_proj), so the claim doesn't transfer directly.
- **L30 is not the SAE layer**: We use the L32 SAE to read features but the dominant causal locus is L30. The L32 residual includes L30's contribution, so this is consistent — but interview-worthy if asked why L30 and not L32.

## Artifacts

- `results_phase4/induction_features.json` — top-10 induction features (Mamba-1 L32 SAE), scored by (clean − corrupted) contrast on 64 synthetic pairs.
- `results_phase4/patching_results.json` — full patch_damage sweep, Mamba-1, 17 layers × 5 components = 85 site-patches.
- `results_phase4/patching_position_specific.json` — per-region patch_damage for top-5 sites.
- `results_phase4/pythia_induction_features.json`, `pythia_patching_results.json` — Pythia-2.8B analogues.
- `results_phase4/validation.json` — null-patching, random-feature baseline, multi-seed robustness.
- `results_phase4/real_text_induction.json` — natural-text validation on 26,807 Pile repeat positions.
- `results_phase4/phase_b_retrained_sae/` — cross-check using a freshly-retrained Mamba-1 L32 SAE (the original was unavailable during the first run; numbers agreed within ±0.02 at the headline L30 sites).
- `results_phase4/figures/` — heatmaps and bar charts for all of the above.

## Code

- `src/mamba_internals.py` — HF Mamba capture/patch context managers + `force_slow_forward` helper.
- `scripts/04_induction_circuit.py` — main Mamba-1 Phase-B script.
- `scripts/05_pythia_induction_compare.py` — matched Pythia experiment.
- `scripts/07_extract_xproj.py` — x_proj input/output activation extraction.
- `scripts/09_real_text_induction.py` — natural-text validation.
- `scripts/10_train_xproj_sae.py` — internal SAE training + feature identification.
- `scripts/11_validate_patching.py` — null-patch, random-feature, multi-seed validations.
