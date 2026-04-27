# Sparse Autoencoders on State Space Models

A controlled cross-architecture study at 2.8B parameter scale, plus a mechanistic localization of the attention-free induction circuit in Mamba-1.

**🔗 Live writeup with figures: [sae-mamba.desmondzhong.com](https://sae-mamba.desmondzhong.com/)**

We trained TopK sparse autoencoders on residual-stream activations from **Mamba-1 2.8B**, **Mamba-2 2.7B**, and **Pythia-2.8B** — all pretrained on The Pile, matched on parameter count and `d_model = 2560` — and used them to (1) compare feature geometry across architectures and (2) reverse-engineer where Mamba performs induction without attention.

---

## Headline findings

1. **Pythia concentrates 8× more.** At middle relative depth, Pythia's effective feature dimensionality (participation ratio) is 8.5× lower than Mamba-1's (674 vs 5,760), and each Pythia feature has 1.75× larger single-feature KL-ablation effect. Transformers pack the same information into far fewer high-variance features.

2. **Mamba-1's induction circuit lives in 16 dimensions at L30.** Of 85 (layer × submodule) sites swept, the 16-dim **C matrix** of `x_proj` at layer 30 carries 80% of the induction-feature signal and 47.5% of next-token logit damage — 8% of the submodule output, 0.6% of one residual layer. Sharply localized; reproduces in Mamba-130M at L15.

3. **Specificity to selective scan, not SSMs in general.** Mamba-2 (state-space duality) has the same 16-dim C matrix but shows 15× weaker induction with B and C causally indistinguishable. The concentration is a property of selective scan, confirmed independently by the behavioral recall benchmarks in Arora et al. (2023).

Two further findings worth flagging:

- **Representation ≠ computation.** A linear probe on `x_proj` slices says induction is encoded in Δ (97.7% decodable) but causal patching says the mechanism is in C (57.5% decodable, 80% causal). Probes locate where information is linearly *readable*, not where it is *used* — a methodological caution for the broader interp literature.

- **Universality at the feature level.** A crosscoder over Mamba-1 L32 and Pythia L16 finds **99.6% of features are shared** between the two models (40,807 of 40,960). Architectures differ in how features are routed and read out, not in which concepts get encoded.

Full writeup with figures: **[sae-mamba.desmondzhong.com](https://sae-mamba.desmondzhong.com/)**.

---

## Repo layout

```
src/                    SAE training, activation caching, Mamba mixer-internals patcher
scripts/01_..03_*.py    Original cross-architecture sweep at 130M-370M (early-stage exploration)
scripts/04_..36_*.py    2.8B-scale experiments: induction localization, controls, validation
web/index.html          Main writeup (source for the live site)
web/index_2.8b.html     Extended dashboard for the 2.8B sweep
CLAUDE.md               Hardware budget + OOM hazards (read before launching long runs)
```

`src/mamba_internals.py` — capture / patcher context managers for Mamba-1 mixer submodules (`in_proj`, `conv1d`, `x_proj`, `dt_proj`, `out_proj` input). This is the workhorse of the induction-localization experiments.

---

## Setup

```bash
pip install torch "transformers>=4.44,<4.50" datasets accelerate safetensors
pip install einops plotly tqdm pandas numpy scipy scikit-learn
pip install mamba-ssm causal-conv1d   # for Mamba-1/Mamba-2 forward passes

export SAE_MAMBA_STORAGE=/path/to/storage   # canonical results root
```

All scripts default `SAE_MAMBA_STORAGE` to `/path/to/storage`. Override before running.

---

## Reproducing the main results

The experiments are heavy (10M Pile tokens at 2.8B; multi-hour H100 jobs). Each script writes its outputs under `$SAE_MAMBA_STORAGE/`. Recommended path:

```bash
# 1. Extract residual-stream activations and train the SAE sweep
bash scripts/run_training.sh

# 2. Localize induction in Mamba-1 (layer × submodule patching)
python scripts/04_induction_circuit.py --layers 4 8 12 16 20 24 28 30 31 32

# 3. Slice-level patching inside x_proj (Δ / B / C)
python scripts/12_slice_patching.py

# 4. Pythia control
python scripts/05_pythia_induction_compare.py --layers 2 4 6 8 10 12 14 15 16

# 5. Mamba-2 control
python scripts/17_mamba2_induction.py

# 6. Linear-probe representation vs computation
python scripts/24_linear_probe_C.py

# 7. Crosscoder (Mamba-1 ↔ Pythia shared dictionary)
python scripts/33_crosscoder.py

# 8. Build the dashboard
python scripts/06_build_2.8b_web.py
```

For validation and robustness checks (multi-seed feature stability, SAE-hyperparameter sweep, natural-text generalization, position specificity, scaling to 130M/370M), see `scripts/11_*` through `scripts/36_*` — file names are descriptive.

---

## Hardware notes

The full 2.8B pipeline was run on a 4× H100 80 GB / 2 TiB CPU RAM box. Activation extraction is the memory-critical step; see `CLAUDE.md` for the OOM hazards and the streaming/memmap rules that keep activation dumps from blowing up CPU RAM.
