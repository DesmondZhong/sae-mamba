# CLAUDE.md — sae-mamba

## Hardware budget
- **CPU RAM: 2.0 TiB, no swap.** Generous but finite — a single badly-written run *can* still OOM, and there is no swap to soak a spike.
- **GPUs: 4× H100 80 GB HBM3.**
- **Disk: 1.5 TB on `/workspace`.** Activation dumps go here.
- `$SAE_MAMBA_STORAGE` (default `/mnt/storage/desmond/excuse/`) is the canonical results root; keep large intermediate files under it or under `/workspace`.

> **User has explicitly said they cannot afford CPU OOM.** Treat CPU RAM as a hard constraint, not a soft one. Budget and measure *before* launching any long run — a crash that loses 2 hours of H100 time is worse than 15 minutes spent sizing the run.

## Known OOM hazards in this repo

1. **`src/activation_cache.py::_extract_with_hooks`** appends every batch's activations to a Python list in **fp32 CPU**, then `torch.cat`s at the end.
   - Per-layer cost ≈ `n_tokens × d_model × 4 B`.
   - Mamba-370M (48 layers, d=1024) at 500K tokens × all 48 layers × fp32 ≈ **94 GB**. Fine here, but 10× any axis starts hurting.
   - The final `torch.cat` **doubles peak memory briefly** (old list + new contiguous tensor both alive).
   - Extracting many layers of the **2.8B model** (d=2560, 64 layers) at 10M tokens × fp32 = **~6.1 TB** — way over. Must stream and/or fp16.

2. **`get_text_data`** builds a Python list of per-example token tensors then `torch.cat`. Fine at 500K tokens, scales linearly.

3. **HF `datasets` non-streaming mode.** The WikiText path calls `load_dataset(...)` without `streaming=True`. OK for WikiText-103 (~500 MB). Never do this for Pile — always `streaming=True`.

4. **`torch.load(path)` without `mmap=True`** materializes the entire tensor in RAM. For big activation shards, always `torch.load(path, mmap=True, weights_only=True)` or use `np.memmap`.

5. **DataLoader `num_workers > 0`** forks workers that get copy-on-write of the parent's tokenized tensors. With a 50 GB tokenized corpus and 8 workers, RAM can balloon when any worker mutates. Keep `num_workers=0` for anything beyond toy sizes; use `num_workers=2` only when the parent footprint is small.

6. **Looping over multiple models in one process** (`scripts/01_run_experiment.py` loads Mamba-130M, Mamba-370M, Pythia-160M sequentially). Without `del model; gc.collect(); torch.cuda.empty_cache()` between models, weights *and* their activation caches stack.

7. **`pip install` resolver** can spike ~1 GB. Irrelevant on this box, but note if ever porting to a tight-RAM container.

## Design rules for new experiments

### Activations (highest-risk path)
- **Stream to disk, don't accumulate.** In the hook, write each batch into a pre-allocated `np.memmap` (fp16) or an append-only shard file. Don't `.append(...)` to a Python list.
  ```python
  path = out_dir / f"layer{li}.fp16.npy"
  mmap = np.memmap(path, dtype='float16', mode='w+', shape=(N_tok_total, d_model))
  cursor = 0
  def hook(module, input, output):
      nonlocal cursor
      h = output[0] + output[1] if backend == "mamba_ssm" and isinstance(output, tuple) else (output[0] if isinstance(output, tuple) else output)
      flat = h.reshape(-1, h.shape[-1]).to(torch.float16).cpu().numpy()
      mmap[cursor:cursor + flat.shape[0]] = flat
      cursor += flat.shape[0]
  mmap.flush()
  ```
- **Store activations in fp16** (or bf16 via `torch.save` of a bf16 tensor). Halves RAM *and* disk. SAE training is fine in fp16 with a standard Adam/float32 optimizer state.
- **Never `torch.cat` a list of activation tensors** for anything > a few GB. Pre-allocate the destination (`np.memmap` on disk, or a GPU tensor) and write slices.
- **Extract layers in separate passes** when the combined budget is tight. One extra forward pass is cheaper than a CPU OOM.
- **Plan the budget up front.** Write the formula in the script's docstring: `peak_cpu_ram ≈ n_tokens × d_model × bytes × n_layers_live × 2 (for cat)`. If that exceeds 200 GB, refactor to streaming/memmap.

### Data loading
- **Big corpora: `streaming=True`** on HF datasets, always.
- **`num_workers=0`** unless you've measured and are confident.
- **Token tensors: int32** is plenty (<2 B distinct token ids on all models here). Halves vs. int64.
- **Shard and tokenize once, then reuse.** Write token shards to disk; subsequent runs mmap them instead of re-tokenizing.

### Model loading
- `torch_dtype=torch.float16` (already done in `get_model_and_tokenizer`). Don't change.
- Between models in a loop:
  ```python
  del model
  del tokenizer
  import gc; gc.collect()
  torch.cuda.empty_cache()
  ```
- **For the 2.8B+ case**, consider running each model in a **subprocess** (`subprocess.run(["python", "run_one_model.py", ...])`). OS reclamation on process exit is the only *guaranteed* RAM release; `del` + `gc.collect` often leaves fragmentation.

### Saving / loading intermediates
- Write with `torch.save(x.to(torch.float16), path)` or `np.memmap` for raw dumps.
- Read with `torch.load(path, mmap=True, weights_only=True)` or re-open the `np.memmap` in `mode='r'`.
- SAE weights themselves are tiny (<500 MB even for 2.8B); regular save/load is fine.

### Before pressing go
- **Always dry-run first** with `n_tokens=10_000`, one layer, `batch_size=2` to shake out hook / shape bugs without cost.
- **Print `free -g` and `nvidia-smi` at script start.** Log them — makes postmortems possible.
- **Estimate peak CPU RAM in the script** and `assert` it's < 60% of `MemAvailable`. Fail fast with a clear error rather than OOM-killer fast.
- **Watch for silent death.** Linux OOM killer sends `SIGKILL`; no Python traceback appears. If a run vanishes with no exception, check `dmesg | tail -50` for `Out of memory: Killed process ...`.

### What this rules out
- No "load all activations of all layers of all models into one dict in one process" (naïve extension of `01_run_experiment.py` to the 2.8B model will OOM).
- No in-memory Pandas DataFrames over full activation tensors.
- No `torch.cat` on >100 GB of per-batch shards — concatenate on disk instead.
