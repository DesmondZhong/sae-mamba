"""Train SAE with warmup, dead feature resampling, and FVE tracking."""

import torch
import json
import time
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from src.sae import create_sae


def train_sae(activations: torch.Tensor, d_hidden: int, sae_type: str = "topk",
              k: int = 64, l1_coeff: float = 1e-3, n_steps: int = 50000,
              batch_size: int = 4096, lr: float = 3e-4, device: str = "cuda",
              save_path: str = None, log_interval: int = 2000,
              warmup_steps: int = 1000, resample_interval: int = 25000,
              resample_dead_threshold: int = 10000,
              aux_dead_threshold: int = 1000,
              k_aux: int = 512, aux_coeff: float = 1.0 / 32.0):
    """Train an SAE with LR warmup, dead feature resampling, and FVE tracking."""
    d_input = activations.shape[1]
    sae = create_sae(d_input, d_hidden, sae_type=sae_type, k=k, l1_coeff=l1_coeff,
                     k_aux=k_aux, aux_coeff=aux_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))

    # LR schedule: linear warmup then cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=4, pin_memory=True)
    loader_iter = iter(loader)

    # Compute activation variance for FVE
    with torch.no_grad():
        sample = activations[:min(50000, len(activations))].to(device)
        act_var = sample.var(dim=0).mean().item()
        act_mean_norm = sample.norm(dim=-1).mean().item()
        del sample

    history = {"recon_loss": [], "l0": [], "total_loss": [], "fve": [], "lr": []}
    if sae_type == "l1":
        history["l1_loss"] = []

    # Track feature firing for dead feature resampling
    feature_last_fired = torch.zeros(d_hidden, dtype=torch.long, device=device)
    step_counter = torch.tensor(0, dtype=torch.long, device=device)

    start_time = time.time()

    for step in range(n_steps):
        try:
            (batch,) = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            (batch,) = next(loader_iter)

        batch = batch.to(device)

        # Compute dead mask for AuxK (features that haven't fired in aux_dead_threshold steps).
        # Only relevant after a warmup period, otherwise most features are "dead".
        if sae_type == "topk" and step > aux_dead_threshold:
            dead_mask = (step_counter - feature_last_fired) > aux_dead_threshold
        else:
            dead_mask = None

        x_hat, z, loss, metrics = sae(batch, dead_mask=dead_mask) if sae_type == "topk" else sae(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Normalize decoder after each step
        sae.normalize_decoder()

        # Track dead features — keep step_counter on-device to avoid CPU sync every step.
        step_counter += 1
        active_mask = (z > 0).any(dim=0)  # (d_hidden,)
        feature_last_fired = torch.where(active_mask, step_counter, feature_last_fired)

        # Dead feature resampling
        if resample_interval > 0 and (step + 1) % resample_interval == 0:
            dead_mask = (step_counter.item() - feature_last_fired) > resample_dead_threshold
            n_dead = dead_mask.sum().item()
            if n_dead > 0:
                _resample_dead_features(sae, activations, dead_mask, optimizer, device, batch_size)
                feature_last_fired[dead_mask] = step_counter.item()
                print(f"  [Step {step+1}] Resampled {n_dead} dead features "
                      f"({n_dead/d_hidden:.1%})", flush=True)

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (step + 1)) * (n_steps - step - 1)
            fve = 1.0 - metrics["recon_loss"] / max(act_var, 1e-10)
            current_lr = scheduler.get_last_lr()[0]

            n_dead = ((step_counter.item() - feature_last_fired) > resample_dead_threshold).sum().item()

            print(f"  Step {step+1}/{n_steps} | recon={metrics['recon_loss']:.6f} | "
                  f"FVE={fve:.4f} | L0={metrics['l0']:.1f} | dead={n_dead} | "
                  f"lr={current_lr:.2e} | ETA={eta/60:.1f}min", flush=True)

            record = {"step": step + 1}
            for k_name, v in metrics.items():
                history.setdefault(k_name, [])
                history[k_name].append({"step": step + 1, "value": v})
            history["fve"].append({"step": step + 1, "value": fve})
            history["lr"].append({"step": step + 1, "value": current_lr})

    # Final stats
    final_fve = 1.0 - metrics["recon_loss"] / max(act_var, 1e-10)
    n_dead_final = ((step_counter.item() - feature_last_fired) > resample_dead_threshold).sum().item()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(sae.state_dict(), save_path)
        hist_path = save_path.replace(".pt", "_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f)

    summary = {
        "final_recon_loss": metrics["recon_loss"],
        "final_fve": final_fve,
        "final_l0": metrics["l0"],
        "final_dead_features": n_dead_final,
        "act_var": act_var,
        "act_mean_norm": act_mean_norm,
    }

    return sae, history, summary


def _resample_dead_features(sae, activations, dead_mask, optimizer, device, batch_size=4096):
    """Resample dead features from high-loss examples (Anthropic's approach)."""
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return

    # Find high-loss examples
    indices = torch.randperm(len(activations))[:min(batch_size * 4, len(activations))]
    sample = activations[indices].to(device)

    with torch.no_grad():
        x_hat, z, loss_tensor, _ = sae(sample)
        per_example_loss = (sample - x_hat).pow(2).sum(dim=-1)  # (batch,)

    # Sample from high-loss examples proportional to loss
    probs = per_example_loss / per_example_loss.sum()
    resample_idx = torch.multinomial(probs, n_dead, replacement=True)
    resampled_inputs = sample[resample_idx]  # (n_dead, d_input)

    # Reinitialize dead encoder weights from high-loss inputs
    dead_indices = dead_mask.nonzero(as_tuple=True)[0]
    alive_mask = ~dead_mask
    with torch.no_grad():
        # Scale resampled encoder rows to 0.2 * mean ||alive encoder rows||.
        if alive_mask.any():
            alive_norm_mean = sae.encoder.weight.data[alive_mask].norm(dim=-1).mean()
        else:
            alive_norm_mean = torch.tensor(1.0, device=sae.encoder.weight.device)
        target_enc_norm = 0.2 * alive_norm_mean

        normed = resampled_inputs / (resampled_inputs.norm(dim=-1, keepdim=True) + 1e-8)
        sae.encoder.weight.data[dead_indices] = normed * target_enc_norm
        sae.encoder.bias.data[dead_indices] = 0.0

        # Set decoder columns to normalized high-loss inputs (unit norm).
        sae.decoder.weight.data[:, dead_indices] = normed.T

    # Reset optimizer state for these parameters
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state:
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()


def train_sae_streaming(model, tokenizer, layer_idx: int, d_hidden: int,
                        sae_type: str = "topk", k: int = 64,
                        n_tokens: int = 100_000_000, n_steps: int = 100000,
                        batch_size: int = 4096, lr: float = 3e-4,
                        device: str = "cuda", save_path: str = None,
                        dataset_name: str = "pile"):
    """Train SAE by streaming activations — avoids caching all activations to disk."""
    from src.activation_cache import extract_activations_streaming

    chunk_size = 10_000_000  # 10M tokens per chunk
    chunks_collected = []

    print(f"Collecting activations for layer {layer_idx}...")
    for chunk_idx, chunk_acts in extract_activations_streaming(
        model, tokenizer, [layer_idx], device=device,
        n_tokens=n_tokens, chunk_size=chunk_size, dataset_name=dataset_name,
        batch_size=2  # small batch for 2.8B models
    ):
        chunks_collected.append(chunk_acts[layer_idx])
        if sum(c.shape[0] for c in chunks_collected) >= n_tokens:
            break

    # Concatenate all chunks
    all_acts = torch.cat(chunks_collected, dim=0)
    print(f"Total activations: {all_acts.shape}")
    del chunks_collected

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Train SAE on collected activations
    return train_sae(all_acts, d_hidden, sae_type=sae_type, k=k,
                     n_steps=n_steps, batch_size=batch_size, lr=lr,
                     device=device, save_path=save_path)
