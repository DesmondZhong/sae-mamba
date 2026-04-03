"""Train SAE on cached activations."""

import torch
import json
import time
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from src.sae import SparseAutoencoder


def train_sae(activations: torch.Tensor, d_hidden: int, l1_coeff: float,
              n_steps: int = 30000, batch_size: int = 4096, lr: float = 1e-4,
              device: str = "cuda", save_path: str = None, log_interval: int = 1000):
    """Train an SAE on pre-extracted activations."""
    d_input = activations.shape[1]
    sae = SparseAutoencoder(d_input, d_hidden, l1_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_iter = iter(loader)

    history = {"recon_loss": [], "l1_loss": [], "l0": [], "total_loss": []}
    start_time = time.time()

    for step in range(n_steps):
        try:
            (batch,) = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            (batch,) = next(loader_iter)

        batch = batch.to(device)
        x_hat, z, loss, metrics = sae(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Normalize decoder after each step
        sae.normalize_decoder()

        if (step + 1) % log_interval == 0:
            import sys
            elapsed = time.time() - start_time
            eta = (elapsed / (step + 1)) * (n_steps - step - 1)
            print(f"  Step {step+1}/{n_steps} | recon={metrics['recon_loss']:.6f} | "
                  f"l1={metrics['l1_loss']:.4f} | L0={metrics['l0']:.1f} | ETA={eta/60:.1f}min",
                  flush=True)
            for k, v in metrics.items():
                history[k].append({"step": step + 1, "value": v})

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(sae.state_dict(), save_path)
        # Save history
        hist_path = save_path.replace(".pt", "_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f)

    return sae, history
