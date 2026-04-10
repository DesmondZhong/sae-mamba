"""Sparse Autoencoder variants: L1, TopK, BatchTopK."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Base SAE with L1 sparsity (Anthropic's original approach)."""

    def __init__(self, d_input: int, d_hidden: int, l1_coeff: float = 1e-3):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        self.sae_type = "l1"

        self.encoder = nn.Linear(d_input, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_input, bias=True)

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x):
        x_centered = x - self.decoder.bias
        return F.relu(self.encoder(x_centered))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)
        l1_loss = z.abs().mean()
        loss = recon_loss + self.l1_coeff * l1_loss
        l0 = (z > 0).float().sum(dim=-1).mean()

        return x_hat, z, loss, {
            "recon_loss": recon_loss.item(),
            "l1_loss": l1_loss.item(),
            "total_loss": loss.item(),
            "l0": l0.item(),
        }

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


class TopKSAE(nn.Module):
    """TopK SAE — forces exactly K active features per example."""

    def __init__(self, d_input: int, d_hidden: int, k: int = 64):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.k = k
        self.sae_type = "topk"

        self.encoder = nn.Linear(d_input, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_input, bias=True)

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x):
        x_centered = x - self.decoder.bias
        pre_acts = self.encoder(x_centered)
        # Keep only top-k activations, zero the rest
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        z = torch.zeros_like(pre_acts)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)
        l0 = (z > 0).float().sum(dim=-1).mean()
        # Auxiliary loss: encourage dead features to activate
        # (no L1 needed since sparsity is enforced by top-k)
        loss = recon_loss

        return x_hat, z, loss, {
            "recon_loss": recon_loss.item(),
            "total_loss": loss.item(),
            "l0": l0.item(),
        }

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


class BatchTopKSAE(nn.Module):
    """BatchTopK SAE — targets average K across the batch, allows per-example variation."""

    def __init__(self, d_input: int, d_hidden: int, k: int = 64):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.k = k
        self.sae_type = "batchtopk"

        self.encoder = nn.Linear(d_input, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_input, bias=True)

        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x):
        x_centered = x - self.decoder.bias
        pre_acts = self.encoder(x_centered)
        pre_relu = F.relu(pre_acts)

        # Flatten batch, take top batch_size * k activations globally
        batch_size = x.shape[0]
        total_k = batch_size * self.k
        flat = pre_relu.reshape(-1)
        if total_k >= flat.shape[0]:
            return pre_relu
        topk_vals, topk_idx = torch.topk(flat, total_k)
        threshold = topk_vals[-1]
        # Zero out everything below the global threshold
        z = pre_relu * (pre_relu >= threshold).float()
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)
        l0 = (z > 0).float().sum(dim=-1).mean()
        loss = recon_loss

        return x_hat, z, loss, {
            "recon_loss": recon_loss.item(),
            "total_loss": loss.item(),
            "l0": l0.item(),
        }

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


def create_sae(d_input: int, d_hidden: int, sae_type: str = "topk",
               k: int = 64, l1_coeff: float = 1e-3):
    """Factory function to create SAE of specified type."""
    if sae_type == "l1":
        return SparseAutoencoder(d_input, d_hidden, l1_coeff=l1_coeff)
    elif sae_type == "topk":
        return TopKSAE(d_input, d_hidden, k=k)
    elif sae_type == "batchtopk":
        return BatchTopKSAE(d_input, d_hidden, k=k)
    else:
        raise ValueError(f"Unknown SAE type: {sae_type}")
