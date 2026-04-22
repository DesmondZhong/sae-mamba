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
        # Per-example L1 sum, then mean over batch — matches Anthropic/OpenAI convention.
        l1_loss = z.abs().sum(dim=-1).mean()
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
    """TopK SAE — forces exactly K active features per example.

    Includes AuxK loss (Gao et al. 2024): reconstruct residual (x - x_hat) using
    the top-k_aux dead features, weighted by aux_coeff. Prevents feature death
    during training, replacing/complementing explicit resampling.
    """

    def __init__(self, d_input: int, d_hidden: int, k: int = 64,
                 k_aux: int = 512, aux_coeff: float = 1.0 / 32.0):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.k = k
        self.k_aux = min(k_aux, d_hidden)
        self.aux_coeff = aux_coeff
        self.sae_type = "topk"

        self.encoder = nn.Linear(d_input, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_input, bias=True)

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def _pre_acts(self, x):
        x_centered = x - self.decoder.bias
        return self.encoder(x_centered)

    def _topk_from_pre(self, pre_acts, k):
        topk_vals, topk_idx = torch.topk(pre_acts, k, dim=-1)
        z = torch.zeros_like(pre_acts)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        return z

    def encode(self, x):
        return self._topk_from_pre(self._pre_acts(x), self.k)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, dead_mask: torch.Tensor = None):
        pre_acts = self._pre_acts(x)
        z = self._topk_from_pre(pre_acts, self.k)
        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)
        l0 = (z > 0).float().sum(dim=-1).mean()

        # AuxK loss: top-k_aux among dead features reconstruct the residual.
        aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        if dead_mask is not None:
            # Mask non-dead features to -inf so topk only selects from dead.
            neg_inf = torch.finfo(pre_acts.dtype).min
            dead_pre = torch.where(
                dead_mask.view(1, -1), pre_acts,
                torch.full_like(pre_acts, neg_inf),
            )
            k_aux_eff = min(self.k_aux, self.d_hidden)
            aux_vals, aux_idx = torch.topk(dead_pre, k_aux_eff, dim=-1)
            aux_vals = F.relu(aux_vals)  # ReLU(-inf) = 0, so undead picks contribute nothing
            z_aux = torch.zeros_like(pre_acts)
            z_aux.scatter_(-1, aux_idx, aux_vals)
            # Reconstruct the residual with decoder weights only (no bias).
            e = (x - x_hat).detach()
            e_hat = F.linear(z_aux, self.decoder.weight)
            aux_loss = F.mse_loss(e_hat, e)

        loss = recon_loss + self.aux_coeff * aux_loss

        return x_hat, z, loss, {
            "recon_loss": recon_loss.item(),
            "aux_loss": aux_loss.item(),
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
               k: int = 64, l1_coeff: float = 1e-3,
               k_aux: int = 512, aux_coeff: float = 1.0 / 32.0):
    """Factory function to create SAE of specified type."""
    if sae_type == "l1":
        return SparseAutoencoder(d_input, d_hidden, l1_coeff=l1_coeff)
    elif sae_type == "topk":
        return TopKSAE(d_input, d_hidden, k=k, k_aux=k_aux, aux_coeff=aux_coeff)
    elif sae_type == "batchtopk":
        return BatchTopKSAE(d_input, d_hidden, k=k)
    else:
        raise ValueError(f"Unknown SAE type: {sae_type}")
