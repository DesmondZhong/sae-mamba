"""Sparse Autoencoder model following Anthropic's approach."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, l1_coeff: float = 1e-3):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.encoder = nn.Linear(d_input, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_input, bias=True)

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x):
        # x: (batch, d_input)
        # Center input by subtracting decoder bias
        x_centered = x - self.decoder.bias
        z = F.relu(self.encoder(x_centered))  # sparse hidden activations
        x_hat = self.decoder(z)

        recon_loss = F.mse_loss(x_hat, x)
        l1_loss = z.abs().mean()
        loss = recon_loss + self.l1_coeff * l1_loss

        # L0: average number of active features per example
        l0 = (z > 0).float().sum(dim=-1).mean()

        return x_hat, z, loss, {
            "recon_loss": recon_loss.item(),
            "l1_loss": l1_loss.item(),
            "total_loss": loss.item(),
            "l0": l0.item(),
        }

    @torch.no_grad()
    def normalize_decoder(self):
        """Project decoder columns to unit norm (maintain during training)."""
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
