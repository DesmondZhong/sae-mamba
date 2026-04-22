"""Internal-activation capture and patching for HF Mamba-1.

The HF `MambaMixer` has these hookable submodules:
  in_proj  (Linear): residual_in → concat(x_inner, gate)  shape (B, 2*d_inner, L)
  conv1d   (Conv1d): short-range mixing of x_inner        shape (B, d_inner, L+pad)
  x_proj   (Linear): x_act → concat(dt, B, C)             shape (B, L, dt_rank + 2*state_size)
  dt_proj  (Linear): dt (rank) → dt (d_inner) pre-softplus shape (B, L, d_inner)
  out_proj (Linear): post-SSM post-gate signal → residual  (hook its input, shape (B, L, d_inner))

The selective scan and the `y * silu(gate)` multiplication are not exposed as
submodules, so we cannot hook them without monkey-patching `MambaMixer.forward`.
The five sites above cover the full in→scan-params→out path.
"""

from typing import Dict, List, Tuple

import torch


def get_mamba_layer(model, layer_idx: int):
    """Return the MambaBlock at layer_idx in an HF MambaForCausalLM."""
    return model.backbone.layers[layer_idx]


def get_mamba_mixer(model, layer_idx: int):
    """Return the MambaMixer at layer_idx in an HF MambaForCausalLM."""
    return get_mamba_layer(model, layer_idx).mixer


COMPONENT_MODULES = {
    "in_proj": lambda mx: mx.in_proj,
    "conv1d": lambda mx: mx.conv1d,
    "x_proj": lambda mx: mx.x_proj,
    "dt_proj": lambda mx: mx.dt_proj,
    # Hook the input to out_proj — that is the post-SSM post-gate tensor.
    "out_proj_in": lambda mx: mx.out_proj,
}

ALL_COMPONENTS = list(COMPONENT_MODULES.keys())


class MambaInternalCapture:
    """Context manager that captures internal mixer activations.

    Usage:
        sites = [(32, "dt_proj"), (28, "conv1d")]
        with MambaInternalCapture(model, sites) as cap:
            model(tokens)
        dt_l32 = cap.captured[(32, "dt_proj")]  # tensor
    """

    def __init__(self, model, sites: List[Tuple[int, str]]):
        self.model = model
        self.sites = sites
        self.captured: Dict[Tuple[int, str], torch.Tensor] = {}
        self._hooks = []

    def __enter__(self):
        for layer_idx, component in self.sites:
            mixer = get_mamba_mixer(self.model, layer_idx)
            module = COMPONENT_MODULES[component](mixer)
            key = (layer_idx, component)

            if component == "out_proj_in":
                def make_pre_hook(k):
                    def hook(module, inputs):
                        self.captured[k] = inputs[0].detach().clone()
                    return hook
                self._hooks.append(module.register_forward_pre_hook(make_pre_hook(key)))
            else:
                def make_hook(k):
                    def hook(module, inputs, output):
                        self.captured[k] = output.detach().clone()
                    return hook
                self._hooks.append(module.register_forward_hook(make_hook(key)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._hooks:
            h.remove()
        self._hooks = []


class MambaInternalPatcher:
    """Context manager that replaces internal mixer activations.

    Usage:
        patches = {(32, "dt_proj"): tensor_from_corrupted_run}
        with MambaInternalPatcher(model, patches):
            model(clean_tokens)   # dt_proj at L32 is overridden
    """

    def __init__(self, model, patches: Dict[Tuple[int, str], torch.Tensor],
                 positions: List[int] = None):
        """
        Args:
            patches: {(layer_idx, component): replacement_tensor}
            positions: if given, only replace activations at these sequence positions.
                       Otherwise replace the entire tensor.
        """
        self.model = model
        self.patches = patches
        self.positions = positions
        self._hooks = []

    def _apply_replacement(self, original: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
        replacement = replacement.to(original.dtype).to(original.device)
        if self.positions is None:
            return replacement
        # Position-specific overwrite. We need to know which dim is the sequence dim.
        # For Linear outputs (shape (B, L, D)) and out_proj input, sequence dim is 1.
        # For Conv1d output (shape (B, C, L_out)), sequence dim is 2.
        # We resolve by comparing against the position indices + known layout at caller.
        # Default to dim=1 here; override via subclass for conv1d if needed.
        if replacement.dim() == 3 and replacement.shape[1] == original.shape[1]:
            out = original.clone()
            idx = torch.tensor(self.positions, device=original.device)
            out[:, idx] = replacement[:, idx]
            return out
        if replacement.dim() == 3 and replacement.shape[2] == original.shape[2]:
            out = original.clone()
            idx = torch.tensor(self.positions, device=original.device)
            out[:, :, idx] = replacement[:, :, idx]
            return out
        return replacement

    def __enter__(self):
        for (layer_idx, component), replacement in self.patches.items():
            mixer = get_mamba_mixer(self.model, layer_idx)
            module = COMPONENT_MODULES[component](mixer)

            if component == "out_proj_in":
                def make_pre_hook(r):
                    def hook(module, inputs):
                        new = self._apply_replacement(inputs[0], r)
                        return (new,) + inputs[1:]
                    return hook
                self._hooks.append(module.register_forward_pre_hook(make_pre_hook(replacement)))
            else:
                def make_hook(r):
                    def hook(module, inputs, output):
                        return self._apply_replacement(output, r)
                    return hook
                self._hooks.append(module.register_forward_hook(make_hook(replacement)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._hooks:
            h.remove()
        self._hooks = []


class ResidualStreamCapture:
    """Captures the post-block residual stream at one or more layers (HF Mamba / HF Transformers)."""

    def __init__(self, model, layer_indices: List[int], backend: str = "transformers"):
        self.model = model
        self.layer_indices = layer_indices
        self.backend = backend
        self.captured: Dict[int, torch.Tensor] = {}
        self._hooks = []

    def __enter__(self):
        for li in self.layer_indices:
            layer = get_mamba_layer(self.model, li) if self.backend != "pythia" else \
                    self.model.gpt_neox.layers[li]

            def make_hook(idx):
                def hook(module, inputs, output):
                    out = output[0] if isinstance(output, tuple) else output
                    self.captured[idx] = out.detach().clone()
                return hook

            self._hooks.append(layer.register_forward_hook(make_hook(li)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._hooks:
            h.remove()
        self._hooks = []
