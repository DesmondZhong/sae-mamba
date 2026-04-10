"""Extract and cache activations from Mamba-1, Mamba-2, and Pythia models."""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer, supporting Mamba-1, Mamba-2, and transformers."""
    print(f"Loading {model_name}...")

    if "mamba2" in model_name.lower():
        # Mamba-2 uses mamba_ssm library (not HF Transformers)
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = MambaLMHeadModel.from_pretrained(model_name, dtype=torch.float16, device=device)
        model.eval()
        model._model_backend = "mamba_ssm"
        return model, tokenizer
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if "mamba" in model_name.lower():
            from transformers import MambaForCausalLM
            model = MambaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

        model = model.to(device).eval()
        model._model_backend = "transformers"
        return model, tokenizer


def get_text_data(n_tokens: int = 10_000_000, seq_len: int = 512, tokenizer=None,
                  dataset_name: str = "pile"):
    """Load and tokenize text data from the Pile or WikiText."""
    from datasets import load_dataset
    print(f"Loading {dataset_name} dataset for {n_tokens:,} tokens...")

    if dataset_name == "pile":
        dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        text_key = "text"
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        text_key = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    all_tokens = []
    total_tokens = 0

    for example in tqdm(dataset, desc="Tokenizing", total=n_tokens // 200):
        text = example[text_key]
        if not text or len(text.strip()) < 50:
            continue
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=seq_len)["input_ids"][0]
        all_tokens.append(tokens)
        total_tokens += len(tokens)
        if total_tokens >= n_tokens:
            break

    all_tokens = torch.cat(all_tokens)[:n_tokens]
    n_seqs = len(all_tokens) // seq_len
    sequences = all_tokens[:n_seqs * seq_len].reshape(n_seqs, seq_len)
    print(f"Prepared {n_seqs:,} sequences of length {seq_len} ({n_seqs * seq_len:,} tokens)")
    return sequences


def _get_layers(model):
    """Get the list of layers from any supported model architecture."""
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        return model.backbone.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers
    else:
        raise ValueError(f"Cannot find layers in model: {type(model)}")


def extract_residual_stream(model, sequences, layer_indices, device="cuda",
                            batch_size=4, model_type="auto"):
    """Extract residual stream activations at specified layers using hooks.

    Always uses hooks (not output_hidden_states) to avoid storing all layers
    in memory. This is critical for large models with many layers.
    """
    return _extract_with_hooks(model, sequences, layer_indices, device, batch_size)


def _extract_with_hidden_states(model, sequences, layer_indices, device, batch_size):
    """Extract using output_hidden_states (HF Transformers models)."""
    activations = {layer: [] for layer in layer_indices}
    n_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="Extracting (HF)"):
        batch = sequences[i * batch_size:(i + 1) * batch_size].to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        for layer in layer_indices:
            if layer < len(hidden_states):
                acts = hidden_states[layer].float().cpu()
                activations[layer].append(acts.reshape(-1, acts.shape[-1]))

        del outputs, hidden_states
        torch.cuda.empty_cache()

    for layer in layer_indices:
        activations[layer] = torch.cat(activations[layer], dim=0)
        print(f"  Layer {layer}: {activations[layer].shape}")

    return activations


def _extract_with_hooks(model, sequences, layer_indices, device, batch_size):
    """Extract using forward hooks (mamba_ssm models).

    Hooks capture the residual stream AFTER each block (norm + mixer + residual).
    """
    activations = {layer: [] for layer in layer_indices}
    hooks = []

    layers = _get_layers(model)

    backend = getattr(model, '_model_backend', 'transformers')

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if backend == "mamba_ssm":
                # mamba_ssm Block: (hidden_states, residual) tuple
                # hidden_states (index 0) is post-norm, residual (index 1) is raw accumulation
                # Use hidden_states (post-norm) for consistent scale across layers
                if isinstance(output, tuple) and len(output) == 2:
                    hidden = output[0]  # post-norm hidden states
                else:
                    hidden = output[0] if isinstance(output, tuple) else output
            else:
                # HF Transformers layer: output is tuple (hidden_states, ...)
                hidden = output[0] if isinstance(output, tuple) else output
            activations[layer_idx].append(hidden.float().cpu())
        return hook_fn

    for layer_idx in layer_indices:
        hook = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    n_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="Extracting (hooks)"):
        batch = sequences[i * batch_size:(i + 1) * batch_size].to(device)
        with torch.no_grad():
            model(batch)
        torch.cuda.empty_cache()

    for hook in hooks:
        hook.remove()

    for layer in layer_indices:
        if activations[layer]:
            combined = torch.cat(activations[layer], dim=0)
            if combined.dim() == 3:
                combined = combined.reshape(-1, combined.shape[-1])
            activations[layer] = combined
            print(f"  Layer {layer}: {activations[layer].shape}")
        else:
            print(f"  Layer {layer}: no activations captured!")
            activations[layer] = torch.empty(0)

    return activations
