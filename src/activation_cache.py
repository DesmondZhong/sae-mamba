"""Extract and cache activations from Mamba and Pythia models."""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, MambaForCausalLM
from datasets import load_dataset
from tqdm import tqdm


def get_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "mamba" in model_name.lower():
        model = MambaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    model = model.to(device).eval()
    return model, tokenizer


def get_text_data(n_tokens: int = 2_000_000, seq_len: int = 512, tokenizer=None):
    """Load and tokenize text data."""
    print(f"Loading dataset for {n_tokens} tokens...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    all_tokens = []
    for example in dataset:
        text = example["text"]
        if len(text.strip()) < 50:
            continue
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=seq_len)["input_ids"][0]
        all_tokens.append(tokens)
        if sum(len(t) for t in all_tokens) >= n_tokens:
            break

    # Concatenate and split into fixed-length sequences
    all_tokens = torch.cat(all_tokens)[:n_tokens]
    n_seqs = len(all_tokens) // seq_len
    sequences = all_tokens[:n_seqs * seq_len].reshape(n_seqs, seq_len)
    print(f"Prepared {n_seqs} sequences of length {seq_len} ({n_seqs * seq_len} tokens)")
    return sequences


def extract_residual_stream(model, sequences, layer_indices, device="cuda",
                            batch_size=8, model_type="mamba"):
    """Extract residual stream activations at specified layers."""
    activations = {layer: [] for layer in layer_indices}
    n_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="Extracting residual stream"):
        batch = sequences[i * batch_size:(i + 1) * batch_size].to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of (batch, seq, d_model)

        for layer in layer_indices:
            if layer < len(hidden_states):
                acts = hidden_states[layer].float().cpu()
                # Flatten: (batch, seq, d_model) -> (batch * seq, d_model)
                activations[layer].append(acts.reshape(-1, acts.shape[-1]))

        del outputs, hidden_states
        torch.cuda.empty_cache()

    # Concatenate
    for layer in layer_indices:
        activations[layer] = torch.cat(activations[layer], dim=0)
        print(f"  Layer {layer}: {activations[layer].shape}")

    return activations


def extract_post_ssm(model, sequences, layer_indices, device="cuda",
                     batch_size=8):
    """Extract post-SSM activations from Mamba using hooks."""
    activations = {layer: [] for layer in layer_indices}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # MambaMixer output is a tuple or tensor
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            activations[layer_idx].append(out.float().cpu())
        return hook_fn

    # Register hooks on the mixer modules
    for layer_idx in layer_indices:
        if hasattr(model, 'backbone'):
            layer = model.backbone.layers[layer_idx]
        elif hasattr(model, 'model'):
            layer = model.model.layers[layer_idx]
        else:
            raise ValueError(f"Cannot find layers in model architecture")

        if hasattr(layer, 'mixer'):
            hook = layer.mixer.register_forward_hook(make_hook(layer_idx))
        elif hasattr(layer, 'mamba'):
            hook = layer.mamba.register_forward_hook(make_hook(layer_idx))
        else:
            raise ValueError(f"Cannot find mixer/mamba in layer {layer_idx}")
        hooks.append(hook)

    n_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="Extracting post-SSM"):
        batch = sequences[i * batch_size:(i + 1) * batch_size].to(device)
        with torch.no_grad():
            model(batch)
        torch.cuda.empty_cache()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Concatenate and flatten
    for layer in layer_indices:
        if activations[layer]:
            acts = torch.cat(activations[layer], dim=0)
            # Shape: (total_batch, seq, d_model) -> (total_batch * seq, d_model)
            if acts.dim() == 3:
                acts = acts.reshape(-1, acts.shape[-1])
            activations[layer] = acts
            print(f"  Post-SSM Layer {layer}: {activations[layer].shape}")
        else:
            print(f"  Post-SSM Layer {layer}: no activations captured!")

    return activations
