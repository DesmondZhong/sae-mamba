#!/usr/bin/env python3
"""Phase-B induction localization for Mamba-2 2.7B.

Mamba-2's mixer has a merged in_proj whose output of 10576 channels is
sliced as [z(5120), x(5120), dt(80), B(128), C(128)]. After SSD, a norm +
out_proj give the residual contribution.

We do:
  1. Identify induction features from Mamba-2 L32 SAE on synthetic patterns.
  2. Patching sweep across layers × slices of in_proj.
  3. Also test the conv1d (short-range mixer) and out_proj-in (post-SSD) sites.

Output: $STORAGE/results_phase4/mamba2_induction.json
"""
import argparse, json, os, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.activation_cache import get_model_and_tokenizer
from src.sae import create_sae

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
CKPT_DIR = STORAGE / "checkpoints_normed"
ACTS_DIR = STORAGE / "activations"
RESULTS_DIR = STORAGE / "results_phase4"

MODEL_NAME = "state-spaces/mamba2-2.7b"
MODEL_KEY = "mamba2_2.7b"
D_MODEL = 2560
MID_LAYER = 32
SAE_EXPANSION = 16
SAE_K = 64
PATTERN_LEN = 8
PREFIX_LEN = 8
MID_LEN = 32


def load_sae_and_norm(device):
    d_hidden = D_MODEL * SAE_EXPANSION
    run_key = f"{MODEL_KEY}_L{MID_LAYER}_x{SAE_EXPANSION}_k{SAE_K}_normed"
    sae = create_sae(D_MODEL, d_hidden, sae_type="topk", k=SAE_K).to(device)
    sae.load_state_dict(torch.load(CKPT_DIR / f"{run_key}.pt",
                                   map_location=device, weights_only=True))
    sae.eval()
    acts_path = ACTS_DIR / MODEL_KEY / f"layer_{MID_LAYER}.pt"
    t = torch.load(acts_path, map_location="cpu", weights_only=True, mmap=True)
    sample = t[:10000].clone().float()
    act_mean = sample.mean(dim=0).to(device)
    act_std = sample.std(dim=0).clamp(min=1e-6).to(device)
    del t, sample
    return sae, act_mean, act_std


def make_induction_batch(tokenizer, n_pairs, seed, device):
    rng = np.random.default_rng(seed)
    vocab = tokenizer.vocab_size
    seq_len = PREFIX_LEN + PATTERN_LEN + MID_LEN + PATTERN_LEN
    clean = np.zeros((n_pairs, seq_len), dtype=np.int64)
    corr = np.zeros_like(clean)
    ind_start = PREFIX_LEN + PATTERN_LEN + MID_LEN
    ind_end = ind_start + PATTERN_LEN
    for i in range(n_pairs):
        prefix = rng.integers(0, vocab, PREFIX_LEN)
        P = rng.integers(0, vocab, PATTERN_LEN)
        mid = rng.integers(0, vocab, MID_LEN)
        while True:
            Pp = rng.integers(0, vocab, PATTERN_LEN)
            if not np.array_equal(Pp, P):
                break
        clean[i, :PREFIX_LEN] = prefix
        clean[i, PREFIX_LEN:PREFIX_LEN + PATTERN_LEN] = P
        clean[i, PREFIX_LEN + PATTERN_LEN:ind_start] = mid
        clean[i, ind_start:ind_end] = P
        corr[i, :ind_start] = clean[i, :ind_start]
        corr[i, ind_start:ind_end] = Pp
    return (torch.from_numpy(clean).to(device),
            torch.from_numpy(corr).to(device),
            ind_start, ind_end)


def get_slice_specs(mixer):
    """Return dict of {slice_name: (start, end)} for in_proj output (10576)."""
    d_inner = mixer.d_inner          # 5120
    nheads = mixer.nheads            # 80
    ngroups = mixer.ngroups          # 1
    dstate = mixer.d_state           # 128
    # Slicing per mamba-2 convention: [z, x, dt, B, C] or similar.
    # Verify by checking in_proj.out_features.
    # Mamba2 in_proj typical output: [z(d_inner), xBC(d_inner + 2*ngroups*dstate), dt(nheads)]
    # But HF returns concatenation. We verify: d_inner*2 + 2*ngroups*dstate + nheads = 10240 + 256 + 80 = 10576 ✓
    z_start = 0
    z_end = d_inner
    x_start = d_inner
    x_end = 2 * d_inner
    B_start = 2 * d_inner
    B_end = 2 * d_inner + ngroups * dstate
    C_start = B_end
    C_end = B_end + ngroups * dstate
    dt_start = C_end
    dt_end = dt_start + nheads
    return {
        "z_gate":    (z_start, z_end),       # 5120
        "x_stream":  (x_start, x_end),       # 5120
        "B_matrix":  (B_start, B_end),       # 128
        "C_matrix":  (C_start, C_end),       # 128
        "dt_step":   (dt_start, dt_end),     # 80
        "B_and_C":   (B_start, C_end),       # 256
        "full":      (0, dt_end),            # 10576
    }


def capture_inproj(model, tokens, layer):
    captured = {}
    def hook(mod, ins, out):
        captured["o"] = out.detach().clone()
    h = model.backbone.layers[layer].mixer.in_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(tokens)
    h.remove()
    return captured["o"]


def encode_residual_at(model, tokens, sae, act_mean, act_std, mid_layer,
                        positions, inproj_patch_value=None, inproj_layer=None,
                        slice_range=None):
    """Forward; optionally patch in_proj output at `inproj_layer` with slice
    from `inproj_patch_value`. Return mean z at induction positions."""
    captured = {}
    def res_hook(mod, ins, out):
        captured["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()

    hooks = [model.backbone.layers[mid_layer].register_forward_hook(res_hook)]
    if inproj_patch_value is not None:
        s, e = slice_range
        def patch_fn(mod, ins, out):
            new_out = out.clone()
            repl = inproj_patch_value[..., s:e].to(new_out.dtype)
            if repl.shape[0] != new_out.shape[0]:
                if repl.shape[0] == 1:
                    repl = repl.expand(new_out.shape[0], -1, -1)
                else:
                    repl = repl[:new_out.shape[0]]
            new_out[..., s:e] = repl
            return new_out
        hooks.append(model.backbone.layers[inproj_layer].mixer.in_proj.register_forward_hook(patch_fn))

    with torch.no_grad():
        model(tokens)
    for h in hooks:
        h.remove()
    res = captured["r"]
    normed = (res.float() - act_mean) / act_std
    _, z, *_ = sae(normed)
    return z[:, positions[0]:positions[1]].mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_pairs", type=int, default=64)
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[0, 8, 16, 24, 28, 30, 31, 32, 33, 34, 36, 40, 48, 56])
    args = ap.parse_args()
    device = args.device

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device)
    sae, act_mean, act_std = load_sae_and_norm(device)

    mixer = model.backbone.layers[0].mixer
    print(f"Mixer type: {type(mixer).__name__}")
    print(f"d_inner={mixer.d_inner}, nheads={mixer.nheads}, "
          f"ngroups={mixer.ngroups}, d_state={mixer.d_state}")
    slices = get_slice_specs(mixer)
    in_proj_out_features = mixer.in_proj.out_features
    print(f"in_proj.out_features = {in_proj_out_features}")
    assert slices["full"][1] == in_proj_out_features, (
        f"slice math mismatch: sum={slices['full'][1]}, actual={in_proj_out_features}"
    )
    for name, (s, e) in slices.items():
        print(f"  {name}: [{s}:{e}]  dim={e-s}")

    clean, corr, ind_start, ind_end = make_induction_batch(
        tokenizer, args.n_pairs, seed=0, device=device)

    # Identify induction features (same score metric as Mamba-1)
    z_clean = encode_residual_at(model, clean, sae, act_mean, act_std,
                                  MID_LAYER, (ind_start, ind_end))
    z_corr = encode_residual_at(model, corr, sae, act_mean, act_std,
                                 MID_LAYER, (ind_start, ind_end))
    score = (z_clean - z_corr).mean(dim=0).detach()
    top_vals, top_idx = score.topk(10)
    top_feats = top_idx.cpu().tolist()
    print(f"\nTop-10 induction features (Mamba-2 L{MID_LAYER} SAE): {top_feats}")

    # Baseline / corrupted target-feature activation
    ti = top_idx
    baseline_act = z_clean[:, ti].mean().item()
    corrupted_act = z_corr[:, ti].mean().item()
    gap = baseline_act - corrupted_act
    print(f"baseline={baseline_act:.4f}, corrupted={corrupted_act:.4f}, gap={gap:.4f}")

    # Patching sweep
    per_site = []
    for L in tqdm(args.layers, desc="layer sweep"):
        inproj_corr = capture_inproj(model, corr, L)
        for slice_name, sl in slices.items():
            z_patched = encode_residual_at(
                model, clean, sae, act_mean, act_std,
                MID_LAYER, (ind_start, ind_end),
                inproj_patch_value=inproj_corr,
                inproj_layer=L, slice_range=sl,
            )
            patched_act = z_patched[:, ti].mean().item()
            patch_damage = 1.0 - (patched_act - corrupted_act) / gap if abs(gap) > 1e-8 else 0.0
            per_site.append({
                "layer": L,
                "slice": slice_name,
                "slice_range": list(sl),
                "baseline_act": baseline_act,
                "corrupted_act": corrupted_act,
                "patched_act": patched_act,
                "patch_damage": patch_damage,
            })

    # Sort by patch_damage
    per_site_sorted = sorted(per_site, key=lambda r: -r["patch_damage"])
    print("\n=== Top-10 Mamba-2 patch_damage sites ===")
    for r in per_site_sorted[:10]:
        print(f"  L{r['layer']:>2d}  {r['slice']:<10s}  patch_damage={r['patch_damage']:+.4f}")

    out = {
        "model": MODEL_NAME,
        "mid_layer": MID_LAYER,
        "n_pairs": args.n_pairs,
        "layers_tested": args.layers,
        "slice_specs": slices,
        "top_induction_features": top_feats,
        "top_induction_scores": top_vals.cpu().tolist(),
        "baseline_act": baseline_act,
        "corrupted_act": corrupted_act,
        "gap": gap,
        "per_site": per_site,
    }
    out_path = RESULTS_DIR / "mamba2_induction.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
