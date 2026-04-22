#!/usr/bin/env python3
"""Plot Phase-4 induction-circuit localization results.

Produces (under $SAE_MAMBA_STORAGE/results_phase4/figures/):
  1. Heatmap of patch_damage over (layer, component) for Mamba-1.
  2. Heatmap for Pythia-2.8B.
  3. Side-by-side bar chart of top-10 sites in Mamba-1 vs. Pythia.
  4. Position-specific bar for top Mamba-1 sites (all / ind_only / pre_ind_only).
  5. Cross-layer emergence line: x_proj patch_damage by layer for Mamba-1.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/workspace/excuse"))
RESULTS = STORAGE / "results_phase4"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def load(fn):
    p = RESULTS / fn
    if not p.exists():
        return None
    return json.load(open(p))


def unwrap_per_site(p):
    return p["per_site"] if isinstance(p, dict) and "per_site" in p else p


def heatmap(per_site, title, out, component_order=None, cmap="viridis"):
    rows = unwrap_per_site(per_site)
    comps = component_order or sorted({r["component"] for r in rows})
    layers = sorted({r["layer"] for r in rows})
    grid = np.full((len(comps), len(layers)), np.nan)
    for r in rows:
        i = comps.index(r["component"])
        j = layers.index(r["layer"])
        grid[i, j] = r["patch_damage"]
    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.6), 0.6 * len(comps) + 2))
    vmax = max(0.9, np.nanmax(grid))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(layers)), labels=[f"L{l}" for l in layers])
    ax.set_yticks(range(len(comps)), labels=comps)
    ax.set_title(title)
    for i in range(len(comps)):
        for j in range(len(layers)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > vmax * 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="patch_damage")
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def top_bar(mamba, pythia, out, top_n=10):
    m_rows = sorted(unwrap_per_site(mamba), key=lambda r: r["patch_damage"], reverse=True)[:top_n]
    p_rows = sorted(unwrap_per_site(pythia), key=lambda r: r["patch_damage"], reverse=True)[:top_n]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.35 * top_n + 2), sharex=True)
    for ax, rows, color, title in [
        (axes[0], m_rows, "#c03", "Mamba-1 2.8B"),
        (axes[1], p_rows, "#36c", "Pythia 2.8B"),
    ]:
        labels = [f"L{r['layer']:>2} {r['component']}" for r in rows][::-1]
        vals = [r["patch_damage"] for r in rows][::-1]
        ax.barh(range(len(labels)), vals, color=color)
        ax.set_yticks(range(len(labels)), labels=labels, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("patch_damage")
        ax.axvline(0, color="k", lw=0.5)
    axes[0].set_xlim(0, max(1.0, max(r["patch_damage"] for r in m_rows + p_rows) * 1.05))
    plt.suptitle(f"Top-{top_n} patching sites: Mamba-1 concentrates, Pythia distributes",
                 y=1.02)
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def position_bar(pos_data, out):
    """pos_data is list of {layer, component, region, patch_damage}."""
    rows = pos_data if isinstance(pos_data, list) else pos_data.get("per_site", [])
    # Unique (layer, component) pairs, ordered by ind_only damage
    pairs = {}
    for r in rows:
        key = (r["layer"], r["component"])
        pairs.setdefault(key, {})[r["patch_region"]] = r["patch_damage"]
    pairs = sorted(pairs.items(),
                   key=lambda kv: -kv[1].get("ind_only", 0))
    pairs = pairs[:8]
    labels = [f"L{k[0]:>2} {k[1]}" for k, _ in pairs][::-1]
    all_v  = [v.get("all", 0) for _, v in pairs][::-1]
    ind_v  = [v.get("ind_only", 0) for _, v in pairs][::-1]
    pre_v  = [v.get("pre_ind_only", 0) for _, v in pairs][::-1]
    y = np.arange(len(labels))
    h = 0.28
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(labels) + 1.5))
    ax.barh(y - h, all_v, h, label="all positions", color="#888")
    ax.barh(y,     ind_v, h, label="induction positions only", color="#c03")
    ax.barh(y + h, pre_v, h, label="pre-induction positions only", color="#36c")
    ax.set_yticks(y, labels=labels)
    ax.set_xlabel("patch_damage")
    ax.set_title("Position-specific patching: induction signal is position-localized")
    ax.legend(loc="lower right")
    ax.axvline(0, color="k", lw=0.5)
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def emergence_line(mamba, mamba_fine, out, component="x_proj"):
    rows_all = unwrap_per_site(mamba) + (unwrap_per_site(mamba_fine) if mamba_fine else [])
    rows = [r for r in rows_all if r["component"] == component]
    # Dedup (layer) keeping first
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["layer"], r["patch_damage"])
    layers = sorted(by_layer)
    vals = [by_layer[l] for l in layers]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, vals, "o-", color="#c03", linewidth=2, markersize=6)
    ax.axhline(0.5, color="k", ls="--", lw=0.7, label="0.5 threshold")
    # First crossing
    cross = next((l for l, v in zip(layers, vals) if v >= 0.5), None)
    if cross is not None:
        ax.axvline(cross, color="#888", ls=":", lw=1)
        ax.text(cross + 0.3, 0.05, f"first crosses 0.5 at L{cross}", fontsize=9)
    ax.set_xlabel("layer")
    ax.set_ylabel(f"patch_damage ({component})")
    ax.set_title(f"Cross-layer emergence of induction signal ({component}, Mamba-1)")
    ax.set_ylim(-0.05, max(1.0, max(vals) * 1.05))
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def slice_bar(slice_data, suff_data, out):
    """Bar chart of x_proj slice-wise patch_damage (necessity) and rescue (sufficiency)."""
    res = slice_data["results"]
    names = ["delta_pre", "B_matrix", "C_matrix", "B_and_C", "full_xproj"]
    dams = [res[n]["patch_damage"] for n in names]
    dims = [res[n]["dim"] for n in names]
    suff = suff_data.get("sufficiency", {}) if suff_data else {}
    rescues = []
    for n in names:
        if n in suff:
            rescues.append(suff[n]["rescue_fraction"])
        else:
            rescues.append(0.0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(names))
    h = 0.35
    ax.barh(y + h / 2, dams, h, color="#c03", label="necessity (corrupted→clean)")
    ax.barh(y - h / 2, rescues, h, color="#36c", label="sufficiency (clean→corrupted)")
    labels = [f"{n}  (dim={d})" for n, d in zip(names, dims)]
    ax.set_yticks(y, labels=labels)
    ax.set_xlabel("patch_damage / rescue fraction")
    ax.set_title("L30 x_proj slice-wise patching: induction lives in C (16-dim)")
    ax.axvline(0, color="k", lw=0.5)
    ax.legend(loc="upper right")
    for i, v in enumerate(dams):
        ax.text(v + 0.01, y[i] + h / 2, f"{v:+.2f}", va="center", fontsize=9)
    for i, v in enumerate(rescues):
        ax.text(v + 0.01, y[i] - h / 2, f"{v:+.2f}", va="center", fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def mamba2_heatmap(data, out):
    rows = data["per_site"]
    slices = ["z_gate", "x_stream", "B_matrix", "C_matrix", "dt_step", "B_and_C", "full"]
    layers = sorted({r["layer"] for r in rows})
    # only show layers with any non-trivial damage
    layers = [l for l in layers if any(abs(r["patch_damage"]) > 0.02
                                        for r in rows if r["layer"] == l)]
    grid = np.full((len(slices), len(layers)), np.nan)
    for r in rows:
        if r["layer"] not in layers or r["slice"] not in slices:
            continue
        i = slices.index(r["slice"]); j = layers.index(r["layer"])
        grid[i, j] = r["patch_damage"]
    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.7), 0.6 * len(slices) + 2))
    im = ax.imshow(grid, aspect="auto", cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(len(layers)), labels=[f"L{l}" for l in layers])
    ax.set_yticks(range(len(slices)), labels=slices)
    ax.set_title(f"Mamba-2 2.7B: patch_damage by (layer, in_proj slice)\n"
                 f"[gap = baseline − corrupted = {data['gap']:.3f}, much smaller than Mamba-1's 3.23]")
    for i in range(len(slices)):
        for j in range(len(layers)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="patch_damage")
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def real_text_bar(rt, out):
    feats = rt["per_feature"]
    labels = [f"feat {p['feature']}" for p in feats][::-1]
    syn = [p["synth_score"] for p in feats][::-1]
    ratios = [min(p["real_repeat_ratio_to_baseline"], 20) for p in feats][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.4 * len(labels) + 2))
    axes[0].barh(range(len(labels)), syn, color="#999")
    axes[0].set_yticks(range(len(labels)), labels=labels)
    axes[0].set_title("Synthetic pattern score (clean − corrupted)")
    axes[0].set_xlabel("score")
    axes[0].axvline(0, color="k", lw=0.5)
    axes[1].barh(range(len(labels)), ratios, color="#c03")
    axes[1].set_yticks(range(len(labels)), labels=labels)
    axes[1].set_title(f"Real-Pile natural-repeat activation ratio  "
                      f"(mean {rt['summary']['mean_ratio_repeat_over_baseline']:.2f}×)")
    axes[1].set_xlabel("ratio (capped at 20)")
    axes[1].axvline(1, color="k", lw=0.5, ls=":")
    axes[1].axvline(2, color="#333", lw=0.8, ls="--")
    plt.suptitle(f"Induction features generalize to natural text "
                 f"(n={rt['n_repeat_positions']:,} repeat positions from Pile)", y=1.02)
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    m = load("patching_results.json")
    p = load("pythia_patching_results.json")
    pos = load("patching_position_specific.json")
    m_fine = load("patching_results_fine.json")  # optional
    slice_data = load("xproj_slice_patching.json")
    suff_data = load("sufficiency_patch.json")
    mamba2_data = load("mamba2_induction.json")
    real_text = load("real_text_induction.json")

    if m is None or p is None:
        print("ERROR: missing patching_results.json or pythia_patching_results.json")
        sys.exit(1)

    # HF Mamba submodule order
    mamba_order = ["in_proj", "conv1d", "x_proj", "dt_proj", "out_proj_in"]
    heatmap(m, "Mamba-1 2.8B: patch_damage by (layer, component)",
            FIGS / "heatmap_mamba1.png", component_order=mamba_order, cmap="Reds")
    pythia_order = sorted({r["component"] for r in unwrap_per_site(p)})
    heatmap(p, "Pythia 2.8B: patch_damage by (layer, component)",
            FIGS / "heatmap_pythia.png", component_order=pythia_order, cmap="Blues")

    top_bar(m, p, FIGS / "top_sites_bar.png", top_n=10)

    if pos is not None:
        position_bar(pos, FIGS / "position_specific_bar.png")

    emergence_line(m, m_fine, FIGS / "emergence_line.png", component="x_proj")

    if slice_data is not None:
        slice_bar(slice_data, suff_data, FIGS / "slice_necessity_sufficiency.png")
    if mamba2_data is not None:
        mamba2_heatmap(mamba2_data, FIGS / "heatmap_mamba2.png")
    if real_text is not None:
        real_text_bar(real_text, FIGS / "real_text_bar.png")


if __name__ == "__main__":
    main()
