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

STORAGE = Path(os.environ.get("SAE_MAMBA_STORAGE", "/path/to/storage"))
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


def gap_sweep_plot(data, out):
    results = data["results"]
    bins = list(results.keys())
    ratios = [results[b]["mean_ratio"] for b in bins]
    ns = [results[b]["n_repeats"] for b in bins]
    mids = [2 ** (4 + i + 0.5) for i in range(len(bins))]  # geometric midpoint approx
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(range(len(bins)), ratios, "o-", color="#c03", markersize=8, linewidth=2)
    ax.set_xticks(range(len(bins)), labels=bins, rotation=20, ha="right")
    ax.axhline(1.0, color="k", ls=":", lw=0.8, label="no-repeat baseline")
    ax.axhline(2.0, color="#888", ls="--", lw=0.8, label="2× threshold")
    for i, (r, n) in enumerate(zip(ratios, ns)):
        ax.text(i, r + 0.12, f"{r:.2f}×\n(n={n:,})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("activation ratio (repeat / baseline)")
    ax.set_xlabel("gap between first and second bigram occurrence")
    ax.set_title("Induction features fire strongly even at long gaps (natural Pile text)")
    ax.set_ylim(0, max(ratios) * 1.25)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def next_token_bar(data, out):
    per = data["per_slice"]
    names = ["delta_pre", "B_matrix", "C_matrix", "B_and_C", "full_xproj"]
    dams = [per[n]["next_token_damage"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(names))
    ax.barh(y, dams, color="#c03")
    ax.set_yticks(y, labels=names)
    ax.set_xlabel("next-token logit damage")
    ax.set_title(f"Behavioral confirmation: C slice drops logit by 47.5% of gap "
                 f"(clean={data['clean_mean_logit']:.2f}, corrupted={data['corrupted_mean_logit']:.2f})")
    ax.axvline(0, color="k", lw=0.5)
    for i, v in enumerate(dams):
        ax.text(v + 0.008, y[i], f"{v:+.3f}", va="center", fontsize=10)
    ax.set_xlim(-0.05, max(dams) * 1.2)
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def mamba2_plen_plot(data, out):
    results = data["results"]
    lengths = [int(l) for l in data["lengths"]]
    gaps = [results[str(l)]["gap"] for l in lengths]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lengths, gaps, "o-", color="#ffd93d", markersize=8, linewidth=2,
            label="Mamba-2 (L32 SAE)")
    ax.axhline(3.23, color="#ff6b6b", ls="--", lw=1.5,
                label="Mamba-1 gap at plen=8 (3.23)")
    ax.set_xlabel("pattern length")
    ax.set_ylabel("clean − corrupted activation gap")
    ax.set_title("Mamba-2 induction is ~15× weaker than Mamba-1 at all pattern lengths")
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels(lengths)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def natural_vs_synthetic_bar(nat, syn_slice, out):
    """Show that natural-text and synthetic patching give the same C-matrix localization."""
    n = nat["results"]
    s = syn_slice["results"]
    slices = ["delta_pre", "B_matrix", "C_matrix", "full_xproj"]
    syn_dam = [s[sl]["patch_damage"] for sl in slices]
    nat_dam = [n[sl]["patch_damage"] for sl in slices]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(slices))
    h = 0.35
    ax.barh(y - h/2, syn_dam, h, color="#c03", label="synthetic induction pairs (±random tokens)")
    ax.barh(y + h/2, nat_dam, h, color="#36c", label="natural Pile bigram repeats")
    ax.set_yticks(y, labels=slices)
    ax.set_xlabel("patch_damage")
    ax.set_title("Natural-text patching replicates synthetic C-matrix localization")
    ax.legend(loc="lower right")
    ax.axvline(0, color="k", lw=0.5)
    for i in range(len(slices)):
        ax.text(syn_dam[i] + 0.01, y[i] - h/2, f"{syn_dam[i]:+.3f}", va="center", fontsize=9)
        ax.text(nat_dam[i] + 0.01, y[i] + h/2, f"{nat_dam[i]:+.3f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def scaling_plot(m130, m370, out):
    """Plot the Mamba-130M and Mamba-370M scaling sweeps (x_proj C by layer)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, d, title in [(axes[0], m130, "Mamba-130M (24 layers)"),
                          (axes[1], m370, "Mamba-370M (48 layers)")]:
        rows = [r for r in d["per_site"] if r["slice"] == "C_matrix"]
        rows.sort(key=lambda r: r["layer"])
        layers = [r["layer"] for r in rows]
        vals = [r["logit_damage"] for r in rows]
        ax.plot(layers, vals, "o-", color="#c03", linewidth=2, markersize=6)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("layer")
        ax.set_ylabel("logit damage (C_matrix patch)")
        ax.set_title(f"{title}  gap={d['gap']:.2f}")
        ax.grid(alpha=0.3)
        # Mark peak
        if vals:
            peak = max(vals)
            peak_L = layers[vals.index(peak)]
            ax.axvline(peak_L, color="#888", ls=":", lw=1)
            ax.text(peak_L + 0.5, peak * 0.9, f"peak L{peak_L}\n{peak:+.3f}",
                    fontsize=9, ha="left", va="top")
    plt.suptitle("Scaling check: C-matrix patch damage by layer (logit metric)")
    plt.tight_layout()
    fig.savefig(out, dpi=140)
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
    gap_data = load("gap_sweep.json")
    ntd_data = load("next_token_damage.json")
    m2_plen = load("mamba2_plen_sweep.json")
    nat_text = load("natural_text_patching.json")
    scaling_130 = load("scaling_mamba130m.json")
    scaling_370 = load("scaling_mamba370m.json")

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
    if gap_data is not None:
        gap_sweep_plot(gap_data, FIGS / "gap_sweep.png")
    if ntd_data is not None:
        next_token_bar(ntd_data, FIGS / "next_token_damage.png")
    if m2_plen is not None:
        mamba2_plen_plot(m2_plen, FIGS / "mamba2_plen.png")
    if nat_text is not None and slice_data is not None:
        natural_vs_synthetic_bar(nat_text, slice_data,
                                  FIGS / "natural_vs_synthetic.png")
    if scaling_130 is not None and scaling_370 is not None:
        scaling_plot(scaling_130, scaling_370, FIGS / "scaling_comparison.png")


if __name__ == "__main__":
    main()
