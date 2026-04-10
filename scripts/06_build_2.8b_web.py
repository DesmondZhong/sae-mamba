#!/usr/bin/env python3
"""Build interactive web visualization for 2.8B SAE comparison."""

import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

STORAGE = Path("/mnt/storage/desmond/excuse")
RESULTS_DIR = STORAGE / "results"
WEB_DIR = Path("/root/sae-mamba/web")
WEB_DIR.mkdir(exist_ok=True)

MODEL_COLORS = {
    "mamba1_2.8b": "#ff6b6b",
    "mamba2_2.7b": "#ffd93d",
    "pythia_2.8b": "#6bcb77",
}

MODEL_LABELS = {
    "mamba1_2.8b": "Mamba-1 2.8B",
    "mamba2_2.7b": "Mamba-2 2.7B",
    "pythia_2.8b": "Pythia 2.8B",
}


def load_results():
    path = RESULTS_DIR / "comprehensive_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Build from individual files
    results = {"sae_stats": {}, "cka": {}, "baselines": {}, "downstream": {}, "monosemanticity": {}}
    for f in RESULTS_DIR.glob("*_stats.json"):
        with open(f) as fh:
            data = json.load(fh)
            results["sae_stats"][data.get("run_key", f.stem)] = data
    for name in ["cka_results", "baselines", "downstream", "monosemanticity"]:
        p = RESULTS_DIR / f"{name}.json"
        if p.exists():
            with open(p) as fh:
                results[name] = json.load(fh)
    return results


def _slim_stats(stats):
    """Strip large fields (feature_frequency, top_features) — viz only needs scalars."""
    slimmed = {}
    keep_keys = {
        "model_key", "layer", "expansion_ratio", "k", "sae_type", "run_key",
        "d_hidden", "n_samples", "dead_features", "alive_features", "dead_frac",
        "l0_mean", "l0_median", "l0_std", "avg_recon_loss", "fve",
        "final_recon_loss", "final_fve", "final_l0", "final_dead_features",
        "act_var", "act_mean_norm", "normalized",
    }
    for k, v in stats.items():
        slimmed[k] = {kk: vv for kk, vv in v.items() if kk in keep_keys}
    return slimmed


def build_html(results):
    # Prepare data for charts (slim stats: drop feature_frequency arrays etc)
    stats = _slim_stats(results.get("sae_stats", {}))
    normed_stats = _slim_stats(results.get("normed_stats", {}))
    cka = results.get("cka", {})
    baselines = results.get("baselines", {})
    downstream = results.get("downstream", {})
    mono = results.get("monosemanticity", {})
    feat_freq = results.get("feature_frequency", {})
    decoder_geo = results.get("decoder_geometry", {})
    coactivation = results.get("coactivation", {})
    within_cka = results.get("within_model_cka", {})
    eff_dim = results.get("effective_dim", {})

    # Load feature examples
    features_data = {}
    for f in RESULTS_DIR.glob("*_features.json"):
        with open(f) as fh:
            features_data[f.stem.replace("_features", "")] = json.load(fh)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAE Analysis: SSM vs Transformer Representations at Scale</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #21262d;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --mamba1: #ff6b6b;
    --mamba2: #ffd93d;
    --pythia: #6bcb77;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    line-height: 1.6;
}}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
header {{
    text-align: center;
    padding: 40px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 40px;
}}
header h1 {{
    font-size: 2.2em;
    background: linear-gradient(135deg, var(--mamba1), var(--accent), var(--pythia));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}}
header p {{ color: var(--text-dim); font-size: 1.1em; max-width: 800px; margin: 0 auto; }}
.legend {{
    display: flex;
    justify-content: center;
    gap: 30px;
    margin: 20px 0;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.95em;
}}
.legend-dot {{
    width: 14px;
    height: 14px;
    border-radius: 50%;
}}
.section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 30px;
}}
.section h2 {{
    font-size: 1.5em;
    margin-bottom: 5px;
    color: var(--accent);
}}
.section .subtitle {{
    color: var(--text-dim);
    font-size: 0.9em;
    margin-bottom: 20px;
}}
.chart-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}}
.chart-box {{
    background: var(--surface2);
    border-radius: 8px;
    padding: 15px;
    min-height: 400px;
}}
.chart-box > div {{ height: 100%; min-height: 370px; }}
.chart-box.full {{ grid-column: 1 / -1; }}
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 20px 0;
}}
.metric-card {{
    background: var(--surface2);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}}
.metric-card .value {{
    font-size: 2em;
    font-weight: bold;
}}
.metric-card .label {{
    color: var(--text-dim);
    font-size: 0.85em;
    margin-top: 5px;
}}
.insight {{
    background: rgba(88, 166, 255, 0.1);
    border-left: 3px solid var(--accent);
    padding: 15px 20px;
    margin: 15px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.95em;
}}
.feature-browser {{
    margin-top: 20px;
}}
.feature-card {{
    background: var(--surface2);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
}}
.feature-card h4 {{
    color: var(--accent);
    margin-bottom: 8px;
}}
.feature-example {{
    font-family: monospace;
    font-size: 0.85em;
    padding: 4px 8px;
    background: var(--bg);
    border-radius: 4px;
    margin: 3px 0;
    color: var(--text-dim);
}}
.feature-example mark {{
    background: rgba(255, 107, 107, 0.3);
    color: var(--text);
    padding: 0 2px;
    border-radius: 2px;
}}
.tabs {{
    display: flex;
    gap: 5px;
    margin-bottom: 15px;
}}
.tab {{
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-dim);
    cursor: pointer;
    font-size: 0.9em;
}}
.tab.active {{
    background: var(--accent);
    color: white;
    border-color: var(--accent);
}}
footer {{
    text-align: center;
    padding: 30px;
    color: var(--text-dim);
    font-size: 0.85em;
    border-top: 1px solid var(--border);
    margin-top: 40px;
}}
@media (max-width: 768px) {{
    .chart-row {{ grid-template-columns: 1fr; }}
    header h1 {{ font-size: 1.5em; }}
}}
</style>
</head>
<body>
<div class="container">

<header>
    <h1>Sparse Autoencoder Analysis: SSM vs Transformer Representations</h1>
    <p>Three-way comparison of learned features in Mamba-1 (2.8B), Mamba-2 (2.7B), and Pythia (2.8B)
       using TopK sparse autoencoders at equal sparsity</p>
    <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:var(--mamba1)"></div>Mamba-1 2.8B (SSM)</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--mamba2)"></div>Mamba-2 2.7B (SSD)</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--pythia)"></div>Pythia 2.8B (Transformer)</div>
    </div>
</header>

<!-- Section 0: Key Findings Summary -->
<div class="section">
    <h2>Key Findings</h2>
    <div class="subtitle">Three-way comparison at 2.8B scale with TopK SAEs (K=64, 16x expansion, 10M tokens from The Pile)</div>
    <div class="insight"><strong>1. SSMs and Transformers are equally sparse-decomposable when properly normalized.</strong> With per-dimension mean/std normalization, both Mamba-1 (FVE 0.67&ndash;0.97) and Pythia (FVE 0.70&ndash;0.79) achieve comparable reconstruction quality. Without normalization, Pythia middle layers catastrophically fail (FVE &lt; -5) while Mamba works out-of-the-box &mdash; revealing that SSM activations have more uniform scale across layers, a practical advantage for interpretability tooling.</div>
    <div class="insight"><strong>2. Cross-architecture representations diverge rapidly with depth.</strong> CKA between Mamba-1 and Pythia drops from 0.69 at the embedding layer to 0.02 by layer 2, recovering to 0.33 at the final layer. Mamba-1 vs Mamba-2 (both SSMs) maintain higher similarity (0.19&ndash;0.45) throughout. SSM variants share more representational structure with each other than either shares with Transformers.</div>
    <div class="insight"><strong>3. Both architectures show U-shaped reconstruction curves.</strong> FVE dips at middle layers then recovers &mdash; Mamba-1 (L32: 0.67) and Pythia (L12: 0.70 normed). Mamba-2 shows a flatter profile (0.73&ndash;0.93). Middle layers compress information maximally, consistent with information bottleneck theory.</div>
    <div class="insight"><strong>4. SAEs massively outperform PCA at equal dimensionality.</strong> At K=64, Mamba-1 SAE FVE (0.67&ndash;0.97) is 2&ndash;4x higher than PCA (0.24&ndash;0.64). Pythia PCA captures more variance (0.81&ndash;0.86) than Mamba PCA, suggesting Transformer representations are more linearly structured but equally amenable to sparse decomposition after normalization.</div>
    <div class="insight"><strong>5. Activation normalization is critical for cross-architecture SAE comparison.</strong> Without normalization, Pythia&rsquo;s 55x variance range across layers (0.13&ndash;7.2) prevents SAE convergence at middle layers. Mamba&rsquo;s narrower range allows convergence with default hyperparameters. This is a methodological finding: future cross-architecture SAE studies must normalize activations.</div>
    <div id="key-metrics" class="metric-grid"></div>
</div>

<!-- Section 2: Layer Sweep -->
<div class="section">
    <h2>Reconstruction Quality by Layer Depth</h2>
    <div class="subtitle">FVE (Fraction of Variance Explained) across layers — higher is better. Compared against PCA and random baselines.</div>
    <div class="chart-row">
        <div class="chart-box" id="chart-fve-depth"></div>
        <div class="chart-box" id="chart-l0-depth"></div>
    </div>
    <div id="insight-depth" class="insight"></div>
</div>

<!-- Section 3: Expansion Sweep -->
<div class="section">
    <h2>Scaling SAE Width</h2>
    <div class="subtitle">How does reconstruction improve as we increase the number of features (expansion ratio)?</div>
    <div class="chart-row">
        <div class="chart-box" id="chart-expansion-fve"></div>
        <div class="chart-box" id="chart-expansion-dead"></div>
    </div>
</div>

<!-- Section 4: Sparsity Sweep -->
<div class="section">
    <h2>Sparsity-Quality Tradeoff</h2>
    <div class="subtitle">At K=32, 64, 128: how much information is preserved?</div>
    <div class="chart-box full" id="chart-k-sweep"></div>
</div>

<!-- Section 5: Normalization Ablation -->
<div class="section">
    <h2>Normalization Ablation: The Critical Preprocessing Step</h2>
    <div class="subtitle">Does activation normalization (subtract mean, divide by std) change SAE quality? This ablation reveals a methodological finding.</div>
    <div class="chart-box full" id="chart-norm-ablation"></div>
    <div id="insight-norm" class="insight"></div>
</div>

<!-- Section 6: CKA -->
<div class="section">
    <h2>Cross-Architecture Representational Similarity (CKA)</h2>
    <div class="subtitle">Do SSMs and Transformers learn similar features? CKA=1 means identical representations.</div>
    <div class="chart-row">
        <div class="chart-box" id="chart-cka-raw"></div>
        <div class="chart-box" id="chart-cka-sae"></div>
    </div>
    <div id="insight-cka" class="insight"></div>
</div>

<!-- Section 6: Downstream -->
<div class="section">
    <h2>Downstream Impact</h2>
    <div class="subtitle">Perplexity when replacing activations with SAE reconstructions</div>
    <div id="chart-downstream" class="chart-box"></div>
</div>

<!-- Section 7: Feature Geometry (Phase 2) -->
<div class="section">
    <h2>Feature Geometry: How Do Learned Features Differ?</h2>
    <div class="subtitle">Analyzing the structure of SAE features beyond reconstruction quality</div>
    <div class="chart-row">
        <div class="chart-box" id="chart-effective-dim"></div>
        <div class="chart-box" id="chart-decoder-ortho"></div>
    </div>
    <div class="chart-row">
        <div class="chart-box" id="chart-zipf"></div>
        <div class="chart-box" id="chart-within-cka"></div>
    </div>
    <div id="insight-geometry" class="insight"></div>
</div>

<!-- Section 8: Feature Browser -->
<div class="section">
    <h2>Feature Browser</h2>
    <div class="subtitle">Explore what individual SAE features detect (max-activating examples)</div>
    <div class="tabs" id="feature-tabs"></div>
    <div class="feature-browser" id="feature-browser"></div>
</div>

<footer>
    <p>SAE Mechanistic Interpretability: Mamba vs Transformer | TopK SAE with dead feature resampling</p>
    <p>Models trained on The Pile | Activations from residual stream</p>
</footer>

</div>

<script>
// Embed data
const allStats = {json.dumps(stats)};
const normedStats = {json.dumps(normed_stats)};
const ckaData = {json.dumps(cka)};
const baselineData = {json.dumps(baselines)};
const downstreamData = {json.dumps(downstream)};
const monoData = {json.dumps(mono)};
const featuresData = {json.dumps(features_data)};
const featFreqData = {json.dumps(feat_freq)};
const decoderGeoData = {json.dumps(decoder_geo)};
const withinCKAData = {json.dumps(within_cka)};
const effDimData = {json.dumps(eff_dim)};

const modelColors = {json.dumps(MODEL_COLORS)};
const modelLabels = {json.dumps(MODEL_LABELS)};

const plotLayout = {{
    paper_bgcolor: '#161b22',
    plot_bgcolor: '#21262d',
    font: {{ color: '#e6edf3', size: 12 }},
    margin: {{ l: 60, r: 30, t: 40, b: 50 }},
    xaxis: {{ gridcolor: '#30363d', zerolinecolor: '#30363d' }},
    yaxis: {{ gridcolor: '#30363d', zerolinecolor: '#30363d' }},
    legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{ size: 11 }} }},
}};

// ---- Key Metrics ----
function renderKeyMetrics() {{
    const container = document.getElementById('key-metrics');
    const metrics = [];

    // Count models and SAEs
    const nSAEs = Object.keys(allStats).length;
    metrics.push({{ value: nSAEs, label: 'SAEs Trained' }});

    // Gather FVE by model
    for (const [model, label] of Object.entries(modelLabels)) {{
        const modelStats = Object.values(allStats).filter(s => s.model_key === model);
        if (modelStats.length > 0) {{
            const avgFVE = modelStats.reduce((s, x) => s + (x.fve || 0), 0) / modelStats.length;
            metrics.push({{ value: avgFVE.toFixed(3), label: `${{label}} Avg FVE` }});
        }}
    }}

    container.innerHTML = metrics.map(m =>
        `<div class="metric-card"><div class="value">${{m.value}}</div><div class="label">${{m.label}}</div></div>`
    ).join('');
}}

// ---- Layer Depth Charts ----
function renderDepthCharts() {{
    const fveTraces = [];
    const l0Traces = [];

    for (const [model, label] of Object.entries(modelLabels)) {{
        const color = modelColors[model];
        const modelStats = Object.values(allStats)
            .filter(s => s.model_key === model && s.k === 64 && s.expansion_ratio === 16)
            .sort((a, b) => a.layer - b.layer);

        if (modelStats.length === 0) continue;

        const nLayers = model.includes('pythia') ? 32 : 64;
        const depths = modelStats.map(s => s.layer / (nLayers - 1));
        const fves = modelStats.map(s => s.fve || 0);
        const l0s = modelStats.map(s => s.l0_mean || 0);

        fveTraces.push({{
            x: depths, y: fves,
            name: label, type: 'scatter', mode: 'lines+markers',
            line: {{ color, width: 2 }}, marker: {{ size: 6 }},
        }});

        l0Traces.push({{
            x: depths, y: l0s,
            name: label, type: 'scatter', mode: 'lines+markers',
            line: {{ color, width: 2 }}, marker: {{ size: 6 }},
        }});
    }}

    // Add PCA baseline if available
    for (const [model, layers] of Object.entries(baselineData)) {{
        const color = modelColors[model];
        const entries = Object.values(layers).sort((a, b) => a.depth - b.depth);
        if (entries.length > 0 && entries[0].pca_fve !== undefined) {{
            fveTraces.push({{
                x: entries.map(e => e.depth),
                y: entries.map(e => e.pca_fve),
                name: `${{modelLabels[model] || model}} PCA`,
                type: 'scatter', mode: 'lines',
                line: {{ color, width: 1, dash: 'dash' }},
            }});
        }}
    }}

    Plotly.newPlot('chart-fve-depth', fveTraces, {{
        ...plotLayout,
        title: 'Fraction of Variance Explained by Depth',
        xaxis: {{ ...plotLayout.xaxis, title: 'Relative Depth (0=early, 1=late)' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'FVE' }},
    }});

    Plotly.newPlot('chart-l0-depth', l0Traces, {{
        ...plotLayout,
        title: 'Active Features (L0) by Depth',
        xaxis: {{ ...plotLayout.xaxis, title: 'Relative Depth' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'L0 (avg active features)' }},
    }});

    // Insight
    const insight = document.getElementById('insight-depth');
    insight.textContent = 'Layer depth analysis shows how information density evolves through each architecture. ' +
        'SSMs (Mamba) process information through recurrent state accumulation, while Transformers (Pythia) ' +
        'build representations through compositional attention layers.';
}}

// ---- Expansion Sweep ----
function renderExpansionCharts() {{
    const fveTraces = [];
    const deadTraces = [];

    for (const [model, label] of Object.entries(modelLabels)) {{
        const color = modelColors[model];
        const modelStats = Object.values(allStats)
            .filter(s => s.model_key === model && s.k === 64)
            .sort((a, b) => (a.expansion_ratio || 0) - (b.expansion_ratio || 0));

        // Group by expansion ratio (take middle layer)
        const byExp = {{}};
        for (const s of modelStats) {{
            const exp = s.expansion_ratio;
            if (!byExp[exp] || Math.abs(s.layer - 32) < Math.abs(byExp[exp].layer - 32)) {{
                byExp[exp] = s;
            }}
        }}

        const exps = Object.keys(byExp).map(Number).sort((a, b) => a - b);
        if (exps.length === 0) continue;

        fveTraces.push({{
            x: exps.map(e => `${{e}}x`),
            y: exps.map(e => byExp[e].fve || 0),
            name: label, type: 'bar',
            marker: {{ color }},
        }});

        deadTraces.push({{
            x: exps.map(e => `${{e}}x`),
            y: exps.map(e => (byExp[e].dead_frac || 0) * 100),
            name: label, type: 'bar',
            marker: {{ color }},
        }});
    }}

    Plotly.newPlot('chart-expansion-fve', fveTraces, {{
        ...plotLayout,
        title: 'FVE by Expansion Ratio',
        xaxis: {{ ...plotLayout.xaxis, title: 'Expansion Ratio' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'FVE' }},
        barmode: 'group',
    }});

    Plotly.newPlot('chart-expansion-dead', deadTraces, {{
        ...plotLayout,
        title: 'Dead Features by Expansion Ratio',
        xaxis: {{ ...plotLayout.xaxis, title: 'Expansion Ratio' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'Dead Features (%)' }},
        barmode: 'group',
    }});
}}

// ---- K Sweep ----
function renderKSweep() {{
    const traces = [];

    for (const [model, label] of Object.entries(modelLabels)) {{
        const color = modelColors[model];
        const modelStats = Object.values(allStats)
            .filter(s => s.model_key === model && s.expansion_ratio === 16)
            .sort((a, b) => (a.k || 0) - (b.k || 0));

        // Group by K (take middle layer)
        const byK = {{}};
        for (const s of modelStats) {{
            const k = s.k;
            if (!byK[k] || Math.abs(s.layer - 32) < Math.abs(byK[k].layer - 32)) {{
                byK[k] = s;
            }}
        }}

        const ks = Object.keys(byK).map(Number).sort((a, b) => a - b);
        if (ks.length === 0) continue;

        traces.push({{
            x: ks,
            y: ks.map(k => byK[k].fve || 0),
            name: label, type: 'scatter', mode: 'lines+markers',
            line: {{ color, width: 2 }}, marker: {{ size: 8 }},
        }});
    }}

    Plotly.newPlot('chart-k-sweep', traces, {{
        ...plotLayout,
        title: 'FVE vs Sparsity Level (TopK)',
        xaxis: {{ ...plotLayout.xaxis, title: 'K (active features per token)', type: 'log' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'FVE' }},
    }});
}}

// ---- CKA ----
function renderCKA() {{
    const rawTraces = [];
    const saeTraces = [];

    const pairColors = {{
        'mamba1_2.8b_vs_pythia_2.8b': '#ff6b6b',
        'mamba2_2.7b_vs_pythia_2.8b': '#ffd93d',
        'mamba1_2.8b_vs_mamba2_2.7b': '#58a6ff',
    }};
    const pairLabels = {{
        'mamba1_2.8b_vs_pythia_2.8b': 'Mamba-1 vs Pythia',
        'mamba2_2.7b_vs_pythia_2.8b': 'Mamba-2 vs Pythia',
        'mamba1_2.8b_vs_mamba2_2.7b': 'Mamba-1 vs Mamba-2',
    }};

    for (const [pair, depths] of Object.entries(ckaData)) {{
        const entries = Object.values(depths).sort((a, b) => a.depth_a - b.depth_a);
        if (entries.length === 0) continue;

        rawTraces.push({{
            x: entries.map(e => e.depth_a),
            y: entries.map(e => e.cka_raw),
            name: pairLabels[pair] || pair,
            type: 'scatter', mode: 'lines+markers',
            line: {{ color: pairColors[pair] || '#888', width: 2 }},
            marker: {{ size: 6 }},
        }});

        saeTraces.push({{
            x: entries.map(e => e.depth_a),
            y: entries.map(e => e.cka_sae),
            name: pairLabels[pair] || pair,
            type: 'scatter', mode: 'lines+markers',
            line: {{ color: pairColors[pair] || '#888', width: 2 }},
            marker: {{ size: 6 }},
        }});
    }}

    Plotly.newPlot('chart-cka-raw', rawTraces, {{
        ...plotLayout,
        title: 'CKA on Raw Activations',
        xaxis: {{ ...plotLayout.xaxis, title: 'Relative Depth' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'CKA', range: [0, 1] }},
    }});

    Plotly.newPlot('chart-cka-sae', saeTraces, {{
        ...plotLayout,
        title: 'CKA on SAE Features',
        xaxis: {{ ...plotLayout.xaxis, title: 'Relative Depth' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'CKA', range: [0, 1] }},
    }});

    const insight = document.getElementById('insight-cka');
    insight.textContent = 'CKA measures how similar two models\\'s internal representations are. ' +
        'Values near 1 indicate the models learn similar features; values near 0 indicate very different representations. ' +
        'Comparing raw activations vs SAE features reveals whether sparsification preserves or destroys cross-architecture similarity.';
}}

// ---- Downstream ----
function renderDownstream() {{
    const models = Object.keys(downstreamData);
    const traces = [];

    if (models.length > 0) {{
        traces.push({{
            x: models.map(m => modelLabels[m] || m),
            y: models.map(m => downstreamData[m].baseline_ppl || 0),
            name: 'Baseline', type: 'bar',
            marker: {{ color: 'rgba(88,166,255,0.5)' }},
        }});
        traces.push({{
            x: models.map(m => modelLabels[m] || m),
            y: models.map(m => downstreamData[m].sae_ppl || 0),
            name: 'With SAE', type: 'bar',
            marker: {{ color: 'rgba(255,107,107,0.5)' }},
        }});
    }}

    Plotly.newPlot('chart-downstream', traces, {{
        ...plotLayout,
        title: 'Perplexity: Baseline vs SAE Reconstruction',
        xaxis: {{ ...plotLayout.xaxis, title: '' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'Perplexity' }},
        barmode: 'group',
    }});
}}

// ---- Feature Browser ----
function renderFeatureBrowser() {{
    const tabsEl = document.getElementById('feature-tabs');
    const browserEl = document.getElementById('feature-browser');

    const models = Object.keys(featuresData);
    if (models.length === 0) {{
        browserEl.innerHTML = '<p style="color:var(--text-dim)">No feature data available yet. Run the experiment first.</p>';
        return;
    }}

    tabsEl.innerHTML = models.map((m, i) =>
        `<button class="tab ${{i === 0 ? 'active' : ''}}" onclick="showFeatures('${{m}}', this)">${{
            modelLabels[m.split('_L')[0]] || m
        }} L${{m.split('_L')[1]?.split('_')[0] || '?'}}</button>`
    ).join('');

    window.showFeatures = function(modelKey, tabEl) {{
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        if (tabEl) tabEl.classList.add('active');

        const features = featuresData[modelKey] || [];
        browserEl.innerHTML = features.slice(0, 15).map(f => {{
            const examples = (f.top_examples || []).slice(0, 5).map(ex => {{
                const text = (ex.text || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                const highlighted = text.replace(/\\[&gt;(.*?)&lt;\\]/g, '<mark>$1</mark>');
                return `<div class="feature-example">${{highlighted}} <span style="color:var(--accent)">(act=${{ex.activation?.toFixed(2) || '?'}})</span></div>`;
            }}).join('');
            return `<div class="feature-card">
                <h4>Feature #${{f.feature_id}} (max activation: ${{f.max_activation?.toFixed(2) || '?'}})</h4>
                ${{examples}}
            </div>`;
        }}).join('');
    }};

    if (models.length > 0) showFeatures(models[0], null);
}}

// ---- Normalization Ablation ----
function renderNormAblation() {{
    const traces = [];

    // Collect unnormed FVE for Mamba-1 and Pythia at matched layers
    const unnormedByModel = {{}};
    for (const [key, s] of Object.entries(allStats)) {{
        if (s.k === 64 && s.expansion_ratio === 16) {{
            const m = s.model_key;
            if (!unnormedByModel[m]) unnormedByModel[m] = {{}};
            unnormedByModel[m][s.layer] = s.fve || 0;
        }}
    }}

    // Collect normed FVE
    const normedByModel = {{}};
    for (const [key, s] of Object.entries(normedStats)) {{
        const m = s.model_key;
        if (!normedByModel[m]) normedByModel[m] = {{}};
        normedByModel[m][s.layer] = s.fve || s.final_fve || 0;
    }}

    // Plot paired bars for each model
    const models = ['mamba1_2.8b', 'pythia_2.8b'];
    const nLayers = {{ 'mamba1_2.8b': 64, 'pythia_2.8b': 32 }};

    for (const model of models) {{
        const unnormed = unnormedByModel[model] || {{}};
        const normed = normedByModel[model] || {{}};
        const layers = [...new Set([...Object.keys(unnormed), ...Object.keys(normed)])]
            .map(Number).sort((a, b) => a - b);
        if (layers.length === 0) continue;

        const nl = nLayers[model] || 64;
        const label = modelLabels[model] || model;
        const color = modelColors[model];

        // Unnormed
        traces.push({{
            x: layers.map(l => `${{label}} L${{l}}`),
            y: layers.map(l => unnormed[l] !== undefined ? unnormed[l] : null),
            name: `${{label}} (raw)`,
            type: 'bar',
            marker: {{ color, opacity: 0.4 }},
        }});

        // Normed
        traces.push({{
            x: layers.map(l => `${{label}} L${{l}}`),
            y: layers.map(l => normed[l] !== undefined ? normed[l] : null),
            name: `${{label}} (normalized)`,
            type: 'bar',
            marker: {{ color, opacity: 1.0 }},
        }});
    }}

    Plotly.newPlot('chart-norm-ablation', traces, {{
        ...plotLayout,
        title: 'FVE: Raw vs Normalized Activations',
        xaxis: {{ ...plotLayout.xaxis, title: '' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'FVE', range: [-2, 1] }},
        barmode: 'group',
        annotations: [{{
            x: 0.5, y: -1.5, xref: 'paper', yref: 'y',
            text: 'Pythia raw FVE drops below -5 at middle layers (clipped for readability)',
            showarrow: false, font: {{ color: '#8b949e', size: 11 }},
        }}],
    }});

    const insight = document.getElementById('insight-norm');
    insight.innerHTML = '<strong>Normalization is essential for Transformers, optional for SSMs.</strong> ' +
        'Without per-dimension normalization, Pythia SAEs catastrophically fail at layers 4\u201316 (FVE < -5, clipped in chart). ' +
        'With normalization, Pythia achieves FVE 0.70\u20130.79 \u2014 comparable to Mamba-1. ' +
        'Mamba-1 FVE is nearly unchanged by normalization (0.67\u20130.97 \u2192 0.67\u20130.97), ' +
        'indicating SSM activation scales are naturally well-conditioned across layers.';
}}

// ---- Phase 2: Feature Geometry Charts ----
function renderFeatureGeometry() {{
    // Effective dimensionality by depth
    const edTraces = [];
    for (const [model, layers] of Object.entries(effDimData)) {{
        const color = modelColors[model];
        const label = modelLabels[model] || model;
        const entries = Object.values(layers).sort((a, b) => a.depth - b.depth);
        if (entries.length === 0) continue;
        edTraces.push({{
            x: entries.map(e => e.depth),
            y: entries.map(e => e.participation_ratio),
            name: label, type: 'scatter', mode: 'lines+markers',
            line: {{ color, width: 2 }}, marker: {{ size: 6 }},
        }});
    }}
    Plotly.newPlot('chart-effective-dim', edTraces, {{
        ...plotLayout,
        title: 'Effective Dimensionality (Participation Ratio)',
        xaxis: {{ ...plotLayout.xaxis, title: 'Relative Depth' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'Participation Ratio (higher = more uniform)' }},
    }});

    // Decoder orthogonality by depth
    const orthoTraces = [];
    for (const [model, layers] of Object.entries(decoderGeoData)) {{
        const color = modelColors[model];
        const label = modelLabels[model] || model;
        const entries = Object.values(layers).sort((a, b) => a.depth - b.depth);
        if (entries.length === 0) continue;
        orthoTraces.push({{
            x: entries.map(e => e.depth),
            y: entries.map(e => e.mean_abs_cosine),
            name: label, type: 'scatter', mode: 'lines+markers',
            line: {{ color, width: 2 }}, marker: {{ size: 6 }},
        }});
    }}
    Plotly.newPlot('chart-decoder-ortho', orthoTraces, {{
        ...plotLayout,
        title: 'Decoder Column Similarity (lower = more orthogonal)',
        xaxis: {{ ...plotLayout.xaxis, title: 'Relative Depth' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'Mean |cosine similarity|' }},
    }});

    // Zipf slope by depth
    const zipfTraces = [];
    for (const [model, layers] of Object.entries(featFreqData)) {{
        const color = modelColors[model];
        const label = modelLabels[model] || model;
        const entries = Object.values(layers).sort((a, b) => a.depth - b.depth);
        if (entries.length === 0) continue;
        zipfTraces.push({{
            x: entries.map(e => e.depth),
            y: entries.map(e => Math.abs(e.zipf_slope)),
            name: label, type: 'scatter', mode: 'lines+markers',
            line: {{ color, width: 2 }}, marker: {{ size: 6 }},
        }});
    }}
    Plotly.newPlot('chart-zipf', zipfTraces, {{
        ...plotLayout,
        title: 'Feature Frequency Power Law (|Zipf slope|)',
        xaxis: {{ ...plotLayout.xaxis, title: 'Relative Depth' }},
        yaxis: {{ ...plotLayout.yaxis, title: '|Zipf slope| (steeper = more unequal)' }},
    }});

    // Within-model CKA
    const wckaTraces = [];
    for (const [model, pairs] of Object.entries(withinCKAData)) {{
        const color = modelColors[model];
        const label = modelLabels[model] || model;
        const entries = Object.values(pairs).sort((a, b) => a.depth_a - b.depth_a);
        if (entries.length === 0) continue;
        wckaTraces.push({{
            x: entries.map(e => e.depth_a),
            y: entries.map(e => e.cka),
            name: label, type: 'scatter', mode: 'lines+markers',
            line: {{ color, width: 2 }}, marker: {{ size: 6 }},
        }});
    }}
    Plotly.newPlot('chart-within-cka', wckaTraces, {{
        ...plotLayout,
        title: 'Within-Model Layer-to-Layer CKA',
        xaxis: {{ ...plotLayout.xaxis, title: 'Depth of Earlier Layer' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'CKA to Next Layer', range: [0, 1] }},
    }});

    const insight = document.getElementById('insight-geometry');
    insight.innerHTML = 'These analyses reveal structural differences in how the two architectures organize learned features. ' +
        'Participation ratio measures how many features effectively contribute (higher = more distributed). ' +
        'Decoder orthogonality indicates how independent the feature directions are. ' +
        'Zipf slope measures inequality in feature usage. ' +
        'Within-model CKA shows how quickly representations change between layers.';
}}

// ---- Render All ----
renderKeyMetrics();
renderDepthCharts();
renderExpansionCharts();
renderKSweep();
renderNormAblation();
renderCKA();
renderDownstream();
renderFeatureGeometry();
renderFeatureBrowser();
</script>
</body>
</html>"""

    output_path = WEB_DIR / "index_2.8b.html"
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Web visualization saved to: {output_path}")
    return output_path


def main():
    results = load_results()
    build_html(results)


if __name__ == "__main__":
    main()
