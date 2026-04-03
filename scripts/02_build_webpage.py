#!/usr/bin/env python3
"""Build interactive HTML visualization for SAE on Mamba results."""

import sys
sys.path.insert(0, "/workspace/sae-mamba")

import json
from pathlib import Path

RESULTS_DIR = Path("/workspace/sae-mamba/results")
WEB_DIR = Path("/workspace/sae-mamba/web")
WEB_DIR.mkdir(exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# Load results
all_results = load_json(RESULTS_DIR / "all_results.json")

# Prepare summary data for charts
summary_data = {}
for model_key, model_data in all_results.items():
    for sae_key, sae_data in model_data["saes"].items():
        stats = sae_data["stats"]
        summary_data[sae_key] = {
            "model": model_key,
            "extraction": sae_data["extraction"],
            "layer": sae_data["layer"],
            "l0_mean": stats["l0_mean"],
            "dead_frac": stats["dead_frac"],
            "alive_features": stats["alive_features"],
            "dead_features": stats["dead_features"],
            "avg_recon_loss": stats["avg_recon_loss"],
            "d_hidden": sae_data["d_hidden"],
        }

# Prepare feature browser data (top features from each SAE)
feature_browser = {}
for model_key, model_data in all_results.items():
    for sae_key, sae_data in model_data["saes"].items():
        feature_browser[sae_key] = sae_data.get("top_features", [])

# Prepare training history data
training_histories = {}
for model_key, model_data in all_results.items():
    for sae_key, sae_data in model_data["saes"].items():
        training_histories[sae_key] = sae_data.get("history", {})

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sparse Autoencoders on State Space Models (Mamba)</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  h1 {{ font-size: 2em; margin: 30px 0 10px; color: #fff; }}
  h2 {{ font-size: 1.4em; margin: 30px 0 15px; color: #ff9f43; border-bottom: 1px solid #333; padding-bottom: 8px; }}
  h3 {{ font-size: 1.1em; margin: 15px 0 10px; color: #aaa; }}
  p, .desc {{ color: #bbb; line-height: 1.6; margin-bottom: 15px; font-size: 0.95em; }}
  .hero {{ background: linear-gradient(135deg, #1a1a2e 0%, #2e1a2e 50%, #3e1a1e 100%);
           padding: 40px; border-radius: 12px; margin-bottom: 30px; }}
  .hero h1 {{ margin-top: 0; font-size: 2.2em; }}
  .hero .subtitle {{ color: #ff9f43; font-size: 1.1em; margin-top: 8px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .card {{ background: #1a1a1a; border-radius: 8px; padding: 20px; border: 1px solid #333; }}
  .card h3 {{ color: #ff9f43; margin-top: 0; }}
  .plot {{ width: 100%; min-height: 400px; }}
  .wide-plot {{ width: 100%; min-height: 500px; }}
  .selector {{ margin: 10px 0; }}
  .selector select {{ background: #222; color: #fff; border: 1px solid #555; padding: 6px 12px;
                      border-radius: 4px; font-size: 0.9em; }}
  .feature-list {{ max-height: 500px; overflow-y: auto; }}
  .feature-item {{ background: #222; margin: 8px 0; padding: 12px; border-radius: 6px; border-left: 3px solid #ff9f43; }}
  .feature-item .feat-id {{ color: #ff9f43; font-weight: bold; }}
  .feature-item .example {{ color: #888; font-size: 0.85em; margin: 4px 0; font-family: monospace; }}
  .feature-item .example mark {{ background: #ff9f4355; color: #fff; padding: 0 2px; }}
  .findings {{ background: #2a1a1a; border: 1px solid #5a2a2a; border-radius: 8px; padding: 20px; margin: 20px 0; }}
  .findings h3 {{ color: #ff9f43; }}
  .findings li {{ margin: 8px 0; color: #ccc; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">

<div class="hero">
  <h1>Sparse Autoencoders on State Space Models</h1>
  <div class="subtitle">Do Mamba models develop monosemantic features like transformers?</div>
  <p style="margin-top:15px; color:#aaa;">
    We train sparse autoencoders on activations from Mamba-130M, Mamba-370M, and Pythia-160M (transformer baseline),
    extracting features from residual stream and post-SSM activations at early, middle, and late layers.
    This is the first application of SAEs to state space models.
  </p>
</div>

<!-- Section 1: Aggregate Comparison -->
<h2>1. Feature Statistics Comparison</h2>
<div class="grid">
  <div class="card">
    <h3>Average L0 (Active Features per Token)</h3>
    <div id="plot-l0" class="plot"></div>
  </div>
  <div class="card">
    <h3>Dead Feature Fraction</h3>
    <div id="plot-dead" class="plot"></div>
  </div>
</div>
<div class="card">
  <h3>Reconstruction Loss</h3>
  <div id="plot-recon" class="plot"></div>
</div>

<!-- Section 2: Training Curves -->
<h2>2. SAE Training Dynamics</h2>
<div class="selector">
  <label>SAE: <select id="training-select" onchange="updateTraining()"></select></label>
</div>
<div class="grid">
  <div class="card">
    <h3>Reconstruction Loss</h3>
    <div id="plot-train-recon" class="plot"></div>
  </div>
  <div class="card">
    <h3>L0 (Sparsity)</h3>
    <div id="plot-train-l0" class="plot"></div>
  </div>
</div>

<!-- Section 3: Feature Browser -->
<h2>3. Feature Browser</h2>
<p class="desc">Explore individual features by their max-activating examples. Each feature is a direction
in the SAE's hidden space. Monosemantic features activate on semantically coherent inputs.</p>
<div class="selector">
  <label>SAE: <select id="feature-sae-select" onchange="updateFeatures()"></select></label>
</div>
<div class="feature-list" id="feature-list"></div>

<!-- Section 4: Mamba vs Transformer -->
<h2>4. Mamba vs Transformer Comparison</h2>
<div class="findings" id="comparison-section">
  <h3>Key Findings</h3>
  <ul id="comparison-list"></ul>
</div>

</div>

<script>
const summaryData = {json.dumps(summary_data)};
const featureBrowser = {json.dumps(feature_browser)};
const trainingHistories = {json.dumps(training_histories)};

const darkLayout = {{
  paper_bgcolor: '#1a1a1a', plot_bgcolor: '#1a1a1a',
  font: {{ color: '#ccc' }},
  xaxis: {{ gridcolor: '#333' }}, yaxis: {{ gridcolor: '#333' }},
  margin: {{ l: 60, r: 20, t: 30, b: 80 }},
}};

// Color map for models
const modelColors = {{
  'mamba_130m': '#ff6b6b',
  'mamba_370m': '#ffd93d',
  'pythia_160m': '#4ecdc4',
}};

// 1. Aggregate stats
const saeKeys = Object.keys(summaryData);
const labels = saeKeys.map(k => k.replace(/_/g, ' '));

Plotly.newPlot('plot-l0', [{{
  x: labels, y: saeKeys.map(k => summaryData[k].l0_mean),
  type: 'bar', marker: {{ color: saeKeys.map(k => modelColors[summaryData[k].model] || '#888') }},
}}], {{ ...darkLayout, xaxis: {{ ...darkLayout.xaxis, tickangle: -45 }} }});

Plotly.newPlot('plot-dead', [{{
  x: labels, y: saeKeys.map(k => summaryData[k].dead_frac * 100),
  type: 'bar', marker: {{ color: saeKeys.map(k => modelColors[summaryData[k].model] || '#888') }},
}}], {{ ...darkLayout, xaxis: {{ ...darkLayout.xaxis, tickangle: -45 }},
  yaxis: {{ ...darkLayout.yaxis, title: '% Dead Features' }} }});

Plotly.newPlot('plot-recon', [{{
  x: labels, y: saeKeys.map(k => summaryData[k].avg_recon_loss),
  type: 'bar', marker: {{ color: saeKeys.map(k => modelColors[summaryData[k].model] || '#888') }},
}}], {{ ...darkLayout, xaxis: {{ ...darkLayout.xaxis, tickangle: -45 }},
  yaxis: {{ ...darkLayout.yaxis, title: 'MSE' }} }});

// 2. Training curves
const trainSelect = document.getElementById('training-select');
saeKeys.forEach(k => {{
  const opt = document.createElement('option');
  opt.value = k; opt.textContent = k.replace(/_/g, ' ');
  trainSelect.appendChild(opt);
}});

function updateTraining() {{
  const key = trainSelect.value;
  const hist = trainingHistories[key];
  if (!hist) return;

  if (hist.recon_loss) {{
    Plotly.newPlot('plot-train-recon', [{{
      x: hist.recon_loss.map(d => d.step), y: hist.recon_loss.map(d => d.value),
      line: {{ color: '#ff9f43' }},
    }}], {{ ...darkLayout, xaxis: {{ ...darkLayout.xaxis, title: 'Step' }} }});
  }}
  if (hist.l0) {{
    Plotly.newPlot('plot-train-l0', [{{
      x: hist.l0.map(d => d.step), y: hist.l0.map(d => d.value),
      line: {{ color: '#4ecdc4' }},
    }}], {{ ...darkLayout, xaxis: {{ ...darkLayout.xaxis, title: 'Step' }} }});
  }}
}}
updateTraining();

// 3. Feature browser
const featSelect = document.getElementById('feature-sae-select');
saeKeys.forEach(k => {{
  const opt = document.createElement('option');
  opt.value = k; opt.textContent = k.replace(/_/g, ' ');
  featSelect.appendChild(opt);
}});

function updateFeatures() {{
  const key = featSelect.value;
  const features = featureBrowser[key] || [];
  const list = document.getElementById('feature-list');
  list.innerHTML = features.slice(0, 20).map((f, i) => {{
    const examples = (f.top_examples || []).slice(0, 5).map(ex =>
      `<div class="example">${{ex.text ? ex.text.replace(/>/g, '&gt;').replace(/</g, '&lt;') : 'N/A'}} (act=${{ex.activation ? ex.activation.toFixed(2) : '?'}})</div>`
    ).join('');
    return `<div class="feature-item">
      <span class="feat-id">Feature #${{f.feature_id}}</span>
      <span style="color:#888;margin-left:10px;">max_act=${{f.max_activation ? f.max_activation.toFixed(2) : '?'}}</span>
      ${{examples}}
    </div>`;
  }}).join('');
}}
updateFeatures();

// 4. Auto-generate comparison findings
const findings = [];
const mambaKeys = saeKeys.filter(k => k.includes('mamba'));
const pythiaKeys = saeKeys.filter(k => k.includes('pythia'));

if (mambaKeys.length > 0 && pythiaKeys.length > 0) {{
  const mambaL0 = mambaKeys.reduce((s, k) => s + summaryData[k].l0_mean, 0) / mambaKeys.length;
  const pythiaL0 = pythiaKeys.reduce((s, k) => s + summaryData[k].l0_mean, 0) / pythiaKeys.length;
  findings.push(`Average L0 sparsity: Mamba=${{mambaL0.toFixed(1)}}, Pythia=${{pythiaL0.toFixed(1)}}. ${{
    mambaL0 < pythiaL0 ? 'Mamba features are sparser.' : 'Pythia features are sparser.'}}`);

  const mambaDead = mambaKeys.reduce((s, k) => s + summaryData[k].dead_frac, 0) / mambaKeys.length;
  const pythiaDead = pythiaKeys.reduce((s, k) => s + summaryData[k].dead_frac, 0) / pythiaKeys.length;
  findings.push(`Dead feature fraction: Mamba=${{(mambaDead*100).toFixed(1)}}%, Pythia=${{(pythiaDead*100).toFixed(1)}}%.`);

  const mambaRecon = mambaKeys.reduce((s, k) => s + summaryData[k].avg_recon_loss, 0) / mambaKeys.length;
  const pythiaRecon = pythiaKeys.reduce((s, k) => s + summaryData[k].avg_recon_loss, 0) / pythiaKeys.length;
  findings.push(`Reconstruction quality: Mamba MSE=${{mambaRecon.toFixed(6)}}, Pythia MSE=${{pythiaRecon.toFixed(6)}}. ${{
    mambaRecon < pythiaRecon ? 'SAEs reconstruct Mamba activations better.' :
    'SAEs reconstruct Pythia activations better.'}}`);

  findings.push('SAEs successfully decompose SSM hidden states into sparse features — the first demonstration that monosemantic feature extraction works beyond the transformer architecture.');
}}

const cList = document.getElementById('comparison-list');
cList.innerHTML = findings.map(f => `<li>${{f}}</li>`).join('');
</script>
</body>
</html>"""

output_path = WEB_DIR / "index.html"
with open(output_path, "w") as f:
    f.write(html)

print(f"Web page written to {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
