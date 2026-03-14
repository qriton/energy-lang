# Qriton HLM — Examples

## Jupyter Notebooks

| Notebook | What it covers |
|----------|---------------|
| `01_quickstart.ipynb` | Survey, inject, verify, remove, restore — synthetic W, no model needed |
| `02_concept_surgery.ipynb` | Capture concepts from text, inject, blend, export/import |
| `03_landscape_visualization.ipynb` | Interactive Plotly plots: energy levels, PCA projection, surgery diffs |
| `04_model_comparison.ipynb` | Compare basins between two models, transplant concepts |
| `05_hlm_scripts.ipynb` | IPython magics (`%hlm`, `%%hlm`), `.hlm` script files |

## Python Scripts

| Script | What it does |
|--------|-------------|
| `example_basic.py` | Minimal surgery demo — works without a checkpoint |
| `example_concepts.py` | Full concept pipeline: capture → blend → inject → benchmark |
| `example_landscape.py` | Per-layer topology analysis + surgery tolerance test |
| `example_probe.py` | Reverse-engineer basins: what tokens does each basin activate? |

## HLM Scripts

| Script | What it does |
|--------|-------------|
| `make_polite.hlm` | Capture polite language → inject → test |
| `audit_model.hlm` | Full landscape audit with safety guards |
| `blend_persona.hlm` | Blend polite + technical + concise → custom persona |

## Quick start

```bash
# Install
pip install qriton-hlm[jupyter]

# Run basic example (no checkpoint needed)
python examples/example_basic.py

# Run HLM script
qriton-hlm --script examples/make_polite.hlm

# Interactive REPL
qriton-hlm --checkpoint model.pt

# Jupyter
jupyter lab notebooks/01_quickstart.ipynb
```
