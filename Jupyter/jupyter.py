"""
qriton_hlm.jupyter — IPython magics & rich display for Jupyter.

Usage:
    %load_ext qriton_hlm.jupyter

    surgeon = %hlm_load model.pt
    %hlm survey 0
    %hlm inject 0 42 0.1
    %%hlm
    capture 5 polite Thank you so much
    capture 5 polite I really appreciate it
    inject-concept 5 polite 0.1
    apply 5
"""

import json
import time
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic, line_cell_magic
from IPython.display import display, HTML, JSON
from IPython import get_ipython

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import ipywidgets as widgets
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

from qriton_hlm.core import BasinSurgeon, compute_energy, find_basins
from qriton_hlm.cli import EnergyLang

# ── Visualization helpers ────────────────────────────────────────────

def plot_landscape(survey_result, title=None):
    """Render basin landscape as interactive Plotly chart."""
    if not HAS_PLOTLY:
        print("pip install plotly for visualizations")
        return

    basins = survey_result['basins']
    energies = survey_result['energies']
    populations = survey_result['populations']
    layer = survey_result['layer']

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Basin Energy Levels', 'Basin Population'),
        column_widths=[0.6, 0.4]
    )

    colors = [f'hsl({i * 360 // len(basins)}, 70%, 50%)' for i in range(len(basins))]

    fig.add_trace(go.Bar(
        x=[f'B{i}' for i in range(len(basins))],
        y=energies,
        marker_color=colors,
        name='Energy',
        hovertemplate='Basin %{x}<br>Energy: %{y:.4f}<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=[f'B{i}' for i in range(len(basins))],
        y=populations,
        marker_color=colors,
        name='Population',
        hovertemplate='Basin %{x}<br>Population: %{y}<extra></extra>',
    ), row=1, col=2)

    fig.update_layout(
        title=title or f'Layer {layer} — {len(basins)} basins',
        template='plotly_dark',
        showlegend=False,
        height=400,
        margin=dict(t=60, b=40, l=40, r=20),
    )
    fig.update_yaxes(title_text='Energy', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=2)

    fig.show()


def plot_landscape_2d(landscape_result, title=None):
    """Render 2D PCA projection of basin locations."""
    if not HAS_PLOTLY:
        print("pip install plotly for visualizations")
        return

    coords = landscape_result.get('coords_2d')
    if coords is None:
        print("Not enough basins for 2D projection")
        return

    basins = landscape_result['basins']
    layer = landscape_result['layer']

    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    energies = [b['energy'] for b in basins]
    populations = [b['population'] for b in basins]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(
            size=[max(8, p * 3) for p in populations],
            color=energies,
            colorscale='Viridis',
            colorbar=dict(title='Energy'),
            line=dict(width=1, color='white'),
        ),
        text=[f'B{b["idx"]}<br>E={b["energy"]:.4f}<br>pop={b["population"]}'
              for b in basins],
        hovertemplate='%{text}<extra></extra>',
    ))

    fig.update_layout(
        title=title or f'Layer {layer} — Energy Landscape (PCA projection)',
        template='plotly_dark',
        xaxis_title='PC1',
        yaxis_title='PC2',
        height=500,
        margin=dict(t=60, b=40, l=40, r=20),
    )
    fig.show()


def plot_surgery_diff(before_survey, after_survey, operation='surgery'):
    """Compare basin landscape before/after surgery."""
    if not HAS_PLOTLY:
        print("pip install plotly for visualizations")
        return

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Before ({len(before_survey["basins"])} basins)',
            f'After ({len(after_survey["basins"])} basins)',
        ),
    )

    for col, survey in enumerate([before_survey, after_survey], 1):
        energies = survey['energies']
        n = len(energies)
        colors = [f'hsl({i * 360 // max(n, 1)}, 70%, 50%)' for i in range(n)]
        fig.add_trace(go.Bar(
            x=[f'B{i}' for i in range(n)],
            y=energies,
            marker_color=colors,
            hovertemplate='Basin %{x}<br>Energy: %{y:.4f}<extra></extra>',
        ), row=1, col=col)

    delta = len(after_survey['basins']) - len(before_survey['basins'])
    sign = '+' if delta >= 0 else ''
    fig.update_layout(
        title=f'{operation}: {sign}{delta} basins',
        template='plotly_dark',
        showlegend=False,
        height=400,
        margin=dict(t=60, b=40, l=40, r=20),
    )
    fig.show()


def plot_convergence(trajectory):
    """Plot energy over convergence iterations."""
    if not HAS_PLOTLY:
        print("pip install plotly for visualizations")
        return

    if isinstance(trajectory[0], dict):
        steps = [t['step'] for t in trajectory]
        energies = [t['energy'] for t in trajectory]
        deltas = [t.get('delta', 0) for t in trajectory]
    else:
        return

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Energy', 'Delta (convergence)'),
                        vertical_spacing=0.15)

    fig.add_trace(go.Scatter(
        x=steps, y=energies, mode='lines+markers',
        name='Energy', line=dict(color='#00d4aa'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=steps, y=deltas, mode='lines+markers',
        name='Delta', line=dict(color='#ff6b6b'),
    ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark', height=500, showlegend=False,
        margin=dict(t=60, b=40, l=40, r=20),
    )
    fig.show()


def plot_concept_space(concepts, surgeon, layer=0):
    """Visualize captured concepts in 2D via PCA."""
    if not HAS_PLOTLY:
        print("pip install plotly for visualizations")
        return

    import torch
    import numpy as np

    all_states = []
    labels = []
    for name, concept in concepts.items():
        for s in concept['states']:
            all_states.append(s.cpu().numpy())
            labels.append(name)
        if concept['centroid'] is not None:
            all_states.append(concept['centroid'].cpu().numpy())
            labels.append(f'{name} [centroid]')

    if len(all_states) < 2:
        print("Need at least 2 concept samples for visualization")
        return

    matrix = np.stack(all_states)
    centered = matrix - matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ Vt[:2].T

    unique_names = list(set(n.replace(' [centroid]', '') for n in labels))
    color_map = {n: f'hsl({i * 360 // len(unique_names)}, 70%, 50%)'
                 for i, n in enumerate(unique_names)}

    fig = go.Figure()
    for i, (label, coord) in enumerate(zip(labels, coords)):
        is_centroid = '[centroid]' in label
        base_name = label.replace(' [centroid]', '')
        fig.add_trace(go.Scatter(
            x=[coord[0]], y=[coord[1]],
            mode='markers+text' if is_centroid else 'markers',
            marker=dict(
                size=15 if is_centroid else 8,
                color=color_map[base_name],
                symbol='diamond' if is_centroid else 'circle',
                line=dict(width=2, color='white') if is_centroid else dict(width=0),
            ),
            text=label if is_centroid else None,
            textposition='top center',
            name=label,
            showlegend=is_centroid,
        ))

    fig.update_layout(
        title='Concept Space (PCA projection)',
        template='plotly_dark',
        height=500,
        xaxis_title='PC1', yaxis_title='PC2',
        margin=dict(t=60, b=40, l=40, r=20),
    )
    fig.show()


# ── HTML display for text results ────────────────────────────────────

def _styled_output(text, title=None):
    """Render CLI output as styled HTML."""
    escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # Highlight keywords
    for kw in ['basin', 'Basin', 'MODIFIED', 'injected', 'removed', 'programmed']:
        escaped = escaped.replace(kw, f'<span style="color:#00d4aa">{kw}</span>')
    for kw in ['Error', 'GONE', 'missing', 'Failed']:
        escaped = escaped.replace(kw, f'<span style="color:#ff6b6b">{kw}</span>')

    header = f'<div style="color:#888;font-size:11px;margin-bottom:4px">{title}</div>' if title else ''

    html = f"""
    <div style="background:#1a1a2e;border:1px solid #333;border-radius:6px;
                padding:12px;font-family:'JetBrains Mono',monospace;font-size:13px;
                color:#e0e0e0;margin:4px 0;overflow-x:auto">
        {header}
        <pre style="margin:0;white-space:pre-wrap">{escaped}</pre>
    </div>
    """
    display(HTML(html))


# ── IPython Magics ───────────────────────────────────────────────────

@magics_class
class HLMMagics(Magics):
    """IPython magics for qriton-hlm Energy Language."""

    def __init__(self, shell):
        super().__init__(shell)
        self._lang = EnergyLang()
        self._surgeon = None

    @line_magic
    def hlm_load(self, line):
        """Load a checkpoint: %hlm_load model.pt"""
        result = self._lang.load_checkpoint(line.strip())
        _styled_output(result, 'load')

        # Also create a BasinSurgeon for the Python API
        path = line.strip().strip('"').strip("'")
        self._surgeon = BasinSurgeon.from_checkpoint(path)
        self.shell.user_ns['surgeon'] = self._surgeon
        self.shell.user_ns['hlm_lang'] = self._lang
        return self._surgeon

    @line_cell_magic
    def hlm(self, line, cell=None):
        """Execute HLM commands.

        Line magic:  %hlm survey 0
        Cell magic:  %%hlm
                     survey 0
                     inject 0 42 0.1
        """
        if cell is not None:
            commands = cell.strip().split('\n')
        else:
            commands = [line.strip()]

        for cmd in commands:
            cmd = cmd.strip()
            if not cmd or cmd.startswith('#'):
                continue
            result = self._lang.execute(cmd)
            if result and result != '__EXIT__':
                _styled_output(result, cmd)

    @line_magic
    def hlm_survey(self, line):
        """Survey with visualization: %hlm_survey 0"""
        layer = int(line.strip()) if line.strip() else 0
        result = self._lang.execute(f'survey {layer}')
        _styled_output(result, f'survey {layer}')

        if self._surgeon:
            survey = self._surgeon.survey(layer)
            plot_landscape(survey)

    @line_magic
    def hlm_landscape(self, line):
        """Full landscape visualization: %hlm_landscape 0"""
        layer = int(line.strip()) if line.strip() else 0

        if self._surgeon:
            landscape = self._surgeon.landscape(layer)
            _styled_output(
                f"Layer {layer}: {landscape['num_basins']} basins\n"
                f"Energy range: [{landscape['energy_range'][0]:.4f}, "
                f"{landscape['energy_range'][1]:.4f}]",
                f'landscape {layer}'
            )
            plot_landscape_2d(landscape)
        else:
            result = self._lang.execute(f'landscape {layer}')
            _styled_output(result, f'landscape {layer}')

    @line_magic
    def hlm_concepts(self, line):
        """Visualize concept space: %hlm_concepts"""
        if self._surgeon and self._surgeon._concepts:
            result = self._lang.execute('concepts')
            _styled_output(result, 'concepts')
            plot_concept_space(self._surgeon._concepts, self._surgeon)
        else:
            _styled_output("No concepts captured yet.", 'concepts')


# ── Interactive widget (optional) ────────────────────────────────────

def basin_explorer(surgeon, layer=0):
    """Interactive basin explorer widget for Jupyter."""
    if not HAS_WIDGETS:
        print("pip install ipywidgets for interactive explorer")
        return

    layer_slider = widgets.IntSlider(
        value=layer, min=0, max=max(surgeon.num_layers() - 1, 0),
        description='Layer:', style={'description_width': '50px'},
    )
    inits_slider = widgets.IntSlider(
        value=200, min=50, max=1000, step=50,
        description='Inits:', style={'description_width': '50px'},
    )
    output = widgets.Output()

    def on_survey(b):
        with output:
            output.clear_output()
            surgeon.params['inits'] = inits_slider.value
            survey = surgeon.survey(layer_slider.value)
            plot_landscape(survey)

    btn = widgets.Button(description='Survey', button_style='primary')
    btn.on_click(on_survey)

    return widgets.VBox([
        widgets.HBox([layer_slider, inits_slider, btn]),
        output,
    ])


# ── Registration ─────────────────────────────────────────────────────

def load_ipython_extension(ipython):
    """Called by %load_ext qriton_hlm.jupyter"""
    ipython.register_magics(HLMMagics)
    print("Qriton HLM magics loaded. Commands: %hlm, %hlm_load, %hlm_survey, %hlm_landscape, %hlm_concepts")
