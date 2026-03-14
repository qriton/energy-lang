"""
Energy Language UI — Interactive basin surgery and visualization.

Inspect, inject, remove, and move basins in trained Hopfield energy landscapes.
The first tool toward "energy minima as a programming language."

Usage:
    python ui.py
    python ui.py --share
    python ui.py --checkpoint path/to/model.pt
"""

import os
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr

from qriton_hlm import (
    find_basins, compute_energy, poly_interaction,
    inject_basin, remove_basin, move_basin,
    verify_basin_exists, load_W_from_checkpoint,
)
from qriton_hlm.theme import qriton_theme, qriton_css, QRITON_JS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -- Available checkpoints ------------------------------------------------

CHECKPOINTS = {
    # -- Language Models --
    'LM: Large+FFN (35.9M, PPL 48.3)': 'D:/HLM2/output-hlm3-large-ffn/model.pt',
    'LM: Large no-FFN (19.15M, PPL 250.7)': 'D:/HLM2/output-hlm3-large/model.pt',
    'LM: Medium (8.9M, PPL 77.7)': 'D:/HLM2/output-hlm3-medium/model.pt',
    'LM: Tiny (1.5M)': 'D:/HLM2/output-hlm3-tiny/model.pt',
    'LM: Baseline Medium (8.9M)': 'D:/HLM2/output-baseline-medium-seed1/model.pt',
    # -- 3D Spatial Models --
    'LIDAR: Segmentation (96.3%)': 'D:/HLM-Spatial/spatial/output-lidar-best/model_best.pt',
    'Medical: Organ Seg (97.7%)': 'D:/HLM-Spatial/spatial/output-medical-seg/model_best.pt',
    'Medical: Anomaly (98.7%)': 'D:/HLM-Spatial/spatial/output-medical-anomaly/model_best.pt',
    'Industrial: Defect (99.5%)': 'D:/HLM-Spatial/spatial/output-industrial-50ep/model_best.pt',
    'Industrial: Safety (100%)': 'D:/HLM-Spatial/spatial/output-safety-50ep/model_best.pt',
    'Predictive: Classify (99.5%)': 'D:/HLM-Spatial/spatial/output-predictive/model_best.pt',
    # -- Audio Models --
    'STT: Speech-to-Text (WER 57.8%)': 'D:/HLM-Audio/output-stt-v5/model_best.pt',
    'TTS: Text-to-Speech (val_loss 0.47)': 'D:/HLM-Audio/output-tts/model_best.pt',
}
# Filter to only existing checkpoints
CHECKPOINTS = {k: v for k, v in CHECKPOINTS.items() if os.path.exists(v)}
if not CHECKPOINTS:
    CHECKPOINTS = {'Random W (dim=256)': '__random_256__'}

# -- Cache ----------------------------------------------------------------

_cache = {}
_w_backups = {}  # {(checkpoint_path, layer): original_W_tensor}
_surgery_state = {}


def get_W(checkpoint, layer):
    """Load and cache a W matrix."""
    key = (checkpoint, layer)
    if key not in _cache:
        if checkpoint.startswith('__random_'):
            dim = int(checkpoint.split('_')[-1].rstrip('_'))
            W = torch.randn(dim, dim, device=DEVICE) * 0.02
            _cache[key] = (W, dim)
        else:
            W, d = load_W_from_checkpoint(checkpoint, layer, DEVICE)
            _cache[key] = (W, d)
    return _cache[key]


def apply_surgery(ckpt_name, layer):
    """Apply surgery W_new to the cached W matrix."""
    if 'W_after' not in _surgery_state:
        return "No surgery result to apply. Run surgery first."
    checkpoint = CHECKPOINTS.get(ckpt_name, '')
    layer = int(layer)
    key = (checkpoint, layer)
    if key not in _w_backups:
        W_orig, d = get_W(checkpoint, layer)
        _w_backups[key] = W_orig.clone()
    W_new = _surgery_state['W_after']
    _cache[key] = (W_new.clone(), W_new.shape[0])
    modified = [l for (cp, l) in _w_backups if cp == checkpoint]
    return (f"**Applied!** {ckpt_name} L{layer} W updated in cache.\n"
            f"Modified layers: {modified}\n"
            f"All subsequent basin operations will use this modified W.")


def restore_surgery(ckpt_name, layer):
    """Restore cached W to original checkpoint value."""
    checkpoint = CHECKPOINTS.get(ckpt_name, '')
    key = (checkpoint, int(layer))
    if key not in _w_backups:
        return f"Layer {layer} has not been modified."
    _cache[key] = (_w_backups[key].clone(), _w_backups[key].shape[0])
    del _w_backups[key]
    remaining = [l for (cp, l) in _w_backups if cp == checkpoint]
    return (f"**Restored** L{layer} to original.\n"
            f"Still modified: {remaining if remaining else 'none'}")


def restore_all_surgery(ckpt_name):
    """Restore ALL modified layers for this checkpoint."""
    checkpoint = CHECKPOINTS.get(ckpt_name, '')
    restored = []
    for (cp, l) in list(_w_backups.keys()):
        if cp == checkpoint:
            _cache[(cp, l)] = (_w_backups[(cp, l)].clone(),
                               _w_backups[(cp, l)].shape[0])
            del _w_backups[(cp, l)]
            restored.append(l)
    if not restored:
        return "No layers were modified."
    return f"**Restored layers {restored}** to original checkpoint values."


def get_num_layers(checkpoint):
    """Count Hopfield layers in checkpoint."""
    if checkpoint.startswith('__random_'):
        return 1
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state', ckpt.get('model', ckpt))
    return len([k for k in state if 'hopfield.W' in k])


# -- Energy landscape computation -----------------------------------------

def compute_2d_landscape(W, center, dir1, dir2, grid_range=2.0, n_points=80):
    """Compute energy on a 2D grid through center along dir1, dir2."""
    alpha = np.linspace(-grid_range, grid_range, n_points)
    beta = np.linspace(-grid_range, grid_range, n_points)
    E = np.zeros((n_points, n_points))

    for i in range(n_points):
        for j in range(n_points):
            x = center + alpha[j] * dir1 + beta[i] * dir2
            E[i, j] = compute_energy(x, W)

    return alpha, beta, E


def get_orthogonal_dirs(basins, d_model):
    """Get two orthogonal directions from basins or random."""
    if len(basins) >= 2:
        d1 = basins[0] / basins[0].norm()
        d2 = basins[1] - (basins[1] @ d1) * d1
        d2 = d2 / d2.norm()
    elif len(basins) == 1:
        d1 = basins[0] / basins[0].norm()
        d2 = torch.randn(d_model, device=basins[0].device)
        d2 = d2 - (d2 @ d1) * d1
        d2 = d2 / d2.norm()
    else:
        d1 = torch.randn(d_model, device=DEVICE)
        d1 = d1 / d1.norm()
        d2 = torch.randn(d_model, device=DEVICE)
        d2 = d2 - (d2 @ d1) * d1
        d2 = d2 / d2.norm()
    return d1, d2


# -- Tab 1: Basin Survey --------------------------------------------------

def survey_basins(checkpoint_name, layer, num_inits, beta):
    """Survey basin structure of a layer."""
    t0 = time.time()
    checkpoint = CHECKPOINTS[checkpoint_name]
    W, d_model = get_W(checkpoint, layer)

    basins, ids, trajectories = find_basins(
        W, d_model, num_inits=num_inits, beta=beta, device=DEVICE)

    elapsed = time.time() - t0

    populations = [ids.count(i) for i in range(len(basins))]
    energies = [compute_energy(b, W) for b in basins]
    conv_iters = [len(t) - 1 for t in trajectories]

    lines = [
        f"## Layer {layer} Basin Survey",
        f"- **{len(basins)} unique basins** from {num_inits} random inits",
        f"- W shape: {W.shape[0]}x{W.shape[1]}",
        f"- W spectral norm: {torch.linalg.norm(W, ord=2).item():.4f}",
        f"- Mean convergence: {np.mean(conv_iters):.1f} iters",
        f"- Time: {elapsed:.1f}s",
        "",
        "| Basin | Energy | Population | Share |",
        "|-------|--------|------------|-------|",
    ]
    for i in range(min(len(basins), 20)):
        pct = 100 * populations[i] / num_inits
        lines.append(f"| {i} | {energies[i]:.4f} | {populations[i]} | {pct:.1f}% |")
    if len(basins) > 20:
        lines.append(f"| ... | ... | ... | ({len(basins)-20} more) |")

    summary = "\n".join(lines)

    # Basin population bar chart
    fig_pop = go.Figure()
    fig_pop.add_trace(go.Bar(
        x=list(range(len(basins))),
        y=populations,
        marker_color=['#2196F3' if p > 1 else '#90CAF9' for p in populations],
        text=[f"E={e:.3f}" for e in energies],
        hovertemplate="Basin %{x}<br>Population: %{y}<br>%{text}<extra></extra>",
    ))
    fig_pop.update_layout(
        title=f"Basin Populations -- Layer {layer} ({len(basins)} basins)",
        xaxis_title="Basin ID", yaxis_title="Inits attracted",
        template="plotly_white", height=400,
    )

    # Energy landscape 2D contour
    d1, d2 = get_orthogonal_dirs(basins, d_model)
    center = torch.zeros(d_model, device=DEVICE)
    alpha, beta_ax, E = compute_2d_landscape(W, center, d1, d2, n_points=60)

    fig_landscape = go.Figure()
    fig_landscape.add_trace(go.Contour(
        x=alpha, y=beta_ax, z=E,
        colorscale='RdYlBu_r', ncontours=40,
        colorbar=dict(title="Energy"),
        hovertemplate="d1=%{x:.2f}<br>d2=%{y:.2f}<br>E=%{z:.4f}<extra></extra>",
    ))

    for i, b in enumerate(basins[:15]):
        bx = float(b @ d1)
        by = float(b @ d2)
        fig_landscape.add_trace(go.Scatter(
            x=[bx], y=[by], mode='markers+text',
            marker=dict(size=12, color='white', line=dict(color='black', width=2)),
            text=[str(i)], textposition='top center',
            textfont=dict(size=10, color='black'),
            showlegend=False,
            hovertemplate=f"Basin {i}<br>E={energies[i]:.4f}<extra></extra>",
        ))

    fig_landscape.update_layout(
        title="Energy Landscape (2D slice through basins)",
        xaxis_title="Direction 1", yaxis_title="Direction 2",
        template="plotly_white", height=500,
    )

    # 3D surface
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Surface(
        x=alpha, y=beta_ax, z=E,
        colorscale='RdYlBu_r', opacity=0.85,
        colorbar=dict(title="Energy"),
        hovertemplate="d1=%{x:.2f}<br>d2=%{y:.2f}<br>E=%{z:.4f}<extra></extra>",
    ))

    for i, b in enumerate(basins[:15]):
        bx = float(b @ d1)
        by = float(b @ d2)
        be = energies[i]
        fig_3d.add_trace(go.Scatter3d(
            x=[bx], y=[by], z=[be],
            mode='markers+text',
            marker=dict(size=6, color='black', symbol='diamond'),
            text=[f"B{i}"], textposition='top center',
            textfont=dict(size=9),
            showlegend=False,
            hovertemplate=f"Basin {i}<br>E={be:.4f}<extra></extra>",
        ))

    fig_3d.update_layout(
        title="3D Energy Surface",
        scene=dict(
            xaxis_title="Direction 1",
            yaxis_title="Direction 2",
            zaxis_title="Energy",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        template="plotly_white", height=600,
    )

    # Energy histogram
    fig_energy = go.Figure()
    fig_energy.add_trace(go.Histogram(
        x=energies, nbinsx=20, marker_color='#4CAF50',
        hovertemplate="Energy: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))
    fig_energy.update_layout(
        title="Basin Energy Distribution",
        xaxis_title="Energy", yaxis_title="Count",
        template="plotly_white", height=300,
    )

    return summary, fig_pop, fig_landscape, fig_3d, fig_energy


# -- Tab 2: All Layers ----------------------------------------------------

def survey_all_layers(checkpoint_name, num_inits, beta):
    """Survey all layers at once."""
    t0 = time.time()
    checkpoint = CHECKPOINTS[checkpoint_name]
    n_layers = get_num_layers(checkpoint)

    layer_data = []
    for l in range(n_layers):
        W, d_model = get_W(checkpoint, l)
        basins, ids, trajs = find_basins(
            W, d_model, num_inits=num_inits, beta=beta, device=DEVICE)
        energies = [compute_energy(b, W) for b in basins]
        conv_iters = [len(t) - 1 for t in trajs]

        learned_beta = None
        if not checkpoint.startswith('__random_'):
            ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
            state = ckpt.get('model_state', ckpt.get('model', ckpt))
            lb_key = f'blocks.{l}.hopfield.log_beta'
            if lb_key in state:
                lb = state[lb_key]
                if lb.dim() == 0:
                    learned_beta = float(torch.exp(lb))
                else:
                    learned_beta = float(torch.exp(lb[0]))

        layer_data.append({
            'layer': l,
            'num_basins': len(basins),
            'min_energy': min(energies) if energies else 0,
            'max_energy': max(energies) if energies else 0,
            'mean_iters': np.mean(conv_iters),
            'learned_beta': learned_beta,
        })

    elapsed = time.time() - t0

    lines = [
        f"## All-Layer Basin Survey ({n_layers} layers)",
        f"- Time: {elapsed:.1f}s",
        "",
        "| Layer | Basins | Energy Range | Mean Iters | Learned Beta |",
        "|-------|--------|-------------|------------|--------------|",
    ]
    for d in layer_data:
        lb = f"{d['learned_beta']:.2f}" if d['learned_beta'] else "--"
        lines.append(
            f"| L{d['layer']} | **{d['num_basins']}** | "
            f"[{d['min_energy']:.2f}, {d['max_energy']:.2f}] | "
            f"{d['mean_iters']:.1f} | {lb} |"
        )

    summary = "\n".join(lines)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"L{d['layer']}" for d in layer_data],
        y=[d['num_basins'] for d in layer_data],
        marker_color='#2196F3',
        text=[str(d['num_basins']) for d in layer_data],
        textposition='outside',
        hovertemplate="Layer %{x}<br>Basins: %{y}<extra></extra>",
    ))

    if any(d['learned_beta'] is not None for d in layer_data):
        betas = [d['learned_beta'] or 0 for d in layer_data]
        fig.add_trace(go.Scatter(
            x=[f"L{d['layer']}" for d in layer_data],
            y=betas,
            name='Learned beta',
            yaxis='y2',
            mode='lines+markers',
            marker=dict(color='#FF5722', size=8),
            line=dict(color='#FF5722', width=2),
        ))

    fig.update_layout(
        title="Basin Count & Learned Beta per Layer",
        xaxis_title="Layer",
        yaxis_title="Unique Basins",
        yaxis2=dict(title="Beta", overlaying='y', side='right',
                    showgrid=False, color='#FF5722'),
        template="plotly_white", height=450,
        legend=dict(x=0.8, y=0.95),
    )

    return summary, fig


# -- Tab 3: Basin Surgery -------------------------------------------------

def do_surgery(checkpoint_name, layer, operation, strength, target_seed, beta):
    """Perform basin surgery and show before/after."""
    t0 = time.time()
    checkpoint = CHECKPOINTS[checkpoint_name]
    W_orig, d_model = get_W(checkpoint, layer)
    W = W_orig.clone()

    basins_before, ids_before, _ = find_basins(
        W, d_model, num_inits=100, beta=beta, device=DEVICE)

    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(target_seed))
    target = torch.randn(d_model, generator=gen).to(DEVICE)
    target = target / target.norm()

    exists_before, _, cos_before, iters_before = verify_basin_exists(
        W, target, beta=beta)

    if operation == "Inject":
        W_new = inject_basin(W, target, strength=strength)
        op_desc = f"Injected basin (strength={strength})"
    elif operation == "Remove closest":
        if basins_before:
            sims = [F.cosine_similarity(target.unsqueeze(0),
                    b.unsqueeze(0)).item() for b in basins_before]
            closest_idx = np.argmax(sims)
            target = basins_before[closest_idx]
            W_new = remove_basin(W, target, strength=strength)
            op_desc = f"Removed basin {closest_idx} (strength={strength})"
        else:
            W_new = W.clone()
            op_desc = "No basins found to remove"
    elif operation == "Move closest":
        if basins_before:
            sims = [F.cosine_similarity(target.unsqueeze(0),
                    b.unsqueeze(0)).item() for b in basins_before]
            closest_idx = np.argmax(sims)
            source = basins_before[closest_idx]
            gen2 = torch.Generator(device='cpu')
            gen2.manual_seed(int(target_seed) + 1000)
            dest = torch.randn(d_model, generator=gen2).to(DEVICE)
            dest = dest / dest.norm()
            W_new = move_basin(W, source, dest, strength=strength)
            target = dest
            op_desc = f"Moved basin {closest_idx} to new location (strength={strength})"
        else:
            W_new = W.clone()
            op_desc = "No basins found to move"

    exists_after, _, cos_after, iters_after = verify_basin_exists(
        W_new, target, beta=beta)
    basins_after, ids_after, _ = find_basins(
        W_new, d_model, num_inits=100, beta=beta, device=DEVICE)

    elapsed = time.time() - t0

    _surgery_state['W_before'] = W
    _surgery_state['W_after'] = W_new
    _surgery_state['target'] = target
    _surgery_state['basins_before'] = basins_before
    _surgery_state['basins_after'] = basins_after
    _surgery_state['d_model'] = d_model

    lines = [
        f"## Surgery Result: {op_desc}",
        f"- Time: {elapsed:.1f}s",
        "",
        "| Metric | Before | After |",
        "|--------|--------|-------|",
        f"| Unique basins | {len(basins_before)} | **{len(basins_after)}** |",
        f"| Target is basin | {'Yes' if exists_before else 'No'} | "
        f"**{'Yes' if exists_after else 'No'}** |",
        f"| Target cos_sim | {cos_before:.4f} | **{cos_after:.4f}** |",
        f"| Convergence iters | {iters_before} | {iters_after} |",
    ]

    delta = len(basins_after) - len(basins_before)
    if operation == "Inject" and exists_after and not exists_before:
        lines.append(f"\n**Basin successfully injected!** (+{delta} net basins)")
    elif operation == "Remove closest" and not exists_after and exists_before:
        lines.append(f"\n**Basin successfully removed!** ({delta} net basins)")
    elif operation == "Move closest":
        lines.append(f"\n**Basin move attempted.** Net change: {delta} basins")

    summary = "\n".join(lines)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["BEFORE Surgery", "AFTER Surgery"])

    pop_before = [ids_before.count(i) for i in range(len(basins_before))]
    pop_after = [ids_after.count(i) for i in range(len(basins_after))]

    fig.add_trace(go.Bar(
        x=list(range(len(basins_before))), y=pop_before,
        marker_color='#90CAF9', name='Before', showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=list(range(len(basins_after))), y=pop_after,
        marker_color='#4CAF50', name='After', showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=f"Basin Populations -- {op_desc}",
        template="plotly_white", height=400,
    )

    return summary, fig


def show_surgery_landscape(grid_res, grid_range):
    """Show before/after energy landscape from last surgery."""
    if 'W_before' not in _surgery_state:
        return (go.Figure().update_layout(title="Run a surgery operation first",
                template="plotly_white"),
                go.Figure().update_layout(title="Run a surgery operation first",
                template="plotly_white"))

    W_before = _surgery_state['W_before']
    W_after = _surgery_state['W_after']
    target = _surgery_state['target']
    d_model = _surgery_state['d_model']
    basins_before = _surgery_state['basins_before']
    basins_after = _surgery_state['basins_after']

    d1 = target / target.norm()
    d2 = torch.randn(d_model, device=DEVICE)
    d2 = d2 - (d2 @ d1) * d1
    d2 = d2 / d2.norm()

    center = torch.zeros(d_model, device=DEVICE)
    n = int(grid_res)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["BEFORE", "AFTER"],
                        horizontal_spacing=0.08)

    for col, W, basins in [(1, W_before, basins_before),
                            (2, W_after, basins_after)]:
        a, b, E = compute_2d_landscape(W, center, d1, d2,
                                        grid_range=grid_range, n_points=n)
        fig.add_trace(go.Contour(
            x=a, y=b, z=E, colorscale='RdYlBu_r', ncontours=40,
            showscale=(col == 2),
            colorbar=dict(title="Energy") if col == 2 else None,
        ), row=1, col=col)

        tx = float(target @ d1)
        ty = float(target @ d2)
        fig.add_trace(go.Scatter(
            x=[tx], y=[ty], mode='markers',
            marker=dict(size=16, symbol='star', color='white',
                        line=dict(color='black', width=2)),
            name='Target' if col == 1 else None,
            showlegend=(col == 1),
        ), row=1, col=col)

        for i, bn in enumerate(basins[:10]):
            bx = float(bn @ d1)
            by = float(bn @ d2)
            fig.add_trace(go.Scatter(
                x=[bx], y=[by], mode='markers',
                marker=dict(size=8, color='white',
                            line=dict(color='black', width=1)),
                showlegend=False,
            ), row=1, col=col)

    fig.update_layout(
        title="Energy Landscape -- Before vs After Surgery (2D)",
        template="plotly_white", height=500,
        legend=dict(x=0.45, y=1.1, orientation='h'),
    )
    fig.update_xaxes(title_text="Target direction", row=1, col=1)
    fig.update_xaxes(title_text="Target direction", row=1, col=2)
    fig.update_yaxes(title_text="Orthogonal", row=1, col=1)

    # 3D surface comparison
    fig_3d = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=["BEFORE (3D)", "AFTER (3D)"],
        horizontal_spacing=0.05,
    )

    for col, W_s, basins_s in [(1, W_before, basins_before),
                                (2, W_after, basins_after)]:
        a, b, E = compute_2d_landscape(W_s, center, d1, d2,
                                        grid_range=grid_range, n_points=n)
        fig_3d.add_trace(go.Surface(
            x=a, y=b, z=E,
            colorscale='RdYlBu_r', opacity=0.85,
            showscale=(col == 2),
            colorbar=dict(title="Energy") if col == 2 else None,
        ), row=1, col=col)

    fig_3d.update_layout(
        title="Energy Surface -- Before vs After Surgery (3D)",
        template="plotly_white", height=550,
    )
    for i in range(1, 3):
        fig_3d.update_scenes(dict(
            xaxis_title="Target dir",
            yaxis_title="Orthogonal",
            zaxis_title="Energy",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ), row=1, col=i)

    return fig, fig_3d


# -- Tab 4: Strength Sweep ------------------------------------------------

def sweep_strength(checkpoint_name, layer, target_seed, beta, num_strengths):
    """Sweep injection strength to find the sweet spot."""
    t0 = time.time()
    checkpoint = CHECKPOINTS[checkpoint_name]
    W, d_model = get_W(checkpoint, layer)

    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(target_seed))
    target = torch.randn(d_model, generator=gen).to(DEVICE)
    target = target / target.norm()

    strengths = np.logspace(-3, 0.5, int(num_strengths))
    results = []

    for s in strengths:
        W_new = inject_basin(W, target, strength=float(s))
        is_basin, _, cos_sim, iters = verify_basin_exists(W_new, target, beta=beta)
        basins, ids, _ = find_basins(
            W_new, d_model, num_inits=50, beta=beta, device=DEVICE)
        results.append({
            'strength': float(s),
            'is_basin': is_basin,
            'cos_sim': cos_sim,
            'iters': iters,
            'num_basins': len(basins),
        })

    elapsed = time.time() - t0

    basins_orig, _, _ = find_basins(W, d_model, num_inits=50, beta=beta, device=DEVICE)
    n_orig = len(basins_orig)

    lines = [
        f"## Injection Strength Sweep",
        f"- Layer {layer}, {len(strengths)} strengths tested",
        f"- Baseline: {n_orig} basins",
        f"- Time: {elapsed:.1f}s",
        "",
        "| Strength | Basin Created | Cos Sim | Iters | Total Basins | Net |",
        "|----------|---------------|---------|-------|--------------|-----|",
    ]
    for r in results:
        net = r['num_basins'] - n_orig
        sign = "+" if net >= 0 else ""
        lines.append(
            f"| {r['strength']:.4f} | "
            f"{'**YES**' if r['is_basin'] else 'no'} | "
            f"{r['cos_sim']:.4f} | {r['iters']} | "
            f"{r['num_basins']} | {sign}{net} |"
        )

    summary = "\n".join(lines)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["Target Cosine Similarity",
                                        "Total Basin Count"],
                        vertical_spacing=0.12)

    strengths_list = [r['strength'] for r in results]
    cos_sims = [r['cos_sim'] for r in results]
    n_basins = [r['num_basins'] for r in results]
    colors = ['#4CAF50' if r['is_basin'] else '#F44336' for r in results]

    fig.add_trace(go.Scatter(
        x=strengths_list, y=cos_sims, mode='lines+markers',
        marker=dict(color=colors, size=8),
        line=dict(color='#2196F3'),
        name='Cos similarity',
    ), row=1, col=1)

    fig.add_hline(y=0.9, line_dash="dash", line_color="gray",
                  annotation_text="Basin threshold", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=strengths_list, y=n_basins, mode='lines+markers',
        marker=dict(color='#FF9800', size=8),
        line=dict(color='#FF9800'),
        name='Total basins',
    ), row=2, col=1)

    fig.add_hline(y=n_orig, line_dash="dash", line_color="gray",
                  annotation_text=f"Baseline ({n_orig})", row=2, col=1)

    fig.update_xaxes(type="log", title_text="Injection Strength", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Injection Strength", row=2, col=1)
    fig.update_yaxes(title_text="Cosine Similarity", row=1, col=1)
    fig.update_yaxes(title_text="Basin Count", row=2, col=1)
    fig.update_layout(template="plotly_white", height=600, showlegend=False)

    return summary, fig


# -- Tab 5: Convergence Trajectories --------------------------------------

def show_trajectories(checkpoint_name, layer, num_inits, beta):
    """Visualize convergence trajectories in 2D projection."""
    t0 = time.time()
    checkpoint = CHECKPOINTS[checkpoint_name]
    W, d_model = get_W(checkpoint, layer)

    basins, ids, trajectories = find_basins(
        W, d_model, num_inits=int(num_inits), beta=beta, device=DEVICE)

    elapsed = time.time() - t0

    d1, d2 = get_orthogonal_dirs(basins, d_model)

    colors = [
        '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
        '#1abc9c', '#e67e22', '#34495e', '#c0392b', '#2980b9',
        '#27ae60', '#d35400', '#8e44ad', '#16a085', '#f1c40f',
    ]

    fig = go.Figure()

    n_show = min(int(num_inits), 50)
    for i in range(n_show):
        traj = trajectories[i]
        basin_id = ids[i]
        xs = [float(t @ d1) for t in traj]
        ys = [float(t @ d2) for t in traj]
        color = colors[basin_id % len(colors)]

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=color, width=1),
            opacity=0.4, showlegend=False,
            hovertemplate=f"Traj {i} -> Basin {basin_id}<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=[xs[0]], y=[ys[0]], mode='markers',
            marker=dict(size=4, color=color, symbol='circle-open'),
            showlegend=False,
        ))

    for i, b in enumerate(basins):
        bx = float(b @ d1)
        by = float(b @ d2)
        pop = ids.count(i)
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=[bx], y=[by], mode='markers+text',
            marker=dict(size=max(8, min(25, pop)), color=color,
                        line=dict(color='black', width=2)),
            text=[f"B{i}"], textposition='top center',
            textfont=dict(size=9),
            name=f"Basin {i} ({pop} inits)",
            hovertemplate=f"Basin {i}<br>Pop: {pop}/{num_inits}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Convergence Trajectories (2D) -- Layer {layer} "
              f"({len(basins)} basins, {elapsed:.1f}s)",
        xaxis_title="Direction 1", yaxis_title="Direction 2",
        template="plotly_white", height=600,
        legend=dict(x=1.02, y=1, font=dict(size=9)),
    )

    # 3D trajectories
    if len(basins) >= 3:
        all_pts = torch.stack(basins)
        mean = all_pts.mean(0, keepdim=True)
        centered = all_pts - mean
        try:
            U, S, V = torch.pca_lowrank(centered, q=3)
            d3 = V[:, 2]
        except Exception:
            d3 = torch.randn(d_model, device=DEVICE)
            d3 = d3 - (d3 @ d1) * d1 - (d3 @ d2) * d2
            d3 = d3 / d3.norm()
    else:
        d3 = torch.randn(d_model, device=DEVICE)
        d3 = d3 - (d3 @ d1) * d1 - (d3 @ d2) * d2
        d3 = d3 / d3.norm()

    fig_3d = go.Figure()
    for i in range(n_show):
        traj = trajectories[i]
        basin_id = ids[i]
        xs = [float(t @ d1) for t in traj]
        ys = [float(t @ d2) for t in traj]
        zs = [float(t @ d3) for t in traj]
        color = colors[basin_id % len(colors)]

        fig_3d.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines',
            line=dict(color=color, width=2),
            opacity=0.5, showlegend=False,
            hovertemplate=f"Traj {i} -> Basin {basin_id}<extra></extra>",
        ))

    for i, b in enumerate(basins):
        bx, by, bz = float(b @ d1), float(b @ d2), float(b @ d3)
        pop = ids.count(i)
        color = colors[i % len(colors)]
        fig_3d.add_trace(go.Scatter3d(
            x=[bx], y=[by], z=[bz], mode='markers+text',
            marker=dict(size=max(4, min(12, pop)), color=color,
                        line=dict(color='black', width=1)),
            text=[f"B{i}"], textposition='top center',
            textfont=dict(size=8),
            name=f"Basin {i} ({pop})",
            hovertemplate=f"Basin {i}<br>Pop: {pop}<extra></extra>",
        ))

    fig_3d.update_layout(
        title=f"3D Convergence Trajectories -- Layer {layer}",
        scene=dict(
            xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        template="plotly_white", height=650,
        legend=dict(x=1.02, y=1, font=dict(size=8)),
    )

    summary = (f"**{len(basins)} basins** from {num_inits} inits. "
               f"Showing {n_show} trajectories. {elapsed:.1f}s.")

    return summary, fig, fig_3d


# -- Build Gradio App -----------------------------------------------------

def build_ui():
    checkpoint_choices = list(CHECKPOINTS.keys())
    default_ckpt = checkpoint_choices[0] if checkpoint_choices else None

    with gr.Blocks(
        title="Energy Language -- Qriton Technologies",
        theme=qriton_theme(),
        css=qriton_css("Energy Language", "Qriton Technologies"),
        js=QRITON_JS,
    ) as demo:

        with gr.Tab("Basin Survey"):
            gr.Markdown("Survey the basin structure of a single layer.")
            with gr.Row():
                ckpt1 = gr.Dropdown(checkpoint_choices, value=default_ckpt,
                                    label="Checkpoint")
                layer1 = gr.Slider(0, 7, value=0, step=1, label="Layer")
                inits1 = gr.Slider(50, 500, value=200, step=50,
                                   label="Random inits")
                beta1 = gr.Slider(1.0, 20.0, value=7.0, step=0.5,
                                  label="Beta")
            btn1 = gr.Button("Survey Basins", variant="primary")
            out1_text = gr.Markdown()
            out1_pop = gr.Plot(label="Basin Populations")
            out1_land = gr.Plot(label="Energy Landscape (2D)")
            out1_3d = gr.Plot(label="Energy Surface (3D)")
            out1_hist = gr.Plot(label="Energy Distribution")
            btn1.click(survey_basins,
                       inputs=[ckpt1, layer1, inits1, beta1],
                       outputs=[out1_text, out1_pop, out1_land, out1_3d, out1_hist])

        with gr.Tab("All Layers"):
            gr.Markdown("Compare basin structure across all layers at once.")
            with gr.Row():
                ckpt2 = gr.Dropdown(checkpoint_choices, value=default_ckpt,
                                    label="Checkpoint")
                inits2 = gr.Slider(50, 300, value=100, step=50,
                                   label="Random inits per layer")
                beta2 = gr.Slider(1.0, 20.0, value=7.0, step=0.5,
                                  label="Beta")
            btn2 = gr.Button("Survey All Layers", variant="primary")
            out2_text = gr.Markdown()
            out2_fig = gr.Plot(label="Basin Count per Layer")
            btn2.click(survey_all_layers,
                       inputs=[ckpt2, inits2, beta2],
                       outputs=[out2_text, out2_fig])

        with gr.Tab("Basin Surgery"):
            gr.Markdown("""
**Inject**, **remove**, or **move** basins in the energy landscape.

This is the core experiment: can we *program* the energy landscape directly?
            """)
            with gr.Row():
                ckpt3 = gr.Dropdown(checkpoint_choices, value=default_ckpt,
                                    label="Checkpoint")
                layer3 = gr.Slider(0, 7, value=0, step=1, label="Layer")
                op3 = gr.Radio(["Inject", "Remove closest", "Move closest"],
                               value="Inject", label="Operation")
            with gr.Row():
                strength3 = gr.Slider(0.01, 2.0, value=0.1, step=0.01,
                                      label="Strength")
                seed3 = gr.Number(value=42, label="Target seed",
                                  precision=0)
                beta3 = gr.Slider(1.0, 20.0, value=7.0, step=0.5,
                                  label="Beta")

            btn3 = gr.Button("Perform Surgery", variant="primary")
            out3_text = gr.Markdown()
            out3_pop = gr.Plot(label="Before/After Populations")

            gr.Markdown("### Energy Landscape Comparison")
            with gr.Row():
                grid_res = gr.Slider(30, 100, value=60, step=10,
                                     label="Grid resolution")
                grid_range = gr.Slider(1.0, 4.0, value=2.0, step=0.5,
                                       label="Grid range")
            btn3b = gr.Button("Show Landscape")
            out3_land = gr.Plot(label="Before vs After (2D)")
            out3_land3d = gr.Plot(label="Before vs After (3D)")

            btn3.click(do_surgery,
                       inputs=[ckpt3, layer3, op3, strength3, seed3, beta3],
                       outputs=[out3_text, out3_pop])
            btn3b.click(show_surgery_landscape,
                        inputs=[grid_res, grid_range],
                        outputs=[out3_land, out3_land3d])

            gr.Markdown("---")
            gr.Markdown("### Apply / Restore")
            gr.Markdown("*Apply the surgery result to the cached W matrix. "
                        "All subsequent operations on this layer will use the modified W.*")
            with gr.Row():
                apply_btn = gr.Button("Apply Surgery", variant="primary")
                restore_btn = gr.Button("Restore This Layer", variant="secondary")
                restore_all_btn = gr.Button("Restore All", variant="stop")
            apply_status = gr.Markdown()
            apply_btn.click(apply_surgery,
                            inputs=[ckpt3, layer3], outputs=[apply_status])
            restore_btn.click(restore_surgery,
                              inputs=[ckpt3, layer3], outputs=[apply_status])
            restore_all_btn.click(restore_all_surgery,
                                  inputs=[ckpt3], outputs=[apply_status])

        with gr.Tab("Strength Sweep"):
            gr.Markdown("""
Find the optimal injection strength -- strong enough to create a basin,
weak enough to preserve existing structure.
            """)
            with gr.Row():
                ckpt4 = gr.Dropdown(checkpoint_choices, value=default_ckpt,
                                    label="Checkpoint")
                layer4 = gr.Slider(0, 7, value=0, step=1, label="Layer")
                seed4 = gr.Number(value=42, label="Target seed",
                                  precision=0)
            with gr.Row():
                beta4 = gr.Slider(1.0, 20.0, value=7.0, step=0.5,
                                  label="Beta")
                n_strengths = gr.Slider(5, 30, value=15, step=1,
                                        label="Number of strengths")
            btn4 = gr.Button("Run Sweep", variant="primary")
            out4_text = gr.Markdown()
            out4_fig = gr.Plot(label="Strength vs Basin Creation")
            btn4.click(sweep_strength,
                       inputs=[ckpt4, layer4, seed4, beta4, n_strengths],
                       outputs=[out4_text, out4_fig])

        with gr.Tab("Trajectories"):
            gr.Markdown("""
Visualize how random initial states converge to basins.
Each color = one basin. Lines = convergence paths. Size = population.
            """)
            with gr.Row():
                ckpt5 = gr.Dropdown(checkpoint_choices, value=default_ckpt,
                                    label="Checkpoint")
                layer5 = gr.Slider(0, 7, value=0, step=1, label="Layer")
                inits5 = gr.Slider(10, 200, value=50, step=10,
                                   label="Random inits")
                beta5 = gr.Slider(1.0, 20.0, value=7.0, step=0.5,
                                  label="Beta")
            btn5 = gr.Button("Show Trajectories", variant="primary")
            out5_text = gr.Markdown()
            out5_fig = gr.Plot(label="Convergence Trajectories (2D)")
            out5_3d = gr.Plot(label="Convergence Trajectories (3D)")
            btn5.click(show_trajectories,
                       inputs=[ckpt5, layer5, inits5, beta5],
                       outputs=[out5_text, out5_fig, out5_3d])

        with gr.Tab("About"):
            gr.Markdown("""
## Energy Language -- Limbaj Energetic

**Three paradigms of neural network control:**

| Paradigm | How you change behavior | Examples |
|----------|------------------------|----------|
| Statistical | Train on data | PyTorch, TensorFlow, JAX |
| Programmatic | Write rules | Expert systems, symbolic AI |
| **Energetic** | **Shape the energy landscape** | **Qriton HLM** |

### What this UI does

**Basin Survey** -- Count and visualize basins in trained models. Each basin
is a stable attractor in the energy landscape -- a "subroutine" the model
learned by optimizing W.

**Basin Surgery** -- The key experiment. Can we *inject* a new basin at a
chosen point? *Remove* an existing one? *Move* one to a new location?
If yes, we can *program* energy landscapes directly.

**Strength Sweep** -- Find the Goldilocks zone: strong enough to create
a basin, weak enough to preserve existing structure.

**Trajectories** -- Watch computation happen. Each trajectory is a random
initial state settling into a basin. The settling IS the computation.

### The math

```
Energy:    E(x) = -1/d * sum|x_i|^d + 1/2 * x^T W f(x)
Update:    x_{t+1} = (1-tau)*x + tau*tanh(beta * W @ f(x))
Injection: W_new = W + s * F(target) @ target^T
```

### Trained models available

This UI can inspect any trained HLM checkpoint:
- **Language models** (HLM3-Large: PPL 48.3, Medium: PPL 77.7)
- **3D Spatial** (LIDAR 96.3%, Medical 97.7%, Industrial 99.5%)
- **Audio** (STT, TTS)
- **Predictive** (99.5% classification)

All models share the same HLM3Block Hopfield core -- only the frontend
and task head change. The energy landscape is the same math everywhere.

### Install

```bash
pip install qriton-hlm
```

### Credits

Built on polynomial Hopfield networks (Krotov & Hopfield, 2016)
and the HLM architecture (Qriton Technologies, 2026).
            """)

    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int, default=7861)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name='0.0.0.0', server_port=args.port,
                share=args.share)
