"""
qriton_hlm.core — Basin surgery for polynomial Hopfield networks.

No external dependencies beyond torch and numpy.
Works with any checkpoint that has `hopfield.W` parameters.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F


# ── Polynomial interaction ───────────────────────────────────────────

def poly_interaction(x, degree=3):
    """Polynomial interaction function for Hopfield dynamics."""
    return torch.sign(x) * torch.abs(x) ** (degree - 1)


# ── Energy computation ──────────────────────────────────────────────

def compute_energy(x, W, degree=3):
    """Compute Hopfield energy at state x. Lower = more stable."""
    fx = poly_interaction(x, degree)
    kinetic = -1.0 / degree * torch.sum(torch.abs(x) ** degree)
    potential = 0.5 * x @ W @ fx
    return (kinetic + potential).item()


# ── Basin discovery ──────────────────────────────────────────────────

def _converge(x, W, degree=3, max_iter=100, tau_start=0.9, tau_end=0.1,
              beta=7.0, eps=5e-3, record=False):
    """Run Krotov-Hopfield dynamics to convergence. Returns (state, iters, trajectory)."""
    trajectory = [x.clone()] if record else None
    for t in range(max_iter):
        tau = tau_start + (tau_end - tau_start) * (t / max_iter)
        fx = poly_interaction(x, degree)
        h = W @ fx
        x_new = (1 - tau) * x + tau * torch.tanh(beta * h)
        if record:
            trajectory.append(x_new.clone())
        delta = (x_new - x).norm() / (x.norm() + 1e-8)
        if delta < eps:
            break
        x = x_new
    return x, t + 1, trajectory


def find_basins(W, d_model, degree=3, num_inits=200, max_iter=100,
                tau_start=0.9, tau_end=0.1, beta=7.0, eps=5e-3,
                device='cpu'):
    """Discover basins by converging from random initial states.

    Returns: (unique_basins, basin_ids, trajectories)
    """
    basins = []
    trajectories = []

    for i in range(num_inits):
        x = torch.randn(d_model, device=device) * 0.5

        x, iters, traj = _converge(x, W, degree, max_iter, tau_start,
                                    tau_end, beta, eps, record=True)
        basins.append(x.detach())
        trajectories.append(traj)

    # Cluster basins by cosine similarity (skip near-zero states)
    unique_basins = []
    basin_ids = []
    for b in basins:
        if b.norm() < 1e-6:
            basin_ids.append(-1)
            continue
        found = False
        for j, ub in enumerate(unique_basins):
            cos_sim = F.cosine_similarity(
                b.unsqueeze(0), ub.unsqueeze(0)).item()
            if cos_sim > 0.95:
                basin_ids.append(j)
                found = True
                break
        if not found:
            basin_ids.append(len(unique_basins))
            unique_basins.append(b)

    return unique_basins, basin_ids, trajectories


# ── Basin surgery operations ─────────────────────────────────────────

def inject_basin(W, target, degree=3, strength=1.0, beta=7.0):
    """Inject/strengthen a basin near target using Hebbian learning.

    First converges from target to find the nearest natural basin,
    then strengthens that basin with a Hebbian outer-product update.
    strength=1.0 adds a perturbation with Frobenius norm equal to W's.

    For concept injection (target from capture()), the target is
    already at the right scale. For unit-normalized targets (from
    random seeds), auto-scales to the dynamics' operating range.
    """
    d = W.shape[0]

    # If target is near unit-norm (from random seed), scale up to
    # the dynamics' operating range before converging
    t = target.clone()
    if t.norm() < 2.0:
        t = t / t.norm() * (0.5 * d ** 0.5)

    # Converge from scaled target to find the nearest natural basin
    basin, _, _ = _converge(t, W, degree, beta=beta)

    # If basin collapsed to zero, use sign pattern at natural scale
    if basin.norm() < 1e-6:
        direction = target / target.norm().clamp(min=1e-8)
        basin = torch.sign(direction) * 0.9

    ft = poly_interaction(basin, degree)
    outer = torch.outer(basin, ft)
    outer_norm = outer.norm()
    if outer_norm < 1e-10:
        return W

    # Scale so delta_W has Frobenius norm = strength * ||W||_F
    w_norm = W.norm().clamp(min=1e-10)
    alpha = strength * w_norm / outer_norm

    return W + alpha * outer


def remove_basin(W, target, degree=3, strength=1.0, beta=7.0):
    """Remove/weaken a basin near target.

    Converges to the nearest basin and applies anti-Hebbian update.
    """
    d = W.shape[0]
    t = target.clone()
    if t.norm() < 2.0:
        t = t / t.norm().clamp(min=1e-8) * (0.5 * d ** 0.5)

    basin, _, _ = _converge(t, W, degree, beta=beta)
    if basin.norm() < 1e-6:
        return W  # nothing to remove

    ft = poly_interaction(basin, degree)
    outer = torch.outer(basin, ft)
    outer_norm = outer.norm()
    if outer_norm < 1e-10:
        return W

    w_norm = W.norm().clamp(min=1e-10)
    alpha = strength * w_norm / outer_norm

    return W - alpha * outer


def move_basin(W, source, destination, degree=3, strength=1.0, beta=7.0):
    """Move a basin from source to destination.

    Equivalent to remove(source) + inject(destination).
    """
    W = remove_basin(W, source, degree, strength, beta)
    W = inject_basin(W, destination, degree, strength, beta)
    return W


# ── Verification ─────────────────────────────────────────────────────

def verify_basin_exists(W, target, degree=3, max_iter=100,
                        tau_start=0.9, tau_end=0.1, beta=7.0,
                        eps=5e-3, cos_threshold=0.9):
    """Verify that a basin exists near target.

    Perturbs the target state by 10% (relative to its norm) and checks
    if convergence returns to target (cos_sim > threshold).

    Returns: (is_basin: bool, final_state, cos_similarity, num_iters)
    """
    # Relative perturbation — 10% of target's scale
    perturb_scale = max(target.norm().item() * 0.1, 0.1)
    x = target + perturb_scale * torch.randn_like(target)

    x, iters, _ = _converge(x, W, degree, max_iter, tau_start,
                             tau_end, beta, eps)

    cos_sim = F.cosine_similarity(
        x.unsqueeze(0), target.unsqueeze(0)).item()
    return cos_sim > cos_threshold, x, cos_sim, iters


# ── Checkpoint loading ───────────────────────────────────────────────

def load_W_from_checkpoint(checkpoint_path, layer=0, device='cpu'):
    """Extract weight matrix W from any checkpoint with hopfield.W keys.

    Works with HLM2, HLM3, HLM-Spatial, HLM-Audio, or any model
    that uses PolyHopfieldLayer.

    Returns: (W tensor, d_model int)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state', ckpt.get('model', ckpt))

    w_key = f'blocks.{layer}.hopfield.W'
    if w_key not in state:
        for key in state:
            if f'.{layer}.' in key and '.W' in key and 'hopfield' in key:
                w_key = key
                break
        else:
            available = [k for k in state if 'W' in k and 'hopfield' in k]
            raise KeyError(f"W not found at {w_key}. Available: {available}")

    W = state[w_key].to(device)
    return W, W.shape[0]


def count_hopfield_layers(checkpoint_path):
    """Count number of Hopfield layers in a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state', ckpt.get('model', ckpt))
    return len([k for k in state if 'hopfield.W' in k])


# ── BasinSurgeon — high-level API ────────────────────────────────────

class BasinSurgeon:
    """High-level API for energy landscape programming.

    Example:
        surgeon = BasinSurgeon.from_checkpoint("model.pt", device="cuda")
        print(surgeon.survey(layer=0))
        surgeon.inject(layer=0, seed=42, strength=0.1)
        surgeon.apply(layer=0)
        print(surgeon.diff(layer=0))
        surgeon.restore(layer=0)
    """

    def __init__(self, device='cpu'):
        self.device = device
        self._w_cache = {}       # layer -> current W
        self._w_backups = {}     # layer -> original W
        self._layer_betas = {}   # layer -> learned beta value
        self._checkpoint_path = None
        self._model = None       # optional: full model for text gen
        self._tokenizer = None
        self._config = None
        self._history = []

        self._concepts = {}     # name -> {'states': [...], 'centroid': tensor}
        self.params = {
            'beta': 7.0,
            'inits': 200,
            'strength': 1.0,
            'degree': 3,
        }

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        """Create a BasinSurgeon from a checkpoint file."""
        surgeon = cls(device=device)
        surgeon._checkpoint_path = path
        return surgeon

    @classmethod
    def from_W(cls, W, device='cpu'):
        """Create a BasinSurgeon from a raw W matrix."""
        surgeon = cls(device=device)
        surgeon._w_cache[0] = W.to(device)
        return surgeon

    def get_W(self, layer=0):
        """Get W matrix for a layer."""
        if layer in self._w_cache:
            return self._w_cache[layer]
        if self._checkpoint_path:
            W, _ = load_W_from_checkpoint(
                self._checkpoint_path, layer, self.device)
            self._w_cache[layer] = W
            return W
        raise ValueError(f"No W available for layer {layer}")

    def num_layers(self):
        if self._checkpoint_path:
            return count_hopfield_layers(self._checkpoint_path)
        return len(self._w_cache)

    def get_beta(self, layer=0):
        """Get beta for a layer (from checkpoint or default)."""
        if layer in self._layer_betas:
            return self._layer_betas[layer]
        # Try to load from checkpoint
        if self._checkpoint_path:
            import math
            ckpt = torch.load(self._checkpoint_path, map_location='cpu',
                              weights_only=False)
            state = ckpt.get('model_state', ckpt.get('model', ckpt))
            for key in state:
                if f'.{layer}.' in key and 'log_beta' in key:
                    log_beta = state[key].float().mean().item()
                    beta = math.exp(log_beta)
                    self._layer_betas[layer] = beta
                    return beta
        return self.params['beta']

    def _make_target(self, seed, d_model):
        gen = torch.Generator(device='cpu')
        gen.manual_seed(int(seed))
        target = torch.randn(d_model, generator=gen).to(self.device)
        return target / target.norm()

    def survey(self, layer=0):
        """Survey basins of a layer. Returns dict with results."""
        W = self.get_W(layer)
        d = W.shape[0]
        beta = self.get_beta(layer)
        t0 = time.time()
        basins, ids, trajectories = find_basins(
            W, d, num_inits=int(self.params['inits']),
            beta=beta, device=self.device)
        elapsed = time.time() - t0

        results = {
            'layer': layer,
            'num_basins': len(basins),
            'num_inits': int(self.params['inits']),
            'basins': basins,
            'basin_ids': ids,
            'trajectories': trajectories,
            'elapsed': elapsed,
            'energies': [compute_energy(b, W) for b in basins],
            'populations': [ids.count(i) for i in range(len(basins))],
        }
        self._history.append(('survey', layer, results['num_basins']))
        return results

    def _to_basin_scale(self, target, W, beta):
        """Scale a unit-normalized target to the dynamics' operating range
        and converge to find the nearest natural basin."""
        d = W.shape[0]
        t = target.clone()
        if t.norm() < 2.0:
            t = t / t.norm().clamp(min=1e-8) * (0.5 * d ** 0.5)
        basin, _, _ = _converge(t, W, beta=beta)
        if basin.norm() < 1e-6:
            return t  # dynamics collapsed, return scaled target
        return basin

    def inject(self, layer=0, seed=42, strength=None):
        """Inject a new basin. Returns dict with before/after stats."""
        strength = strength or self.params['strength']
        W = self.get_W(layer)
        d = W.shape[0]
        beta = self.get_beta(layer)
        target = self._make_target(seed, d)

        # Find the nearest basin to this direction
        basin_before = self._to_basin_scale(target, W, beta)
        exists_before, _, cos_before, _ = verify_basin_exists(
            W, basin_before, beta=beta)

        W_new = inject_basin(W, target, strength=strength, beta=beta)

        # After injection, find the basin again on the new W
        basin_after = self._to_basin_scale(target, W_new, beta)
        exists_after, _, cos_after, _ = verify_basin_exists(
            W_new, basin_after, beta=beta)

        self._w_cache[layer] = W_new
        self._history.append(('inject', layer, seed, strength))

        return {
            'layer': layer, 'seed': seed, 'strength': strength,
            'existed_before': exists_before, 'cos_before': cos_before,
            'exists_after': exists_after, 'cos_after': cos_after,
            'W_before': W, 'W_after': W_new, 'target': target,
        }

    def remove(self, layer=0, seed=42, strength=None):
        """Remove closest basin to seed target."""
        strength = strength or self.params['strength']
        W = self.get_W(layer)
        d = W.shape[0]
        beta = self.get_beta(layer)
        target = self._make_target(seed, d)

        basins, _, _ = find_basins(
            W, d, num_inits=100, beta=beta, device=self.device)
        if basins:
            sims = [F.cosine_similarity(target.unsqueeze(0), b.unsqueeze(0)).item()
                    for b in basins]
            target = basins[int(np.argmax(sims))]

        W_new = remove_basin(W, target, strength=strength, beta=beta)
        exists_after, _, cos_after, _ = verify_basin_exists(
            W_new, target, beta=beta)

        self._w_cache[layer] = W_new
        self._history.append(('remove', layer, seed, strength))

        return {
            'layer': layer, 'exists_after': exists_after,
            'cos_after': cos_after, 'W_after': W_new,
        }

    def move(self, layer=0, seed=42, strength=None):
        """Move closest basin to a new location."""
        strength = strength or self.params['strength']
        W = self.get_W(layer)
        d = W.shape[0]
        beta = self.get_beta(layer)

        source_target = self._make_target(seed, d)
        basins, _, _ = find_basins(
            W, d, num_inits=100, beta=beta, device=self.device)
        source = source_target
        if basins:
            sims = [F.cosine_similarity(source_target.unsqueeze(0), b.unsqueeze(0)).item()
                    for b in basins]
            source = basins[int(np.argmax(sims))]

        dest = self._make_target(int(seed) + 1000, d)
        W_new = move_basin(W, source, dest, strength=strength, beta=beta)
        self._w_cache[layer] = W_new
        self._history.append(('move', layer, seed, strength))
        return {'layer': layer, 'W_after': W_new}

    def verify(self, layer=0, seed=42):
        """Check if a basin exists near the target seed direction."""
        W = self.get_W(layer)
        d = W.shape[0]
        beta = self.get_beta(layer)
        target = self._make_target(seed, d)
        basin = self._to_basin_scale(target, W, beta)
        exists, final, cos, iters = verify_basin_exists(
            W, basin, beta=beta)
        energy = compute_energy(final, W)
        return {
            'is_basin': exists, 'cos': cos, 'iters': iters, 'energy': energy,
        }

    def apply(self, layer=0):
        """Apply cached W to a live model (if loaded).

        For standalone usage, this is a no-op since surgery already
        modifies the cached W. For HLM3 models loaded with load_model(),
        this writes W back to model.blocks[layer].hopfield.W.data.
        """
        if self._model is not None and hasattr(self._model, 'blocks'):
            W_new = self.get_W(layer)
            W_live = self._model.blocks[layer].hopfield.W
            if layer not in self._w_backups:
                self._w_backups[layer] = W_live.data.clone()
            with torch.no_grad():
                W_live.data.copy_(W_new)
            self._history.append(('apply', layer))
            return True
        return False

    def restore(self, layer=0):
        """Restore a layer to its original checkpoint W."""
        if layer in self._w_backups:
            if self._model and hasattr(self._model, 'blocks'):
                with torch.no_grad():
                    self._model.blocks[layer].hopfield.W.data.copy_(
                        self._w_backups[layer])
            self._w_cache[layer] = self._w_backups[layer].clone()
            del self._w_backups[layer]
            self._history.append(('restore', layer))
            return True
        # If no backup but we have checkpoint, reload from disk
        if self._checkpoint_path and layer in self._w_cache:
            W, _ = load_W_from_checkpoint(
                self._checkpoint_path, layer, self.device)
            self._w_cache[layer] = W
            self._history.append(('restore', layer))
            return True
        return False

    def restore_all(self):
        """Restore all layers."""
        restored = []
        for l in list(self._w_backups.keys()):
            self.restore(l)
            restored.append(l)
        return restored

    def capture(self, layer, text, concept_name=None):
        """Run text through the model and capture the converged Hopfield state.

        This is how you extract what a concept "looks like" in the energy
        landscape. The converged state at a given layer IS the concept's
        representation — the basin the input falls into.

        Args:
            layer: which Hopfield layer to capture from
            text: input text (requires model + tokenizer loaded)
            concept_name: if given, accumulate into a named concept

        Returns: dict with converged state, energy, basin info
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("capture requires a loaded model. Use load_model() first.")

        # Tokenize
        tokens = self._tokenizer.encode(text)
        input_ids = torch.tensor([tokens], device=self.device)

        # Hook to capture the Hopfield state after convergence
        captured = {}
        def hook_fn(module, input, output):
            if isinstance(output, dict) and 'state' in output:
                captured['state'] = output['state'].detach()
            elif isinstance(output, tuple):
                # PolyHopfieldLayer.forward() returns (settled, energy, iters)
                captured['state'] = output[0].detach()
            elif isinstance(output, torch.Tensor):
                captured['state'] = output.detach()

        block = self._model.blocks[layer]
        handle = block.hopfield.register_forward_hook(hook_fn)

        with torch.no_grad():
            self._model(input_ids)

        handle.remove()

        if 'state' not in captured:
            raise RuntimeError(f"Could not capture state from layer {layer}")

        # Average across sequence positions to get a single concept vector
        state = captured['state']
        if state.dim() == 3:
            state = state.squeeze(0).mean(dim=0)  # [seq, dim] -> [dim]
        elif state.dim() == 2:
            state = state.mean(dim=0)
        # Keep natural scale — don't normalize to unit norm.
        # The state's magnitude reflects the dynamics' operating range.

        W = self.get_W(layer)
        beta = self.get_beta(layer)
        energy = compute_energy(state, W)
        is_basin, _, cos, iters = verify_basin_exists(
            W, state, beta=beta)

        result = {
            'state': state,
            'energy': energy,
            'is_basin': is_basin,
            'cos_similarity': cos,
            'converge_iters': iters,
            'text': text,
            'layer': layer,
        }

        # Accumulate into named concept
        if concept_name:
            if concept_name not in self._concepts:
                self._concepts[concept_name] = {'states': [], 'centroid': None}
            self._concepts[concept_name]['states'].append(state)
            # Recompute centroid (average states, preserving natural scale)
            states = torch.stack(self._concepts[concept_name]['states'])
            centroid = states.mean(dim=0)
            self._concepts[concept_name]['centroid'] = centroid
            result['concept'] = concept_name
            result['num_samples'] = len(self._concepts[concept_name]['states'])

        self._history.append(('capture', layer, text[:40], concept_name))
        return result

    def inject_concept(self, layer, concept_name, strength=None):
        """Inject a captured concept as a new basin.

        First capture examples with capture(), then inject the averaged
        concept vector as a new attractor in the energy landscape.

        Args:
            layer: target layer
            concept_name: name of previously captured concept
            strength: injection strength (default: self.params['strength'])
        """
        if concept_name not in self._concepts:
            raise ValueError(f"Unknown concept '{concept_name}'. "
                           f"Use capture() first. Known: {list(self._concepts.keys())}")

        concept = self._concepts[concept_name]
        if concept['centroid'] is None:
            raise ValueError(f"Concept '{concept_name}' has no samples yet")

        strength = strength or self.params['strength']
        target = concept['centroid'].to(self.device)
        W = self.get_W(layer)
        beta = self.get_beta(layer)

        exists_before, _, cos_before, _ = verify_basin_exists(
            W, target, beta=beta)
        W_new = inject_basin(W, target, strength=strength, beta=beta)
        exists_after, _, cos_after, _ = verify_basin_exists(
            W_new, target, beta=beta)

        self._w_cache[layer] = W_new
        self._history.append(('inject_concept', layer, concept_name, strength))

        return {
            'concept': concept_name,
            'num_samples': len(concept['states']),
            'layer': layer,
            'strength': strength,
            'existed_before': exists_before,
            'cos_before': cos_before,
            'exists_after': exists_after,
            'cos_after': cos_after,
        }

    def remove_concept(self, layer, concept_name, strength=None):
        """Remove a concept's basin from the energy landscape."""
        if concept_name not in self._concepts:
            raise ValueError(f"Unknown concept '{concept_name}'")

        strength = strength or self.params['strength']
        target = self._concepts[concept_name]['centroid'].to(self.device)
        W = self.get_W(layer)
        beta = self.get_beta(layer)

        W_new = remove_basin(W, target, strength=strength, beta=beta)
        exists_after, _, cos_after, _ = verify_basin_exists(
            W_new, target, beta=beta)

        self._w_cache[layer] = W_new
        self._history.append(('remove_concept', layer, concept_name, strength))

        return {
            'concept': concept_name,
            'exists_after': exists_after,
            'cos_after': cos_after,
        }

    def list_concepts(self):
        """List all captured concepts."""
        return {name: len(c['states']) for name, c in self._concepts.items()}

    def transplant(self, source_surgeon, layer, concept_name, strength=None):
        """Copy a concept from another model into this one.

        Args:
            source_surgeon: BasinSurgeon with the concept captured
            layer: target layer
            concept_name: concept to transplant
            strength: injection strength
        """
        if concept_name not in source_surgeon._concepts:
            raise ValueError(f"Concept '{concept_name}' not found in source model")

        concept = source_surgeon._concepts[concept_name]
        # Copy concept to this surgeon
        self._concepts[concept_name] = {
            'states': [s.to(self.device) for s in concept['states']],
            'centroid': concept['centroid'].to(self.device),
        }
        return self.inject_concept(layer, concept_name, strength)

    def blend(self, concept_a, concept_b, new_name, ratio=0.5):
        """Blend two concepts into a new one.

        Args:
            concept_a: first concept name
            concept_b: second concept name
            new_name: name for blended concept
            ratio: weight for concept_a (1-ratio for concept_b)
        """
        if concept_a not in self._concepts or concept_b not in self._concepts:
            raise ValueError(f"Both concepts must exist. Have: {list(self._concepts.keys())}")

        ca = self._concepts[concept_a]['centroid']
        cb = self._concepts[concept_b]['centroid']
        blended = ratio * ca + (1 - ratio) * cb
        blended = blended / blended.norm()

        self._concepts[new_name] = {
            'states': [],
            'centroid': blended,
        }
        self._history.append(('blend', concept_a, concept_b, new_name, ratio))
        return {'concept': new_name, 'ratio': f"{ratio:.0%} {concept_a} + {1-ratio:.0%} {concept_b}"}

    def strengthen(self, layer=0, seed=42, factor=2.0):
        """Strengthen an existing basin (make it deeper/more attractive)."""
        W = self.get_W(layer)
        d = W.shape[0]
        beta = self.get_beta(layer)
        target = self._make_target(seed, d)

        # Find closest actual basin
        basins, _, _ = find_basins(W, d, num_inits=100,
                                   beta=beta, device=self.device)
        if basins:
            sims = [F.cosine_similarity(target.unsqueeze(0), b.unsqueeze(0)).item()
                    for b in basins]
            target = basins[int(np.argmax(sims))]

        strength = self.params['strength'] * factor
        W_new = inject_basin(W, target, strength=strength, beta=beta)
        self._w_cache[layer] = W_new
        self._history.append(('strengthen', layer, seed, factor))

        e_before = compute_energy(target, W)
        e_after = compute_energy(target, W_new)
        return {
            'layer': layer, 'energy_before': e_before, 'energy_after': e_after,
            'deepened_by': e_before - e_after,
        }

    def weaken(self, layer=0, seed=42, factor=0.5):
        """Weaken an existing basin (make it shallower)."""
        W = self.get_W(layer)
        d = W.shape[0]
        beta = self.get_beta(layer)
        target = self._make_target(seed, d)

        basins, _, _ = find_basins(W, d, num_inits=100,
                                   beta=beta, device=self.device)
        if basins:
            sims = [F.cosine_similarity(target.unsqueeze(0), b.unsqueeze(0)).item()
                    for b in basins]
            target = basins[int(np.argmax(sims))]

        strength = self.params['strength'] * factor
        W_new = remove_basin(W, target, strength=strength, beta=beta)
        self._w_cache[layer] = W_new
        self._history.append(('weaken', layer, seed, factor))

        e_before = compute_energy(target, W)
        e_after = compute_energy(target, W_new)
        return {
            'layer': layer, 'energy_before': e_before, 'energy_after': e_after,
            'raised_by': e_after - e_before,
        }

    def trace(self, layer, text):
        """Show the convergence trajectory for an input at a given layer.

        Returns the full path from initial state to final basin,
        with energy at each step.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("trace requires a loaded model")

        tokens = self._tokenizer.encode(text)
        input_ids = torch.tensor([tokens], device=self.device)

        trajectory = []
        def hook_fn(module, inp, output):
            if hasattr(module, '_trajectory'):
                trajectory.extend(module._trajectory)

        block = self._model.blocks[layer]
        handle = block.hopfield.register_forward_hook(hook_fn)

        with torch.no_grad():
            self._model(input_ids)
        handle.remove()

        # If no trajectory from hook, compute manually
        W = self.get_W(layer)
        beta = self.get_beta(layer)
        if not trajectory:
            # Use the last hidden state as starting point
            target = self._make_target(hash(text) % 2**31, W.shape[0])
            x = target.clone()
            for t in range(100):
                tau = 0.9 + (0.1 - 0.9) * (t / 100)
                fx = poly_interaction(x, self.params['degree'])
                h = W @ fx
                x_new = (1 - tau) * x + tau * torch.tanh(beta * h)
                trajectory.append({
                    'step': t,
                    'energy': compute_energy(x_new, W),
                    'delta': (x_new - x).norm().item(),
                })
                if (x_new - x).norm().item() / (x.norm().item() + 1e-8) < 5e-3:
                    break
                x = x_new

        self._history.append(('trace', layer, text[:40]))
        return {'layer': layer, 'text': text, 'steps': len(trajectory), 'trajectory': trajectory}

    def energy(self, layer, seed=42):
        """Measure energy at a specific point."""
        W = self.get_W(layer)
        d = W.shape[0]
        beta = self.get_beta(layer)
        target = self._make_target(seed, d)
        basin = self._to_basin_scale(target, W, beta)
        e = compute_energy(basin, W)
        is_basin, _, cos, iters = verify_basin_exists(W, basin, beta=beta)
        return {'energy': e, 'is_basin': is_basin, 'cos': cos, 'iters': iters}

    def export_concept(self, concept_name, path):
        """Export a concept as a portable file."""
        if concept_name not in self._concepts:
            raise ValueError(f"Unknown concept '{concept_name}'")
        concept = self._concepts[concept_name]
        data = {
            'name': concept_name,
            'centroid': concept['centroid'].cpu(),
            'num_samples': len(concept['states']),
            'states': [s.cpu() for s in concept['states']],
        }
        torch.save(data, path)
        self._history.append(('export_concept', concept_name, path))
        return {'path': path, 'concept': concept_name, 'samples': len(concept['states'])}

    def import_concept(self, path):
        """Import a concept from a file."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        name = data['name']
        self._concepts[name] = {
            'centroid': data['centroid'].to(self.device),
            'states': [s.to(self.device) for s in data.get('states', [])],
        }
        self._history.append(('import_concept', name, path))
        return {'concept': name, 'samples': len(self._concepts[name]['states'])}

    def batch_capture(self, layer, concept_name, texts):
        """Capture multiple examples for a concept at once."""
        results = []
        for text in texts:
            r = self.capture(layer, text, concept_name)
            results.append(r)
        return {
            'concept': concept_name,
            'total_samples': len(self._concepts[concept_name]['states']),
            'new_samples': len(texts),
        }

    def compare(self, other_surgeon, layer=0):
        """Compare basins between this model and another."""
        W_self = self.get_W(layer)
        W_other = other_surgeon.get_W(layer)
        d = W_self.shape[0]
        beta_self = self.get_beta(layer)
        beta_other = other_surgeon.get_beta(layer)

        basins_self, _, _ = find_basins(W_self, d, num_inits=100,
                                        beta=beta_self, device=self.device)
        basins_other, _, _ = find_basins(W_other, d, num_inits=100,
                                         beta=beta_other, device=other_surgeon.device)

        # Find shared vs unique basins
        shared, only_self, only_other = 0, 0, 0
        matched = set()
        for i, bs in enumerate(basins_self):
            found = False
            for j, bo in enumerate(basins_other):
                if j not in matched:
                    cos = F.cosine_similarity(bs.unsqueeze(0), bo.unsqueeze(0).to(self.device)).item()
                    if cos > 0.9:
                        shared += 1
                        matched.add(j)
                        found = True
                        break
            if not found:
                only_self += 1
        only_other = len(basins_other) - shared

        return {
            'layer': layer,
            'basins_self': len(basins_self),
            'basins_other': len(basins_other),
            'shared': shared,
            'only_self': only_self,
            'only_other': only_other,
        }

    def diff(self, layer=0):
        """Show W diff stats vs original."""
        if layer not in self._w_backups:
            if not self._checkpoint_path:
                return None
            W_orig, _ = load_W_from_checkpoint(
                self._checkpoint_path, layer, self.device)
        else:
            W_orig = self._w_backups[layer]

        W_curr = self.get_W(layer)
        delta = W_curr - W_orig
        return {
            'frobenius': delta.norm().item(),
            'max_abs': delta.abs().max().item(),
            'mean_abs': delta.abs().mean().item(),
            'spectral_orig': torch.linalg.norm(W_orig, ord=2).item(),
            'spectral_curr': torch.linalg.norm(W_curr, ord=2).item(),
            'relative_pct': delta.norm().item() / W_orig.norm().item() * 100,
        }

    def probe(self, layer, basin_idx=0, num_tokens=20):
        """Given a basin, generate text that activates it (reverse of capture).

        Finds what input converges to this basin by using the basin state
        as initial hidden state and projecting through the model's output head.
        """
        if self._model is None:
            raise RuntimeError("probe requires a loaded model")

        W = self.get_W(layer)
        d = W.shape[0]
        beta = self.get_beta(layer)
        basins, _, _ = find_basins(W, d, num_inits=int(self.params['inits']),
                                   beta=beta, device=self.device)
        if basin_idx >= len(basins):
            raise ValueError(f"Basin {basin_idx} not found. Have {len(basins)} basins.")

        basin_state = basins[basin_idx]
        energy = compute_energy(basin_state, W)

        # Try to decode the basin through the model's output head
        tokens = []
        if hasattr(self._model, 'lm_head'):
            with torch.no_grad():
                # Project basin state through output head
                logits = self._model.lm_head(basin_state.unsqueeze(0))
                top_ids = torch.topk(logits, min(num_tokens, logits.shape[-1]), dim=-1).indices[0]
                if self._tokenizer:
                    tokens = [self._tokenizer.decode([tid.item()]) for tid in top_ids]

        self._history.append(('probe', layer, basin_idx))
        return {
            'layer': layer,
            'basin_idx': basin_idx,
            'energy': energy,
            'top_tokens': tokens,
            'state_norm': basin_state.norm().item(),
        }

    def landscape(self, layer, resolution=50):
        """Map the full energy landscape of a layer.

        Samples the energy at many points and returns a 2D projection
        (via PCA of basin locations) for visualization.
        """
        W = self.get_W(layer)
        d = W.shape[0]
        beta = self.get_beta(layer)
        basins, ids, _ = find_basins(W, d, num_inits=int(self.params['inits']),
                                     beta=beta, device=self.device)

        # Compute energies for all basins
        basin_data = []
        for i, b in enumerate(basins):
            e = compute_energy(b, W)
            pop = ids.count(i)
            basin_data.append({
                'idx': i, 'energy': e, 'population': pop,
                'norm': b.norm().item(),
            })

        # 2D projection via PCA if enough basins
        coords_2d = None
        if len(basins) >= 2:
            basin_matrix = torch.stack(basins).cpu().numpy()
            mean = basin_matrix.mean(axis=0)
            centered = basin_matrix - mean
            try:
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                coords_2d = (centered @ Vt[:2].T).tolist()
            except Exception:
                pass

        self._history.append(('landscape', layer))
        return {
            'layer': layer,
            'num_basins': len(basins),
            'basins': basin_data,
            'coords_2d': coords_2d,
            'energy_range': (min(b['energy'] for b in basin_data),
                           max(b['energy'] for b in basin_data)) if basin_data else (0, 0),
        }

    def guard(self, layer, max_remove_pct=10.0):
        """Set guard constraints for a layer.

        Prevents removing more than max_remove_pct% of basins.
        Returns current basin count and threshold.
        """
        W = self.get_W(layer)
        d = W.shape[0]
        beta = self.get_beta(layer)
        basins, _, _ = find_basins(W, d, num_inits=int(self.params['inits']),
                                   beta=beta, device=self.device)
        threshold = max(1, int(len(basins) * (1 - max_remove_pct / 100)))

        if not hasattr(self, '_guards'):
            self._guards = {}
        self._guards[layer] = {
            'baseline_count': len(basins),
            'min_count': threshold,
            'max_remove_pct': max_remove_pct,
        }
        self._history.append(('guard', layer, max_remove_pct))
        return {
            'layer': layer,
            'current_basins': len(basins),
            'min_allowed': threshold,
            'max_remove_pct': max_remove_pct,
        }

    # ── Causal discovery & intervention ─────────────────────────────

    def causal_scan(self, layer=0, strength=None, num_inits=50, threshold=0.15):
        """Discover causal links between basins by systematic knockout.

        For each basin: temporarily remove it, re-survey, measure which
        other basins shifted. If basin A's removal causes basin B to move
        (cosine drift > threshold), there's a causal link A → B.

        Returns a directed causal graph as an adjacency dict + edge list.
        """
        strength = strength or self.params['strength']
        W_orig = self.get_W(layer).clone()
        d = W_orig.shape[0]
        beta = self.get_beta(layer)

        # Baseline: discover all basins on original W
        basins, ids, _ = find_basins(
            W_orig, d, num_inits=num_inits, beta=beta, device=self.device)
        n = len(basins)

        if n < 2:
            return {
                'layer': layer, 'num_basins': n, 'edges': [],
                'adjacency': {}, 'basins': basins,
            }

        # For each basin, knock it out and measure drift on all others
        edges = []
        adjacency = {i: [] for i in range(n)}

        for i in range(n):
            # Remove basin i
            W_knocked = remove_basin(W_orig, basins[i], strength=strength, beta=beta)

            # Check how all other basins changed
            for j in range(n):
                if i == j:
                    continue

                # Does basin j still exist after removing basin i?
                exists, final_j, cos_j, _ = verify_basin_exists(
                    W_knocked, basins[j], beta=beta)

                drift = 1.0 - cos_j  # 0 = no change, 1 = completely different

                if drift > threshold:
                    edge = {
                        'source': i,
                        'target': j,
                        'drift': drift,
                        'target_survived': exists,
                        'type': 'destroyed' if not exists else 'shifted',
                    }
                    edges.append(edge)
                    adjacency[i].append(j)

            # Restore original W in cache
            self._w_cache[layer] = W_orig.clone()

        self._history.append(('causal_scan', layer, n, len(edges)))

        return {
            'layer': layer,
            'num_basins': n,
            'edges': edges,
            'adjacency': adjacency,
            'basins': basins,
        }

    def causal_intervene(self, layer=0, basin_idx=0, operation='remove',
                          strength=None, num_inits=50):
        """Perform a do-operator intervention on a specific basin.

        do(X) = remove or replace basin X, then measure the full
        downstream effect on the landscape.

        Returns before/after basin counts, which basins were affected,
        and the energy landscape delta.
        """
        strength = strength or self.params['strength']
        W_orig = self.get_W(layer).clone()
        d = W_orig.shape[0]
        beta = self.get_beta(layer)

        # Before
        basins_before, ids_before, _ = find_basins(
            W_orig, d, num_inits=num_inits, beta=beta, device=self.device)

        if basin_idx >= len(basins_before):
            raise ValueError(f"Basin {basin_idx} not found. Have {len(basins_before)}.")

        target = basins_before[basin_idx]
        e_target_before = compute_energy(target, W_orig)

        # Intervene
        if operation == 'remove':
            W_after = remove_basin(W_orig, target, strength=strength, beta=beta)
        elif operation == 'weaken':
            W_after = remove_basin(W_orig, target, strength=strength * 0.5, beta=beta)
        elif operation == 'strengthen':
            W_after = inject_basin(W_orig, target, strength=strength, beta=beta)
        else:
            raise ValueError(f"Unknown operation: {operation}. Use: remove, weaken, strengthen")

        # After
        basins_after, ids_after, _ = find_basins(
            W_after, d, num_inits=num_inits, beta=beta, device=self.device)

        # Measure effect on each original basin
        affected = []
        for j, b in enumerate(basins_before):
            if j == basin_idx:
                continue
            exists, final, cos, _ = verify_basin_exists(W_after, b, beta=beta)
            drift = 1.0 - cos
            if drift > 0.05:  # any measurable change
                affected.append({
                    'basin': j,
                    'drift': drift,
                    'survived': exists,
                    'energy_before': compute_energy(b, W_orig),
                    'energy_after': compute_energy(final, W_after),
                })

        # Apply the intervention (leave W modified)
        self._w_cache[layer] = W_after

        self._history.append(('causal_intervene', layer, basin_idx, operation))

        return {
            'layer': layer,
            'basin_idx': basin_idx,
            'operation': operation,
            'basins_before': len(basins_before),
            'basins_after': len(basins_after),
            'target_energy_before': e_target_before,
            'affected': affected,
            'num_affected': len(affected),
        }

    def causal_counterfactual(self, layer=0, basin_idx=0, modification='invert',
                               strength=None, num_inits=50):
        """Counterfactual: "What if basin X had been different?"

        Creates a modified copy of basin X, injects it, and measures
        downstream effects. Then restores original state.

        Modifications:
          'invert'  — flip the basin (negate the state vector)
          'weaken'  — reduce basin depth by 50%
          'shift'   — move basin to a random nearby location

        Returns the counterfactual effects WITHOUT persisting the change.
        """
        strength = strength or self.params['strength']
        W_orig = self.get_W(layer).clone()
        d = W_orig.shape[0]
        beta = self.get_beta(layer)

        basins, _, _ = find_basins(
            W_orig, d, num_inits=num_inits, beta=beta, device=self.device)

        if basin_idx >= len(basins):
            raise ValueError(f"Basin {basin_idx} not found. Have {len(basins)}.")

        original_basin = basins[basin_idx]

        # Create counterfactual version
        if modification == 'invert':
            cf_basin = -original_basin
        elif modification == 'weaken':
            cf_basin = original_basin * 0.5
        elif modification == 'shift':
            noise = torch.randn_like(original_basin) * 0.3
            cf_basin = original_basin + noise
            cf_basin = cf_basin / cf_basin.norm() * original_basin.norm()
        else:
            raise ValueError(f"Unknown modification: {modification}. Use: invert, weaken, shift")

        # Apply counterfactual: remove original, inject modified
        W_cf = remove_basin(W_orig, original_basin, strength=strength, beta=beta)
        W_cf = inject_basin(W_cf, cf_basin, strength=strength, beta=beta)

        # Measure effects
        basins_cf, _, _ = find_basins(
            W_cf, d, num_inits=num_inits, beta=beta, device=self.device)

        affected = []
        for j, b in enumerate(basins):
            if j == basin_idx:
                continue
            exists, final, cos, _ = verify_basin_exists(W_cf, b, beta=beta)
            drift = 1.0 - cos
            if drift > 0.05:
                affected.append({
                    'basin': j,
                    'drift': drift,
                    'survived': exists,
                })

        # DO NOT persist — restore original W
        self._w_cache[layer] = W_orig

        self._history.append(('causal_counterfactual', layer, basin_idx, modification))

        return {
            'layer': layer,
            'basin_idx': basin_idx,
            'modification': modification,
            'basins_original': len(basins),
            'basins_counterfactual': len(basins_cf),
            'affected': affected,
            'num_affected': len(affected),
        }

    def benchmark(self, texts=None, max_tokens=50):
        """Measure model quality before/after surgery.

        Computes perplexity on sample texts to quantify impact.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("benchmark requires a loaded model")

        if texts is None:
            texts = [
                "The capital of France is",
                "In the beginning there was",
                "The meaning of life is",
                "Once upon a time in a",
                "The quick brown fox jumps",
            ]

        import math
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                tokens = self._tokenizer.encode(text)
                if len(tokens) < 2:
                    continue
                input_ids = torch.tensor([tokens], device=self.device)
                result = self._model(input_ids)
                logits = result['logits']
                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1))
                total_loss += loss.item()
                total_tokens += shift_labels.numel()

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))

        self._history.append(('benchmark', ppl))
        return {
            'perplexity': ppl,
            'avg_loss': avg_loss,
            'num_texts': len(texts),
            'total_tokens': total_tokens,
        }
