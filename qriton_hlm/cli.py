#!/usr/bin/env python3
"""
Qriton HLM — Energy Language CLI

An interactive shell for programming Hopfield energy landscapes.
Inspect, inject, remove, move basins. Apply changes to live models.
Chain operations into scripts. The assembler for energy computation.

Usage:
    qriton-hlm                               # interactive REPL
    qriton-hlm --checkpoint model.pt         # load specific model
    qriton-hlm --script program.hlm          # run a script
    echo "survey 0" | qriton-hlm -c model.pt # pipe commands

Commands:
    load <path>           Load a checkpoint
    info                  Show loaded model info
    survey <layer>        Find all basins in a layer
    survey-all            Survey all layers
    inject <layer> <seed> [strength]   Inject a basin
    remove <layer> <seed> [strength]   Remove closest basin
    move <layer> <seed> [strength]     Move closest basin
    verify <layer> <seed>              Check if target is a basin
    apply <layer>         Apply last surgery to live model
    restore <layer>       Restore layer to original
    restore-all           Restore all layers
    status                Show modification status
    generate <prompt>     Generate text (if LM loaded)
    set <param> <value>   Set parameter (beta, inits, strength, temp, tokens, topk)
    save <path>           Save modified W matrices
    diff <layer>          Show W diff stats vs original
    history               Show operation history
    guard <type> <value>  Set guard (max-basins, min-basins, perplexity-delta, strength-cap, cosine-drift)
    guards                Show active guards
    help                  Show this help
    quit / exit           Exit
"""

import sys
import os
import argparse
import time
import json
try:
    import readline
except ImportError:
    readline = None
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from qriton_hlm.core import (
    find_basins, compute_energy, poly_interaction,
    inject_basin, remove_basin, move_basin,
    verify_basin_exists, load_W_from_checkpoint,
    count_hopfield_layers,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VERSION = '0.9.4'

# ── ANSI colors ──────────────────────────────────────────────────────

class C:
    """ANSI color codes. Disabled when not a TTY."""
    _enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def _w(cls, code, text):
        return f"\033[{code}m{text}\033[0m" if cls._enabled else text

    @classmethod
    def bold(cls, t):    return cls._w("1", t)
    @classmethod
    def dim(cls, t):     return cls._w("2", t)
    @classmethod
    def green(cls, t):   return cls._w("32", t)
    @classmethod
    def red(cls, t):     return cls._w("31", t)
    @classmethod
    def yellow(cls, t):  return cls._w("33", t)
    @classmethod
    def blue(cls, t):    return cls._w("34", t)
    @classmethod
    def cyan(cls, t):    return cls._w("36", t)
    @classmethod
    def magenta(cls, t): return cls._w("35", t)
    @classmethod
    def gray(cls, t):    return cls._w("90", t)


# ── Banner ───────────────────────────────────────────────────────────

def _make_logo():
    """Build the logo with ANSI colors if enabled."""
    if C._enabled:
        BG = "\033[44m"  # blue background
        WH = "\033[47m"  # white background (the dot)
        B  = "\033[1m"   # bold
        D  = "\033[2m"   # dim
        R  = "\033[0m"   # reset
        return f"""
       {BG}  ▀▀▀▀▀▀▀▀▀▀▀▀▀▀{R}
      {BG}                  {R}     {B}____       _ __{R}
      {BG}                  {R}    {B}/ __ \\_____(_) /_____  ____{R}
      {BG}                  {R}   {B}/ / / / ___/ / __/ __ \\/ __ \\{R}
      {BG}                  {R}   {B}/ /_/ / /  / / /_/ /_/ / / / /{R}
      {BG}              {WH}  {BG}  {R}   {B}\\___\\_/_/ /_/\\__/\\____/_/ /_/{R}
      {BG}                  {R}
      {BG}▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄{R}   {D}Energy Language{R}
"""
    else:
        return r"""
        ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
      ██████████████████     ____       _ __
      ██████████████████    / __ \_____(_) /_____  ____
      ██████████████████   / / / / ___/ / __/ __ \/ __ \
      ██████████████████   / /_/ / /  / / /_/ /_/ / / / /
      ██████████████◉███   \___\_/_/ /_/\__/\____/_/ /_/
      ██████████████████
      ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄   Energy Language
"""

def find_checkpoints():
    """Auto-discover model checkpoints in common locations."""
    search_paths = [
        '.',
        os.path.expanduser('~'),
        os.path.expanduser('~/Desktop'),
        os.path.expanduser('~/models'),
        os.path.expanduser('~/Desktop/HLM2'),
        os.path.expanduser('~/Desktop/HLM3'),
    ]
    found = []
    for base in search_paths:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            # Don't go too deep
            depth = root.replace(base, '').count(os.sep)
            if depth > 3:
                dirs.clear()
                continue
            # Skip hidden dirs, venvs, node_modules
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                       ('node_modules', '__pycache__', 'venv', '.venv', 'env')]
            for f in files:
                if f.endswith('.pt') or f.endswith('.pth'):
                    full = os.path.join(root, f)
                    try:
                        size = os.path.getsize(full)
                        if size > 1_000_000:  # > 1MB, likely a real model
                            found.append((full, size))
                    except OSError:
                        pass
    # Sort by size descending, deduplicate
    found = sorted(set(found), key=lambda x: -x[1])
    return found[:10]  # max 10


def print_banner(checkpoint_path=None):
    """Print startup banner with system info."""
    print(_make_logo())
    print(f"  {C.bold('Qriton HLM')} v{VERSION}  {C.dim('— Qriton Technologies S.R.L.')}")
    print(f"  {C.dim('Apache 2.0 — hlm.qriton.com — github.com/qriton/energy-lang')}")
    print()
    print(f"  Device: {C.cyan(DEVICE)}   Python: {C.cyan(sys.version.split()[0])}   Torch: {C.cyan(torch.__version__)}")
    if DEVICE == 'cuda':
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU:    {C.cyan(gpu)} ({mem:.0f} GB)")
    print()

    if not checkpoint_path:
        # Show auto-discovered checkpoints
        checkpoints = find_checkpoints()
        if checkpoints:
            print(f"  {C.bold('Models found:')}")
            for i, (path, size) in enumerate(checkpoints):
                size_str = f"{size/1e6:.0f}M" if size < 1e9 else f"{size/1e9:.1f}G"
                short = path.replace(os.path.expanduser('~'), '~')
                print(f"    {C.cyan(str(i+1))}  {short}  {C.dim(f'({size_str})')}")
            print()
            print(f"  {C.dim('Quick start:')}  {C.bold('load <path>')}  or  {C.bold('load 1')} to load first match")
        else:
            print(f"  {C.dim('No models found. Use:')}  {C.bold('load <path/to/model.pt>')}")
        print()

    print(f"  {C.dim('Type')} {C.bold('help')} {C.dim('for commands,')} {C.bold('quit')} {C.dim('to exit.')}")
    print(f"  {C.dim('Tab:')} completion   {C.dim('↑/↓:')} history   {C.dim('Ctrl+C:')} cancel   {C.dim('Ctrl+D:')} exit")
    print()


# ── Tab completion ───────────────────────────────────────────────────

COMMANDS = [
    'load', 'info', 'survey', 'survey-all', 'inject', 'remove', 'move',
    'verify', 'apply', 'restore', 'restore-all', 'status', 'generate',
    'set', 'save', 'diff', 'history', 'capture', 'inject-concept',
    'remove-concept', 'concepts', 'blend', 'strengthen', 'weaken',
    'energy', 'export-concept', 'import-concept', 'probe', 'landscape',
    'guard', 'guards', 'benchmark',
    'causal', 'causal scan', 'causal intervene', 'causal counterfactual',
    'help', 'quit', 'exit',
]

CAUSAL_SUBCOMMANDS = ['scan', 'intervene', 'counterfactual']

SET_PARAMS = ['beta', 'inits', 'strength', 'degree', 'eps', 'temp', 'tokens', 'topk']
GUARD_TYPES = ['max-basins', 'min-basins', 'perplexity-delta', 'strength-cap', 'cosine-drift']


def setup_completer(lang_instance):
    """Setup tab completion for the REPL."""
    if readline is None:
        return

    def completer(text, state):
        line = readline.get_line_buffer().lstrip()
        parts = line.split()

        if len(parts) <= 1:
            # Complete command name
            options = [c + ' ' for c in COMMANDS if c.startswith(text)]
        elif parts[0] == 'set':
            if len(parts) == 2:
                options = [p + ' ' for p in SET_PARAMS if p.startswith(text)]
            else:
                options = []
        elif parts[0] == 'guard':
            if len(parts) == 2:
                options = [g + ' ' for g in GUARD_TYPES if g.startswith(text)]
            else:
                options = []
        elif parts[0] == 'causal':
            if len(parts) == 2:
                options = [s + ' ' for s in CAUSAL_SUBCOMMANDS if s.startswith(text)]
            else:
                options = []
        elif parts[0] in ('inject-concept', 'remove-concept', 'export-concept'):
            if len(parts) == 2 or (len(parts) == 3 and parts[0] == 'inject-concept'):
                concepts = list(lang_instance._concepts.keys())
                options = [c + ' ' for c in concepts if c.startswith(text)]
            else:
                options = []
        elif parts[0] == 'blend':
            concepts = list(lang_instance._concepts.keys())
            options = [c + ' ' for c in concepts if c.startswith(text)]
        elif parts[0] == 'load':
            # File path completion
            import glob
            options = glob.glob(text + '*')
            options = [(o + '/' if os.path.isdir(o) else o + ' ') for o in options]
        else:
            options = []

        return options[state] if state < len(options) else None

    readline.set_completer(completer)
    readline.set_completer_delims(' \t\n')
    readline.parse_and_bind('tab: complete')


# ── Guard system ─────────────────────────────────────────────────────

class GuardSystem:
    """Pre-execution safety checks. Operations that violate guards do not execute."""

    def __init__(self):
        self.guards = {}  # type -> value
        self._defaults = {
            'strength-cap': 0.5,  # default max alpha
        }
        self.guards.update(self._defaults)

    def set_guard(self, guard_type, value):
        if guard_type not in GUARD_TYPES:
            return f"{C.red('Unknown guard:')} {guard_type}. Available: {', '.join(GUARD_TYPES)}"
        self.guards[guard_type] = float(value)
        return f"{C.green('Guard set:')} {guard_type} = {value}"

    def list_guards(self):
        if not self.guards:
            return C.dim("No guards active.")
        lines = [C.bold("Active guards:")]
        for g, v in sorted(self.guards.items()):
            default = " (default)" if g in self._defaults and self._defaults[g] == v else ""
            lines.append(f"  {C.cyan(g):30s} {v}{C.dim(default)}")
        return "\n".join(lines)

    def check(self, operation, lang, layer, strength=None, force=False, reason=None):
        """Check guards before an operation. Returns (allowed, message)."""
        violations = []

        # strength-cap
        if strength and 'strength-cap' in self.guards:
            cap = self.guards['strength-cap']
            if abs(strength) > cap:
                violations.append(
                    f"strength-cap: |{strength}| > {cap}")

        # max-basins (only for inject)
        if operation == 'inject' and 'max-basins' in self.guards:
            W = lang._get_W(layer)
            d = W.shape[0]
            basins, _, _ = find_basins(W, d, num_inits=50, beta=lang._get_beta(layer), device=DEVICE)
            if len(basins) >= self.guards['max-basins']:
                violations.append(
                    f"max-basins: {len(basins)} >= {int(self.guards['max-basins'])}")

        # min-basins (only for remove)
        if operation == 'remove' and 'min-basins' in self.guards:
            W = lang._get_W(layer)
            d = W.shape[0]
            basins, _, _ = find_basins(W, d, num_inits=50, beta=lang._get_beta(layer), device=DEVICE)
            if len(basins) <= self.guards['min-basins']:
                violations.append(
                    f"min-basins: {len(basins)} <= {int(self.guards['min-basins'])}")

        if not violations:
            return True, ""

        if force:
            if not reason:
                return False, (
                    f"{C.red('BLOCKED')} — guard violation(s):\n"
                    + "\n".join(f"  {C.yellow(v)}" for v in violations)
                    + f"\n\n  --force requires --reason. Usage: {operation} ... --force --reason \"justification\"")
            # Forced override — log it
            msg = (f"{C.yellow('GUARD OVERRIDE')} ({', '.join(violations)})\n"
                   f"  Reason: {reason}")
            return True, msg
        else:
            msg = (f"{C.red('BLOCKED')} — guard violation(s):\n"
                   + "\n".join(f"  {C.yellow(v)}" for v in violations)
                   + f"\n\n  {C.dim('Override:')} add --force --reason \"justification\" to bypass")
            return False, msg


# ── Main CLI class ───────────────────────────────────────────────────

class EnergyLang:
    """Energy Language interpreter / REPL."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.checkpoint_path = None
        self.model_label = None

        # Cached W matrices (layer -> W tensor)
        self._w_cache = {}
        self._w_backups = {}  # layer -> original W

        # Surgery state
        self._last_surgery = {}
        self._concepts = {}     # name -> {'states': [...], 'centroid': tensor}
        self._history = []

        # Guard system
        self.guard_system = GuardSystem()

        # Parameters
        self.params = {
            'beta': 7.0,
            'inits': 200,
            'strength': 0.1,
            'degree': 3,
            'eps': 5e-3,
            'temp': 0.8,
            'tokens': 50,
            'topk': 40,
        }

    # ── Model loading ────────────────────────────────────────────────

    def load_checkpoint(self, path):
        """Load a model checkpoint. Accepts path or number from discovered list."""
        path = path.strip().strip('"').strip("'")

        # Check if it's a number referencing discovered checkpoints
        if path.isdigit() and hasattr(self, '_discovered_checkpoints') and self._discovered_checkpoints:
            idx = int(path) - 1
            if 0 <= idx < len(self._discovered_checkpoints):
                path = self._discovered_checkpoints[idx][0]
            else:
                return f"{C.red('Invalid index:')} {path}. Have {len(self._discovered_checkpoints)} models."

        if not os.path.exists(path):
            return f"{C.red('File not found:')} {path}"

        self.checkpoint_path = path
        self._w_cache.clear()
        self._w_backups.clear()
        self._last_surgery.clear()

        # Try loading as full HLM3 model
        try:
            from hlm3_model import HLM3
            from data_utils import Tokenizer

            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            config = ckpt.get('config', {})
            state = ckpt.get('model', ckpt.get('model_state', {}))

            for k in list(state.keys()):
                if 'log_beta' in k and state[k].dim() == 0:
                    state[k] = state[k].unsqueeze(0)

            if 'vocabSize' in config:
                model = HLM3(config).to(DEVICE)
                model.load_state_dict(state, strict=False)
                model.eval()
                self.model = model
                self.config = config

                tok_dir = os.path.dirname(path)
                tok_path = os.path.join(tok_dir, 'tokenizer.json')
                if os.path.exists(tok_path):
                    self.tokenizer = Tokenizer.from_json(tok_path)

                n_layers = len(model.blocks)
                n_params = sum(p.numel() for p in model.parameters())
                d_model = config.get('dModel', '?')
                self.model_label = os.path.basename(os.path.dirname(path))
                self._log(f"load {path}")
                return (
                    f"{C.green('Loaded HLM3 model:')}\n"
                    f"  {C.bold(f'{n_params/1e6:.1f}M')} params, d={d_model}, {n_layers} layers\n"
                    f"  Device: {C.cyan(str(DEVICE))}\n"
                    f"  Text gen: {C.green('enabled') if self.tokenizer else C.yellow('no tokenizer')}")
        except Exception:
            pass

        # Fallback: W matrices only
        try:
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            state = ckpt.get('model', ckpt.get('model_state', ckpt))
            w_keys = sorted([k for k in state if 'hopfield.W' in k])
            if w_keys:
                self.model = None
                self.tokenizer = None
                self.config = {'_w_keys': w_keys, '_state': state}
                n_layers = len(w_keys)
                d_model = state[w_keys[0]].shape[0]
                self.model_label = os.path.basename(os.path.dirname(path))
                self._log(f"load {path}")
                return (
                    f"{C.green('Loaded W matrices:')}\n"
                    f"  {n_layers} layers, d={d_model}\n"
                    f"  Text gen: {C.dim('disabled (W-only mode)')}")
            return f"{C.red('No hopfield.W keys found in')} {path}"
        except Exception as e:
            return f"{C.red('Failed:')} {e}"

    def _get_beta(self, layer):
        import math
        layer = int(layer)
        if self.model is not None and hasattr(self.model, 'blocks'):
            try:
                log_beta = self.model.blocks[layer].hopfield.log_beta.data
                return math.exp(log_beta.float().mean().item())
            except (AttributeError, IndexError):
                pass
        if self.config and '_state' in self.config:
            state = self.config['_state']
            for key in state:
                if f'.{layer}.' in key and 'log_beta' in key:
                    return math.exp(state[key].float().mean().item())
        return self.params['beta']

    def _get_W(self, layer):
        layer = int(layer)
        if layer in self._w_cache:
            return self._w_cache[layer]

        if self.model is not None and hasattr(self.model, 'blocks'):
            W = self.model.blocks[layer].hopfield.W.data.clone()
            self._w_cache[layer] = W
            return W

        if self.config and '_state' in self.config:
            w_keys = self.config['_w_keys']
            if layer < len(w_keys):
                W = self.config['_state'][w_keys[layer]].to(DEVICE)
                self._w_cache[layer] = W
                return W

        raise ValueError(f"Cannot access layer {layer}")

    def _num_layers(self):
        if self.model is not None and hasattr(self.model, 'blocks'):
            return len(self.model.blocks)
        if self.config and '_w_keys' in self.config:
            return len(self.config['_w_keys'])
        if self.checkpoint_path:
            return count_hopfield_layers(self.checkpoint_path)
        return 0

    def _log(self, cmd, status='OK', detail=None):
        entry = {
            'time': time.strftime('%H:%M:%S'),
            'cmd': cmd,
            'status': status,
        }
        if detail:
            entry['detail'] = detail
        self._history.append(entry)

    def _make_target(self, seed, d_model):
        gen = torch.Generator(device='cpu')
        gen.manual_seed(int(seed))
        target = torch.randn(d_model, generator=gen).to(DEVICE)
        return target / target.norm()

    # ── Parse --force --reason from args ─────────────────────────────

    def _parse_force(self, args):
        """Extract --force and --reason from argument list."""
        force = False
        reason = None
        clean_args = []
        i = 0
        while i < len(args):
            if args[i] == '--force':
                force = True
            elif args[i] == '--reason':
                # Collect everything after --reason as the reason string
                reason_parts = args[i + 1:]
                reason = " ".join(reason_parts).strip('"').strip("'")
                break
            elif args[i].startswith('--reason='):
                reason = args[i].split('=', 1)[1].strip('"').strip("'")
            else:
                clean_args.append(args[i])
            i += 1
        return clean_args, force, reason

    # ── Commands ─────────────────────────────────────────────────────

    def cmd_info(self):
        if not self.checkpoint_path:
            return f"{C.yellow('No model loaded.')} Use: load <path>"
        lines = [C.bold(f"Model: {self.model_label}")]
        lines.append(f"  Checkpoint: {C.dim(self.checkpoint_path)}")
        lines.append(f"  Layers:     {self._num_layers()}")
        if self.model:
            n_params = sum(p.numel() for p in self.model.parameters())
            lines.append(f"  Params:     {n_params/1e6:.1f}M")
            if self.config:
                d = self.config.get('dModel', self.config.get('d_model', '?'))
                lines.append(f"  d_model:    {d}")
        lines.append(f"  Device:     {DEVICE}")
        lines.append(f"  Text gen:   {'yes' if self.tokenizer else 'no'}")
        modified = [l for l in self._w_backups]
        if modified:
            lines.append(f"  Modified:   {C.yellow(str(modified))}")
        else:
            lines.append(f"  Modified:   {C.dim('none')}")
        return "\n".join(lines)

    def cmd_survey(self, layer):
        layer = int(layer)
        W = self._get_W(layer)
        d_model = W.shape[0]
        beta = self._get_beta(layer)
        num_inits = int(self.params['inits'])

        t0 = time.time()
        basins, ids, trajectories = find_basins(
            W, d_model, num_inits=num_inits, beta=beta, device=DEVICE)
        elapsed = time.time() - t0

        conv_iters = [len(t) - 1 for t in trajectories]

        lines = [
            C.bold(f"Layer {layer}: {len(basins)} basins") +
            C.dim(f"  ({num_inits} inits, β={beta:.1f}, {elapsed:.1f}s)"),
            C.dim(f"  W: {d_model}×{d_model}, spectral={torch.linalg.norm(W, ord=2).item():.4f}, "
                  f"avg convergence: {np.mean(conv_iters):.1f} iters"),
            "",
        ]
        for i in range(min(len(basins), 20)):
            e = compute_energy(basins[i], W)
            pop = ids.count(i)
            pct = 100 * pop / num_inits
            bar = C.cyan("█" * min(int(pct / 3), 30))
            lines.append(f"  B{i:2d}  E={e:+.4f}  pop={pop:3d} ({pct:4.1f}%)  {bar}")
        if len(basins) > 20:
            lines.append(C.dim(f"  ... and {len(basins) - 20} more"))

        self._log(f"survey {layer}")
        return "\n".join(lines)

    def cmd_survey_all(self):
        n = self._num_layers()
        lines = [C.bold(f"Surveying {n} layers..."), ""]
        t0 = time.time()
        for l in range(n):
            W = self._get_W(l)
            d = W.shape[0]
            basins, ids, trajs = find_basins(
                W, d, num_inits=int(self.params['inits']),
                beta=self._get_beta(l), device=DEVICE)
            mean_iters = np.mean([len(t) - 1 for t in trajs])

            beta_str = ""
            if self.model and hasattr(self.model, 'blocks'):
                lb = self.model.blocks[l].hopfield.log_beta.data
                beta_val = float(torch.exp(lb[0] if lb.dim() > 0 else lb))
                beta_str = f"  β={beta_val:.2f}"

            mod = C.yellow(" [MODIFIED]") if l in self._w_backups else ""
            bar = C.cyan("█" * min(len(basins), 30))
            lines.append(
                f"  L{l}: {C.bold(f'{len(basins):3d}')} basins  "
                f"avg_iters={mean_iters:.0f}{beta_str}{mod}  {bar}")

        elapsed = time.time() - t0
        lines.append(C.dim(f"\nTotal: {elapsed:.1f}s"))
        self._log("survey-all")
        return "\n".join(lines)

    def cmd_inject(self, layer, seed, strength=None, force=False, reason=None):
        layer = int(layer)
        strength = float(strength) if strength else self.params['strength']
        W = self._get_W(layer)
        d = W.shape[0]
        target = self._make_target(seed, d)

        # Guard check
        allowed, guard_msg = self.guard_system.check('inject', self, layer, strength, force, reason)
        if not allowed:
            self._log(f"inject {layer} {seed} {strength}", status='BLOCKED', detail=guard_msg)
            return guard_msg

        exists_before, _, cos_before, _ = verify_basin_exists(
            W, target, beta=self._get_beta(layer))
        basins_before, _, _ = find_basins(
            W, d, num_inits=100, beta=self._get_beta(layer), device=DEVICE)

        W_new = inject_basin(W, target, strength=strength)

        exists_after, _, cos_after, _ = verify_basin_exists(
            W_new, target, beta=self._get_beta(layer))
        basins_after, _, _ = find_basins(
            W_new, d, num_inits=100, beta=self._get_beta(layer), device=DEVICE)

        self._last_surgery = {
            'layer': layer, 'W_before': W, 'W_after': W_new,
            'target': target, 'operation': 'inject',
        }
        if layer not in self._w_backups:
            self._w_backups[layer] = W.clone()
        self._w_cache[layer] = W_new

        delta = len(basins_after) - len(basins_before)
        self._log(f"inject {layer} {seed} {strength}")

        lines = []
        if guard_msg:
            lines.append(guard_msg)
        lines.extend([
            C.bold(f"Inject L{layer}") + f" seed={seed} α={strength}",
            f"  Before: {len(basins_before)} basins, target is basin: {exists_before} (cos={cos_before:.4f})",
            f"  After:  {len(basins_after)} basins ({'+' if delta >= 0 else ''}{delta}), "
            f"target is basin: {exists_after} (cos={cos_after:.4f})",
        ])
        if exists_after and not exists_before:
            lines.append(f"  {C.green('>> Basin successfully programmed!')}")
        return "\n".join(lines)

    def cmd_remove(self, layer, seed, strength=None, force=False, reason=None):
        layer = int(layer)
        strength = float(strength) if strength else self.params['strength']
        W = self._get_W(layer)
        d = W.shape[0]

        # Guard check
        allowed, guard_msg = self.guard_system.check('remove', self, layer, strength, force, reason)
        if not allowed:
            self._log(f"remove {layer} {seed} {strength}", status='BLOCKED', detail=guard_msg)
            return guard_msg

        basins_before, _, _ = find_basins(
            W, d, num_inits=100, beta=self._get_beta(layer), device=DEVICE)

        target = self._make_target(seed, d)
        if basins_before:
            sims = [F.cosine_similarity(target.unsqueeze(0), b.unsqueeze(0)).item()
                    for b in basins_before]
            closest = basins_before[int(np.argmax(sims))]
            target = closest

        W_new = remove_basin(W, target, strength=strength)
        exists_after, _, cos_after, _ = verify_basin_exists(
            W_new, target, beta=self._get_beta(layer))
        basins_after, _, _ = find_basins(
            W_new, d, num_inits=100, beta=self._get_beta(layer), device=DEVICE)

        self._last_surgery = {
            'layer': layer, 'W_before': W, 'W_after': W_new,
            'target': target, 'operation': 'remove',
        }
        if layer not in self._w_backups:
            self._w_backups[layer] = W.clone()
        self._w_cache[layer] = W_new

        delta = len(basins_after) - len(basins_before)
        self._log(f"remove {layer} {seed} {strength}")

        lines = []
        if guard_msg:
            lines.append(guard_msg)
        lines.extend([
            C.bold(f"Remove L{layer}") + f" seed={seed} α={strength}",
            f"  Before: {len(basins_before)} basins",
            f"  After:  {len(basins_after)} basins ({'+' if delta >= 0 else ''}{delta})",
            f"  Target still exists: {exists_after} (cos={cos_after:.4f})",
        ])
        if not exists_after:
            lines.append(f"  {C.green('>> Basin successfully removed!')}")
        return "\n".join(lines)

    def cmd_move(self, layer, seed, strength=None):
        layer = int(layer)
        strength = float(strength) if strength else self.params['strength']
        W = self._get_W(layer)
        d = W.shape[0]

        basins_before, _, _ = find_basins(
            W, d, num_inits=100, beta=self._get_beta(layer), device=DEVICE)

        source_target = self._make_target(seed, d)
        if basins_before:
            sims = [F.cosine_similarity(source_target.unsqueeze(0), b.unsqueeze(0)).item()
                    for b in basins_before]
            source = basins_before[int(np.argmax(sims))]
        else:
            source = source_target

        dest = self._make_target(int(seed) + 1000, d)
        W_new = move_basin(W, source, dest, strength=strength)

        src_exists, _, src_cos, _ = verify_basin_exists(W_new, source, beta=self._get_beta(layer))
        dst_exists, _, dst_cos, _ = verify_basin_exists(W_new, dest, beta=self._get_beta(layer))
        basins_after, _, _ = find_basins(
            W_new, d, num_inits=100, beta=self._get_beta(layer), device=DEVICE)

        self._last_surgery = {
            'layer': layer, 'W_before': W, 'W_after': W_new,
            'target': dest, 'operation': 'move',
        }
        if layer not in self._w_backups:
            self._w_backups[layer] = W.clone()
        self._w_cache[layer] = W_new

        self._log(f"move {layer} {seed} {strength}")
        src_status = C.green('GONE') if not src_exists else C.yellow('still exists')
        dst_status = C.green('EXISTS') if dst_exists else C.red('missing')
        return (
            f"{C.bold(f'Move L{layer}')} seed={seed} α={strength}\n"
            f"  Source: {src_status} (cos={src_cos:.4f})\n"
            f"  Dest:   {dst_status} (cos={dst_cos:.4f})\n"
            f"  Basins: {len(basins_before)} → {len(basins_after)}")

    def cmd_verify(self, layer, seed):
        layer = int(layer)
        W = self._get_W(layer)
        d = W.shape[0]
        target = self._make_target(seed, d)

        exists, final, cos, iters = verify_basin_exists(W, target, beta=self._get_beta(layer))
        energy = compute_energy(final, W)

        status = C.green("YES — stable basin") if exists else C.red("NO — not a basin")
        self._log(f"verify {layer} {seed}")
        return (
            f"{C.bold(f'Verify L{layer}')} seed={seed}\n"
            f"  Is basin:    {status}\n"
            f"  Cosine sim:  {cos:.6f}\n"
            f"  Converged:   {iters} iters\n"
            f"  Energy:      {energy:.4f}")

    def cmd_apply(self, layer):
        layer = int(layer)
        if self.model is None or not hasattr(self.model, 'blocks'):
            return f"{C.yellow('No full model loaded.')} Apply only works with HLM3 models."

        if layer not in self._w_cache:
            return f"{C.yellow('No modified W for layer')} {layer}. Run surgery first."

        W_new = self._w_cache[layer]
        W_live = self.model.blocks[layer].hopfield.W

        if W_new.shape != W_live.shape:
            return f"{C.red('Shape mismatch:')} {list(W_new.shape)} vs {list(W_live.shape)}"

        if layer not in self._w_backups:
            self._w_backups[layer] = W_live.data.clone()

        with torch.no_grad():
            W_live.data.copy_(W_new)

        self._log(f"apply {layer}")
        modified = sorted(self._w_backups.keys())
        return (
            f"{C.green('Applied')} L{layer} to live model.\n"
            f"  Modified layers: {C.yellow(str(modified))}\n"
            f"  Text generation now uses modified W.")

    def cmd_restore(self, layer):
        layer = int(layer)
        if layer not in self._w_backups:
            return f"Layer {layer} has not been modified."

        if self.model and hasattr(self.model, 'blocks'):
            with torch.no_grad():
                self.model.blocks[layer].hopfield.W.data.copy_(self._w_backups[layer])

        self._w_cache[layer] = self._w_backups[layer].clone()
        del self._w_backups[layer]

        self._log(f"restore {layer}")
        remaining = sorted(self._w_backups.keys())
        return (
            f"{C.green('Restored')} L{layer} to original.\n"
            f"  Still modified: {remaining if remaining else C.dim('none')}")

    def cmd_restore_all(self):
        if not self._w_backups:
            return "No layers modified."
        restored = sorted(self._w_backups.keys())
        for l in list(self._w_backups.keys()):
            if self.model and hasattr(self.model, 'blocks'):
                with torch.no_grad():
                    self.model.blocks[l].hopfield.W.data.copy_(self._w_backups[l])
            self._w_cache[l] = self._w_backups[l].clone()
            del self._w_backups[l]

        self._log("restore-all")
        return f"{C.green('Restored')} layers {restored} to original."

    def cmd_status(self):
        if not self.checkpoint_path:
            return f"{C.yellow('No model loaded.')}"
        n = self._num_layers()
        lines = [C.bold(f"Model: {self.model_label}") + f" ({n} layers)"]
        for l in range(n):
            if l in self._w_backups:
                lines.append(f"  L{l}: {C.yellow('MODIFIED')}")
            else:
                lines.append(f"  L{l}: {C.dim('original')}")
        return "\n".join(lines)

    def cmd_capture(self, layer, text, concept_name=None):
        layer = int(layer)
        if self.model is None or self.tokenizer is None:
            return f"{C.yellow('No model+tokenizer loaded.')}"

        tokens = self.tokenizer.encode(text)
        input_ids = torch.tensor([tokens], device=DEVICE)

        captured = {}
        def hook_fn(module, inp, output):
            if isinstance(output, dict) and 'state' in output:
                captured['state'] = output['state'].detach()
            elif isinstance(output, tuple):
                captured['state'] = output[0].detach()
            elif isinstance(output, torch.Tensor):
                captured['state'] = output.detach()

        block = self.model.blocks[layer]
        handle = block.hopfield.register_forward_hook(hook_fn)

        with torch.no_grad():
            self.model(input_ids)
        handle.remove()

        if 'state' not in captured:
            return f"{C.red('Failed:')} could not capture state from layer {layer}"

        state = captured['state']
        if state.dim() == 3:
            state = state.squeeze(0).mean(dim=0)
        elif state.dim() == 2:
            state = state.mean(dim=0)
        state = state / state.norm()

        W = self._get_W(layer)
        energy = compute_energy(state, W)
        is_basin, _, cos, iters = verify_basin_exists(W, state, beta=self._get_beta(layer))

        lines = [
            C.bold(f"Captured L{layer}:") + f" \"{text[:50]}\"",
            f"  Energy: {energy:.4f} | Basin: {is_basin} (cos={cos:.4f}, {iters} iters)",
        ]

        if concept_name:
            if concept_name not in self._concepts:
                self._concepts[concept_name] = {'states': [], 'centroid': None}
            self._concepts[concept_name]['states'].append(state)
            states = torch.stack(self._concepts[concept_name]['states'])
            centroid = states.mean(dim=0)
            self._concepts[concept_name]['centroid'] = centroid / centroid.norm()
            n = len(self._concepts[concept_name]['states'])
            lines.append(f"  → concept {C.cyan(concept_name)} ({n} samples)")

        self._log(f"capture {layer} {concept_name or ''} {text[:30]}")
        return "\n".join(lines)

    def cmd_inject_concept(self, layer, concept_name, strength=None, force=False, reason=None):
        layer = int(layer)
        strength = float(strength) if strength else self.params['strength']

        if concept_name not in self._concepts:
            available = list(self._concepts.keys()) if self._concepts else "none"
            return f"{C.red('Unknown concept')} '{concept_name}'. Available: {available}"

        concept = self._concepts[concept_name]
        if concept['centroid'] is None:
            return f"Concept '{concept_name}' has no samples."

        # Guard check
        allowed, guard_msg = self.guard_system.check('inject', self, layer, strength, force, reason)
        if not allowed:
            self._log(f"inject-concept {layer} {concept_name} {strength}", status='BLOCKED')
            return guard_msg

        target = concept['centroid'].to(DEVICE)
        W = self._get_W(layer)
        d = W.shape[0]

        exists_before, _, cos_before, _ = verify_basin_exists(W, target, beta=self._get_beta(layer))
        basins_before, _, _ = find_basins(W, d, num_inits=100, beta=self._get_beta(layer), device=DEVICE)

        W_new = inject_basin(W, target, strength=strength)

        exists_after, _, cos_after, _ = verify_basin_exists(W_new, target, beta=self._get_beta(layer))
        basins_after, _, _ = find_basins(W_new, d, num_inits=100, beta=self._get_beta(layer), device=DEVICE)

        self._w_cache[layer] = W_new
        if layer not in self._w_backups:
            self._w_backups[layer] = W.clone()

        delta = len(basins_after) - len(basins_before)
        n = len(concept['states'])
        self._log(f"inject-concept {layer} {concept_name} {strength}")

        lines = []
        if guard_msg:
            lines.append(guard_msg)
        lines.extend([
            C.bold(f"Inject concept") + f" '{C.cyan(concept_name)}' ({n} samples) into L{layer} α={strength}",
            f"  Before: {len(basins_before)} basins, concept is basin: {exists_before} (cos={cos_before:.4f})",
            f"  After:  {len(basins_after)} basins ({'+' if delta >= 0 else ''}{delta}), "
            f"concept is basin: {exists_after} (cos={cos_after:.4f})",
        ])
        if exists_after and not exists_before:
            lines.append(f"  {C.green('>> Concept successfully injected!')}")
        return "\n".join(lines)

    def cmd_remove_concept(self, layer, concept_name, strength=None, force=False, reason=None):
        layer = int(layer)
        strength = float(strength) if strength else self.params['strength']

        if concept_name not in self._concepts:
            return f"{C.red('Unknown concept')} '{concept_name}'"

        # Guard check
        allowed, guard_msg = self.guard_system.check('remove', self, layer, strength, force, reason)
        if not allowed:
            self._log(f"remove-concept {layer} {concept_name} {strength}", status='BLOCKED')
            return guard_msg

        target = self._concepts[concept_name]['centroid'].to(DEVICE)
        W = self._get_W(layer)

        W_new = remove_basin(W, target, strength=strength)
        exists_after, _, cos_after, _ = verify_basin_exists(W_new, target, beta=self._get_beta(layer))

        self._w_cache[layer] = W_new
        if layer not in self._w_backups:
            self._w_backups[layer] = W.clone()

        self._log(f"remove-concept {layer} {concept_name} {strength}")
        status = C.green("GONE") if not exists_after else C.yellow(f"still exists (cos={cos_after:.4f})")
        lines = []
        if guard_msg:
            lines.append(guard_msg)
        lines.append(f"{C.bold('Removed concept')} '{C.cyan(concept_name)}' from L{layer}: {status}")
        return "\n".join(lines)

    def cmd_concepts(self):
        if not self._concepts:
            return C.dim("No concepts captured. Use: capture <layer> <concept> <text>")
        lines = [C.bold("Captured concepts:")]
        for name, c in self._concepts.items():
            lines.append(f"  {C.cyan(name)}: {len(c['states'])} samples")
        return "\n".join(lines)

    def cmd_blend(self, concept_a, concept_b, new_name, ratio=None):
        ratio = float(ratio) if ratio else 0.5
        if concept_a not in self._concepts or concept_b not in self._concepts:
            return f"{C.red('Both concepts must exist.')} Have: {list(self._concepts.keys())}"
        ca = self._concepts[concept_a]['centroid']
        cb = self._concepts[concept_b]['centroid']
        blended = ratio * ca + (1 - ratio) * cb
        blended = blended / blended.norm()
        self._concepts[new_name] = {'states': [], 'centroid': blended}
        self._log(f"blend {concept_a} {concept_b} -> {new_name} ({ratio})")
        return (
            f"{C.green('Blended')} '{C.cyan(concept_a)}' ({ratio:.0%}) + "
            f"'{C.cyan(concept_b)}' ({1-ratio:.0%}) → '{C.cyan(new_name)}'")

    def cmd_strengthen(self, layer, seed, factor=None):
        layer, seed = int(layer), int(seed)
        factor = float(factor) if factor else 2.0
        W = self._get_W(layer)
        d = W.shape[0]
        target = self._make_target(seed, d)
        basins, _, _ = find_basins(W, d, num_inits=100, beta=self._get_beta(layer), device=DEVICE)
        if basins:
            sims = [F.cosine_similarity(target.unsqueeze(0), b.unsqueeze(0)).item() for b in basins]
            target = basins[int(np.argmax(sims))]
        e_before = compute_energy(target, W)
        W_new = inject_basin(W, target, strength=self.params['strength'] * factor)
        e_after = compute_energy(target, W_new)
        self._w_cache[layer] = W_new
        if layer not in self._w_backups:
            self._w_backups[layer] = W.clone()
        self._log(f"strengthen {layer} {seed} {factor}")
        return (
            f"{C.bold(f'Strengthen L{layer}')} seed={seed} (factor={factor}×)\n"
            f"  Energy: {e_before:.4f} → {e_after:.4f} (deepened by {e_before - e_after:.4f})")

    def cmd_weaken(self, layer, seed, factor=None):
        layer, seed = int(layer), int(seed)
        factor = float(factor) if factor else 0.5
        W = self._get_W(layer)
        d = W.shape[0]
        target = self._make_target(seed, d)
        basins, _, _ = find_basins(W, d, num_inits=100, beta=self._get_beta(layer), device=DEVICE)
        if basins:
            sims = [F.cosine_similarity(target.unsqueeze(0), b.unsqueeze(0)).item() for b in basins]
            target = basins[int(np.argmax(sims))]
        e_before = compute_energy(target, W)
        W_new = remove_basin(W, target, strength=self.params['strength'] * factor)
        e_after = compute_energy(target, W_new)
        self._w_cache[layer] = W_new
        if layer not in self._w_backups:
            self._w_backups[layer] = W.clone()
        self._log(f"weaken {layer} {seed} {factor}")
        return (
            f"{C.bold(f'Weaken L{layer}')} seed={seed} (factor={factor}×)\n"
            f"  Energy: {e_before:.4f} → {e_after:.4f} (raised by {e_after - e_before:.4f})")

    def cmd_energy(self, layer, seed):
        layer, seed = int(layer), int(seed)
        W = self._get_W(layer)
        d = W.shape[0]
        target = self._make_target(seed, d)
        e = compute_energy(target, W)
        is_basin, _, cos, iters = verify_basin_exists(W, target, beta=self._get_beta(layer))
        return (
            f"{C.bold(f'Energy L{layer}')} seed={seed}: {C.cyan(f'{e:.6f}')}\n"
            f"  Basin: {is_basin} (cos={cos:.4f}, {iters} iters)")

    def cmd_export_concept(self, concept_name, path):
        if concept_name not in self._concepts:
            return f"{C.red('Unknown concept')} '{concept_name}'"
        concept = self._concepts[concept_name]
        data = {
            'name': concept_name,
            'centroid': concept['centroid'].cpu(),
            'num_samples': len(concept['states']),
            'states': [s.cpu() for s in concept['states']],
            'dim': concept['centroid'].shape[0],
            'version': VERSION,
            'exported_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_checkpoint': self.checkpoint_path,
        }
        torch.save(data, path)
        self._log(f"export {concept_name} {path}")
        return (
            f"{C.green('Exported')} '{C.cyan(concept_name)}' → {path}\n"
            f"  {len(concept['states'])} samples, dim={data['dim']}, v{VERSION}")

    def cmd_import_concept(self, path):
        if not os.path.exists(path):
            return f"{C.red('File not found:')} {path}"
        data = torch.load(path, map_location=DEVICE, weights_only=False)
        name = data['name']
        self._concepts[name] = {
            'centroid': data['centroid'].to(DEVICE),
            'states': [s.to(DEVICE) for s in data.get('states', [])],
        }
        meta = ""
        if 'version' in data:
            meta += f", exported v{data['version']}"
        if 'source_checkpoint' in data:
            meta += f", from {os.path.basename(str(data.get('source_checkpoint', '?')))}"
        self._log(f"import {name} {path}")
        return (
            f"{C.green('Imported')} '{C.cyan(name)}' from {path}\n"
            f"  {len(self._concepts[name]['states'])} samples, dim={data.get('dim', '?')}{meta}")

    def cmd_probe(self, layer, basin_idx=None):
        layer = int(layer)
        basin_idx = int(basin_idx) if basin_idx else 0
        if self.model is None:
            return f"{C.yellow('No model loaded.')}"

        W = self._get_W(layer)
        d = W.shape[0]
        basins, _, _ = find_basins(W, d, num_inits=int(self.params['inits']),
                                   beta=self._get_beta(layer), device=DEVICE)
        if basin_idx >= len(basins):
            return f"Basin {basin_idx} not found. Have {len(basins)} basins."

        basin_state = basins[basin_idx]
        energy = compute_energy(basin_state, W)

        tokens = []
        if hasattr(self.model, 'lm_head'):
            with torch.no_grad():
                logits = self.model.lm_head(basin_state.unsqueeze(0))
                top_k = min(10, logits.shape[-1])
                values, indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(values, dim=-1)
                if self.tokenizer:
                    for i in range(top_k):
                        tok = self.tokenizer.decode([indices[0, i].item()])
                        tokens.append(f"  {tok!r:20s} {probs[0, i].item():.3f}")

        self._log(f"probe {layer} {basin_idx}")
        lines = [C.bold(f"Probe L{layer} basin #{basin_idx}") + f" (E={energy:.4f})"]
        if tokens:
            lines.append(C.dim("Top tokens:"))
            lines.extend(tokens)
        else:
            lines.append(C.dim("  (no output head available)"))
        return "\n".join(lines)

    def cmd_landscape(self, layer):
        layer = int(layer)
        W = self._get_W(layer)
        d = W.shape[0]
        basins, ids, _ = find_basins(W, d, num_inits=int(self.params['inits']),
                                     beta=self._get_beta(layer), device=DEVICE)
        lines = [C.bold(f"Landscape L{layer}: {len(basins)} basins") +
                 C.dim(f" from {int(self.params['inits'])} inits")]
        for i, b in enumerate(basins):
            e = compute_energy(b, W)
            pop = ids.count(i)
            bar = C.cyan("█" * min(pop, 40))
            lines.append(f"  basin {i:3d}: E={e:8.4f} pop={pop:3d} {bar}")

        energies = [compute_energy(b, W) for b in basins]
        lines.append(f"\n  Energy range: [{min(energies):.4f}, {max(energies):.4f}]")
        lines.append(f"  Deepest:      basin {np.argmin(energies)} (E={min(energies):.4f})")
        most_pop = max(range(len(basins)), key=lambda i: ids.count(i))
        lines.append(f"  Most popular: basin {most_pop} (pop={ids.count(most_pop)})")
        self._log(f"landscape {layer}")
        return "\n".join(lines)

    def cmd_guard(self, guard_type, value):
        return self.guard_system.set_guard(guard_type, value)

    def cmd_guards(self):
        return self.guard_system.list_guards()

    def cmd_benchmark(self):
        if self.model is None or self.tokenizer is None:
            return f"{C.yellow('No model+tokenizer loaded.')}"
        import math
        texts = [
            "The capital of France is",
            "In the beginning there was",
            "The meaning of life is",
            "Once upon a time in a",
            "The quick brown fox jumps",
        ]
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer.encode(text)
                if len(tokens) < 2:
                    continue
                input_ids = torch.tensor([tokens], device=DEVICE)
                result = self.model(input_ids)
                logits = result['logits']
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1))
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        self._log(f"benchmark ppl={ppl:.2f}")
        return (
            f"{C.bold('Benchmark:')} PPL={C.cyan(f'{ppl:.2f}')} (loss={avg_loss:.4f})\n"
            f"  {len(texts)} texts, {total_tokens} tokens")

    def cmd_generate(self, prompt):
        if self.model is None or self.tokenizer is None:
            return f"{C.yellow('No model+tokenizer loaded.')}"

        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        input_ids = tokenizer.encode(prompt)
        ids = torch.tensor([input_ids], device=DEVICE)

        max_tokens = int(self.params['tokens'])
        temperature = self.params['temp']
        top_k = int(self.params['topk'])
        generated = []

        t0 = time.time()
        with torch.no_grad():
            for _ in range(max_tokens):
                max_seq = config.get('maxSeqLen', 512)
                inp = ids if ids.shape[1] <= max_seq else ids[:, -max_seq:]
                result = model(inp)
                logits = result['logits'][:, -1, :] / max(temperature, 0.01)
                k = min(top_k, config.get('vocabSize', logits.shape[-1]))
                values, indices = torch.topk(logits, k, dim=-1)
                probs = F.softmax(values, dim=-1)
                sampled = torch.multinomial(probs, 1)
                next_token = indices.gather(-1, sampled)
                token_id = next_token.item()
                generated.append(token_id)
                ids = torch.cat([ids, next_token], dim=1)
                if token_id in (0, 3):
                    break

        elapsed = time.time() - t0
        text = tokenizer.decode(generated)
        tok_s = len(generated) / max(elapsed, 0.01)

        self._log(f"generate {prompt[:30]}...")
        return (
            f"{C.bold(prompt)}{text}\n"
            f"  {C.dim(f'[{len(generated)} tokens, {elapsed:.1f}s, {tok_s:.0f} tok/s]')}")

    def cmd_set(self, param, value):
        if param not in self.params:
            return f"{C.red('Unknown param:')} {param}. Available: {list(self.params.keys())}"
        self.params[param] = float(value)
        self._log(f"set {param} {value}")
        return f"  {C.cyan(param)} = {self.params[param]}"

    def cmd_save(self, path):
        if not self._w_backups and not self._w_cache:
            return "Nothing to save."

        save_dict = {}
        for layer, W in self._w_cache.items():
            save_dict[f'layer_{layer}_W'] = W.cpu()
        if self._w_backups:
            save_dict['backups'] = {
                f'layer_{l}': W.cpu() for l, W in self._w_backups.items()
            }
        save_dict['checkpoint'] = self.checkpoint_path
        save_dict['history'] = self._history

        torch.save(save_dict, path)
        self._log(f"save {path}")
        return f"{C.green('Saved')} {len(self._w_cache)} W matrices to {path}"

    def cmd_diff(self, layer):
        layer = int(layer)
        if layer not in self._w_backups:
            return f"Layer {layer} has not been modified."

        W_orig = self._w_backups[layer]
        W_curr = self._get_W(layer)
        diff = W_curr - W_orig
        rel_change = diff.norm().item() / W_orig.norm().item() * 100

        return (
            f"{C.bold(f'Diff L{layer}:')}\n"
            f"  Frobenius norm Δ:  {diff.norm().item():.6f}\n"
            f"  Max absolute Δ:    {diff.abs().max().item():.6f}\n"
            f"  Mean absolute Δ:   {diff.abs().mean().item():.8f}\n"
            f"  Spectral (orig):   {torch.linalg.norm(W_orig, ord=2).item():.4f}\n"
            f"  Spectral (curr):   {torch.linalg.norm(W_curr, ord=2).item():.4f}\n"
            f"  Relative change:   {C.yellow(f'{rel_change:.2f}%') if rel_change > 1 else C.green(f'{rel_change:.2f}%')}")

    def cmd_history(self):
        if not self._history:
            return C.dim("No history.")
        lines = [C.bold("Time      Status    Command")]
        for entry in self._history:
            ts = entry['time']
            cmd = entry['cmd']
            status = entry.get('status', 'OK')
            if status == 'BLOCKED':
                status_str = C.red('BLOCKED')
            elif status == 'OK':
                status_str = C.green('OK     ')
            else:
                status_str = status
            lines.append(f"  {C.dim(ts)}  {status_str}  {cmd}")
        return "\n".join(lines)

    # ── Causal commands ───────────────────────────────────────────

    def cmd_causal_scan(self, layer, threshold=None):
        """Discover causal links between basins by systematic knockout."""
        layer = int(layer)
        threshold = float(threshold) if threshold else 0.15

        # Use BasinSurgeon's causal_scan via direct W operations
        W_orig = self._get_W(layer).clone()
        d = W_orig.shape[0]
        beta = self._get_beta(layer)

        basins, ids, _ = find_basins(
            W_orig, d, num_inits=int(self.params['inits']),
            beta=beta, device=DEVICE)
        n = len(basins)

        if n < 2:
            self._log(f"causal scan {layer}")
            return f"{C.yellow('Need at least 2 basins for causal scan.')} Found: {n}"

        lines = [
            C.bold(f"Causal scan L{layer}") +
            f": {n} basins, threshold={threshold}",
            C.dim(f"  Knocking out each basin and measuring downstream drift..."),
            "",
        ]

        edges = []
        for i in range(n):
            W_knocked = remove_basin(W_orig, basins[i], strength=self.params['strength'], beta=beta)
            for j in range(n):
                if i == j:
                    continue
                exists, final, cos, _ = verify_basin_exists(W_knocked, basins[j], beta=beta)
                drift = 1.0 - cos
                if drift > threshold:
                    edge_type = 'destroyed' if not exists else 'shifted'
                    edges.append((i, j, drift, edge_type))

        if edges:
            lines.append(C.bold(f"  Causal edges found: {len(edges)}"))
            lines.append("")
            lines.append(f"  {'Source':>8s}  →  {'Target':<8s}  {'Drift':>8s}  {'Effect'}")
            lines.append(f"  {'─'*8}     {'─'*8}  {'─'*8}  {'─'*10}")
            for src, tgt, drift, etype in sorted(edges, key=lambda e: -e[2]):
                effect = C.red("DESTROYED") if etype == 'destroyed' else C.yellow("shifted")
                lines.append(f"  {'B'+str(src):>8s}  →  {'B'+str(tgt):<8s}  {drift:8.4f}  {effect}")
        else:
            lines.append(C.dim("  No causal links found above threshold."))

        # Show adjacency summary
        if edges:
            lines.append("")
            sources = set(e[0] for e in edges)
            for s in sorted(sources):
                targets = [f"B{e[1]}" for e in edges if e[0] == s]
                lines.append(f"  B{s} causes → {', '.join(targets)}")

        self._log(f"causal scan {layer}", detail=f"{len(edges)} edges")
        return "\n".join(lines)

    def cmd_causal_intervene(self, layer, basin_idx, operation=None):
        """do(X) — intervene on a basin and measure downstream effects."""
        layer = int(layer)
        basin_idx = int(basin_idx)
        operation = operation or 'remove'

        W_orig = self._get_W(layer).clone()
        d = W_orig.shape[0]
        beta = self._get_beta(layer)

        basins_before, _, _ = find_basins(
            W_orig, d, num_inits=int(self.params['inits']),
            beta=beta, device=DEVICE)

        if basin_idx >= len(basins_before):
            return f"{C.red('Basin')} {basin_idx} not found. Have {len(basins_before)}."

        target = basins_before[basin_idx]
        e_before = compute_energy(target, W_orig)

        # Intervene
        if operation == 'remove':
            W_after = remove_basin(W_orig, target, strength=self.params['strength'], beta=beta)
        elif operation == 'weaken':
            W_after = remove_basin(W_orig, target, strength=self.params['strength'] * 0.5, beta=beta)
        elif operation == 'strengthen':
            W_after = inject_basin(W_orig, target, strength=self.params['strength'], beta=beta)
        else:
            return f"{C.red('Unknown operation:')} {operation}. Use: remove, weaken, strengthen"

        basins_after, _, _ = find_basins(
            W_after, d, num_inits=int(self.params['inits']),
            beta=beta, device=DEVICE)

        # Measure effect on each other basin
        affected = []
        for j, b in enumerate(basins_before):
            if j == basin_idx:
                continue
            exists, final, cos, _ = verify_basin_exists(W_after, b, beta=beta)
            drift = 1.0 - cos
            if drift > 0.05:
                affected.append((j, drift, exists))

        # Persist the change
        if layer not in self._w_backups:
            self._w_backups[layer] = W_orig.clone()
        self._w_cache[layer] = W_after

        lines = [
            C.bold(f"do(B{basin_idx}) = {operation}") + f"  L{layer}",
            f"  Energy before: {e_before:.4f}",
            f"  Basins: {len(basins_before)} → {len(basins_after)} "
            f"({'+' if len(basins_after) >= len(basins_before) else ''}"
            f"{len(basins_after) - len(basins_before)})",
            "",
        ]
        if affected:
            lines.append(C.bold(f"  Downstream effects: {len(affected)} basins affected"))
            for j, drift, survived in sorted(affected, key=lambda x: -x[1]):
                status = C.green("ok") if survived else C.red("gone")
                bar = C.yellow("█" * min(int(drift * 30), 30))
                lines.append(f"    B{j:3d}  drift={drift:.4f}  {status}  {bar}")
        else:
            lines.append(C.dim("  No downstream effects detected."))

        self._log(f"causal intervene {layer} B{basin_idx} {operation}")
        return "\n".join(lines)

    def cmd_causal_counterfactual(self, layer, basin_idx, modification=None):
        """What if basin X had been different? Non-destructive."""
        layer = int(layer)
        basin_idx = int(basin_idx)
        modification = modification or 'invert'

        W_orig = self._get_W(layer).clone()
        d = W_orig.shape[0]
        beta = self._get_beta(layer)

        basins, _, _ = find_basins(
            W_orig, d, num_inits=int(self.params['inits']),
            beta=beta, device=DEVICE)

        if basin_idx >= len(basins):
            return f"{C.red('Basin')} {basin_idx} not found. Have {len(basins)}."

        original = basins[basin_idx]

        # Create counterfactual
        if modification == 'invert':
            cf = -original
        elif modification == 'weaken':
            cf = original * 0.5
        elif modification == 'shift':
            noise = torch.randn_like(original) * 0.3
            cf = original + noise
            cf = cf / cf.norm() * original.norm()
        else:
            return f"{C.red('Unknown modification:')} {modification}. Use: invert, weaken, shift"

        # Apply counterfactual temporarily
        W_cf = remove_basin(W_orig, original, strength=self.params['strength'], beta=beta)
        W_cf = inject_basin(W_cf, cf, strength=self.params['strength'], beta=beta)

        basins_cf, _, _ = find_basins(
            W_cf, d, num_inits=int(self.params['inits']),
            beta=beta, device=DEVICE)

        affected = []
        for j, b in enumerate(basins):
            if j == basin_idx:
                continue
            exists, final, cos, _ = verify_basin_exists(W_cf, b, beta=beta)
            drift = 1.0 - cos
            if drift > 0.05:
                affected.append((j, drift, exists))

        # DO NOT persist — this is counterfactual only
        # W stays unchanged

        lines = [
            C.bold(f"Counterfactual: B{basin_idx} → {modification}") + f"  L{layer}",
            C.dim(f"  (non-destructive — original landscape preserved)"),
            f"  Basins: {len(basins)} → {len(basins_cf)}",
            "",
        ]
        if affected:
            lines.append(C.bold(f"  Would affect: {len(affected)} basins"))
            for j, drift, survived in sorted(affected, key=lambda x: -x[1]):
                status = C.green("ok") if survived else C.red("gone")
                lines.append(f"    B{j:3d}  drift={drift:.4f}  {status}")
        else:
            lines.append(C.dim("  No downstream effects predicted."))

        self._log(f"causal counterfactual {layer} B{basin_idx} {modification}")
        return "\n".join(lines)

    def cmd_help(self):
        sections = [
            (C.bold("OBSERVE"), [
                ("survey <layer>",         "Find all basins in a layer"),
                ("survey-all",             "Survey all layers"),
                ("verify <layer> <seed>",  "Check if target is a stable basin"),
                ("energy <layer> <seed>",  "Measure energy at a point"),
                ("probe <layer> [idx]",    "What tokens does basin #idx activate?"),
                ("landscape <layer>",      "Map full energy landscape"),
            ]),
            (C.bold("MODIFY"), [
                ("inject <layer> <seed> [α]",    "Inject a new basin"),
                ("remove <layer> <seed> [α]",    "Remove closest basin"),
                ("move <layer> <seed> [α]",      "Move closest basin to new location"),
                ("strengthen <layer> <seed> [f]", "Deepen a basin (factor)"),
                ("weaken <layer> <seed> [f]",     "Shallow a basin (factor)"),
            ]),
            (C.bold("CONCEPTS"), [
                ("capture <layer> <name> <text>", "Capture concept from text"),
                ("inject-concept <layer> <name>", "Inject concept as basin"),
                ("remove-concept <layer> <name>", "Remove concept basin"),
                ("blend <a> <b> <new> [ratio]",   "Blend two concepts"),
                ("concepts",                      "List captured concepts"),
                ("export-concept <name> <path>",  "Export concept to file"),
                ("import-concept <path>",         "Import concept from file"),
            ]),
            (C.bold("CONTROL"), [
                ("load <path>",       "Load a checkpoint"),
                ("apply <layer>",     "Write modified W to live model"),
                ("restore <layer>",   "Undo modifications to a layer"),
                ("restore-all",       "Undo all modifications"),
                ("save <path>",       "Save modified W matrices"),
                ("status",            "Show modification status"),
                ("info",              "Show model info"),
            ]),
            (C.bold("CAUSAL"), [
                ("causal scan <layer> [threshold]",          "Discover causal links between basins"),
                ("causal intervene <layer> <basin> [op]",    "do(X) — intervene and measure effects"),
                ("causal counterfactual <layer> <basin> [m]","What if X had been different? (safe)"),
            ]),
            (C.bold("SAFETY"), [
                ("guard <type> <val>",  "Set guard (max-basins, min-basins, strength-cap, ...)"),
                ("guards",              "Show active guards"),
                ("diff <layer>",        "Show W change statistics"),
                ("history",             "Show operation log"),
                ("benchmark",           "Measure perplexity"),
            ]),
            (C.bold("OTHER"), [
                ("generate <prompt>",      "Generate text (requires model+tokenizer)"),
                ("set <param> <value>",    "Set parameter (beta, strength, temp, ...)"),
                ("help",                   "This help"),
                ("quit / exit / Ctrl+D",   "Exit"),
            ]),
        ]

        lines = [
            "",
            C.bold("Qriton HLM — Energy Language Commands"),
            C.dim("─" * 60),
            "",
        ]
        for header, cmds in sections:
            lines.append(f"  {header}")
            for cmd, desc in cmds:
                lines.append(f"    {cmd:35s} {C.dim(desc)}")
            lines.append("")

        lines.extend([
            C.dim("─" * 60),
            f"  {C.dim('Guards:')}  --force --reason \"...\" to override blocked operations",
            f"  {C.dim('Scripts:')} qriton-hlm --script program.hlm",
            f"  {C.dim('Pipe:')}    echo \"survey 0\" | qriton-hlm -c model.pt",
            "",
            f"  {C.dim('Docs:')}    hlm.qriton.com/energy-language/operations",
            f"  {C.dim('Source:')}   github.com/qriton/energy-lang",
            f"  {C.dim('PyPI:')}    pip install qriton-hlm",
            "",
        ])
        return "\n".join(lines)

    # ── Dispatcher ───────────────────────────────────────────────────

    def execute(self, line):
        """Execute a single command line."""
        line = line.strip()
        if not line or line.startswith('#'):
            return ""

        parts = line.split()
        cmd = parts[0].lower()
        all_args = parts[1:]

        # Extract --force / --reason from args
        args, force, reason = self._parse_force(all_args)
        args_str = " ".join(args)

        try:
            if cmd == 'load':
                return self.load_checkpoint(args_str)
            elif cmd == 'info':
                return self.cmd_info()
            elif cmd == 'survey' and args:
                return self.cmd_survey(args[0])
            elif cmd in ('survey-all', 'surveyall'):
                return self.cmd_survey_all()
            elif cmd == 'inject' and len(args) >= 2:
                return self.cmd_inject(args[0], args[1],
                                       args[2] if len(args) > 2 else None,
                                       force=force, reason=reason)
            elif cmd == 'remove' and len(args) >= 2:
                return self.cmd_remove(args[0], args[1],
                                       args[2] if len(args) > 2 else None,
                                       force=force, reason=reason)
            elif cmd == 'move' and len(args) >= 2:
                return self.cmd_move(args[0], args[1],
                                     args[2] if len(args) > 2 else None)
            elif cmd == 'verify' and len(args) >= 2:
                return self.cmd_verify(args[0], args[1])
            elif cmd == 'apply' and args:
                return self.cmd_apply(args[0])
            elif cmd == 'restore' and args:
                return self.cmd_restore(args[0])
            elif cmd in ('restore-all', 'restoreall'):
                return self.cmd_restore_all()
            elif cmd == 'status':
                return self.cmd_status()
            elif cmd == 'capture' and len(args) >= 2:
                layer_str = args[0]
                rest = args_str[len(layer_str):].strip()
                rest_parts = rest.split(None, 1)
                if len(rest_parts) >= 2 and not rest_parts[0].startswith('"'):
                    return self.cmd_capture(layer_str, rest_parts[1], rest_parts[0])
                else:
                    return self.cmd_capture(layer_str, rest)
            elif cmd in ('inject-concept', 'inject_concept') and len(args) >= 2:
                return self.cmd_inject_concept(args[0], args[1],
                                               args[2] if len(args) > 2 else None,
                                               force=force, reason=reason)
            elif cmd in ('remove-concept', 'remove_concept') and len(args) >= 2:
                return self.cmd_remove_concept(args[0], args[1],
                                               args[2] if len(args) > 2 else None,
                                               force=force, reason=reason)
            elif cmd == 'concepts':
                return self.cmd_concepts()
            elif cmd == 'blend' and len(args) >= 3:
                return self.cmd_blend(args[0], args[1], args[2],
                                     args[3] if len(args) > 3 else None)
            elif cmd == 'strengthen' and len(args) >= 2:
                return self.cmd_strengthen(args[0], args[1],
                                          args[2] if len(args) > 2 else None)
            elif cmd == 'weaken' and len(args) >= 2:
                return self.cmd_weaken(args[0], args[1],
                                      args[2] if len(args) > 2 else None)
            elif cmd == 'energy' and len(args) >= 2:
                return self.cmd_energy(args[0], args[1])
            elif cmd in ('export-concept', 'export_concept') and len(args) >= 2:
                return self.cmd_export_concept(args[0], args[1])
            elif cmd in ('import-concept', 'import_concept') and args:
                return self.cmd_import_concept(args[0])
            elif cmd == 'probe' and args:
                return self.cmd_probe(args[0], args[1] if len(args) > 1 else None)
            elif cmd == 'landscape' and args:
                return self.cmd_landscape(args[0])
            elif cmd == 'guard' and len(args) >= 2:
                return self.cmd_guard(args[0], args[1])
            elif cmd == 'guards':
                return self.cmd_guards()
            elif cmd == 'benchmark':
                return self.cmd_benchmark()
            elif cmd == 'causal' and args:
                subcmd = args[0]
                if subcmd == 'scan' and len(args) >= 2:
                    return self.cmd_causal_scan(args[1],
                                                args[2] if len(args) > 2 else None)
                elif subcmd == 'intervene' and len(args) >= 3:
                    return self.cmd_causal_intervene(args[1], args[2],
                                                     args[3] if len(args) > 3 else None)
                elif subcmd == 'counterfactual' and len(args) >= 3:
                    return self.cmd_causal_counterfactual(args[1], args[2],
                                                          args[3] if len(args) > 3 else None)
                else:
                    return (f"  {C.bold('causal scan')} <layer> [threshold]       — discover causal graph\n"
                            f"  {C.bold('causal intervene')} <layer> <basin> [op] — do(X) intervention\n"
                            f"  {C.bold('causal counterfactual')} <layer> <basin> [mod] — what if X was different?")
            elif cmd in ('generate', 'gen') and args_str:
                return self.cmd_generate(args_str)
            elif cmd == 'set' and len(args) >= 2:
                return self.cmd_set(args[0], args[1])
            elif cmd == 'save' and args:
                return self.cmd_save(args[0])
            elif cmd == 'diff' and args:
                return self.cmd_diff(args[0])
            elif cmd == 'history':
                return self.cmd_history()
            elif cmd == 'help':
                return self.cmd_help()
            elif cmd in ('quit', 'exit', 'q'):
                return '__EXIT__'
            else:
                return f"{C.red('Unknown:')} {cmd}. Type {C.bold('help')} for commands."
        except Exception as e:
            return f"{C.red('Error:')} {e}"

    # ── REPL ─────────────────────────────────────────────────────────

    def repl(self):
        """Interactive read-eval-print loop."""
        self._discovered_checkpoints = find_checkpoints() if not self.checkpoint_path else []
        print_banner(self.checkpoint_path)

        if self.checkpoint_path:
            print(f"  {C.dim('Model:')} {self.checkpoint_path}\n")

        setup_completer(self)

        while True:
            try:
                label = self.model_label or 'no-model'
                mod = '*' if self._w_backups else ''
                prompt = f"hlm:{C.cyan(label)}{C.yellow(mod)}> "
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                break

            result = self.execute(line)
            if result == '__EXIT__':
                print(C.dim("Goodbye."))
                break
            if result:
                print(result)

    def run_script(self, script_path):
        """Execute a .hlm script file."""
        with open(script_path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                print(f"{C.dim(f'[{line_no}]')} {line}")
                result = self.execute(line)
                if result == '__EXIT__':
                    break
                if result:
                    print(result)
                print()


def main():
    parser = argparse.ArgumentParser(
        description='Qriton HLM — Energy Language CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qriton-hlm                                    # interactive REPL
  qriton-hlm --checkpoint output-hlm3/model.pt  # load model
  qriton-hlm --script program.hlm               # run script
  echo "survey 0" | qriton-hlm -c model.pt      # pipe commands

Docs:    hlm.qriton.com/energy-language
Source:  github.com/qriton/energy-lang
        """)
    parser.add_argument('--checkpoint', '-c', type=str,
                        help='Load checkpoint on startup')
    parser.add_argument('--script', '-s', type=str,
                        help='Run a .hlm script file')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    parser.add_argument('--version', '-v', action='version',
                        version=f'qriton-hlm {VERSION}')
    args = parser.parse_args()

    # Disable colors if not TTY or explicitly requested
    if args.no_color or not sys.stdout.isatty():
        C.disable()

    lang = EnergyLang()

    if args.checkpoint:
        result = lang.load_checkpoint(args.checkpoint)
        print(result)
        print()

    if args.script:
        lang.run_script(args.script)
    elif sys.stdin.isatty():
        lang.repl()
    else:
        # Piped input
        for line in sys.stdin:
            result = lang.execute(line)
            if result == '__EXIT__':
                break
            if result:
                print(result)


if __name__ == '__main__':
    main()
