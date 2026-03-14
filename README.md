# Qriton HLM — Energy Language

**Program neural networks by shaping energy landscapes.**

The first tool that lets you directly program a neural network's behavior —
not train, not fine-tune, not prompt-engineer. **Program.**

> *"Energy minima as a programming language — in a completely new fashion."*
> — **John J. Hopfield**, March 2026

```
$ qriton-hlm -c model.pt

hlm:model> capture 5 polite Thank you so much for your help
  Captured L5: "Thank you so much for your help"
  Energy: -12.34 | Basin: True (cos=0.97, 23 iters)
  -> Added to concept 'polite' (1 samples)

hlm:model> capture 5 polite I really appreciate your patience
  -> Added to concept 'polite' (2 samples)

hlm:model> inject-concept 5 polite 0.1
  Before: 200 basins, concept is basin: False
  After:  201 basins (+1), concept is basin: True
  >> Concept successfully injected!

hlm:model> apply 5
hlm:model> generate Tell me about the weather
  I'd be happy to share! The weather today is...
```

## Install

```bash
pip install qriton-hlm
```

## What is this?

Every AI framework today has one way to change model behavior: **training**.
Qriton HLM adds a second: **surgery**.

This only works because HLM uses polynomial Hopfield dynamics that create
**multiple stable basins** per layer. Transformers can't do this — their
softmax attention collapses to a single attractor.

## Operations — Energy Language

### Observe (read the landscape)

| Command | What it does |
|---------|-------------|
| `survey <layer>` | Map all basins in a layer |
| `survey-all` | Survey every layer |
| `verify <layer> <seed>` | Check if a point is a basin |
| `energy <layer> <seed>` | Measure energy at a point |
| `probe <layer> [basin_idx]` | What tokens does a basin activate? (reverse capture) |
| `landscape <layer>` | Full energy map with population bars |

### Modify (change the landscape)

| Command | What it does |
|---------|-------------|
| `inject <layer> <seed> [str]` | Create a new attractor |
| `remove <layer> <seed> [str]` | Destroy an attractor |
| `move <layer> <seed> [str]` | Relocate an attractor |
| `strengthen <layer> <seed> [f]` | Deepen an existing basin |
| `weaken <layer> <seed> [f]` | Make a basin shallower |

### Concept (semantic operations)

| Command | What it does |
|---------|-------------|
| `capture <layer> <concept> <text>` | Extract what a concept looks like in the model |
| `inject-concept <layer> <concept> [s]` | Program a captured concept as a new attractor |
| `remove-concept <layer> <concept> [s]` | Remove a concept from the model |
| `blend <a> <b> <new> [ratio]` | Mix two concepts (e.g. 70% polite + 30% formal) |
| `concepts` | List all captured concepts |
| `export-concept <name> <path>` | Save concept as portable file |
| `import-concept <path>` | Load concept from file |

### Control (flow & persistence)

| Command | What it does |
|---------|-------------|
| `load <path>` | Load a checkpoint |
| `apply <layer>` | Write modified W to live model |
| `restore <layer>` | Undo modifications |
| `restore-all` | Restore all layers |
| `save <path>` | Save modified W matrices |
| `status` | Show which layers are modified |
| `set <param> <value>` | Set parameter (beta, strength, ...) |
| `info` | Show model info |

### Causal (discovery & intervention)

| Command | What it does |
|---------|-------------|
| `causal scan <layer> [threshold]` | Discover causal links between basins (systematic knockout) |
| `causal intervene <layer> <basin> [op]` | do(X) — intervene on a basin, measure downstream effects |
| `causal counterfactual <layer> <basin> [mod]` | "What if X was different?" (non-destructive) |

Each basin = a causal node. Surgery = the do-operator. The energy landscape becomes a programmable structural causal model. See [Causal Programming Docs](https://github.com/qriton/energy-lang/blob/master/docs/causal.md) for the full framework.

### Safety (guards & audit)

| Command | What it does |
|---------|-------------|
| `guard <type> <value>` | Set guard (max-basins, min-basins, strength-cap, cosine-drift, perplexity-delta) |
| `guards` | Show active guards |
| `diff <layer>` | Show W matrix change stats |
| `benchmark` | Measure perplexity impact of surgery |
| `history` | Show operation log with OK/BLOCKED status |

Guards are pre-execution checks. If violated, the operation does not start — weights are never touched. Override with `--force --reason "justification"` (logged permanently).

### Verify & Generate

| Command | What it does |
|---------|-------------|
| `generate <prompt>` | Generate text with current model |

### Scripts

Write `.hlm` scripts to chain operations:

```bash
# make_polite.hlm
load model.pt
capture 5 polite Thank you so much
capture 5 polite I really appreciate it
capture 5 polite That's very kind of you
inject-concept 5 polite 0.1
apply 5
benchmark
generate Tell me about the weather
restore 5
```

Run with: `qriton-hlm --script make_polite.hlm`

## Python API

```python
from qriton_hlm import BasinSurgeon

surgeon = BasinSurgeon.from_checkpoint("model.pt", device="cuda")

# Capture what a concept looks like in the model
surgeon.capture(layer=5, text="Thank you so much", concept_name="polite")
surgeon.capture(layer=5, text="I really appreciate it", concept_name="polite")

# Inject that concept as a new attractor
result = surgeon.inject_concept(layer=5, concept_name="polite", strength=0.1)
print(f"Concept injected: {result['exists_after']}")

# Blend two concepts
surgeon.blend("polite", "formal", "professional", ratio=0.6)

# Export concept for sharing
surgeon.export_concept("polite", "polite.concept")

# Transplant concept from another model
other = BasinSurgeon.from_checkpoint("other_model.pt")
surgeon.transplant(other, layer=5, concept_name="humor")

# Apply to live model and benchmark
surgeon.apply(layer=5)
result = surgeon.benchmark()
print(f"PPL after surgery: {result['perplexity']:.2f}")

# Probe: what does basin #3 represent?
probe = surgeon.probe(layer=5, basin_idx=3)
print(f"Top tokens: {probe['top_tokens']}")

# Compare basins between models
diff = surgeon.compare(other, layer=5)
print(f"Shared: {diff['shared']}, unique: {diff['only_self']}")

# --- Causal discovery ---

# Discover causal graph: which basins cause which?
graph = surgeon.causal_scan(layer=5, threshold=0.15)
for edge in graph['edges']:
    print(f"B{edge['source']} → B{edge['target']}  drift={edge['drift']:.3f}")

# do(X) — intervene on basin 3, measure downstream effects
result = surgeon.causal_intervene(layer=5, basin_idx=3, operation='remove')
print(f"Affected: {result['num_affected']} basins")

# Counterfactual: what if basin 3 had been inverted? (non-destructive)
cf = surgeon.causal_counterfactual(layer=5, basin_idx=3, modification='invert')
print(f"Would affect: {cf['num_affected']} basins")
```

## Compatibility

Works with any PyTorch checkpoint that contains `hopfield.W` parameters:
- HLM2/HLM3 language models
- HLM-Spatial (LIDAR, Medical3D, Industrial3D)
- HLM-Audio (STT, TTS)
- Any model using PolyHopfieldLayer

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- NumPy

## License

**Business Source License 1.1** — free for research, education, evaluation, and non-commercial use. Commercial use requires a license. Converts to Apache 2.0 on April 1, 2030.

See [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md) for details.

Commercial licensing: license@qriton.com

Qriton Technologies S.R.L. — qriton.com
