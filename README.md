# Qriton HLM — Energy Language

Energy Language is a toolkit for inspecting and editing polynomial Hopfield
networks. It surveys the basins (stable states) of a Hopfield layer and edits
the layer's weights directly: inject, remove or move a basin. Every write made
through the CLI or `BasinSurgeon` is checked before it is kept, and `restore`
returns a layer to its original weights exactly.

```
$ qriton-hlm -c model.pt

hlm:model> survey 0
Layer 0: 15 basins  (200 inits, β=7.0, 0.2s)
  W: 64×64, spectral=1.3867, avg convergence: 10.8 iters

  B 0  E=+13.6172  pop= 33 (16.5%)  █████
  B 1  E=+14.8441  pop= 24 (12.0%)  ████
  ...

hlm:model> inject 0 3 auto
Inject L0 seed=3 α=auto
  Before: 13 basins, target is basin: False (cos=-0.0951)
  After:  14 basins (+1), target is basin: True (cos=1.0000)
  >> Basin successfully programmed!

hlm:model> inject 0 10 auto
Inject L0 seed=10 α=auto
  REFUSED: decisions_changed
  kept 14/14 protected basins
  decisions changed: 3.0% of 896 probes (limit 2%)
  Other decisions would change with this update.
  Override: --force --reason "..." to write anyway
```

Real output on a synthetic checkpoint (d=64, 5 stored concepts). The first
inject was kept. The second would have changed 3.0% of the existing basins'
decisions, more than the 2% budget, so the command printed `REFUSED` and left
the layer unchanged. Basin counts come from random starts, so they vary between
runs. Validation so far is on synthetic landscapes. Validation on trained HLM3
checkpoints is in progress; early results show the decision guard refusing
edits that would change more than its budget. Details will follow.

## Install

```bash
pip install qriton-hlm
```

### Try it without a checkpoint

The `load __random_<dim>` form spins up a single random Hopfield W, so you can exercise every
surgery command with no weights to download:

```bash
qriton-hlm
hlm> load __random_256      # one random W at d=256
hlm> survey 0               # map its basins
hlm> energy 0 3             # inspect a point
```

Runnable walkthroughs live in [`examples/`](examples/), including the HLM3-Mix 35M K=16 preview.

## What is this?

Changing a trained model's behaviour usually means more training. Energy
Language takes a different route for Hopfield layers: it edits their weights
directly, one basin at a time (basin surgery).

This relies on polynomial Hopfield dynamics, which can hold **several stable
basins** per layer.

## How an edit works (0.13)

- **It writes at the target.** An inject adds the target's sign pattern as an
  attractor. The write is projected so that the field at every *protected*
  basin, `W f(b)`, does not change. Remove and move work the same way.
- **It picks its own strength.** With `strength="auto"` (the Python default;
  pass `auto` in the CLI) the smallest strength that does the edit is used.
- **It is gated.** A write is kept only if the protected basins still exist and
  at most **2%** of their decisions change. Decisions here are noisy cues of
  each basin, settled by the model's own dynamics. Otherwise the edit is
  refused and the layer stays as it was. Python results say `applied: True` or
  `False`. The gate covers the CLI and the `BasinSurgeon` methods. The
  module-level `inject_basin`, `remove_basin` and `move_basin` return a
  proposed W without checking it (check it with `certify_decisions`), and
  `force=True` keeps a refused write.

Measured on synthetic landscapes with known concepts (20 per row, see
[RESULTS.md](RESULTS.md)):

| | 0.12 | 0.13 |
|---|---|---|
| inject reaches its target | 0% (any strength tried) | 100% (auto strength) |
| other concepts' cues changed by an inject | 37–39% (strength 0.5) | 0% (auto), 1% (0.5) |
| remove reaches its target / other cues changed | 100% / 2% at 0.5, but `BasinSurgeon.remove` was always refused | 100% / 1% (auto) |

These are synthetic results. Validation on trained HLM3 checkpoints is in
progress; early results show the decision guard refusing edits that would change
more than its budget. Results will be published separately. By
default the protected basins are everything a random-start survey finds,
spurious mixtures included, and that makes the gate strict: at auto strength it
accepted 15–50% of edits, against 90–100% when only the real concepts were
protected. Use `guard(layer, protected=...)` to protect what matters. Edits also
cost more than in 0.12 because the gate re-settles its probes.

Upgrading from 0.12: see [MIGRATION.md](MIGRATION.md). `mode="legacy"` keeps
the 0.12 behaviour.

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

Landscape writes (`inject`, `remove`, `move`, `strengthen`, `weaken`,
`inject-concept`, `remove-concept`, `causal intervene`) are applied only if the
protected basins survive and at most 2% of their decisions change. Otherwise the
command is refused and the layer is left unchanged. `--force --reason`
overrides this for experiments on `inject`, `remove`, `inject-concept` and
`remove-concept`.

See [How to edit](docs/how-to-edit.md) for what the gate checks and what to do
when it refuses an edit.

### Modify (change the landscape)

| Command | What it does |
|---------|-------------|
| `inject <layer> <seed> [str\|auto]` | Create a new attractor |
| `remove <layer> <seed> [str\|auto]` | Destroy an attractor |
| `move <layer> <seed> [str\|auto]` | Relocate an attractor |
| `strengthen <layer> <seed> [f]` | Deepen an existing basin (strength × f) |
| `weaken <layer> <seed> [f]` | Make a basin shallower (strength × f) |

`auto` uses the smallest strength that does the edit. Without it, the strength
is `set strength` (default 0.1), which is usually too weak to create a new
attractor (0 of 40 targets at 0.1 on the synthetic landscapes).

### Concept (semantic operations)

| Command | What it does |
|---------|-------------|
| `capture <layer> <concept> <text>` | Extract what a concept looks like in the model |
| `inject-concept <layer> <concept> [s\|auto]` | Program a captured concept as a new attractor |
| `remove-concept <layer> <concept> [s\|auto]` | Remove a concept from the model |
| `blend <a> <b> <new> [ratio]` | Mix two concepts (e.g. 70% polite + 30% formal) |
| `concepts` | List all captured concepts |
| `export-concept <name> <path>` | Save concept as portable file |
| `import-concept <path>` | Load concept from file |

### Concept Algebra <sup>v0.10</sup>

| Command | What it does |
|---------|-------------|
| `similarity <a> <b>` | Cosine similarity between two concept centroids |
| `add <a> <b> <new>` | Vector addition: A + B |
| `subtract <a> <b> <new>` | Vector subtraction: what's in A but not B |
| `analogy <a> <b> <c> <new>` | A is to B as C is to ? (king - man + woman = queen) |
| `compose <names> <weights> <new>` | Weighted sum of N concepts |
| `interpolate <a> <b> [steps]` | Measure similarity along A-to-B path |

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

Each basin is treated as a causal node and an edit as the do-operator. See [Causal Programming Docs](https://github.com/qriton/energy-lang/blob/master/docs/causal.md) for the method.

### Consolidation (Sleep Cycle) <sup>v0.10</sup>

| Command | What it does |
|---------|-------------|
| `consolidate <layer> [str_f] [prune_t]` | Memory consolidation: strengthen popular basins, prune weak ones |
| `dream [cycles]` | Full sleep cycle across all layers, multiple passes |

Loosely modelled on memory consolidation during sleep: high-population basins (frequently visited) are strengthened and low-population basins (rarely visited) are pruned. The whole pass is gated: kept entirely or not at all.

Concept algebra and consolidation are Python API methods (see [Python API](#python-api)); the 0.13.0 CLI does not include these commands.

### Watermark (experimental) <sup>v0.10</sup>

| Python API | What it does |
|------------|-------------|
| `surgeon.watermark_inject(key, num_marks, strength)` | Write signature basins at states derived from a secret key |
| `surgeon.watermark_verify(key, num_marks)` | Check whether the key's marks are present |
| `surgeon.watermark_strip_attempt(key, num_marks)` | Try to remove your own marks (self-attack) |

Each mark is an ordinary gated `inject` at a seed derived from the key with SHA-256, so the same key always gives the same marks. At the default strength (0.05) a 0.13 write rarely creates the mark: on the synthetic landscapes even 0.1 reached 0 of 40 targets. Pass `strength="auto"`, then check each mark's `applied` (the gate kept it) and `exists_after`. `watermark_verify` tests each mark at the state its inject writes, so it depends on the key: on synthetic landscapes a wrong key and an unmarked model found no marks. In 0.12 it found marks for any key, and models signed with 0.12 or earlier generally do not verify now; sign them again. Effects on model quality and robustness have not been measured, and watermarks have not been tested on trained checkpoints.

### Safety (guards & audit)

| Command | What it does |
|---------|-------------|
| `guard <type> <value>` | Set guard (max-basins, min-basins, strength-cap) |
| `guards` | Show active guards |
| `diff <layer>` | Show W matrix change stats |
| `benchmark` | Measure perplexity impact of surgery |
| `history` | Show operation log with OK/BLOCKED status |

Guards are checked before `inject`, `remove`, `inject-concept` and `remove-concept` run: if one is violated, the command does not run and the layer is unchanged. Override with `--force --reason "justification"` (logged permanently). `strength-cap` does not apply to `auto`. After a write is computed, the decision gate described above checks it as well.

### Verify & Generate

| Command | What it does |
|---------|-------------|
| `generate <prompt>` | Generate text with current model |

### Multi-Model <sup>v0.10</sup>

| Python API | What it does |
|------------|-------------|
| `surgeon.compare(other, layer)` | Compare basin structure: shared, unique-self, unique-other |
| `surgeon.transplant(source, layer, concept)` | Copy concept from another model into this one |

### Database Sync

| Python API | What it does |
|------------|-------------|
| `HLMSync.sync_row(table, row)` | Sync one DB row as a named concept basin |
| `HLMSync.sync_batch(table, rows)` | Sync multiple rows |
| `HLMSync.full_sync_table(table)` | Sync all rows from a table |
| `HLMSync.delete_row(table, row_id)` | Remove a row's concept from the landscape |
| `HLMSync.checkpoint()` | Save state to disk |
| `BasinSurgeon.save_checkpoint(path)` | Save W matrices + concepts + history |
| `BasinSurgeon.load_session(path)` | Restore full session state |

### Scripts

Write `.hlm` scripts to chain operations:

```bash
# make_polite.hlm
load model.pt
capture 5 polite Thank you so much
capture 5 polite I really appreciate it
capture 5 polite That's very kind of you
inject-concept 5 polite auto
apply 5
benchmark
generate Tell me about the weather
restore 5
```

Run with: `qriton-hlm --script make_polite.hlm`. If the gate refuses the
inject, the script continues and `apply` writes the unchanged layer.

## Python API

```python
from qriton_hlm import BasinSurgeon

surgeon = BasinSurgeon.from_checkpoint("model.pt", device="cuda")

# Capture what a concept looks like in the model
surgeon.capture(layer=5, text="Thank you so much", concept_name="polite")
surgeon.capture(layer=5, text="I really appreciate it", concept_name="polite")

# Inject that concept as a new attractor (strength="auto" by default).
# The write is kept only if it passes the gate; otherwise W is unchanged.
result = surgeon.inject_concept(layer=5, concept_name="polite")
print(f"Kept: {result['applied']} ({result['retention']['reason']})")
if not result["applied"]:
    print("Probes that would move:", result["retention"]["decision"]["changed"][:10])

# Protect only the basins you know are real concepts (default: every basin
# a random-start survey finds, which makes the gate strict)
basins = surgeon.survey(layer=5)["basins"]
surgeon.guard(layer=5, protected=[basins[0], basins[3]])

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

# Save session (W matrices + concepts + history)
surgeon.save_checkpoint("session.pt")

# Load it back later
surgeon.load_session("session.pt")

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

# do(X) — intervene on basin 3, measure downstream effects.
# Kept only if it passes the gate (force=True to keep it anyway); strength
# comes from surgeon.params['strength'] (1.0) unless given.
result = surgeon.causal_intervene(layer=5, basin_idx=3, operation='remove', strength=0.3)
print(f"Affected: {result['num_affected']} basins, kept: {result['applied']}")

# Counterfactual: what if basin 3 had been inverted? (non-destructive)
cf = surgeon.causal_counterfactual(layer=5, basin_idx=3, modification='invert')
print(f"Would affect: {cf['num_affected']} basins")

# --- Concept Algebra ---

# Similarity between concepts
sim = surgeon.concept_similarity("polite", "formal")
print(f"Similarity: {sim['similarity']:.4f}")

# Vector arithmetic on concepts
surgeon.concept_add("polite", "formal", "professional")       # A + B
surgeon.concept_subtract("polite", "casual", "formality")     # A - B
surgeon.concept_analogy("man", "king", "woman", "queen")      # A:B::C:?
surgeon.concept_compose(["polite", "formal", "warm"], [0.5, 0.3, 0.2], "ideal_tone")

# Interpolation path
path = surgeon.concept_interpolate("polite", "rude", num_steps=10)
for step in path['steps']:
    print(f"  ratio={step['ratio']:.1f}  sim_A={step['similarity_to_a']:.3f}")

# --- Consolidation (Sleep Cycle) ---

# Single layer: strengthen important basins, prune weak ones.
# The whole pass is gated: kept entirely or not at all.
report = surgeon.consolidate(layer=5, strengthen_factor=1.5, prune_threshold=0.3)
print(f"Basins: {report['basins_before']} -> {report['basins_after']}, kept: {report['applied']}")
print(f"Strengthened: {report['strengthened']}, Pruned: {report['pruned']}")

# Full sleep: all layers, multiple passes
dream = surgeon.dream(cycles=3)
print(f"Dream complete: +{dream['total_strengthened']} -{dream['total_pruned']}")

# --- Watermark (experimental) ---

# Sign your model with watermark basins (each mark is a gated inject)
wm = surgeon.watermark_inject("my-secret-key-2026", num_marks=5, strength="auto")
print(f"Marks present: {sum(m['exists_after'] for m in wm['marks'])}/{wm['total_marks']}")

# Verify a model loaded in a surgeon (e.g. a suspected leak). Each mark is
# tested at its own state; another key or an unmarked model finds none.
result = surgeon.watermark_verify("my-secret-key-2026", num_marks=5)
print(f"Verified: {result['verified']} ({result['marks_found']}/{result['marks_total']} marks found)")

# Test robustness — try stripping your own watermarks
strip = surgeon.watermark_strip_attempt("my-secret-key-2026", num_marks=5)
print(f"Removed: {strip['removed']}, Survived: {strip['survived']}, Absent: {strip['absent']}")
```

## Database Sync — Neural Database Extension

Mirror rows of a SQL database into an energy landscape as named concept basins:

```python
from qriton_hlm.db import HLMSync

with HLMSync.from_config("hlm_sync_config.json") as syncer:
    # Sync rows — each becomes a named concept basin
    syncer.sync_row("Products", {"id": 1, "name": "Widget", "price": 9.99})
    syncer.full_sync_table("Products")
    syncer.delete_row("Products", 1)

    print(syncer.stats)
    # {'synced': 5, 'deleted': 1, 'errors': 0, 'refused': 0, ...}
```

Every sync and delete is a gated edit. A write the gate refuses leaves the
landscape unchanged, is counted in `stats['refused']` and is logged as a
warning.

```bash
# Install with MSSQL support
pip install qriton-hlm[db]
```

SQL stays the source of truth. See the [Database Sync docs](https://hlm.qriton.com/energy-language/database-sync) for configuration and sync strategies.

## Compatibility

Loads any PyTorch checkpoint that contains `hopfield.W` parameters (edits have
been measured on synthetic landscapes so far; see "How an edit works" above):
- HLM2/HLM3 language models
- HLM-Spatial (LIDAR, Medical3D, Industrial3D)
- HLM-Audio (STT, TTS)
- Any model using PolyHopfieldLayer

Surveys and edits need only the W matrices. `capture`, `generate` and
`benchmark` also need a full model checkpoint and its model code (`hlm3_model`
and `data_utils` importable); that code is not part of this package.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- NumPy

## License

**Business Source License 1.1** (source-available) — free for research, education, evaluation, and non-commercial use. Commercial use requires a license. Converts to Apache 2.0 on April 1, 2030.

See [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md) for details.

Commercial licensing: license@qriton.com

Qriton Technologies S.R.L. — qriton.com
