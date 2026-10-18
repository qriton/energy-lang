# Changelog

All notable changes to `qriton-hlm` are documented here. Follows [Semantic Versioning](https://semver.org/).

## v0.13.0 (2026-10-18)

Edits now write at their target and leave the protected basins' field unchanged,
and every write path goes through a decision-stability guard. Default behaviour
changes, hence the 0.x minor bump. Every API change is listed in
[MIGRATION.md](MIGRATION.md); measurements are in [RESULTS.md](RESULTS.md).

### Measured

On synthetic landscapes that store known concepts (d=64/K=5 and d=128/K=10;
20 landscapes per row). No trained HLM checkpoint has been measured yet.

| | 0.12.0 inject (strength 0.5) | 0.13.0 inject (strength auto, ≈0.43) |
|---|---|---|
| target became an attractor | 0% (0/120 at any strength tried) | 100% |
| other concepts' cues changed | 37–39% | 0% |
| other concepts still recalled | 62–66% | 98–99% |
| protected basins' field `W f(b)` | changed | unchanged (exact, tested) |

- remove (auto, ≈0.29): target removed 100%, other concepts' cues changed 1%.
  `BasinSurgeon.remove` was refused 60/60 in 0.12.0.
- move (auto, ≈0.48): target reached 95%, other concepts' cues changed 1%.
- At auto strength the guard accepts 15–50% of edits when it protects every
  visible basin (spurious mixtures included) and 90–100% when it protects only
  the stored concepts.
- Edits cost more: the guard converges the probe set twice. The test suite went
  from about 7 s to about 1 min on the machine where the fix was measured. On
  the release machine, under heavy load, its CPU time went from 16 s (0.12.0, 69
  tests) to 35 s (0.13.0, 96 tests).

### Changed (default behaviour)

- `inject_basin` / `remove_basin` / `move_basin` default to `mode="projected"`.
  Inject writes the target's sign pattern (×0.9) instead of reinforcing the basin
  the target falls into, and every write is projected away from the protected
  basins (by default every basin `find_basins` sees from 64 random starts), so
  their field `W f(b)` is unchanged. `mode="legacy"` reproduces 0.12.0's writes.
- `BasinSurgeon.inject`, `remove`, `move`, `inject_concept` and `remove_concept`
  use `strength="auto"` when no strength is given (projected mode).
- Every `BasinSurgeon` write goes through the gate: the retention certificate
  plus `certify_decisions` (at most 2% of the protected basins' cue decisions may
  change). Newly gated: `remove_concept`, `causal_intervene`, `consolidate`
  (and so `dream`). A refused edit leaves W unchanged and returns
  `applied: False`.
- CLI: `remove-concept`, `strengthen`, `weaken` and `causal intervene` are now
  gated; every write is projected and uses the layer's beta.

### Fixed

- `inject_basin` wrote on whatever basin the target converged to (an existing
  concept, its mirror or a spurious mixture), never at the target.
- `remove_basin` / `move_basin` now protect the other basins the same way; basins
  built from a removed target (`dependent_basins`) are not protected.
- `BasinSurgeon.remove` was refused every time (the target was in its own
  protected set). `remove_concept`, `consolidate`, `causal_intervene` and the
  CLI `remove-concept`, `strengthen`, `weaken` and `causal intervene` wrote
  without any check.
- CLI inject/remove/move ignored the layer's beta.
- `BasinSurgeon.guard()` settings were stored but never read; `max_changed_pct`
  is now enforced.
- Projected writes no longer write rounding noise when the protected states span
  every direction (at least as many protected states as dimensions). They
  return W unchanged instead.
- CLI `--version`, the banner and exported concept files reported 0.9.4.
- CLI "target is basin" and `verify` checked the unit-norm target, which settles
  to zero, and printed False after a successful inject. They now check the
  target's sign pattern, the state the write creates.

### Added

- `certify_decisions(W, W_new, ...)`: decision-stability guard. It measures the
  fraction of probe decisions (noisy cues of protected basins) whose settling
  point moves and returns the changed probes. Results carry it under
  `retention["decision"]`.
- `strength="auto"`: the smallest strength on `AUTO_STRENGTHS` that does the edit.
- `mode="legacy"` (functions) and `params['edit_mode'] = 'legacy'` (`BasinSurgeon`).
- `BasinSurgeon.guard(layer, protected=[...])` pins the protected set for a layer,
  e.g. to protect only real concepts.
- Guard sample size: `certify_decisions(..., n_cue=64, max_probes=4096)`,
  `default_probes(..., max_probes=None)` and `BasinSurgeon.params['guard_probes']`.
  The default cap never binds for the default protected set (at most 64 basins).
- `HLMSync.stats["refused"]` counts sync/delete writes the gate refused, with a
  warning in the log.
- `measure_edits.py` and `RESULTS.md`: before/after measurements.

### Upgrading

- Read [MIGRATION.md](MIGRATION.md). Check `result["applied"]` after every
  `BasinSurgeon` edit. A refusal is normal and leaves W unchanged;
  `result["retention"]["decision"]["changed"]` lists the probes that would move.
- To reproduce 0.12.0, pass `mode="legacy"` to the functions or set
  `surgeon.params['edit_mode'] = 'legacy'`. In legacy mode `remove_concept`,
  `causal_intervene` and `consolidate` are still gated by the 0.12.0 certificate;
  add `force=True` to write unconditionally as 0.12.0 did.
- If edits are refused because spurious basins are protected, protect only the
  real concepts: `surgeon.guard(layer, protected=concept_states)`, or
  `protected=` on the functions.
- `strengthen`, `weaken`, `consolidate` and the `causal_*` methods do not use
  `auto`. They derive strength from `params['strength']` (1.0 by default), which
  the guard usually refuses. Lower it (measured auto strengths were 0.3–0.5).
- CLI: pass `auto` as the strength (`inject 0 42 auto`). The CLI default is
  `set strength` (0.1), usually too weak to create an attractor.

## v0.12.0 (2026-09-02)

### Changed

- Landscape writes (`inject`, `remove`, `move`, `strengthen`, `weaken`,
  `inject_concept`) now check that existing attractors still survive before the
  new weights are kept. If they would not, the layer is left unchanged and the
  command is refused.
- CLI refuses the same way unless `--force --reason "..."` is passed.
- Low-level `inject_basin` / `remove_basin` are unchanged for callers that
  apply their own checks.

### Added

- `certify_compact_edit(W, W_new, protected_basins)` — retention check used by
  the high-level API. Results include `retention` (`accepted`, `applied`).
- [How to edit](docs/how-to-edit.md) — compact write if it certifies; otherwise
  land as an explicit slot. The trunk stays frozen.

## v0.11.2 (2026-08-08)

### Added

- README: documented the no-checkpoint sandbox (`load __random_<dim>`) and the `examples/` walkthroughs.

## v0.11.1 (2026-08-08)

### Changed

- Docs: README example cleanup. No code changes.

## v0.11.0 (2026-08-08)

### Added

- **No-checkpoint sandbox**: `load __random_<dim>` instantiates a single random Hopfield W,
  so the surgery commands can be exercised without downloading any weights.
- **HLM3-Mix research-preview walkthroughs**: `examples/hlm3_mix_35m_k16.hlm` (DSL) and
  `examples/hlm3_mix_35m_k16.py` (BasinSurgeon API), plus `examples/README.md`.

### Fixed

- Checkpoints with 0-dim scalar `.gate` buffers now load cleanly (previously only `log_beta`
  scalars were unsqueezed).
- The load/`info` summary reports the correct `d` for checkpoints whose config uses
  `embeddingDim` (or `d_model`) rather than `dModel`.

## v0.10.0 (2026-04-06)

> Reconciled into git on 2026-08-08 — 0.10.0 was published to PyPI from an offline working
> copy and never committed (git master remained at 0.9.5); the source here was restored from
> the published sdist so the repository matches what is live.

### Added

- **Concept algebra**: `similarity`, `add`, `subtract`, `analogy`, `compose`, `interpolate`.
- **Consolidation (sleep cycle)**: `consolidate`, `dream`.
- **Watermarking**: `watermark_inject`, `watermark_verify`, `watermark_strip_attempt`.
- `load_model` and additional surgery API surface.

## v0.9.5 (April 2026)

### Added

- **Database sync extension** (`qriton_hlm.db`) — bridge SQL databases with HLM energy landscapes
  - `HLMSync` class: `sync_row()`, `sync_batch()`, `full_sync_table()`, `delete_row()`
  - Supports SQLite and MSSQL (via pyodbc)
  - Thread-safe with concept accumulation guard
  - `CheckpointWorker` for periodic background saves
  - `SyncConfig.from_file()` for JSON-based configuration
- **Persistence methods** on `BasinSurgeon`
  - `save_checkpoint(path)` — saves W matrices, captured concepts, and operation history
  - `load_session(path)` — restores full state for round-trip workflows
- **Optional dependency groups**: `pip install qriton-hlm[db]`, `pip install qriton-hlm[agent]`
- **Test suite** — 65 tests covering core operations, persistence, DB sync, and causal operations
- **CI/CD** — GitHub Actions workflow for multi-OS, multi-Python testing
- Spatial and audio model documentation

### Improved

- `--version` and `--no-color` CLI flags now documented
- `trace()` method added to Python API reference

## v0.9.4 (April 2026)

### Added

- **Causal programming** — 3 new operations for causal discovery and intervention
  - `causal_scan()`, `causal_intervene()`, `causal_counterfactual()`
- **Safety system** — 5 guard types with `--force --reason` override
- Integration support for DoWhy and CausalNex

## v0.9.3 (March 2026)

### Added

- `blend` operation, `export-concept` / `import-concept`, `transplant`
- Jupyter `%%hlm_landscape` magic

## v0.9.2 (February 2026)

### Added

- `guard`, `history`, `diff` operations
- Gradio web UI

## v0.9.1 (January 2026)

### Added

- `probe`, `landscape`, `strengthen`, `weaken` operations
- HLM Scripts (`.hlm` files)

## v0.9.0 (December 2025)

### Initial release

- Core `BasinSurgeon` with 26 operations
- CLI / REPL, Python API, Jupyter integration
- HLM3, HLM-Spatial, HLM-Audio checkpoint support
