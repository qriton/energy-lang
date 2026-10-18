# Changelog

All notable changes to `qriton-hlm` are documented here. Follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Fixed

- `inject_basin` wrote on whatever basin the target converged to (an existing
  concept, its mirror or a spurious mixture), never at the target. It now writes
  the target's sign pattern and projects the update away from protected basins,
  so their field is unchanged exactly (`mode="projected"`, the default).
- `remove_basin` / `move_basin` protect the other basins the same way; basins
  built from a removed target (`dependent_basins`) are not protected.
- `BasinSurgeon.remove` was refused every time (the target was in its own
  protected set). `remove_concept`, `consolidate`, `causal_intervene` and the
  CLI `remove-concept`, `strengthen`, `weaken` and `causal intervene` wrote
  without any check. Every write path now goes through the gate.
- CLI inject/remove/move ignored the layer's beta.
- `BasinSurgeon.guard()` settings were stored but never read; `max_changed_pct`
  is now enforced.

### Added

- `certify_decisions(W, W_new, ...)` — decision-stability guard: the fraction
  of probe decisions (noisy cues of protected basins) whose settling point
  moves. The gate refuses edits above `params['max_changed']` (2%) and returns
  the changed probes. Results carry it under `retention["decision"]`.
- `strength="auto"` — the smallest strength on `AUTO_STRENGTHS` that does the
  edit. Default for `BasinSurgeon` projected edits without an explicit strength.
- `mode="legacy"` (functions) and `params['edit_mode'] = 'legacy'`
  (`BasinSurgeon`) restore 0.12.0 behaviour exactly.
- `measure_edits.py` and `RESULTS.md`: measured before/after.

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
