# Changelog

All notable changes to `qriton-hlm` are documented here. Follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- **No-checkpoint sandbox**: `load __random_<dim>` instantiates a single random
  Hopfield W matrix, letting first-time users exercise the surgery commands
  without downloading any weights.
- **Examples for the HLM3-Mix research-preview checkpoint**
  - `examples/hlm3_mix_35m_k16.hlm` — DSL walkthrough (load → survey-all →
    baseline generate → capture / guard / inject-concept / apply / diff →
    generate → restore).
  - `examples/hlm3_mix_35m_k16.py` — same workflow via the `BasinSurgeon`
    Python API.
  - `examples/README.md` — index of all walkthroughs.
- README section listing the runnable example walkthroughs.

### Fixed

- Checkpoints with 0-dim scalar `.gate` buffers now load cleanly (previously
  only `log_beta` scalars were unsqueezed; some HLM3-Mix variants persist
  gate parameters as 0-dim tensors).
- The `info` summary now reports the correct `d` value for checkpoints whose
  config uses `embeddingDim` rather than `dModel` (was showing `d=?`).

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
