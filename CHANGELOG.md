# Changelog

All notable changes to `qriton-hlm` are documented here. Follows [Semantic Versioning](https://semver.org/).

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
