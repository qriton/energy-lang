"""
qriton_hlm.db — Database sync extension for Qriton HLM.

Bridges SQL databases (SQLite / MSSQL) with BasinSurgeon's energy landscape.
Each DB row becomes a named concept (attractor basin) that can be edited,
strengthened, weakened, or removed without touching the source database.

Architecture:
    SQL DB (source of truth) → HLMSync → BasinSurgeon (editable semantic layer)

The sync requires a loaded model because capture() runs text through the
network to find the natural attractor state. Without model inference, the
"concept" would be an arbitrary vector, not a real energy minimum.

Usage:
    from qriton_hlm.db import HLMSync, SyncConfig

    config = SyncConfig.from_file("hlm_sync_config.json")
    with HLMSync(config) as syncer:
        syncer.sync_row("Products", {"id": 1, "name": "Widget", "price": 9.99})
        syncer.full_sync_table("Products")
        syncer.checkpoint()
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Sequence

from qriton_hlm.core import BasinSurgeon

logger = logging.getLogger("qriton_hlm.db")


class DBType(Enum):
    SQLITE = "sqlite"
    MSSQL = "mssql"


class SyncError(Exception):
    pass


class HLMUnavailableError(SyncError):
    pass


@dataclass
class SyncConfig:
    """Configuration for DB-to-HLM sync."""
    db_type: DBType
    db_connection_string: str
    hlm_checkpoint_path: str
    allowed_tables: list[str]
    batch_size: int = 50
    max_retries: int = 3
    retry_delay_seconds: float = 0.5
    checkpoint_interval_seconds: float = 300.0
    default_strength: float = 0.85
    default_layer: int = 0
    # Path to the HLM model checkpoint (for capture — requires model inference)
    model_checkpoint_path: Optional[str] = None
    device: str = "cpu"

    @classmethod
    def from_file(cls, path: str) -> "SyncConfig":
        with open(path) as f:
            raw = json.load(f)
        raw["db_type"] = DBType(raw["db_type"])
        return cls(**raw)


def _make_concept_name(table: str, row_id: Any) -> str:
    """Deterministic concept name from table + row ID."""
    return f"{table}:{row_id}"


def _serialize_row(row: Dict[str, Any]) -> str:
    """Convert a DB row to text that BasinSurgeon.capture() can process."""
    return json.dumps(row, ensure_ascii=False, default=str)


def _retry(fn: Callable, max_retries: int, delay: float, stats: dict) -> Any:
    """Retry with linear backoff."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            stats["retries"] = stats.get("retries", 0) + 1
            logger.warning("Attempt %d/%d failed: %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(delay * attempt)
    stats["errors"] = stats.get("errors", 0) + 1
    raise SyncError(f"All {max_retries} attempts failed") from last_exc


class HLMSync:
    """Thread-safe sync layer between SQL DB and Qriton HLM.

    Requires a BasinSurgeon with a loaded model for capture() to work.
    If no model is available, use inject_seed_mode=True to create basins
    from deterministic seeds (no text content, just addressable positions).
    """

    def __init__(self, config: SyncConfig, surgeon: Optional[BasinSurgeon] = None):
        self._config = config
        self._lock = threading.Lock()
        self._dirty = False
        self._synced_concepts: set[str] = set()
        self._allowed_tables: set[str] = set(config.allowed_tables)
        self._stats = {
            "synced": 0, "deleted": 0, "errors": 0,
            "retries": 0, "overwrite_warnings": 0,
        }

        # Use provided surgeon or create from checkpoint
        if surgeon is not None:
            self._surgeon = surgeon
        else:
            self._surgeon = BasinSurgeon.from_checkpoint(
                config.hlm_checkpoint_path, device=config.device,
            )
            # Try to load a previous session (saved concepts + W matrices)
            try:
                self._surgeon.load_session(config.hlm_checkpoint_path + ".session")
                logger.info("Loaded previous sync session")
            except (FileNotFoundError, Exception):
                pass

        self._has_model = (self._surgeon._model is not None
                           and self._surgeon._tokenizer is not None)
        self._db_conn = self._connect_db()

    @classmethod
    def from_config(cls, config_path: str) -> "HLMSync":
        return cls(SyncConfig.from_file(config_path))

    def _validate_table(self, table: str) -> None:
        if table not in self._allowed_tables:
            raise ValueError(
                f"Table '{table}' not in allowed_tables. "
                f"Allowed: {sorted(self._allowed_tables)}"
            )

    def _connect_db(self):
        cfg = self._config
        if cfg.db_type == DBType.SQLITE:
            conn = sqlite3.connect(cfg.db_connection_string, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            return conn
        elif cfg.db_type == DBType.MSSQL:
            import pyodbc
            return pyodbc.connect(cfg.db_connection_string, autocommit=False)
        raise ValueError(f"Unsupported db_type: {cfg.db_type}")

    def _do_sync(self, table: str, row_id: Any, row_text: str, strength: float) -> None:
        """Core sync operation: capture row text as concept, inject as basin."""
        concept_name = _make_concept_name(table, row_id)
        layer = self._config.default_layer

        with self._lock:
            # Clear existing concept to prevent sample accumulation
            if concept_name in self._synced_concepts:
                try:
                    self._surgeon.remove_concept(layer, concept_name, strength)
                except (ValueError, KeyError):
                    # Concept not in surgeon's registry — that's OK on first run
                    pass
                self._stats["overwrite_warnings"] += 1
                logger.debug("Cleared existing concept %s before re-sync", concept_name)

            if self._has_model:
                # Full capture: run text through model, get converged Hopfield state
                self._surgeon.capture(layer, row_text, concept_name=concept_name)
                self._surgeon.inject_concept(layer, concept_name, strength)
            else:
                # Seed mode: create basin at deterministic position (no text content)
                # The basin is addressable but doesn't encode the text semantically
                seed = hash(concept_name) & 0x7FFFFFFF
                self._surgeon.inject(layer, seed=seed, strength=strength)
                logger.warning(
                    "No model loaded — %s synced as seed basin only (no text content). "
                    "Load a model for semantic capture.", concept_name
                )

            self._synced_concepts.add(concept_name)
            self._dirty = True

    # ── Public API ──────────────────────────────────────────────

    def sync_row(
        self,
        table: str,
        row: Dict[str, Any],
        id_column: str = "id",
        strength: Optional[float] = None,
    ) -> str:
        """Sync one DB row into HLM as a named concept basin.

        Returns the concept name (e.g. "Products:42").
        """
        self._validate_table(table)
        row_id = row.get(id_column) or row.get(id_column.capitalize())
        if row_id is None:
            raise ValueError(f"Missing id column '{id_column}'")

        row_text = _serialize_row(row)
        strength = strength or self._config.default_strength

        def _op():
            self._do_sync(table, row_id, row_text, strength)

        _retry(_op, self._config.max_retries, self._config.retry_delay_seconds, self._stats)
        self._stats["synced"] += 1
        concept = _make_concept_name(table, row_id)
        logger.info("Synced %s (layer %d)", concept, self._config.default_layer)
        return concept

    def sync_batch(
        self,
        table: str,
        rows: Sequence[Dict[str, Any]],
        id_column: str = "id",
    ) -> list[str]:
        """Sync multiple rows. Continues on individual failures."""
        concepts = []
        for row in rows:
            try:
                concepts.append(self.sync_row(table, row, id_column))
            except SyncError:
                logger.error("Batch: failed row %s", row.get(id_column))
        return concepts

    def delete_row(self, table: str, row_id: Any) -> None:
        """Remove a row's concept from the energy landscape."""
        self._validate_table(table)
        concept_name = _make_concept_name(table, row_id)
        layer = self._config.default_layer

        with self._lock:
            try:
                self._surgeon.remove_concept(layer, concept_name)
            except (ValueError, KeyError):
                # Concept not captured — weaken by seed instead
                seed = hash(concept_name) & 0x7FFFFFFF
                self._surgeon.weaken(layer, seed=seed, factor=0.1)
                logger.warning(
                    "Concept %s not in registry. Weakened by seed fallback.", concept_name
                )
            self._synced_concepts.discard(concept_name)
            self._dirty = True

        self._stats["deleted"] += 1
        logger.info("Deleted %s", concept_name)

    def full_sync_table(
        self,
        table: str,
        id_column: str = "id",
    ) -> int:
        """Sync all rows from a table."""
        self._validate_table(table)
        cursor = self._db_conn.cursor()
        cursor.execute(f"SELECT * FROM [{table}]")
        columns = [desc[0] for desc in cursor.description]

        count = 0
        batch: list[dict] = []
        for db_row in cursor:
            batch.append(dict(zip(columns, db_row)))
            if len(batch) >= self._config.batch_size:
                self.sync_batch(table, batch, id_column)
                count += len(batch)
                batch = []
        if batch:
            self.sync_batch(table, batch, id_column)
            count += len(batch)

        self.checkpoint()
        logger.info("Full sync %s: %d rows", table, count)
        return count

    def checkpoint(self, path: Optional[str] = None) -> None:
        """Save modified W matrices and captured concepts to disk."""
        dest = path or (self._config.hlm_checkpoint_path + ".session")
        with self._lock:
            if self._dirty:
                self._surgeon.save_checkpoint(dest)
                self._dirty = False
                logger.info("Checkpoint saved to %s", dest)

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    @property
    def synced_concepts(self) -> set[str]:
        return set(self._synced_concepts)

    def close(self) -> None:
        self.checkpoint()
        self._db_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class CheckpointWorker:
    """Daemon thread that periodically saves HLM state."""

    def __init__(self, syncer: HLMSync, interval: Optional[float] = None):
        self._syncer = syncer
        self._interval = interval or syncer._config.checkpoint_interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="hlm-db-checkpoint",
        )
        self._thread.start()
        logger.info("CheckpointWorker started (interval=%ds)", self._interval)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._syncer.checkpoint()
            except Exception as e:
                logger.error("Background checkpoint failed: %s", e)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("CheckpointWorker stopped")
