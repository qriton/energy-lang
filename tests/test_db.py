"""Tests for qriton_hlm.db — SQL-to-HLM sync extension."""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile

import pytest
import torch

import qriton_hlm
from qriton_hlm.core import BasinSurgeon, poly_interaction
from qriton_hlm.db import (
    HLMSync,
    SyncConfig,
    SyncError,
    HLMUnavailableError,
    CheckpointWorker,
    DBType,
    _concept_seed,
    _make_concept_name,
    _serialize_row,
)


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def demo_db(tmp_path):
    """Create a minimal SQLite DB for testing."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE Products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL
        )
    """)
    conn.executemany(
        "INSERT INTO Products (id, name, price) VALUES (?, ?, ?)",
        [(1, "Widget", 9.99), (2, "Gadget", 19.99), (3, "Doohickey", 4.99)],
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def checkpoint_path(tmp_path):
    """Create a fake checkpoint."""
    path = str(tmp_path / "model.pt")
    d = 64
    W = torch.randn(d, d)
    W = (W + W.T) / 2
    state = {"blocks.0.hopfield.W": W}
    torch.save({"model_state": state}, path)
    return path


@pytest.fixture
def sync_config(demo_db, checkpoint_path):
    return SyncConfig(
        db_type=DBType.SQLITE,
        db_connection_string=demo_db,
        hlm_checkpoint_path=checkpoint_path,
        allowed_tables=["Products"],
        batch_size=2,
        max_retries=2,
        retry_delay_seconds=0.01,
        default_layer=0,
    )


@pytest.fixture
def config_file(sync_config, tmp_path):
    """Write config to JSON file."""
    path = str(tmp_path / "config.json")
    data = {
        "db_type": sync_config.db_type.value,
        "db_connection_string": sync_config.db_connection_string,
        "hlm_checkpoint_path": sync_config.hlm_checkpoint_path,
        "allowed_tables": sync_config.allowed_tables,
        "batch_size": sync_config.batch_size,
        "max_retries": sync_config.max_retries,
        "retry_delay_seconds": sync_config.retry_delay_seconds,
        "default_layer": sync_config.default_layer,
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ── Helpers ─────────────────────────────────────────────────────

class TestHelpers:
    def test_make_concept_name(self):
        assert _make_concept_name("Products", 42) == "Products:42"
        assert _make_concept_name("Users", "abc") == "Users:abc"

    def test_serialize_row(self):
        row = {"id": 1, "name": "Widget", "price": 9.99}
        result = _serialize_row(row)
        parsed = json.loads(result)
        assert parsed["name"] == "Widget"
        assert parsed["price"] == 9.99


# ── SyncConfig ──────────────────────────────────────────────────

class TestSyncConfig:
    def test_from_file(self, config_file):
        config = SyncConfig.from_file(config_file)
        assert config.db_type == DBType.SQLITE
        assert config.allowed_tables == ["Products"]

    def test_from_file_missing(self):
        with pytest.raises(FileNotFoundError):
            SyncConfig.from_file("/nonexistent/config.json")


# ── HLMSync ─────────────────────────────────────────────────────

class TestHLMSync:
    def test_init_with_config(self, sync_config):
        syncer = HLMSync(sync_config)
        assert syncer.stats["synced"] == 0
        syncer.close()

    def test_init_with_custom_surgeon(self, sync_config):
        surgeon = BasinSurgeon.from_checkpoint(sync_config.hlm_checkpoint_path)
        syncer = HLMSync(sync_config, surgeon=surgeon)
        assert syncer.stats["synced"] == 0
        syncer.close()

    def test_from_config(self, config_file):
        syncer = HLMSync.from_config(config_file)
        assert syncer.stats["synced"] == 0
        syncer.close()

    def test_validate_table_rejects_unknown(self, sync_config):
        syncer = HLMSync(sync_config)
        with pytest.raises(ValueError, match="not in allowed_tables"):
            syncer.sync_row("NotAllowed", {"id": 1})
        syncer.close()

    def test_sync_row_missing_id(self, sync_config):
        syncer = HLMSync(sync_config)
        with pytest.raises(ValueError, match="Missing id column"):
            syncer.sync_row("Products", {"name": "Widget"})
        syncer.close()

    def test_sync_row_seed_mode(self, sync_config):
        """Without a loaded model, sync falls back to seed-mode injection."""
        syncer = HLMSync(sync_config)
        concept = syncer.sync_row(
            "Products", {"id": 1, "name": "Widget", "price": 9.99},
        )
        assert concept == "Products:1"
        assert syncer.stats["synced"] == 1
        syncer.close()

    def test_sync_batch(self, sync_config):
        syncer = HLMSync(sync_config)
        rows = [
            {"id": 1, "name": "Widget", "price": 9.99},
            {"id": 2, "name": "Gadget", "price": 19.99},
        ]
        concepts = syncer.sync_batch("Products", rows)
        assert len(concepts) == 2
        assert syncer.stats["synced"] == 2
        syncer.close()

    def test_delete_row(self, sync_config):
        syncer = HLMSync(sync_config)
        syncer.sync_row("Products", {"id": 1, "name": "Widget", "price": 9.99})
        syncer.delete_row("Products", 1)
        assert syncer.stats["deleted"] == 1
        assert "Products:1" not in syncer.synced_concepts
        syncer.close()

    def test_full_sync_table(self, sync_config):
        syncer = HLMSync(sync_config)
        count = syncer.full_sync_table("Products")
        assert count == 3  # 3 rows in demo DB
        assert syncer.stats["synced"] == 3
        syncer.close()

    def test_checkpoint(self, sync_config, tmp_path):
        syncer = HLMSync(sync_config)
        syncer.sync_row("Products", {"id": 1, "name": "Widget", "price": 9.99})
        syncer.checkpoint()
        session_path = sync_config.hlm_checkpoint_path + ".session"
        assert os.path.exists(session_path)
        syncer.close()

    def test_context_manager(self, sync_config):
        with HLMSync(sync_config) as syncer:
            syncer.sync_row("Products", {"id": 1, "name": "W", "price": 1.0})
            assert syncer.stats["synced"] == 1

    def test_resync_same_row(self, sync_config):
        """Re-syncing the same row should increment overwrite warnings."""
        syncer = HLMSync(sync_config)
        syncer.sync_row("Products", {"id": 1, "name": "Widget", "price": 9.99})
        syncer.sync_row("Products", {"id": 1, "name": "Widget v2", "price": 12.99})
        assert syncer.stats["synced"] == 2
        assert syncer.stats["overwrite_warnings"] >= 1
        syncer.close()

    def test_refused_write_is_counted(self, sync_config):
        """A write the edit gate refuses is counted, and W stays unchanged."""
        gen = torch.Generator().manual_seed(0)
        xi = torch.randint(0, 2, (5, 64), generator=gen).float() * 2 - 1
        W = (xi.T @ poly_interaction(xi, 3)) / 64
        surgeon = BasinSurgeon.from_W(W)
        surgeon.guard(0, max_changed_pct=0.0)
        syncer = HLMSync(sync_config, surgeon=surgeon)
        syncer.sync_row("Products", {"id": 1, "name": "W", "price": 1.0}, strength=2.0)
        assert syncer.stats["refused"] == 1
        assert torch.equal(surgeon.get_W(0), W)
        syncer.close()

    def test_synced_concepts_property(self, sync_config):
        syncer = HLMSync(sync_config)
        syncer.sync_row("Products", {"id": 1, "name": "W", "price": 1.0})
        syncer.sync_row("Products", {"id": 2, "name": "G", "price": 2.0})
        assert syncer.synced_concepts == {"Products:1", "Products:2"}
        syncer.close()


# ── Seed mode positions ─────────────────────────────────────────

class _SeedRecorder(BasinSurgeon):
    """BasinSurgeon that records the seeds HLMSync writes at."""

    def __init__(self, device="cpu"):
        super().__init__(device)
        self.seeds = []

    def inject(self, layer=0, seed=42, strength=None, force=False):
        self.seeds.append(("inject", seed))
        return super().inject(layer, seed, strength, force)

    def weaken(self, layer=0, seed=42, factor=0.5):
        self.seeds.append(("weaken", seed))
        return super().weaken(layer, seed, factor)


class TestSeedMode:
    def test_concept_seed_is_the_same_in_every_process(self):
        """hash() is salted per process (PYTHONHASHSEED); the seed must not be."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(qriton_hlm.__file__)))
        code = "from qriton_hlm.db import _concept_seed; print(_concept_seed('Products:42'))"
        seeds = set()
        for hash_seed in ("1", "2"):
            path = [root] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
            env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=os.pathsep.join(path))
            out = subprocess.run([sys.executable, "-c", code], env=env,
                                 capture_output=True, text=True, check=True)
            seeds.add(int(out.stdout))
        assert seeds == {_concept_seed("Products:42")}

    def test_delete_weakens_the_seed_sync_row_injected(self, sync_config):
        surgeon = _SeedRecorder.from_checkpoint(sync_config.hlm_checkpoint_path)
        syncer = HLMSync(sync_config, surgeon=surgeon)
        syncer.sync_row("Products", {"id": 7, "name": "W", "price": 1.0})
        syncer.delete_row("Products", 7)
        syncer.close()
        seed = _concept_seed("Products:7")
        assert surgeon.seeds == [("inject", seed), ("weaken", seed)]


# ── CheckpointWorker ────────────────────────────────────────────

class TestCheckpointWorker:
    def test_start_stop(self, sync_config):
        syncer = HLMSync(sync_config)
        worker = CheckpointWorker(syncer, interval=0.1)
        worker.start()
        assert worker._thread is not None
        assert worker._thread.is_alive()
        worker.stop()
        assert not worker._thread.is_alive()
        syncer.close()
