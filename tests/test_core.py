"""Tests for qriton_hlm.core — the foundation of energy landscape surgery."""

import os
import tempfile

import numpy as np
import pytest
import torch

from qriton_hlm.core import (
    poly_interaction,
    compute_energy,
    find_basins,
    inject_basin,
    remove_basin,
    move_basin,
    verify_basin_exists,
    load_W_from_checkpoint,
    BasinSurgeon,
)


# ── Fixtures ────────────────────────────────────────���───────────

@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def W(d_model):
    """Random symmetric weight matrix (typical Hopfield initialization)."""
    torch.manual_seed(42)
    W = torch.randn(d_model, d_model)
    W = (W + W.T) / 2  # symmetrize
    return W


@pytest.fixture
def surgeon(W, tmp_path):
    """BasinSurgeon from a raw W matrix."""
    s = BasinSurgeon.from_W(W)
    return s


@pytest.fixture
def checkpoint_path(W, tmp_path):
    """Save a fake checkpoint for testing load_W_from_checkpoint."""
    path = str(tmp_path / "test_model.pt")
    state = {"blocks.0.hopfield.W": W}
    torch.save({"model_state": state}, path)
    return path


# ── poly_interaction ────────────────────────────────────────────

class TestPolyInteraction:
    def test_output_shape(self, d_model):
        x = torch.randn(d_model)
        result = poly_interaction(x, degree=3)
        assert result.shape == x.shape

    def test_degree_2_is_identity_sign(self):
        x = torch.tensor([1.0, -2.0, 3.0, -0.5])
        result = poly_interaction(x, degree=2)
        expected = torch.sign(x) * torch.abs(x) ** 1  # = x
        assert torch.allclose(result, expected)

    def test_zero_input(self):
        x = torch.zeros(10)
        result = poly_interaction(x, degree=3)
        assert torch.all(result == 0)

    def test_preserves_sign(self):
        x = torch.tensor([-3.0, 2.0, -1.0, 4.0])
        result = poly_interaction(x, degree=3)
        assert torch.all(torch.sign(result) == torch.sign(x))


# ── compute_energy ──────────────────────────────────────────────

class TestComputeEnergy:
    def test_returns_scalar(self, W, d_model):
        x = torch.randn(d_model)
        e = compute_energy(x, W)
        assert isinstance(e, float)

    def test_zero_state_zero_energy(self, W):
        x = torch.zeros(W.shape[0])
        e = compute_energy(x, W)
        assert e == 0.0

    def test_energy_is_finite(self, W, d_model):
        x = torch.randn(d_model)
        e = compute_energy(x, W)
        assert np.isfinite(e)


# ── find_basins ─────────────────────────────────────────────────

class TestFindBasins:
    def test_finds_at_least_one_basin(self, W, d_model):
        basins, ids, trajectories = find_basins(
            W, d_model, num_inits=50, max_iter=50,
        )
        assert len(basins) >= 1

    def test_basin_ids_match_basins(self, W, d_model):
        basins, ids, trajectories = find_basins(
            W, d_model, num_inits=50, max_iter=50,
        )
        # Every valid id should be < num unique basins
        valid_ids = [i for i in ids if i >= 0]
        if valid_ids:
            assert max(valid_ids) < len(basins)

    def test_trajectories_returned(self, W, d_model):
        basins, ids, trajectories = find_basins(
            W, d_model, num_inits=20, max_iter=50,
        )
        assert len(trajectories) == 20


# ── inject_basin ────────────────────────────────────────────────

class TestInjectBasin:
    def test_modifies_W(self, W, d_model):
        target = torch.randn(d_model)
        W_new = inject_basin(W, target, strength=0.5)
        assert not torch.equal(W, W_new)
        assert W_new.shape == W.shape

    def test_inject_changes_energy_at_target(self, W, d_model):
        target = torch.randn(d_model)
        target = target / target.norm() * (0.5 * d_model ** 0.5)
        e_before = compute_energy(target, W)
        W_new = inject_basin(W, target, strength=1.0)
        e_after = compute_energy(target, W_new)
        # Energy at the target should change after Hebbian injection
        assert abs(e_after - e_before) > 1e-6

    def test_zero_strength_no_change(self, W, d_model):
        target = torch.randn(d_model)
        W_new = inject_basin(W, target, strength=0.0)
        assert torch.allclose(W, W_new, atol=1e-6)


# ── remove_basin ────────────────────────────────────────────────

class TestRemoveBasin:
    def test_modifies_W(self, W, d_model):
        target = torch.randn(d_model)
        W_new = remove_basin(W, target, strength=0.5)
        assert not torch.equal(W, W_new)

    def test_remove_changes_landscape(self, W, d_model):
        target = torch.randn(d_model)
        target = target / target.norm() * (0.5 * d_model ** 0.5)
        W_new = remove_basin(W, target, strength=1.0)
        # The W matrix should change (anti-Hebbian update applied)
        delta = (W_new - W).norm().item()
        assert delta > 0


# ── move_basin ──────────────────────────────────────────────────

class TestMoveBasin:
    def test_modifies_W(self, W, d_model):
        source = torch.randn(d_model)
        dest = torch.randn(d_model)
        W_new = move_basin(W, source, dest, strength=0.5)
        assert not torch.equal(W, W_new)


# ── verify_basin_exists ─────────────────────────────────────────

class TestVerifyBasinExists:
    def test_returns_tuple(self, W, d_model):
        target = torch.randn(d_model)
        result = verify_basin_exists(W, target)
        assert len(result) == 4
        is_basin, final_state, cos_sim, iters = result
        assert isinstance(is_basin, bool)
        assert isinstance(cos_sim, float)
        assert isinstance(iters, int)

    def test_injected_basin_verifies(self, W, d_model):
        target = torch.randn(d_model)
        target = target / target.norm() * (0.5 * d_model ** 0.5)
        W_new = inject_basin(W, target, strength=2.0)
        # Converge to find the actual basin location
        from qriton_hlm.core import _converge
        basin, _, _ = _converge(target, W_new, beta=7.0)
        is_basin, _, cos, _ = verify_basin_exists(W_new, basin)
        assert is_basin or cos > 0.8  # should be a basin or very close


# ── load_W_from_checkpoint ──────────────────────────────────────

class TestLoadCheckpoint:
    def test_loads_W(self, checkpoint_path, d_model):
        W, d = load_W_from_checkpoint(checkpoint_path, layer=0)
        assert W.shape == (d_model, d_model)
        assert d == d_model

    def test_missing_layer_raises(self, checkpoint_path):
        with pytest.raises(KeyError):
            load_W_from_checkpoint(checkpoint_path, layer=99)


# ── BasinSurgeon ────────────────────────────────────────────────

class TestBasinSurgeon:
    def test_from_W(self, W):
        s = BasinSurgeon.from_W(W)
        W_back = s.get_W(layer=0)
        assert torch.equal(W_back, W)

    def test_from_checkpoint(self, checkpoint_path):
        s = BasinSurgeon.from_checkpoint(checkpoint_path)
        W = s.get_W(layer=0)
        assert W.shape[0] == W.shape[1]

    def test_survey(self, surgeon):
        result = surgeon.survey(layer=0)
        assert "num_basins" in result
        assert "basins" in result
        assert "energies" in result
        assert result["num_basins"] >= 0

    def test_inject(self, surgeon):
        result = surgeon.inject(layer=0, seed=42, strength=0.1)
        assert "exists_after" in result
        assert "cos_after" in result

    def test_remove(self, surgeon):
        result = surgeon.remove(layer=0, seed=42, strength=0.1)
        assert "exists_after" in result

    def test_verify(self, surgeon):
        result = surgeon.verify(layer=0, seed=42)
        assert "is_basin" in result
        assert "cos" in result
        assert "energy" in result

    def test_inject_then_verify(self, surgeon):
        surgeon.inject(layer=0, seed=99, strength=0.5)
        result = surgeon.verify(layer=0, seed=99)
        # After injection, the target direction should be closer to a basin
        assert "is_basin" in result

    def test_energy(self, surgeon):
        result = surgeon.energy(layer=0, seed=42)
        assert "energy" in result
        assert np.isfinite(result["energy"])

    def test_diff_no_backup_no_checkpoint(self):
        s = BasinSurgeon()
        result = s.diff(layer=0)
        assert result is None

    def test_diff_after_inject(self, surgeon):
        # Need to set backup first
        W_orig = surgeon.get_W(0).clone()
        surgeon._w_backups[0] = W_orig
        surgeon.inject(layer=0, seed=42, strength=0.1)
        result = surgeon.diff(layer=0)
        assert result is not None
        assert result["frobenius"] > 0

    def test_restore(self, surgeon):
        W_orig = surgeon.get_W(0).clone()
        surgeon._w_backups[0] = W_orig.clone()
        surgeon.inject(layer=0, seed=42, strength=0.5)
        assert not torch.equal(surgeon.get_W(0), W_orig)
        surgeon.restore(layer=0)
        assert torch.allclose(surgeon.get_W(0), W_orig)

    def test_landscape(self, surgeon):
        result = surgeon.landscape(layer=0)
        assert "num_basins" in result
        assert "basins" in result

    def test_guard(self, surgeon):
        result = surgeon.guard(layer=0, max_remove_pct=20.0)
        assert "current_basins" in result
        assert "min_allowed" in result

    def test_strengthen(self, surgeon):
        result = surgeon.strengthen(layer=0, seed=42, factor=2.0)
        assert "energy_before" in result
        assert "energy_after" in result

    def test_weaken(self, surgeon):
        result = surgeon.weaken(layer=0, seed=42, factor=0.5)
        assert "energy_before" in result
        assert "energy_after" in result


# ── BasinSurgeon concepts (without model — seed mode) ──────────

class TestBasinSurgeonConcepts:
    def test_list_concepts_empty(self, surgeon):
        assert surgeon.list_concepts() == {}

    def test_blend_requires_both(self, surgeon):
        with pytest.raises(ValueError, match="Both concepts must exist"):
            surgeon.blend("a", "b", "c")

    def test_inject_concept_unknown_raises(self, surgeon):
        with pytest.raises(ValueError, match="Unknown concept"):
            surgeon.inject_concept(0, "nonexistent")

    def test_remove_concept_unknown_raises(self, surgeon):
        with pytest.raises(ValueError, match="Unknown concept"):
            surgeon.remove_concept(0, "nonexistent")

    def test_export_unknown_raises(self, surgeon):
        with pytest.raises(ValueError, match="Unknown concept"):
            surgeon.export_concept("nonexistent", "/tmp/test.pt")


# ── BasinSurgeon persistence ───────────────────────────────────

class TestBasinSurgeonPersistence:
    def test_save_checkpoint(self, surgeon, tmp_path):
        surgeon.inject(layer=0, seed=42, strength=0.1)
        path = str(tmp_path / "session.pt")
        result = surgeon.save_checkpoint(path)
        assert result["path"] == path
        assert result["layers"] >= 1
        assert os.path.exists(path)

    def test_save_and_load_session(self, surgeon, tmp_path):
        # Inject a basin
        surgeon.inject(layer=0, seed=42, strength=0.3)
        W_modified = surgeon.get_W(0).clone()

        # Save
        path = str(tmp_path / "session.pt")
        surgeon.save_checkpoint(path)

        # Load into fresh surgeon
        s2 = BasinSurgeon()
        result = s2.load_session(path)
        assert result["layers"] >= 1

        W_loaded = s2.get_W(0)
        assert torch.allclose(W_modified, W_loaded)

    def test_save_with_concepts(self, surgeon, tmp_path):
        # Manually add a concept (normally done via capture with a model)
        d = surgeon.get_W(0).shape[0]
        state = torch.randn(d)
        surgeon._concepts["test_concept"] = {
            "states": [state],
            "centroid": state,
        }

        path = str(tmp_path / "session.pt")
        result = surgeon.save_checkpoint(path)
        assert result["concepts"] == 1

        # Load and verify concept survived
        s2 = BasinSurgeon()
        s2.load_session(path)
        assert "test_concept" in s2._concepts
        assert torch.allclose(
            s2._concepts["test_concept"]["centroid"], state
        )

    def test_export_import_concept(self, surgeon, tmp_path):
        d = surgeon.get_W(0).shape[0]
        state = torch.randn(d)
        surgeon._concepts["exportable"] = {
            "states": [state],
            "centroid": state,
        }

        path = str(tmp_path / "concept.pt")
        surgeon.export_concept("exportable", path)
        assert os.path.exists(path)

        s2 = BasinSurgeon.from_W(torch.randn(d, d))
        result = s2.import_concept(path)
        assert result["concept"] == "exportable"
        assert "exportable" in s2._concepts


# ── Causal operations ──────────────────────────────────────────

class TestCausal:
    def test_causal_scan_returns_graph(self, surgeon):
        result = surgeon.causal_scan(layer=0, num_inits=30)
        assert "num_basins" in result
        assert "edges" in result
        assert "adjacency" in result

    def test_causal_intervene(self, surgeon):
        # Need at least one basin
        survey = surgeon.survey(layer=0)
        if survey["num_basins"] > 0:
            result = surgeon.causal_intervene(
                layer=0, basin_idx=0, operation="remove", num_inits=30,
            )
            assert "basins_before" in result
            assert "basins_after" in result

    def test_causal_intervene_invalid_op(self, surgeon):
        with pytest.raises(ValueError, match="Unknown operation"):
            surgeon.causal_intervene(
                layer=0, basin_idx=0, operation="explode", num_inits=30,
            )
