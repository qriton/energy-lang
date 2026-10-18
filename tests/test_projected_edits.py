"""Projected edits and the decision-stability guard.

Landscapes with known stored concepts: W = (1/d) sum_k xi_k f(xi_k)^T.
"""

import pytest
import torch
import torch.nn.functional as F

from qriton_hlm.core import (
    GUARD_MAX_PROBES,
    BasinSurgeon,
    _converge,
    certify_decisions,
    default_probes,
    find_basins,
    inject_basin,
    poly_interaction,
    remove_basin,
)

BETA = 7.0


def landscape(d=64, K=5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    xi = torch.randint(0, 2, (K, d), generator=gen).float() * 2 - 1
    W = (xi.T @ poly_interaction(xi, 3)) / d
    return W, xi


def recalls(W, pattern, seed=1, trials=5):
    gen = torch.Generator().manual_seed(seed)
    ok = 0
    for _ in range(trials):
        x = pattern + 0.1 * pattern.norm() * torch.randn(pattern.shape, generator=gen)
        x, _, _ = _converge(x, W, beta=BETA)
        ok += F.cosine_similarity(x, pattern, dim=0).item() > 0.95
    return ok / trials


@pytest.fixture
def stored():
    torch.manual_seed(0)
    W, xi = landscape()
    return W, xi, list(xi * 0.9)


def new_pattern(d=64, seed=99):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (d,), generator=gen).float() * 2 - 1) * 0.9


class TestProjectedInject:
    def test_target_becomes_attractor(self, stored):
        W, xi, protected = stored
        t = new_pattern()
        assert recalls(W, torch.sign(t)) == 0
        W_new = inject_basin(W, t, strength="auto", beta=BETA, protected=protected)
        assert recalls(W_new, torch.sign(t)) == 1.0

    def test_legacy_inject_misses_target(self, stored):
        W, xi, _ = stored
        t = new_pattern()
        W_new = inject_basin(W, t, strength=0.5, beta=BETA, mode="legacy")
        assert recalls(W_new, torch.sign(t)) == 0

    def test_protected_concepts_still_recalled(self, stored):
        W, xi, protected = stored
        W_new = inject_basin(W, new_pattern(), strength="auto", beta=BETA, protected=protected)
        for k in range(xi.shape[0]):
            assert recalls(W_new, xi[k]) == 1.0

    def test_protected_field_unchanged(self, stored):
        W, xi, protected = stored
        W_new = inject_basin(W, new_pattern(), strength=1.0, beta=BETA, protected=protected)
        for b in protected:
            fb = poly_interaction(b, 3)
            assert torch.allclose(W_new @ fb, W @ fb, atol=1e-4)

    def test_collateral_below_threshold_at_auto_strength(self, stored):
        W, xi, protected = stored
        t = new_pattern()
        W_new = inject_basin(W, t, strength="auto", beta=BETA, protected=protected)
        cert = certify_decisions(W, W_new, protected=protected, allowed=[t], beta=BETA)
        assert cert["changed_fraction"] <= 0.02
        assert cert["accepted"]

    def test_unknown_mode_raises(self, stored):
        W, _, _ = stored
        with pytest.raises(ValueError):
            inject_basin(W, new_pattern(), mode="nope")


class TestNoRoom:
    """With at least as many protected states as dimensions, no write can
    leave them all unchanged; projection must not write rounding noise."""

    @pytest.mark.parametrize("K", [12, 16, 24])
    def test_protected_field_unchanged_or_no_write(self, K):
        d = 16
        gen = torch.Generator().manual_seed(K)
        xi = (torch.randint(0, 2, (K, d), generator=gen).float() * 2 - 1) * 0.9
        W = (xi.T @ poly_interaction(xi, 3)) / d
        t = torch.randn(d, generator=gen)
        W_new = inject_basin(W, t, strength=0.5, beta=BETA, protected=list(xi))
        F_prot = poly_interaction(xi, 3).T
        assert torch.allclose(W_new @ F_prot, W @ F_prot, atol=1e-4)
        if torch.linalg.matrix_rank(F_prot) == d:
            assert W_new is W

    def test_auto_and_remove_return_W_without_room(self):
        d = 16
        gen = torch.Generator().manual_seed(0)
        xi = (torch.randint(0, 2, (40, d), generator=gen).float() * 2 - 1) * 0.9
        W = (xi.T @ poly_interaction(xi, 3)) / d
        t = torch.randn(d, generator=gen)
        assert inject_basin(W, t, strength="auto", beta=BETA, protected=list(xi)) is W
        assert remove_basin(W, xi[0], strength=0.5, beta=BETA, protected=list(xi)) is W


class TestProjectedRemove:
    def test_removes_target_keeps_others(self, stored):
        W, xi, protected = stored
        W_new = remove_basin(W, xi[0] * 0.9, strength="auto", beta=BETA, protected=protected)
        assert recalls(W_new, xi[0]) == 0
        for k in range(1, xi.shape[0]):
            assert recalls(W_new, xi[k]) == 1.0


class TestDecisionGuard:
    def test_identity_edit_changes_nothing(self, stored):
        W, _, protected = stored
        cert = certify_decisions(W, W.clone(), protected=protected, beta=BETA)
        assert cert["accepted"] and cert["changed_fraction"] == 0.0

    def test_refuses_destructive_edit(self, stored):
        W, xi, protected = stored
        W_new = inject_basin(W, new_pattern(), strength=2.0, beta=BETA, mode="legacy")
        cert = certify_decisions(W, W_new, protected=protected, beta=BETA)
        assert not cert["accepted"]
        assert cert["changed"]  # names the probes that moved


class TestGuardBudget:
    def test_default_probes_cap(self, stored):
        W, _, protected = stored  # 5 basins
        assert default_probes(W, protected).shape[0] == 5 * 64
        assert default_probes(W, protected, max_probes=100).shape[0] == 100
        assert default_probes(W, protected, max_probes=3).shape[0] == 5  # one cue each

    def test_default_cap_keeps_64_cues_per_basin(self, stored):
        W, _, protected = stored
        assert GUARD_MAX_PROBES >= 64 * 64
        cert = certify_decisions(W, W.clone(), protected=protected, beta=BETA)
        assert cert["n_probes"] == 64 * len(protected)

    def test_capped_guard_still_refuses_destructive_edit(self, stored):
        W, _, protected = stored
        W_new = inject_basin(W, new_pattern(), strength=2.0, beta=BETA, mode="legacy")
        cert = certify_decisions(W, W_new, protected=protected, beta=BETA, max_probes=50)
        assert cert["n_probes"] == 50
        assert not cert["accepted"]

    def test_surgeon_passes_guard_probes(self, stored):
        W, xi, protected = stored
        s = BasinSurgeon.from_W(W)
        s._basin_cache[0] = protected
        s.params['guard_probes'] = 50
        res = s.inject(0, seed=3)
        assert res['retention']['decision']['n_probes'] == 50


class TestSurgeonGate:
    def _surgeon(self, stored):
        W, xi, protected = stored
        s = BasinSurgeon.from_W(W)
        s._basin_cache[0] = protected  # protect the known concepts
        return s, xi

    def test_every_remove_path_goes_through_guard(self, stored):
        s, xi = self._surgeon(stored)
        s._concepts['c0'] = {'states': [xi[0] * 0.9], 'centroid': xi[0] * 0.9}
        results = [
            s.remove(0, seed=1),
            s.remove_concept(0, 'c0'),
            s.weaken(0, seed=2),
            s.causal_intervene(0, basin_idx=0, operation='remove', num_inits=30),
            s.consolidate(0, num_inits=30),
        ]
        for res in results:
            assert 'decision' in res['retention']

    def test_remove_concept_refused_leaves_W(self, stored):
        s, xi = self._surgeon(stored)
        s._concepts['c0'] = {'states': [xi[0] * 0.9], 'centroid': xi[0] * 0.9}
        s.guard(0, max_changed_pct=0.0)
        s.params['auto_strength'] = False
        s.params['strength'] = 3.0
        W0 = s.get_W(0).clone()
        res = s.remove_concept(0, 'c0')
        assert res['applied'] is False
        assert res['retention']['reason'] == 'decisions_changed'
        assert torch.equal(s.get_W(0), W0)

    def test_remove_concept_accepted_removes(self, stored):
        s, xi = self._surgeon(stored)
        s._concepts['c0'] = {'states': [xi[0] * 0.9], 'centroid': xi[0] * 0.9}
        res = s.remove_concept(0, 'c0')
        assert res['applied'] is True
        assert recalls(s.get_W(0), xi[0]) == 0

    def test_remove_is_no_longer_always_refused(self, stored):
        s, xi = self._surgeon(stored)
        res = s.remove(0, seed=1)
        assert 'decision' in res['retention']
        assert res['retention']['reason'] != 'protected_basin_lost'

    def test_guard_protected_is_used_and_kept(self, stored):
        W, xi, protected = stored
        s = BasinSurgeon.from_W(W)
        info = s.guard(0, protected=protected)
        assert info['num_protected'] == len(protected)
        res = s.inject(0, seed=3)
        assert res['applied'] is True
        assert res['retention']['decision']['n_probes'] == 64 * len(protected)
        # a committed edit resets the find_basins cache, not the guard's set
        res = s.inject(0, seed=4)
        assert res['retention']['decision']['n_probes'] == 64 * len(protected)
        s.guard(0)  # replaces the settings: back to the default protected set
        assert s._guards[0]['protected'] is None

    def test_guard_protected_empty_protects_nothing(self, stored):
        W, _, _ = stored
        s = BasinSurgeon.from_W(W)
        s.guard(0, protected=[])
        res = s.inject(0, seed=3)
        assert res['retention']['decision']['n_probes'] == 0

    def test_legacy_mode_keeps_old_gate(self, stored):
        s, xi = self._surgeon(stored)
        s.params['edit_mode'] = 'legacy'
        res = s.inject(0, seed=3, strength=0.1)
        assert 'decision' not in res['retention']
