"""Measure what Energy Language edits actually do on landscapes with known concepts.

Synthetic polynomial-Hopfield landscapes (degree 3) store K random +-1 concepts:
W = (1/d) * sum_k xi_k f(xi_k)^T. Then inject / remove / move are applied and
scored on:

  hit        inject/move: converging from a 10%-noised target lands on the
             target pattern sign(t) (cos > 0.95); remove: the concept is gone.
  reinforced shipped inject only: what the edit actually wrote on
             (existing concept, its mirror -xi, or a spurious mixture).
  collateral fraction of a fixed probe set (noisy cues of every stored concept
             + random starts) whose converged state moved (cos < 0.9), not
             counting cues of the concept the edit was aimed at.
  guard      what the gate would decide: the shipped retention certificate
             (24 random-start basins, as BasinSurgeon uses) and, when the
             package provides it, the decision-stability guard.

Usage:
  python measure_edits.py --mode shipped --out results/shipped.json
  python measure_edits.py --mode projected --out results/projected.json
"""

import argparse
import inspect
import json
import time

import torch
import torch.nn.functional as F

import qriton_hlm.core as core
from qriton_hlm.core import (
    _converge, certify_compact_edit, find_basins, inject_basin, poly_interaction,
    remove_basin, move_basin,
)

BETA = 7.0
DEGREE = 3
HAS_MODE = "mode" in inspect.signature(inject_basin).parameters


def converge_batch(X, W, max_iter=100, tau_start=0.9, tau_end=0.1, beta=BETA, eps=5e-3):
    """Row-wise identical to core._converge, vectorised over probes."""
    X = X.clone()
    active = torch.ones(X.shape[0], dtype=torch.bool)
    for t in range(max_iter):
        if not active.any():
            break
        tau = tau_start + (tau_end - tau_start) * (t / max_iter)
        Xa = X[active]
        H = poly_interaction(Xa, DEGREE) @ W.T
        Xn = (1 - tau) * Xa + tau * torch.tanh(beta * H)
        delta = (Xn - Xa).norm(dim=1) / (Xa.norm(dim=1) + 1e-8)
        done = delta < eps
        idx = active.nonzero(as_tuple=True)[0]
        X[idx[~done]] = Xn[~done]
        active[idx[done]] = False
    return X


def make_landscape(d, K, gen):
    xi = torch.randint(0, 2, (K, d), generator=gen).float() * 2 - 1
    W = (xi.T @ poly_interaction(xi, DEGREE)) / d
    return W, xi


def make_probes(xi, n_cue, n_rand, gen, flip=0.15):
    K, d = xi.shape
    cues, owner = [], []
    for k in range(K):
        for _ in range(n_cue):
            mask = torch.rand(d, generator=gen) < flip
            c = xi[k].clone()
            c[mask] *= -1
            cues.append(c * 0.9)
            owner.append(k)
    rand = torch.randn(n_rand, d, generator=gen) * 0.5
    X = torch.cat([torch.stack(cues), rand])
    owner = torch.tensor(owner + [-1] * n_rand)
    return X, owner


def noisy(v, gen, rel=0.1):
    return v + max(v.norm().item() * rel, 0.1) * torch.randn(v.shape, generator=gen)


def lands_on(W, pattern, gen, trials=5):
    """Fraction of 10%-noised starts at `pattern` that converge back onto it."""
    ok = 0
    for _ in range(trials):
        x, _, _ = _converge(noisy(pattern, gen), W, DEGREE, beta=BETA)
        if x.norm() > 1e-6 and F.cosine_similarity(x, pattern, dim=0).item() > 0.95:
            ok += 1
    return ok / trials


def classify(state, xi):
    if state.norm() < 1e-6:
        return "zero"
    cos = F.cosine_similarity(state.unsqueeze(0), xi)
    k = int(cos.abs().argmax())
    if cos[k] > 0.95:
        return "existing"
    if cos[k] < -0.95:
        return "mirror"
    return "spurious"


def split(moved, keep, owner, concept_start):
    """Collateral on every probe, on concept decisions (cues of other concepts
    plus random starts that settled on a stored concept), on cues only, and
    on random starts that settled on a spurious mixture."""
    def frac(m):
        return moved[m].float().mean().item() if m.any() else 0.0
    cue = keep & (owner >= 0)
    concept = keep & ((owner >= 0) | concept_start)
    spurious = keep & (owner < 0) & ~concept_start
    return {
        "collateral": frac(keep),
        "collateral_decisions": frac(concept),
        "collateral_cues": frac(cue),
        "collateral_spurious": frac(spurious),
    }


def eff(W, W_new):
    return ((W_new - W).norm() / W.norm()).item()


def to_scale(t, d):
    return t / t.norm() * (0.5 * d ** 0.5) if t.norm() < 2.0 else t.clone()


def shipped_guard(W, W_new):
    protected, _, _ = find_basins(W, W.shape[0], num_inits=24, max_iter=40, beta=BETA)
    return bool(certify_compact_edit(W, W_new, protected, beta=BETA)["accepted"])


def decision_guard(W, W_new, protected, allowed, exclude=None):
    """Changed fraction of cue decisions under the package's guard, set up the
    way BasinSurgeon sets it up (removed targets and the basins built from
    them may change)."""
    fn = getattr(core, "certify_decisions", None)
    if fn is None:
        return None
    if exclude:
        _, allowed = core._gate_sets(protected, allowed, exclude)
    return fn(W, W_new, protected=protected, allowed=allowed, beta=BETA)["changed_fraction"]


def edit_kwargs(mode, protected):
    if not HAS_MODE:
        return {}
    kw = {"mode": mode}
    if mode != "legacy":
        kw["protected"] = protected
    return kw


def run(mode, dims, trials, strengths, seed):
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for d, K in dims:
        for trial in range(trials):
            W, xi = make_landscape(d, K, gen)
            probes, owner = make_probes(xi, n_cue=200 if K <= 6 else 120, n_rand=800, gen=gen)
            before = converge_batch(probes, W)
            concept_start = F.cosine_similarity(
                before.unsqueeze(1), xi.unsqueeze(0), dim=2).abs().max(1).values > 0.95
            # Protected set = what a user can see: basins found from random starts.
            found, _, _ = find_basins(W, d, num_inits=64, max_iter=60, beta=BETA)
            concepts = [b for b in found if classify(b, xi) in ("existing", "mirror")]

            for target_kind in ("seed", "pattern"):
                if target_kind == "seed":   # inject <layer> <seed>: random unit direction
                    t = torch.randn(d, generator=gen)
                    t = t / t.norm()
                else:                       # a captured, already-converged concept
                    t = (torch.randint(0, 2, (d,), generator=gen).float() * 2 - 1) * 0.9
                ideal = torch.sign(t)
                for s in strengths:
                    landed, _, _ = _converge(to_scale(t, d), W, DEGREE, beta=BETA)
                    W_new = inject_basin(W, t, strength=s, beta=BETA, **edit_kwargs(mode, found))
                    after = converge_batch(probes, W_new)
                    moved = F.cosine_similarity(before, after, dim=1) < 0.9
                    # Probes that now land on the new concept are the edit working, not collateral.
                    on_target = F.cosine_similarity(after, ideal.unsqueeze(0), dim=1).abs() > 0.95
                    # Only random starts that used to settle on a spurious mixture may
                    # join the new concept for free; concept decisions that switch
                    # to it are collateral.
                    free = on_target & (owner < 0) & ~concept_start
                    coll = split(moved, ~free, owner, concept_start)
                    kept = sum(lands_on(W_new, xi[k], gen, 3) >= 2 / 3 for k in range(K)) / K
                    rows.append({
                        "op": "inject", "d": d, "K": K, "trial": trial, "target": target_kind,
                        "strength": s, "eff_strength": eff(W, W_new), "hit": lands_on(W_new, ideal, gen) >= 0.6,
                        "reinforced": classify(landed, xi) if mode in ("shipped", "legacy") else "target",
                        **coll, "concepts_kept": kept,
                        "shipped_guard": shipped_guard(W, W_new),
                        "decision_guard": decision_guard(W, W_new, found, [t]),
                        "decision_guard_concepts": decision_guard(W, W_new, concepts, [t]),
                    })

            for s in strengths:
                k = int(torch.randint(0, K, (1,), generator=gen))
                aim = noisy(xi[k] * 0.9, gen, 0.2)           # point roughly at concept k
                others = [b for b in found if abs(F.cosine_similarity(b, xi[k], dim=0)) < 0.95]
                W_new = remove_basin(W, aim, strength=s, beta=BETA, **edit_kwargs(mode, others))
                after = converge_batch(probes, W_new)
                moved = F.cosine_similarity(before, after, dim=1) < 0.9
                # Decisions that went to concept k (or its mirror, which shares its
                # weights) are the edit's intended effect, not collateral.
                was_k = F.cosine_similarity(before, xi[k].unsqueeze(0), dim=1).abs() > 0.95
                mask = (owner != k) & ~was_k
                gone = lands_on(W_new, xi[k], gen) < 0.4 and lands_on(W_new, -xi[k], gen) < 0.4
                rows.append({
                    "op": "remove", "d": d, "K": K, "trial": trial, "target": "concept",
                    "strength": s, "eff_strength": eff(W, W_new), "hit": gone,
                    **split(moved, mask, owner, concept_start),
                    "concepts_kept": sum(lands_on(W_new, xi[j], gen, 3) >= 2 / 3
                                         for j in range(K) if j != k) / (K - 1),
                    "shipped_guard": shipped_guard(W, W_new),
                    "decision_guard": decision_guard(W, W_new, found, [xi[k]], [xi[k]]),
                    "decision_guard_concepts": decision_guard(W, W_new, concepts, [xi[k]], [xi[k]]),
                })

                k = int(torch.randint(0, K, (1,), generator=gen))
                dest = (torch.randint(0, 2, (d,), generator=gen).float() * 2 - 1) * 0.9
                others = [b for b in found if abs(F.cosine_similarity(b, xi[k], dim=0)) < 0.95]
                W_new = move_basin(W, xi[k] * 0.9, dest, strength=s, beta=BETA,
                                   **edit_kwargs(mode, others))
                after = converge_batch(probes, W_new)
                moved = F.cosine_similarity(before, after, dim=1) < 0.9
                on_dest = F.cosine_similarity(after, torch.sign(dest).unsqueeze(0), dim=1).abs() > 0.95
                was_k = F.cosine_similarity(before, xi[k].unsqueeze(0), dim=1).abs() > 0.95
                free = on_dest & (owner < 0) & ~concept_start
                mask = (owner != k) & ~was_k & ~free
                rows.append({
                    "op": "move", "d": d, "K": K, "trial": trial, "target": "concept",
                    "strength": s, "eff_strength": eff(W, W_new),
                    "hit": lands_on(W_new, torch.sign(dest), gen) >= 0.6
                    and lands_on(W_new, xi[k], gen) < 0.4,
                    **split(moved, mask, owner, concept_start),
                    "concepts_kept": sum(lands_on(W_new, xi[j], gen, 3) >= 2 / 3
                                         for j in range(K) if j != k) / (K - 1),
                    "shipped_guard": shipped_guard(W, W_new),
                    "decision_guard": decision_guard(W, W_new, found, [xi[k], dest], [xi[k]]),
                    "decision_guard_concepts": decision_guard(W, W_new, concepts, [xi[k], dest], [xi[k]]),
                })
    return rows


def summarise(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["op"], r["target"], r["d"], r["strength"]), []).append(r)
    out = []
    for (op, target, d, s), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        n = len(rs)
        coll = sorted(r["collateral"] for r in rs)
        accepted = [r for r in rs if r["shipped_guard"]]
        item = {
            "op": op, "target": target, "d": d, "strength": s, "n": n,
            "hit_rate": sum(r["hit"] for r in rs) / n,
            "eff_strength": sum(r["eff_strength"] for r in rs) / n,
            "collateral_mean": sum(coll) / n,
            "collateral_max": coll[-1],
            "collateral_decisions": sum(r["collateral_decisions"] for r in rs) / n,
            "collateral_cues": sum(r["collateral_cues"] for r in rs) / n,
            "collateral_spurious": sum(r["collateral_spurious"] for r in rs) / n,
            "concepts_kept": sum(r["concepts_kept"] for r in rs) / n,
            "shipped_guard_accepts": len(accepted) / n,
            "shipped_guard_accepts_hits": sum(r["shipped_guard"] and r["hit"] for r in rs),
            "max_collateral_shipped_guard_accepted": max((r["collateral"] for r in accepted), default=None),
        }
        if "reinforced" in rs[0]:
            kinds = [r["reinforced"] for r in rs]
            item["reinforced"] = {k: kinds.count(k) for k in sorted(set(kinds))}
        for key in ("decision_guard", "decision_guard_concepts"):
            dg = [r[key] for r in rs if r.get(key) is not None]
            if dg:
                item[key + "_changed_mean"] = sum(dg) / len(dg)
                item[key + "_accepts"] = sum(v <= 0.02 for v in dg) / len(dg)
        out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="shipped", choices=["shipped", "legacy", "projected"])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--auto", action="store_true", help="also run strength='auto'")
    a = ap.parse_args()
    if a.mode == "projected" and not HAS_MODE:
        raise SystemExit("this qriton_hlm has no projected mode")
    torch.manual_seed(a.seed)  # find_basins draws from the global RNG
    t0 = time.time()
    rows = run(a.mode, dims=[(64, 5), (128, 10)], trials=a.trials,
               strengths=[0.1, 0.5, 1.0] + (["auto"] if a.auto else []), seed=a.seed)
    result = {
        "mode": a.mode, "seed": a.seed, "trials": a.trials,
        "package_has_mode": HAS_MODE, "seconds": round(time.time() - t0, 1),
        "summary": summarise(rows), "rows": rows,
    }
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    for s in result["summary"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == "__main__":
    main()
