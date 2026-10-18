# Migrating from 0.12.0 to 0.13.0

0.13.0 changes what an edit does by default. Inject used to reinforce whatever
basin the target fell into. It now writes the target itself, projected so that
the field at the protected basins does not change. Every write path also goes
through a decision-stability guard. Code that calls the edit functions or the
`BasinSurgeon` write methods keeps working, but gets different weights, more
refusals and slower edits.

This file lists every public API change. Measurements are in
[RESULTS.md](RESULTS.md) and the release summary in [CHANGELOG.md](CHANGELOG.md).

## Checklist

1. After every `BasinSurgeon` edit, check `result["applied"]`. A refused edit
   leaves W unchanged. That is now a normal outcome, not an error: nothing is
   raised.
2. Stop passing small fixed strengths (0.05–0.1) to inject. Under projected
   writes they rarely create the target (measured: 0 of 40 at 0.1). Pass no
   strength (`BasinSurgeon`), or `strength="auto"`.
3. If you call `inject_basin` / `remove_basin` / `move_basin` directly, pass
   `protected=` explicitly. Otherwise each call runs `find_basins`, which costs
   time and advances torch's global RNG. To keep 0.12.0 weights, pass
   `mode="legacy"`.
4. If most edits are refused, protect only real concepts with
   `surgeon.guard(layer, protected=concept_states)` rather than raising the
   budget.
5. `strengthen`, `weaken`, `consolidate`, `causal_intervene`, `causal_scan` and
   `causal_counterfactual` take their strength from `params['strength']`
   (default 1.0), never `auto`. With default params on 6 synthetic landscapes
   (d=64, 5 concepts) the guard refused `strengthen` 6/6, `weaken` 4/6,
   `consolidate` 5/6 and `causal_intervene` 3/6. Set a lower
   `params['strength']` (measured auto strengths were 0.3–0.5).
6. CLI scripts: write `auto` as the strength argument (`inject 0 42 auto`,
   `inject-concept 5 polite auto`).

## 1. Package root (`qriton_hlm`)

| | 0.12.0 | 0.13.0 |
|---|---|---|
| `__version__` | `"0.12.0"` | `"0.13.0"` |
| `__all__` | — | adds `"certify_decisions"` (also importable from `qriton_hlm`) |

## 2. Module functions (`qriton_hlm.core`)

These functions return a proposal and never check it. Gate the result yourself
with `certify_decisions` (and `certify_compact_edit`), or use `BasinSurgeon`.

### `inject_basin`

```python
# 0.12.0
inject_basin(W, target, degree=3, strength=1.0, beta=7.0) -> Tensor
# 0.13.0
inject_basin(W, target, degree=3, strength=1.0, beta=7.0,
             mode="projected", protected=None) -> Tensor
```

- **Default behaviour changed.** `mode="projected"` writes the outer product
  of `pattern = sign(target) * 0.9` with `f(pattern)`, after removing from
  `f(pattern)` every component along `f(b)` for the protected states `b`. The
  field `W_new @ f(b)` equals `W @ f(b)` for every protected `b` (exact up to
  float error; tested). The size of the edit is unchanged:
  `||W_new - W||_F = strength * ||W||_F`.
- `protected=None` (default) protects the basins `find_basins(W, d,
  degree=degree, num_inits=64, max_iter=60, beta=beta)` finds. States with
  `|cos| >= 0.95` to `pattern` (the target and its mirror) are never
  protected, so writing at an existing basin strengthens it. `protected=[]`
  disables protection, which is the 0.13.0 "direct" write.
- `strength="auto"` (new; any mode) tries `AUTO_STRENGTHS = (0.05, 0.1, 0.15,
  0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0)` in order. It returns the first write
  after which 5 seeded starts at `pattern` plus 10% noise all settle back onto
  it (cos > 0.95). If no rung succeeds it returns the 2.0 write without raising.
  In legacy mode the check uses `sign(target) * 0.9` too.
- Returns `W` itself (same object, unchanged) when `target` has norm < 1e-8,
  or when there is **no room**: the protected `f(b)` span every direction the
  write needs, typically with at least as many protected states as dimensions.
- `mode` other than `"projected"` / `"legacy"` raises `ValueError`.
- `mode="legacy"` reproduces 0.12.0 bit for bit (240/240 rows in RESULTS.md):
  converge from the target scaled to norm `0.5 * sqrt(d)`, then reinforce
  wherever it lands. `protected` is ignored.

### `remove_basin`

```python
# 0.12.0
remove_basin(W, target, degree=3, strength=1.0, beta=7.0) -> Tensor
# 0.13.0
remove_basin(W, target, degree=3, strength=1.0, beta=7.0,
             mode="projected", protected=None) -> Tensor
```

- It still converges from the target to find the basin `b*` and subtracts the
  scaled outer product `b* f(b*)^T`. In projected mode `f(b*)` is first
  projected away from the protected states, except `dependent_basins(b*,
  protected)`. Those are the states with `|cos| >= DEPENDENT_COS = 0.4` to
  `b*`: the basin itself, its mirror, and mixtures built from it. A removal
  cannot leave them unchanged.
- `strength="auto"` returns the first rung after which neither `b*` nor `-b*`
  attracts (0 of 5 seeded noisy starts settle back).
- Returns `W` unchanged if `b*` collapses to zero (as before), or if there is
  no room.

### `move_basin`

```python
# 0.12.0
move_basin(W, source, destination, degree=3, strength=1.0, beta=7.0) -> Tensor
# 0.13.0
move_basin(W, source, destination, degree=3, strength=1.0, beta=7.0,
           mode="projected", protected=None) -> Tensor
```

- This is `remove_basin(source)` followed by `inject_basin(destination)`. In
  projected mode the protected set is found once, on the original W. The
  inject step then protects that set minus the source's dependent basins.
  `strength` applies to each step separately; `"auto"` searches each step.

### New: `certify_decisions` (exported)

```python
certify_decisions(W, W_new, probes=None, protected=None, allowed=None,
                  max_changed=0.02, degree=3, beta=7.0, cos_threshold=0.9,
                  max_iter=100, n_cue=64, max_probes=4096) -> dict
```

It converges a fixed, seeded probe set on `W` and on `W_new`. A probe has
changed when its two final states have cos < `cos_threshold`, unless both are
zero. Two kinds of probe are not counted, because their change is the intended
effect:

- probes that settled on an `allowed` state before the edit (what a remove or
  move targets);
- probes that settled on no protected basin before and on an `allowed` state
  after (a new concept taking over spurious territory).

A probe that switches from a protected basin to the new concept is counted.
`allowed` entries are compared through their sign pattern (|cos| > 0.95).

- `probes=None` builds `default_probes(W, protected, n_cue=n_cue,
  max_probes=max_probes)`: noisy copies of each protected basin's sign pattern
  × 0.9, with 15% of signs flipped, seed 0.
- `protected=None` runs `find_basins` (64 starts, global RNG).

Returns:

```python
{
    "accepted": bool,           # changed_fraction <= max_changed
    "changed_fraction": float,  # len(changed) / max(1, n_probes - n_intended)
    "changed": list[int],       # indices (into the probe set) of counted probes that moved
    "n_probes": int,
    "n_intended": int,          # probes not counted (the intended effect)
    "max_changed": float,
}
```

With no protected basins the probe set is empty, and the edit is accepted
with `changed_fraction == 0.0`.

### New names in `qriton_hlm.core` (not re-exported)

```python
EDIT_MODES = ("projected", "legacy")
AUTO_STRENGTHS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0)
DEPENDENT_COS = 0.4
GUARD_MAX_PROBES = 4096
default_probes(W, protected=None, n_cue=64, n_random=0, flip=0.15, seed=0,
               degree=3, beta=7.0, max_probes=None) -> Tensor  # (n, d), W's device/dtype
converge_batch(X, W, degree=3, max_iter=100, tau_start=0.9, tau_end=0.1,
               beta=7.0, eps=5e-3) -> Tensor  # row-wise _converge
dependent_basins(basin, states, cos_threshold=DEPENDENT_COS) -> list
```

`default_probes(max_probes=M)`: with `K` protected basins (norm ≥ 1e-8), each
gets `min(n_cue, M // K)` cues, at least 1. `n_random` random starts are added
on top of the cap.

Unchanged: `poly_interaction`, `compute_energy`, `find_basins`,
`verify_basin_exists`, `certify_compact_edit`, `load_W_from_checkpoint`,
`count_hopfield_layers`.

## 3. `BasinSurgeon`

### `params`

| key | 0.12.0 | 0.13.0 | meaning |
|---|---|---|---|
| `beta`, `inits`, `degree` | 7.0, 200, 3 | unchanged | |
| `strength` | 1.0 | 1.0 | Used when no strength is given in legacy mode or with `auto_strength=False`, and always by `strengthen`/`weaken`/`consolidate`/`causal_*` (× their factor) |
| `edit_mode` | — | `"projected"` | `"legacy"` restores 0.12.0 writes and the 0.12.0 certificate |
| `max_changed` | — | `0.02` | guard budget (fraction); per layer, `guard(max_changed_pct=)` overrides it |
| `auto_strength` | — | `True` | in projected mode, `strength=None` means `"auto"` |
| `guard_probes` | — | `4096` | guard sample size cap (`max_probes`); `None` = 64 cues per protected basin, no cap |

### What the gate does

Every write method builds `W_new` and keeps it only if it passes:

- **Projected mode:** the 0.12.0 retention certificate (`certify_compact_edit`)
  runs over the protected set, minus the edit's removed targets and their
  dependent basins. On top of it, `certify_decisions` runs with the layer's
  budget and `max_probes=params['guard_probes']`. Both must accept.
- **Legacy mode:** only the 0.12.0 certificate, over the protected set.

The protected set of a layer is chosen in this order:

1. the set pinned with `guard(layer, protected=...)`;
2. the basins stored by the last `survey(layer)` (as in 0.12.0);
3. `find_basins` with 64 starts and 60 iterations (projected), or 24 and 40
   (legacy).

The stored set is cleared by every committed edit; a pinned set is not.

A refused edit leaves `W` unchanged and returns `applied: False`; nothing is
raised. `force=True` keeps it anyway, with `retention["forced"] = True` and
`reason = "forced_despite_retention_failure"`.

### `result["retention"]`

| key | type | notes |
|---|---|---|
| `accepted` | bool | the gate's verdict |
| `applied` | bool | W was replaced (accepted, or forced) |
| `forced` | bool | |
| `reason` | str | `"retention_ok"`, `"no_op"`, `"no_protected_basins"`, `"protected_basin_lost"`, **`"decisions_changed"` (new)**, `"forced_despite_retention_failure"` |
| `compact_edit` | str | `"accepted"`, `"refused_promoted"`, `"forced"` |
| `promoted` | bool | True when refused |
| `K_protected`, `K_kept` | int | certificate set size and survivors. In projected mode this excludes the edit's removed targets and their dependents |
| `lost` | list of `{"index", "cos", "still_basin"}` | |
| `frobenius` | float | `||W_new - W||_F` |
| `nk_ratio` | None | present only for `no_op` / `no_protected_basins` (unchanged) |
| `layer` | `"cold"` | unchanged |
| **`decision`** | dict | **new, projected mode only.** The `certify_decisions` result above |

`applied` is `True` for a no-op write (`reason == "no_op"`), as in 0.12.0. A
projected edit with no room returns W unchanged and so reports `no_op`.

### Methods

| method | signature change | behaviour change | return change |
|---|---|---|---|
| `inject(layer=0, seed=42, strength=None, force=False)` | none | projected write; `strength=None` → `"auto"`; guard added | none (`strength` may be `"auto"`) |
| `remove(layer=0, seed=42, strength=None, force=False)` | **adds `force=False`** | was refused every time in 0.12.0; now excludes its target from protection; auto; guard | none |
| `move(layer=0, seed=42, strength=None, force=False)` | **adds `force=False`** | projected; auto; guard | none |
| `inject_concept(layer, concept_name, strength=None)` | none (no `force`) | projected write at `sign(centroid) * 0.9`; auto; guard | none |
| `remove_concept(layer, concept_name, strength=None, force=False)` | **adds `force=False`** | **now gated** (0.12.0 wrote unconditionally); auto; also creates the restore backup | **adds `applied`, `retention`** |
| `strengthen(layer=0, seed=42, factor=2.0)` | none | projected; strength `params['strength'] * factor` (2.0 by default: refused 6/6 in the check above) | none |
| `weaken(layer=0, seed=42, factor=0.5)` | none | projected; `params['strength'] * factor` | none |
| `causal_intervene(layer=0, basin_idx=0, operation='remove', strength=None, num_inits=50, force=False)` | **adds `force=False`** | **now gated**; projected; strength `strength or params['strength']` | **adds `applied`, `retention`** |
| `consolidate(layer=0, strengthen_factor=1.5, prune_threshold=0.3, num_inits=100, min_population=2, force=False)` | **adds `force=False`** | **now gated**, all or nothing: every prune and strengthen is kept, or none is | **adds `applied`, `retention`** (not on the early "no basins found" return) |
| `dream(layers=None, cycles=1, **kwargs)` | none (`force` passes through) | each pass is a gated `consolidate` | per-pass reports carry `applied`/`retention`. `total_strengthened`/`total_pruned` count proposals, refused ones included |
| `guard(layer, max_remove_pct=10.0, max_changed_pct=None, protected=None)` | **adds `max_changed_pct`, `protected`** | `max_changed_pct` (percent) is enforced; `protected` pins the protected set (see §6); each call replaces the layer's settings | **adds `max_changed_pct` (float, %), `num_protected` (int or None)** |
| `causal_scan(...)`, `causal_counterfactual(...)` | none | knockouts and counterfactual writes use the edit mode. In projected mode the other protected basins' field is preserved, so graphs differ from 0.12.0 (measured on 3 landscapes: 32/43/95 edges vs 36/50/118). Nothing is committed (unchanged) | none |
| `transplant(...)` | none | goes through `inject_concept` | none |
| `watermark_inject(secret_key, num_marks=3, strength=0.05, layers=None)` | none | each mark is a gated `inject` at `strength`. At 0.05 a projected write rarely creates the mark; pass `strength="auto"` and check each mark's `exists_after` | none |
| `watermark_strip_attempt(...)` | none | `remove(strength=1.0)` through the gate (usually refused) | none |

Result fields keep their 0.12.0 meaning:

- `inject`'s `exists_after` / `cos_after` and `verify()` test the basin the
  seed direction settles into once scaled to the operating range. They do not
  prove the target itself is an attractor. To check that, test its sign
  pattern: `verify_basin_exists(W, torch.sign(result["target"]) * 0.9,
  beta=beta)[0]`.
- `inject_concept` and `remove_concept` perturb the concept centroid by 10% and
  check that it settles back near itself.
- `inject` / `remove` / `move` report on the W that was kept.
  `causal_intervene`'s `basins_after` and `affected` describe the proposed W,
  even when it was refused.

`strength=0` now means zero. 0.12.0 evaluated `strength or params['strength']`,
which turned 0 into 1.0. 0.13.0 treats only `None` as "use the default".

## 4. CLI (`qriton-hlm`)

- `--version` prints `qriton-hlm 0.13.0` (it printed `qriton-hlm 0.9.4`). The
  banner and `export-concept` metadata use the same version.
- `inject`, `remove`, `move`, `inject-concept`, `remove-concept` accept `auto`
  as the strength argument. Without one they use `set strength` (default 0.1),
  which is usually too weak for a projected inject. `set strength auto` is not
  accepted (it must be a number).
- Every write is projected. The protected set is `find_basins(W, d,
  num_inits=100, beta=layer_beta)` on the layer before the edit. Writes now use
  the layer's beta (0.12.0 used 7.0).
- Gate: the retention certificate plus the decision guard at a fixed 2% (not
  configurable from the CLI). A refusal prints (real output, synthetic 64-d
  checkpoint with 5 stored concepts):

  ```
  Inject L0 seed=8 α=0.5
    REFUSED: decisions_changed
    kept 13/13 protected basins
    decisions changed: 4.1% of 832 probes (limit 2%)
    Other decisions would change with this update.
    Override: --force --reason "..." to write anyway
  ```

  Newly gated: `remove-concept`, `strengthen`, `weaken`, `causal intervene`.
  `--force --reason` overrides refusals for `inject`, `remove`,
  `inject-concept`, `remove-concept`. `move`, `strengthen`, `weaken` and
  `causal intervene` have no override.
- "target is basin" (inject, inject-concept), the move destination check and
  `verify <layer> <seed>` now test the target's sign pattern × 0.9. In 0.12.0
  they tested the unit-norm target, which settles to zero.
- The `strength-cap` guard (default 0.5) does not apply to `auto`. The ladder
  can reach 2.0, but the decision guard still applies.
- There is no legacy switch in the CLI. Use the Python API to reproduce 0.12.0.

## 5. `qriton_hlm.db.HLMSync`

Signatures and return values are unchanged. Behaviour follows `BasinSurgeon`:

- `sync_row` injects with `strength` or `SyncConfig.default_strength` (0.85),
  through the gate, in seed or capture mode. That is a projected write at 0.85,
  often refused. Pass `strength="auto"` (or set `default_strength="auto"`).
- `delete_row` calls `remove_concept`, which is **newly gated**, or falls back
  to `weaken(factor=0.1)`. A refused delete leaves the basin in place.
- New: refused writes add 1 to `stats["refused"]` and log a warning. The row
  still counts in `synced` / `deleted`, and the concept is still added to or
  dropped from `synced_concepts`. Check `stats["refused"]` if you need to know
  whether the landscape changed.

## 6. Protecting only real concepts

The default protected set is every basin that random starts find, spurious
mixtures included. Mixture basins are shallow and their cues move easily, so
the guard refuses most edits:

- function level, auto strength: accepted 15–50% (RESULTS.md);
- `BasinSurgeon.inject`, d=128 with 10 stored concepts, 10 landscapes:
  accepted 0/10.

With only the real concepts protected, acceptance was 90–100% and 10/10
(target reached 10/10).

```python
survey = surgeon.survey(layer=5)
concepts = [survey["basins"][i] for i in (0, 3, 4)]   # the basins you know are real
surgeon.guard(layer=5, protected=concepts)              # pinned until the next guard() call
result = surgeon.inject_concept(layer=5, concept_name="polite")
```

- Functions: `inject_basin(W, t, protected=concepts)`, then
  `certify_decisions(W, W_new, protected=concepts, allowed=[t])`.
- Pass basin states (`survey()["basins"]`, `find_basins`). The projection keeps
  the field exactly at the states you pass.
- A newly injected concept is not added to the pinned set. Pin it again if
  later edits must preserve it.
- Unprotected basins get no guarantee. That is the trade-off.

Raising the budget, `guard(layer, max_changed_pct=5)`, also lets more edits
through, but it relaxes the check for the real concepts as well.

## 7. Keeping 0.12.0 behaviour

| what | how |
|---|---|
| 0.12.0 weights from the functions | `inject_basin(..., mode="legacy")` (same for `remove_basin`, `move_basin`) |
| 0.12.0 `BasinSurgeon` writes and gate | `surgeon.params['edit_mode'] = 'legacy'`. inject/remove/move/inject_concept/strengthen/weaken then behave as in 0.12.0, including `remove` being refused |
| 0.12.0 unconditional `remove_concept` / `causal_intervene` / `consolidate` | legacy mode **and** `force=True` (in legacy mode they are gated by the 0.12.0 certificate) |
| CLI | not available. Use the Python API |

Legacy mode never picks `auto` by itself, but an explicit `strength="auto"`
works there too.

## 8. Cost of the guard, and the probe budget

A gated edit runs:

- `find_basins` for the protected set, once per layer, cached until a
  committed edit;
- `remove` / `move` / `strengthen` / `weaken` also run `find_basins` (100
  starts) to pick their target, as in 0.12.0;
- the auto ladder: up to 11 rungs, 5–10 short convergences each;
- `certify_decisions`, which converges `n_cue × K` probes twice, up to 100
  iterations each. Its cost grows with the number of protected basins `K` and
  with `d²`.

Measured on this machine (2 threads, shared with other jobs):

- whole edits on synthetic landscapes: 0.05–0.2 s at d=64–128, about
  0.3–0.45 s at d=256;
- `certify_decisions` alone at d=512 with 64 protected basins: about 2.2 s at
  4,096 probes, 1.0 s at 2,048, 0.7 s at 1,024.

The release notes compare test-suite times.

- `max_probes` (functions) / `params['guard_probes']` (`BasinSurgeon`) cap the
  total. Each protected basin gets `min(64, cap // K)` cues, at least 1.
- **Default 4096 = 64 × 64.** `BasinSurgeon`'s default protected set comes from
  64 random starts, so it never has more than 64 basins and the default cap
  never binds. Default edits are probed exactly as in the measurements. The
  cap only limits larger sets: a `survey()` set (up to 200 basins), the CLI's
  100-start set, or a long pinned list.
- For faster, coarser checks, lower the cap: `surgeon.params['guard_probes'] =
  1024`. The budget then rests on fewer samples. At 1,024 probes and a true
  change rate of 2%, the estimate's standard error is about 0.44 percentage
  points (0.22 at 4,096).
- `params['guard_probes'] = None` removes the cap (64 cues per basin).

## 9. Determinism

- `find_basins` draws its starts from torch's global RNG. With `protected=None`,
  `inject_basin` / `remove_basin` / `move_basin` / `certify_decisions` call it.
  Their result then depends on the global RNG state, and calling them advances
  it. Pass `protected=` for RNG-free, repeatable calls.
- The `auto` ladder and `default_probes` use their own seeded generators.
- `verify_basin_exists` adds noise from the global RNG (unchanged).

## 10. Recommended settings

- `BasinSurgeon`: leave `edit_mode="projected"` and `auto_strength=True`, and
  call the write methods without a strength. Keep `max_changed=0.02`.
- Pin the real concepts per layer with `guard(layer, protected=...)` wherever
  you know them. Re-pin after injecting a concept that must stay protected.
- Set `params['strength']` to about 0.3 if you use `strengthen`, `weaken`,
  `consolidate` or `causal_*`, then check `applied`.
- Leave `guard_probes` at 4096; lower it only when edit latency matters more
  than resolution.
- Functions: always pass `protected=`, then gate with `certify_decisions`
  before keeping `W_new`.
- CLI and `.hlm` scripts: use `auto`.

## 11. Known limits

- Every number here comes from synthetic landscapes with random targets.
  Validation on trained HLM weights is in progress and will be appended to the
  release notes.
- Protection covers only basins in the protected set. Basins that
  `find_basins` misses get no guarantee.
- The 2% budget is measured on the guard's own probes (15% of signs flipped).
  Inputs further from a concept change more. The external cross-check in the
  release notes measured 2.5% and 7.5% at 25% flips for the two edits the
  guard accepted.
- Thresholds are fixed choices: cos 0.95 for "reached", cos 0.9 for "moved",
  15% cue noise, the 2% budget, `DEPENDENT_COS = 0.4`.
