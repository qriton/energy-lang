# qriton-hlm 0.13.0: edits write at their target, and every write is gated

Energy Language edits now do what they say. An inject used to reinforce
whatever basin its target happened to fall into, and never created the target.
It now writes the target itself. The write is projected so that the field at
the protected basins does not change, and every write path goes through a
decision-stability guard that refuses edits which would move more than 2% of
the protected basins' decisions.

The default behaviour changes, so this is a 0.x minor release. Upgrade notes,
with every API change, are in [MIGRATION.md](MIGRATION.md).
`mode="legacy"` reproduces 0.12.0's writes.

## Measured

On synthetic landscapes that store known concepts (d=64/K=5 and d=128/K=10,
20 landscapes per row; [RESULTS.md](RESULTS.md)):

| | 0.12.0 inject (strength 0.5) | 0.13.0 inject (strength auto, ≈0.43) |
|---|---|---|
| target became an attractor | 0% (0/120 at any strength tried) | 100% |
| other concepts' cues changed | 37–39% | 0% |
| other concepts still recalled | 62–66% | 98–99% |
| protected basins' field `W f(b)` | changed | unchanged (exact, tested) |

- Remove at auto strength (≈0.29) took out its target 100% of the time and
  changed 1% of other concepts' cues. In 0.12.0, `BasinSurgeon.remove` was
  refused 60/60 times by its own gate.
- Move at auto strength (≈0.48) reached its target 95% of the time, changing 1%
  of other concepts' cues.
- When the guard protects every visible basin, spurious mixtures included, it
  accepts 15–50% of auto edits. When it protects only the stored concepts, it
  accepts 90–100%.
- The 0.13.0 code reproduces the measured rows bit for bit (32/32 re-run rows).

### External cross-check (`edit_locality.py`, 3 seeds)

This independent kit runs the library's own `core.py` on landscapes with 8
stored concepts in d=64. It probes decisions with 25% of signs flipped, harder
than the guard's 15%. Its "shipped" column calls `inject_basin(W, target,
strength=s)`, which in 0.13.0 is the new default: a projected write with
auto-detected protected basins, a fixed strength, and no gate. The kit's
stored 0.12.0 run uses the same seeds; its direct and projected rows match this
run 30/30, so the 0.12.0 column is a like-for-like comparison.

| strength | 0.12.0 inject: target is an attractor / decisions changed | 0.13.0 inject: target is an attractor / decisions changed |
|---|---|---|
| 0.1 | 0% / 18.3% | 0% / 1.5% |
| 0.25 | 0% / 50.2% | 0% / 3.5% |
| 0.5 | 0% / 67.3% | 100% / 11.9% |
| 1.0 | 0% / 79.6% | 100% / 57.1% |
| 2.0 | 0% / 82.9% | 100% / 81.0% |

Before any edit, 81.9% of decisions were correct (mean of the 3 seeds). The old retention
certificate accepted the 0.13.0 writes up to strength 1.0, while 57% of
decisions changed. That is the gap the new decision guard closes. Raw rows are
in `energy-lang-locality-check.json`.

The same landscapes went through the full released pipeline (auto strength,
then the guard; `energy-lang-locality-gated.json`):

- Protecting every visible basin (16–24 basins): auto picked 0.4–0.7. The
  target became an attractor 3/3 times, and the guard refused all three (its
  probes changed 3.5–12.8%).
- Protecting the 8 concepts: auto picked 0.3–0.5. The target became an
  attractor 3/3 times and the guard accepted 2/3. The two accepted edits
  changed 1.6% and 0.8% of the guard's probes, but 2.5% and 7.5% of the kit's
  harder 25%-noise decisions.

The 2% limit holds on the guard's own probes. Inputs further from a concept
move more, and denser landscapes (8 concepts in 64 dimensions) leave less room.

## Changes

**Default behaviour**

- `inject_basin` / `remove_basin` / `move_basin` default to
  `mode="projected"`. They also accept `strength="auto"` and `protected=`.
- `BasinSurgeon` edits without a strength use `"auto"`.
- `remove_concept`, `causal_intervene` and `consolidate` are now gated, as are
  the CLI's `remove-concept`, `strengthen`, `weaken` and `causal intervene`.
  A refused edit leaves W unchanged and returns `applied: False`.

**New**

- `certify_decisions(...)`, the decision-stability guard. Refusals report
  `reason: "decisions_changed"` and list the probes that would move.
- `BasinSurgeon.guard(layer, max_changed_pct=..., protected=[...])`. The budget
  is enforced, and you can pin the protected set, for example to the real
  concepts only.
- A guard sample-size budget: `certify_decisions(max_probes=4096, n_cue=64)`
  and `params['guard_probes']`. The default cap never binds for the default
  protected set (at most 64 basins); it bounds survey-sized or user-supplied
  sets.
- `force=` on `remove`, `move`, `remove_concept`, `causal_intervene` and
  `consolidate`.
- `HLMSync.stats["refused"]`: sync and delete writes the gate refused, also
  logged as warnings.

**Fixed**

- Inject never wrote at its target. `BasinSurgeon.remove` was always refused.
  Several write paths had no check at all.
- CLI inject/remove/move ignored the layer's beta.
- `guard()` settings were never read.
- Projected writes wrote rounding noise onto the protected basins when those
  spanned every direction (at least as many protected states as dimensions).
  They now return W unchanged.
- The CLI reported version 0.9.4.
- CLI "target is basin" and `verify` tested a state that always collapses to
  zero, so they printed False even after a successful inject.

## Cost

The guard settles its probe set twice per edit.

- Whole edits on synthetic landscapes: 0.05–0.2 s at d=64–128, about
  0.3–0.45 s at d=256 (2 threads).
- The guard alone, at d=512 with 64 protected basins: about 2.2 s at 4,096
  probes, 1.0 s at 2,048, 0.7 s at 1,024.
- Test-suite CPU time on the release machine, one run each under heavy load
  (repeat runs varied by tens of percent): 16 s for 0.12.0 (69 tests), 38 s
  for the fix commit (83 tests), 35 s for 0.13.0 (96 tests).

Lower `params['guard_probes']` to trade resolution for speed.

## Known limits

- Every number above comes from synthetic landscapes with random targets.
- Only basins in the protected set are protected. The default set includes
  shallow spurious mixtures, which makes the guard strict. Pin the real
  concepts with `guard(layer, protected=...)`.
- `strengthen`, `weaken`, `consolidate` and the `causal_*` methods do not
  search for a strength. With the default `params['strength']` of 1.0 they are
  usually refused, so lower it.
- In the CLI, pass `auto` (`inject 0 42 auto`). The CLI default strength of
  0.1 rarely creates an attractor.

## Validation on trained HLM weights

Pending. A validation run on real HLM checkpoints is in progress, and its
results will be added here.

## Install

```bash
pip install --upgrade qriton-hlm==0.13.0
```

Artifacts: `qriton_hlm-0.13.0.tar.gz` and `qriton_hlm-0.13.0-py3-none-any.whl`
(`twine check --strict` passed). SHA-256 sums are in `dist/SHA256SUMS`.
