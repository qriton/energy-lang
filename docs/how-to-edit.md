# How to edit

Write to the compact matrix only when the write passes the gate.

Try the compact matrix first. If the write would disturb what is already stored, it is refused and the matrix is unchanged. The memory can then be kept as an explicit slot outside the matrix; `qriton-hlm` does not provide explicit slots.

## Compact store

One matrix. Every write is a proposal, kept only if it passes both checks:

1. The protected wells still settle where they were (retention certificate).
2. At most 2% of their decisions change: noisy cues of each protected well, settled by the model's own dynamics (decision guard, `certify_decisions`).

If either check fails, the matrix is left unchanged.

```python
from qriton_hlm import BasinSurgeon

surgeon = BasinSurgeon.from_W(W)
result = surgeon.inject(layer=0, seed=42)          # strength="auto"

if result["applied"]:
    print("compact write kept")
else:
    print("refused:", result["retention"]["reason"])   # W is unchanged
    decision = result["retention"]["decision"]
    print(f"{decision['changed_fraction']:.1%} of decisions would change")
```

```bash
hlm:model> inject 0 8 auto
Inject L0 seed=8 α=auto
  REFUSED: decisions_changed
  kept 13/13 protected basins
  decisions changed: 7.9% of 832 probes (limit 2%)
  Other decisions would change with this update.
  Override: --force --reason "..." to write anyway
```

A refusal for lost wells reads `REFUSED: protected_basin_lost` and
`Existing attractors would not survive this update.`

`--force --reason "..."` writes the matrix anyway. Use it for experiments only: a forced write carries no guarantee.

## What a kept write does

Since 0.13 an inject writes at its target: the target's sign pattern becomes an
attractor. The write is projected so that the field at every protected well,
`W f(b)`, stays exactly as it was. Remove and move protect the other wells the
same way.

That does not mean nothing else changes. Inputs between wells can settle
somewhere new, and only the protected wells are protected. What was measured,
on synthetic landscapes with known concepts (20 per row, [RESULTS.md](../RESULTS.md)):

| at auto strength | inject | remove | move |
| --- | --- | --- | --- |
| target reached | 100% | 100% | 95% |
| other concepts' cues changed | 0% | 1% | 1% |
| other concepts still recalled | 98–99% | 98% | 98% |

A kept write changed at most 2% of the guard's probes, because the guard
enforces that limit. Inputs further from a concept move more: in an external
check with 25% of signs flipped, the two accepted edits changed 2.5% and 7.5%.
On trained HLM3 checkpoints no tested edit was local enough to keep; see
[Trained weights](../RELEASE_NOTES.md#trained-weights) in the release notes.

## When the guard refuses too much

By default the protected wells are every basin a random-start survey finds,
spurious mixtures included. Mixtures are shallow, so the guard refuses most
edits. Measured at auto strength: 15–50% accepted, against 90–100% with only
the real concepts protected. Protect what matters:

```python
basins = surgeon.survey(layer=0)["basins"]
surgeon.guard(layer=0, protected=[basins[i] for i in known_concepts])
```

Unprotected wells get no guarantee. Raising the budget,
`guard(layer, max_changed_pct=...)`, relaxes the check for every well, the real
ones included.

## Explicit slots

When a compact write is refused, the memory can be stored as an explicit slot instead, which leaves the matrix untouched. A slot can be inspected before it is committed, and corrected, forgotten or rolled back later.

Explicit slots are not part of `qriton-hlm`: the CLI and the Python API stop at the refusal.

## Same write, two outcomes

| Write the matrix anyway | Refuse, then land explicit |
| --- | --- |
| Ungated update on the compact store | Matrix unchanged; the memory can be kept as a slot |
| Existing wells and decisions can change | Existing wells and decisions unchanged |

Do not treat the compact matrix as a general editor. If a write would disturb what is stored, refuse it and store the memory explicitly.
