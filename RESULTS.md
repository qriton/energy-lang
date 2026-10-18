# Energy Language edits: measured before and after the fix

qriton-hlm 0.12.0 (`v0.12.0`, measured at dba627e, which adds only docs to the tag)
compared with the fix released in 0.13.0, measured at commit af316dd. The later
0.13.0 changes leave these rows as they are:
re-running `measure_edits.py --mode projected --trials 2 --auto` on the 0.13.0 code
reproduces the d=64 rows of trials 0–1 bit for bit (32/32). The probe cap never binds
here (at most 64 protected basins) and the no-room case does not occur.
Script: `measure_edits.py`. Raw rows: `results/*.json`. Tables: `results/make_table.py`.

## Setup

- **Landscapes:** synthetic degree-3 polynomial Hopfield landscapes that store
  K random ±1 concepts, `W = (1/d) Σ ξ f(ξ)ᵀ`. There are 10 landscapes each at
  d=64/K=5 and d=128/K=10, so every row below has n=20. Beta is 7, the
  package's own dynamics are used, and runs are seeded with seed 0.
- **Targets:** "seed" targets are random unit directions, as with `inject <seed>`.
  "Pattern" targets are ±0.9 patterns, like a captured, already converged concept.
- **Probes:** each landscape gets a fixed probe set: 120–200 noisy cues per
  concept (15% of signs flipped) plus 800 random starts.

**Metrics:**
- **target reached:** a 10%-noised start at the target converges onto it
  (cos > 0.95). For remove, the concept and its mirror no longer attract.
- **concept cues changed:** noisy cues of *other* concepts whose settling point moved.
- **concept decisions changed:** the same, plus random starts that used to settle on a stored concept.
- **concepts kept:** other stored concepts still recalled.
- **0.12 gate accepts:** the 0.12.0 retention certificate, unchanged: 24
  random-start basins must survive.
- **new guard accepts:** `certify_decisions` at its 2% limit. It is run once
  protecting every visible basin and once protecting only the stored concepts.

The rows for 0.12.0 come from `mode="legacy"`. This reproduces 0.12.0 exactly:
240 of 240 rows are identical to unmodified 0.12.0 (`results/shipped.json`).

## Headline

| | 0.12.0 inject (strength 0.5) | fixed inject (strength auto, ≈0.43) |
|---|---|---|
| target became an attractor | 0% (0/120 at any strength) | 100% |
| other concepts' cues changed | 37–39% | 0% |
| other concepts still recalled | 62–66% | 98–99% |
| protected basins' field `W f(b)` | changed | unchanged exactly (tested) |

In 0.12.0, inject wrote onto the basin the target fell into. Across 120 trials
that was an existing concept 45 times, its mirror 42 times and a spurious
mixture 33 times.

## Full tables

**0.12.0 as shipped**

| edit | strength | target reached | concept cues changed | concept decisions changed | concepts kept | 0.12 gate accepts | new guard accepts (all / concepts) |
|---|---|---|---|---|---|---|---|
| inject (seed) | 0.1 | 0% | 4% | 19% | 97% | 15% | 0% / 65% |
| inject (seed) | 0.5 | 0% | 37% | 51% | 66% | 0% | 0% / 0% |
| inject (seed) | 1.0 | 0% | 58% | 66% | 43% | 0% | 0% / 0% |
| inject (pattern) | 0.1 | 0% | 7% | 21% | 94% | 15% | 0% / 55% |
| inject (pattern) | 0.5 | 0% | 39% | 51% | 62% | 0% | 0% / 0% |
| inject (pattern) | 1.0 | 0% | 61% | 66% | 41% | 5% | 0% / 0% |
| remove | 0.1 | 45% | 1% | 6% | 99% | 0% | 15% / 95% |
| remove | 0.5 | 100% | 2% | 12% | 99% | 0% | 10% / 90% |
| remove | 1.0 | 95% | 5% | 21% | 95% | 0% | 10% / 80% |
| move | 0.1 | 0% | 4% | 17% | 99% | 10% | 0% / 70% |
| move | 0.5 | 0% | 34% | 48% | 68% | 0% | 0% / 0% |
| move | 1.0 | 65% | 58% | 68% | 42% | 0% | 0% / 0% |

**Fixed (`mode="projected"`, default)**

| edit | strength | target reached | concept cues changed | concept decisions changed | concepts kept | 0.12 gate accepts | new guard accepts (all / concepts) |
|---|---|---|---|---|---|---|---|
| inject (seed) | 0.1 | 0% | 0% | 1% | 98% | 100% | 100% / 100% |
| inject (seed) | 0.5 | 100% | 1% | 16% | 98% | 85% | 5% / 95% |
| inject (seed) | 1.0 | 100% | 11% | 38% | 88% | 65% | 0% / 0% |
| inject (seed) | auto (≈0.43) | 100% | 0% | 12% | 98% | 95% | 15% / 95% |
| inject (pattern) | 0.1 | 0% | 0% | 1% | 98% | 95% | 100% / 100% |
| inject (pattern) | 0.5 | 100% | 1% | 17% | 97% | 85% | 10% / 95% |
| inject (pattern) | 1.0 | 100% | 11% | 37% | 88% | 65% | 0% / 0% |
| inject (pattern) | auto (≈0.43) | 100% | 0% | 12% | 99% | 95% | 15% / 95% |
| remove | 0.1 | 25% | 0% | 4% | 99% | 50% | 80% / 95% |
| remove | 0.5 | 100% | 3% | 12% | 97% | 5% | 20% / 95% |
| remove | 1.0 | 90% | 5% | 19% | 94% | 20% | 5% / 85% |
| remove | auto (≈0.29) | 100% | 1% | 7% | 98% | 10% | 50% / 90% |
| move | 0.1 | 0% | 0% | 4% | 98% | 65% | 80% / 95% |
| move | 0.5 | 95% | 3% | 20% | 97% | 15% | 15% / 95% |
| move | 1.0 | 100% | 29% | 51% | 68% | 0% | 0% / 0% |
| move | auto (≈0.48) | 95% | 1% | 13% | 98% | 20% | 35% / 100% |

## What the gates did in 0.12.0

- **The retention certificate tests the wrong thing.** It checks whether about
  24 basins found from random starts still exist, not whether decisions stay put.
  - Of 120 injects it accepted 7, and none of them reached its target.
  - The edits it accepted moved up to 47% of concept decisions.
- **`BasinSurgeon.remove` was refused 60 times out of 60.** The basin being
  removed was in its own protected set.
- **Removal was the only edit that worked,** and it worked anyway: 95–100% at
  strength 0.5–1.0, with 1–5% of other cues changed. The low-level remove was
  fine; the gate blocked it.
- **These paths wrote with no check at all:** `remove_concept`, `consolidate`
  (pruning) and `causal_intervene`, plus the CLI's `remove-concept`,
  `strengthen`, `weaken` and `causal intervene`.
- **`BasinSurgeon.guard()` was never enforced.** It stored a threshold that
  nothing read.

## Reading the fixed numbers

- **"Concept decisions changed" at auto strength (7–13%) is almost all random
  starts.** Cues of other concepts change by 0–1%. A random start is a
  Gaussian vector far from every concept. A new concept takes a share of that
  space; a removed one hands its share to the rest.
- **The new guard is strict when it protects every visible basin.** At auto
  strength it accepts only 15–50% of edits, because spurious mixture basins are
  shallow and their cues move. Protecting only real concepts, it accepts 90–100%.
  - In a real checkpoint you can't tell concepts from mixtures, so the default
    protects everything and refuses.
  - To relax it, use `guard(layer, max_changed_pct=...)`, or pass a protected list.
- **Strength 1.0, the default before this fix, is too strong for the projected
  write.** The target is reached, but 11% of cues and 37% of decisions change,
  and the guard refuses it. `strength="auto"` picks about 0.3–0.5.

## Caveats

- **The landscapes are synthetic, with random targets.** No trained HLM
  checkpoint is in these measurements. Captured concepts in a trained W may
  behave differently. On trained HLM3 checkpoints no tested edit was local enough to keep;
  see Trained weights in RELEASE_NOTES.md.
- **The thresholds are mine:** cos 0.95 to count as reached, cos 0.9 to count
  as moved, 15% cue noise, a 2% guard limit and `DEPENDENT_COS` 0.4.
- **Fixed-mode protection only covers basins `find_basins` can see.** Basins it
  misses are not protected.
- **Edits are slower.** Projected edits that are not given a protected list run
  `find_basins` (64 starts), and the guard converges about 64 cues per basin
  twice. The test suite went from 7 s to about 70 s.
- **The 0.12 gate column for fixed-mode removes uses the old certificate.** That
  certificate counts the removed target as lost, so it understates acceptance.
  The library now leaves the target and the basins built from it out of
  retention.
