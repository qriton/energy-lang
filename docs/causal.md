# Energy-Minima Causal Programming

Causal discovery and intervention on polynomial Hopfield networks, using energy basins as causal nodes and basin surgery as the do-operator.

## Core Idea

In a transformer, "what causes what" is opaque. Attention weights correlate with output, but correlation is not causation. You cannot remove a single learned behavior and measure what else breaks — the representations are entangled across softmax's single attractor.

HLM's polynomial Hopfield layers create **multiple discrete basins** per layer. Each basin is a stable fixed point of the energy landscape — a distinct attractor that captures a specific pattern the model has learned. This discreteness gives us something transformers lack: **individually addressable causal units**.

The causal framework maps directly to Pearl's do-calculus:

| Pearl's Framework | HLM Equivalent |
|---|---|
| Variable X | Basin B_i |
| do(X = x) | `remove_basin(W, B_i)` or `inject_basin(W, B_i')` |
| P(Y \| do(X)) | Basin drift of B_j after intervening on B_i |
| Counterfactual X' | Modified basin (inverted, weakened, shifted) |
| Causal graph | Adjacency from systematic knockout scan |

The key insight: because basins are **stable fixed points**, removing one and measuring drift on the others is a clean intervention, not a noisy ablation. The dynamics either converge to the same attractor (no causal link) or they don't (causal link). There's no gradient to confound — it's topological.

## Commands

### `causal scan` — Discover the causal graph

Systematically knocks out each basin and measures which other basins shift or disappear. Produces a directed causal graph.

**How it works:**
1. Survey all basins on the original W matrix (baseline)
2. For each basin B_i: temporarily remove it via anti-Hebbian update
3. For each other basin B_j: re-converge from B_j's location on the modified W
4. Measure cosine drift: `drift = 1 - cos(B_j_original, B_j_after_knockout)`
5. If drift > threshold, record edge B_i -> B_j
6. Restore original W before next knockout

```
hlm:model> causal scan 5
Causal scan L5: 7 basins, threshold=0.15
  Knocking out each basin and measuring downstream drift...

  Causal edges found: 3

    Source  ->  Target      Drift  Effect
    --------     --------  --------  ----------
        B0  ->  B3          0.4231  DESTROYED
        B0  ->  B5          0.1823  shifted
        B2  ->  B6          0.2104  shifted

  B0 causes -> B3, B5
  B2 causes -> B6
```

```
hlm:model> causal scan 5 0.25
```

The optional second argument sets the drift threshold (default: 0.15). Higher = only strong causal links. Lower = more sensitive, more edges.

### `causal intervene` — Apply do(X)

Performs Pearl's do-operator: intervene on a specific basin and measure the full downstream effect. Unlike `causal scan`, this **persists** the change.

**Operations:**
- `remove` (default) — delete the basin entirely
- `weaken` — reduce basin depth by 50%
- `strengthen` — deepen the basin

```
hlm:model> causal intervene 5 0
do(B0) = remove  L5
  Energy before: -12.3456
  Basins: 7 -> 6 (-1)

  Downstream effects: 2 basins affected
    B  3  drift=0.4231  gone  ████████████
    B  5  drift=0.1823  ok    █████

hlm:model> causal intervene 5 2 strengthen
do(B2) = strengthen  L5
  Energy before: -8.9012
  Basins: 6 -> 6 (+0)

  Downstream effects: 1 basins affected
    B  4  drift=0.0892  ok    ██
```

Use `restore <layer>` to undo the intervention.

### `causal counterfactual` — What if X had been different?

Non-destructive. Asks "what would happen if basin X were modified?" without actually changing the model. The original landscape is preserved.

**Modifications:**
- `invert` (default) — flip the basin (negate state vector)
- `weaken` — scale basin state to 50%
- `shift` — move basin to a random nearby location (adds Gaussian noise, preserves norm)

```
hlm:model> causal counterfactual 5 0 invert
Counterfactual: B0 -> invert  L5
  (non-destructive -- original landscape preserved)
  Basins: 7 -> 8

  Would affect: 3 basins
    B  3  drift=0.5102  gone
    B  5  drift=0.2341  ok
    B  1  drift=0.0712  ok

hlm:model> causal counterfactual 5 2 shift
Counterfactual: B2 -> shift  L5
  (non-destructive -- original landscape preserved)
  Basins: 7 -> 7

  No downstream effects predicted.
```

## Python API

### `BasinSurgeon.causal_scan()`

```python
from qriton_hlm import BasinSurgeon

surgeon = BasinSurgeon.from_checkpoint("model.pt", device="cuda")

# Discover causal graph in layer 5
result = surgeon.causal_scan(layer=5, threshold=0.15, num_inits=50)

print(f"Basins: {result['num_basins']}")
print(f"Edges:  {len(result['edges'])}")

# Iterate edges
for edge in result['edges']:
    src, tgt = edge['source'], edge['target']
    print(f"  B{src} -> B{tgt}  drift={edge['drift']:.4f}  ({edge['type']})")

# Adjacency dict: {basin_idx: [list of downstream basin indices]}
for node, targets in result['adjacency'].items():
    if targets:
        print(f"  B{node} causes -> {['B'+str(t) for t in targets]}")
```

**Parameters:**
- `layer` (int) — layer index
- `strength` (float, optional) — surgery strength for knockout (default: from `surgeon.params`)
- `num_inits` (int) — random initializations for basin discovery (default: 50)
- `threshold` (float) — minimum cosine drift to register a causal link (default: 0.15)

**Returns:** dict with keys `layer`, `num_basins`, `edges`, `adjacency`, `basins`

### `BasinSurgeon.causal_intervene()`

```python
# Remove basin 0 and measure effects
result = surgeon.causal_intervene(layer=5, basin_idx=0, operation='remove')

print(f"Basins: {result['basins_before']} -> {result['basins_after']}")
print(f"Target energy before: {result['target_energy_before']:.4f}")
print(f"Affected basins: {result['num_affected']}")

for a in result['affected']:
    print(f"  B{a['basin']}  drift={a['drift']:.4f}  "
          f"survived={a['survived']}  "
          f"energy: {a['energy_before']:.4f} -> {a['energy_after']:.4f}")

# The change is persisted in the surgeon's W cache.
# To undo:
surgeon.restore(layer=5)
```

**Parameters:**
- `layer` (int) — layer index
- `basin_idx` (int) — which basin to intervene on
- `operation` (str) — `'remove'`, `'weaken'`, or `'strengthen'`
- `strength` (float, optional) — surgery strength
- `num_inits` (int) — random initializations for basin discovery (default: 50)

**Returns:** dict with keys `layer`, `basin_idx`, `operation`, `basins_before`, `basins_after`, `target_energy_before`, `affected`, `num_affected`

### `BasinSurgeon.causal_counterfactual()`

```python
# What if basin 0 had been inverted?
result = surgeon.causal_counterfactual(layer=5, basin_idx=0, modification='invert')

print(f"Basins: {result['basins_original']} -> {result['basins_counterfactual']}")
print(f"Would affect: {result['num_affected']} basins")

for a in result['affected']:
    print(f"  B{a['basin']}  drift={a['drift']:.4f}  survived={a['survived']}")

# W is NOT modified. The surgeon's state is unchanged.
```

**Parameters:**
- `layer` (int) — layer index
- `basin_idx` (int) — which basin to modify in the counterfactual
- `modification` (str) — `'invert'`, `'weaken'`, or `'shift'`
- `strength` (float, optional) — surgery strength
- `num_inits` (int) — random initializations for basin discovery (default: 50)

**Returns:** dict with keys `layer`, `basin_idx`, `modification`, `basins_original`, `basins_counterfactual`, `affected`, `num_affected`

## Pipeline: Discovery to Verification

A realistic workflow for understanding and modifying model behavior:

```python
from qriton_hlm import BasinSurgeon

surgeon = BasinSurgeon.from_checkpoint("model.pt", device="cuda")

# 1. DISCOVERY — map the causal structure
graph = surgeon.causal_scan(layer=5, threshold=0.15)
print(f"Found {graph['num_basins']} basins, {len(graph['edges'])} causal links")

# Identify hub basins (high out-degree = many downstream effects)
hub_basins = [
    node for node, targets in graph['adjacency'].items()
    if len(targets) >= 2
]
print(f"Hub basins (high influence): {hub_basins}")

# 2. COUNTERFACTUAL — test hypotheses before committing
for hub in hub_basins:
    cf = surgeon.causal_counterfactual(layer=5, basin_idx=hub, modification='invert')
    print(f"  B{hub}: inverting would affect {cf['num_affected']} basins")

# 3. INTERVENTION — apply the change
target_basin = hub_basins[0]  # pick the most influential
result = surgeon.causal_intervene(layer=5, basin_idx=target_basin, operation='remove')
print(f"Removed B{target_basin}: {result['basins_before']} -> {result['basins_after']} basins")

# 4. VERIFICATION — confirm the effect
surgeon.apply(layer=5)
bench = surgeon.benchmark()
print(f"Perplexity after intervention: {bench['perplexity']:.2f}")

# If the result is bad, roll back
surgeon.restore(layer=5)
```

Or as a CLI script (`causal_pipeline.hlm`):

```bash
load model.pt
survey 5
causal scan 5 0.15
causal counterfactual 5 0 invert
causal intervene 5 0 remove
apply 5
benchmark
generate Tell me about the weather
restore 5
```

## Advantages Over Transformer-Based Causal Analysis

**Clean interventions.** In transformers, ablating a head or neuron is noisy — the representation is distributed, so removing one piece causes unpredictable cascading failures. In HLM, each basin is a discrete attractor. Removing it is a topologically clean operation: either other basins still converge to the same point, or they don't. There's no "partial activation" to confuse the picture.

**Intrinsic causal units.** Transformer causal analysis requires choosing what to treat as a "variable" — individual neurons, attention heads, layers, circuits. The choice is arbitrary and results depend heavily on it. HLM basins are natural causal units: they emerge from the dynamics, not from the researcher's decomposition.

**Counterfactuals without retraining.** Testing "what if this behavior were different" in a transformer requires activation patching or fine-tuning — both are approximate and expensive. In HLM, `causal_counterfactual` modifies the energy landscape analytically and measures the result. No gradient computation, no training loop.

**Directionality from knockout, not correlation.** Transformer probing tells you that two representations are correlated, not which causes which. The knockout scan produces directed edges: "removing B_i changes B_j" is an asymmetric relationship.

**Composable with surgery.** Once you identify a causal link, you can act on it immediately using the same `BasinSurgeon` API — inject, remove, strengthen, weaken. The causal analysis and the intervention use the same substrate.

## Limitations

**Basin discovery is stochastic.** `find_basins` uses random initializations. With `num_inits=50`, you may miss shallow or narrow basins. Results vary across runs. Increase `num_inits` for more reliable graphs, at the cost of O(n^2) runtime per layer.

**Causal graph is within a single layer.** The current `causal_scan` operates on one layer's W matrix at a time. Cross-layer causal links (basin in layer 3 influences basin in layer 7) are not captured. You can scan each layer independently and reason about the full pipeline, but there's no automated cross-layer causal discovery yet.

**No data-driven graph learning.** The knockout scan is an interventional method — it discovers which basins depend on which by perturbation. It does not learn a causal graph from observational data (e.g., from a corpus of inputs and their basin trajectories). For data-driven structure learning, you need external tools.

**Threshold sensitivity.** The drift threshold (default 0.15) determines what counts as a causal link. Too low and you get noise edges from numerical jitter. Too high and you miss real but weak links. There's no principled way to set it — you have to calibrate per model.

**Knockout strength matters.** The `strength` parameter controls how aggressively a basin is removed. If strength is too low, the basin partially survives and downstream effects are muted. If too high, the W matrix perturbation is large enough to create spurious effects beyond the target basin.

**Quadratic cost.** `causal_scan` is O(n^2) in the number of basins: for each of n basins, it checks drift on all n-1 others. At 20+ basins per layer, this becomes slow. On CPU, expect seconds per layer for small models; on GPU, it's faster but still scales quadratically.

## Integration with DoWhy and CausalNex

The causal graph from `causal_scan` is a plain adjacency dict. You can export it to standard causal inference libraries for further analysis.

### DoWhy

```python
import networkx as nx
import dowhy
from dowhy import CausalModel

# Run causal scan
result = surgeon.causal_scan(layer=5, threshold=0.15)

# Build NetworkX graph
G = nx.DiGraph()
for edge in result['edges']:
    G.add_edge(f"B{edge['source']}", f"B{edge['target']}",
               weight=edge['drift'], effect_type=edge['type'])

# Convert to DoWhy format
# You need observational data — basin activations across inputs
# basin_data: DataFrame with columns B0, B1, ..., Bn (activation counts or energies)
model = CausalModel(
    data=basin_data,
    treatment=f"B{target_basin}",
    outcome=f"B{downstream_basin}",
    graph=G,
)

# Estimate causal effect
estimate = model.identify_effect()
result = model.estimate_effect(estimate, method_name="backdoor.linear_regression")
print(f"Causal effect: {result.value:.4f}")

# Refutation
refutation = model.refute_estimate(estimate, result,
                                    method_name="random_common_cause")
print(f"Refutation p-value: {refutation.refutation_result['p_value']:.4f}")
```

### CausalNex (structure learning)

```python
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas

# If you have observational basin activation data,
# CausalNex can learn the graph structure directly
# (complementing the knockout-based discovery)

# basin_activations: DataFrame, rows=inputs, cols=basin energies per layer
sm = from_pandas(basin_activations, tabu_edges=[], w_threshold=0.3)

# Compare learned graph with knockout graph
knockout_edges = {(e['source'], e['target']) for e in result['edges']}
learned_edges = set(sm.edges())

shared = knockout_edges & learned_edges
print(f"Edges found by both methods: {len(shared)}")
print(f"Knockout-only: {len(knockout_edges - learned_edges)}")
print(f"Data-driven-only: {len(learned_edges - knockout_edges)}")
```

The knockout scan (interventional) and data-driven learning (observational) are complementary. The knockout scan gives you ground-truth directed edges for the basins it can find. Data-driven methods can reveal softer statistical dependencies and cross-layer patterns that knockout misses. Use both when precision matters.
