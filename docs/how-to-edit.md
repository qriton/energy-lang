# How to edit

One compact well if it certifies. Everything else explicit.

Compact store first. If the well would not survive, the write is refused and the memory still lands as an explicit slot. The trunk stays frozen.

## Compact store

One matrix. A write is kept only when existing wells still settle. If they would not, the matrix is left unchanged.

```python
from qriton_hlm import BasinSurgeon

surgeon = BasinSurgeon.from_W(W)
result = surgeon.inject(layer=0, seed=42, strength=0.1)

if result["applied"]:
    print("compact write kept")
else:
    print("refused — existing wells would not survive")
    # W is unchanged
```

```bash
hlm:model> inject 0 42 0.1
  REFUSED: protected_basin_lost
  Existing attractors would not survive this update.
```

`--force --reason "..."` writes the matrix anyway. That is the damage path, not the product path.

## Explicit slots

When a compact write is refused, the fact still stores: append a slot. No touch of the matrix.

Inspect the edit. If it is admitted, commit the slot. Correct, forget, or roll back that slot. The frozen trunk is unchanged.

The `qriton-hlm` CLI stops at the refusal. Product editors then land the same memory on an explicit slot.

## Same write, two outcomes

| Write the matrix anyway | Refuse, then land explicit |
| --- | --- |
| Ungated update on the compact store | Matrix unchanged. Memory still stored. |
| Existing wells can disappear | Trunk stays frozen |

Do not treat the compact matrix as a general editor. If a well would not survive, refuse the write and store the memory explicitly.
