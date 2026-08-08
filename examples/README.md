# Energy Language Examples

Each example is a self-contained walkthrough of the same workflow:
**observe → modify → apply → verify → restore**.

| File | What it shows | Target model |
|---|---|---|
| [`hello_basins.hlm`](hello_basins.hlm) | Minimum viable energy-language program. Single-layer inject / apply / restore. | Any HLM3 checkpoint |
| [`multi_layer_surgery.hlm`](multi_layer_surgery.hlm) | Inject at shallow and deep layers; observe the cumulative effect. | Any HLM3 checkpoint |
| [`hlm3_mix_35m_k16.hlm`](hlm3_mix_35m_k16.hlm) | Concept-capture and injection workflow on the public HLM3-Mix 35M K=16 research-preview checkpoint. | HLM3-Mix 35M K=16 research preview |
| [`hlm3_mix_35m_k16.py`](hlm3_mix_35m_k16.py) | Same workflow as above, but using the `BasinSurgeon` Python API directly. | HLM3-Mix 35M K=16 research preview |

## Running

```bash
# DSL form
qriton-hlm --script examples/hlm3_mix_35m_k16.hlm

# Python form
python examples/hlm3_mix_35m_k16.py \
  --checkpoint hlm3-mix-35m-k16-research-preview-2026-05-14/model/model.pt
```

## Getting the research-preview checkpoint

The HLM3-Mix 35M K=16 research preview is distributed as a separate package
under a research-only license. Extract its zip so that `model/model.pt` is
reachable from the path used in the example above, then run the script
against it.

> **Run from the extracted release directory.** The `generate` command (CLI)
> and any sample that calls it needs `hlm3_model.py`, `model.py`, and
> `data_utils.py` importable — those ship inside the research-preview package
> and are picked up automatically when the script runs from that folder. Pure
> W-matrix surgery (`survey`, `inject`, `inject-concept`, `apply`, `diff`,
> `restore`) works from anywhere — it only needs the checkpoint file.

Verified locally against the released checkpoint
(`model/model.pt`, 27.6M trainable params, 8 Hopfield layers): full load,
`survey-all`, `capture` / `inject-concept` / `apply` / `restore`, and
`generate` all work end-to-end.
