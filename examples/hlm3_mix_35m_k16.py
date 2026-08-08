"""Operate on the HLM3-Mix 35M K=16 research preview from Python.

Companion to ``examples/hlm3_mix_35m_k16.hlm``. Same workflow, but using the
``BasinSurgeon`` Python API directly so it can be embedded in a notebook or
batch pipeline.

Run:
    python examples/hlm3_mix_35m_k16.py \
        --checkpoint hlm3-mix-35m-k16-research-preview-2026-05-14/model/model.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from qriton_hlm import BasinSurgeon


DEFAULT_CHECKPOINT = (
    "hlm3-mix-35m-k16-research-preview-2026-05-14/model/model.pt"
)

FORMAL_SAMPLES = [
    "The committee convened to deliberate the proposal",
    "The findings were duly noted and entered into the record",
    "Per the established protocol the matter was referred onward",
]


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def survey_all(surgeon: BasinSurgeon) -> None:
    _print_header("Survey: basins and beta per layer")
    for layer in range(surgeon.num_layers()):
        result = surgeon.survey(layer=layer)
        n_basins = result.get("num_basins")
        beta = surgeon.get_beta(layer=layer)
        print(f"  L{layer}: basins={n_basins}  beta={beta:.3f}")


def capture_concept(
    surgeon: BasinSurgeon, layer: int, name: str, samples: list[str]
) -> None:
    _print_header(f"Capture concept '{name}' at L{layer} ({len(samples)} samples)")
    for text in samples:
        surgeon.capture(layer=layer, text=text, concept_name=name)
    print(f"  Captured concepts: {surgeon.list_concepts()}")


def inject_and_verify(
    surgeon: BasinSurgeon, layer: int, name: str, strength: float
) -> None:
    _print_header(f"Inject '{name}' at L{layer} (strength={strength})")
    surgeon.guard(layer=layer, max_remove_pct=10.0)
    surgeon.inject_concept(layer=layer, concept_name=name, strength=strength)
    surgeon.apply(layer=layer)
    diff = surgeon.diff(layer=layer)
    print(f"  Frobenius delta: {diff.get('frobenius_delta'):.4f}")
    print(f"  Affected entries: {diff.get('pct_changed', 0):.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--concept", default="formal")
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--prompt",
        default="The history of the city of",
        help="Prompt used for before / after generation comparison.",
    )
    parser.add_argument("--tokens", type=int, default=30)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Download the research-preview package and extract it next to this "
            "script, or pass --checkpoint pointing at model/model.pt."
        )

    _print_header(f"Loading {checkpoint_path}")
    surgeon = BasinSurgeon.from_checkpoint(str(checkpoint_path), device=args.device)
    print(f"  Hopfield layers found: {surgeon.num_layers()}")

    survey_all(surgeon)

    _print_header("Baseline generation (no surgery)")
    print(surgeon.benchmark(texts=[args.prompt], max_tokens=args.tokens))

    capture_concept(surgeon, args.layer, args.concept, FORMAL_SAMPLES)
    inject_and_verify(surgeon, args.layer, args.concept, args.strength)

    _print_header("Generation after surgery")
    print(surgeon.benchmark(texts=[args.prompt], max_tokens=args.tokens))

    _print_header(f"Restoring L{args.layer}")
    surgeon.restore(layer=args.layer)
    print(surgeon.benchmark(texts=[args.prompt], max_tokens=args.tokens))


if __name__ == "__main__":
    main()
