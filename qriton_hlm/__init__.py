"""
Qriton HLM — Energy Language

Program neural networks by shaping energy landscapes.

Usage:
    from qriton_hlm import BasinSurgeon

    surgeon = BasinSurgeon.from_checkpoint("model.pt", device="cuda")

    # Capture what "polite" looks like in the model's brain
    surgeon.capture(layer=5, text="Thank you so much", concept_name="polite")
    surgeon.capture(layer=5, text="I really appreciate it", concept_name="polite")

    # Inject that concept as a new attractor
    surgeon.inject_concept(layer=5, concept_name="polite", strength=0.1)

    # Apply to live model
    surgeon.apply(layer=5)
"""

__version__ = "0.9.5"

from qriton_hlm.core import (
    poly_interaction,
    find_basins,
    compute_energy,
    inject_basin,
    remove_basin,
    move_basin,
    verify_basin_exists,
    load_W_from_checkpoint,
    BasinSurgeon,
)

__all__ = [
    "poly_interaction",
    "find_basins",
    "compute_energy",
    "inject_basin",
    "remove_basin",
    "move_basin",
    "verify_basin_exists",
    "load_W_from_checkpoint",
    "BasinSurgeon",
    # db module available as qriton_hlm.db
]
