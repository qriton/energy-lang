"""
example_landscape.py — Deep analysis of energy landscapes across layers.

Compares basin topology, identifies critical layers, measures surgery tolerance.
Run: python example_landscape.py --checkpoint model.pt
"""

import argparse
import torch
from qriton_hlm import BasinSurgeon

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    surgeon = BasinSurgeon.from_checkpoint(args.checkpoint, device=args.device)
    n_layers = surgeon.num_layers()
    print(f"Model: {n_layers} Hopfield layers\n")

    # --- Per-layer topology ---
    print("Layer | Basins | E_min     | E_max     | Top basin pop")
    print("------|--------|-----------|-----------|-------------")
    for layer in range(n_layers):
        landscape = surgeon.landscape(layer)
        n = landscape['num_basins']
        e_min, e_max = landscape['energy_range']
        top_pop = max(b['population'] for b in landscape['basins']) if landscape['basins'] else 0
        print(f"  {layer:3d} | {n:5d}  | {e_min:+9.4f} | {e_max:+9.4f} | {top_pop}")

    # --- Surgery tolerance test ---
    print("\nSurgery tolerance (inject strength=0.1, measure basin count delta):")
    for layer in range(n_layers):
        before = surgeon.survey(layer)
        surgeon.inject(layer=layer, seed=42, strength=0.1)
        after = surgeon.survey(layer)
        delta = after['num_basins'] - before['num_basins']
        surgeon.restore(layer)
        print(f"  L{layer}: {before['num_basins']} -> {after['num_basins']} ({delta:+d})")

if __name__ == '__main__':
    main()
