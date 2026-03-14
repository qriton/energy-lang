"""
example_probe.py — Reverse-engineer what basins represent.

Given a basin index, probe what tokens it activates.
Use this to understand what the model has learned.
Run: python example_probe.py --checkpoint model.pt
"""

import argparse
import torch
from qriton_hlm import BasinSurgeon

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', required=True)
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--num-basins', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    surgeon = BasinSurgeon.from_checkpoint(args.checkpoint, device=args.device)

    # Survey to find basins
    survey = surgeon.survey(layer=args.layer)
    print(f"Layer {args.layer}: {survey['num_basins']} basins\n")

    # Probe each basin
    for i in range(min(args.num_basins, survey['num_basins'])):
        try:
            probe = surgeon.probe(layer=args.layer, basin_idx=i)
            tokens_str = ', '.join(probe['top_tokens'][:5]) if probe['top_tokens'] else '(no decoder)'
            print(f"  B{i:2d}  E={probe['energy']:+.4f}  tokens=[{tokens_str}]")
        except Exception as e:
            print(f"  B{i:2d}  error: {e}")

if __name__ == '__main__':
    main()
