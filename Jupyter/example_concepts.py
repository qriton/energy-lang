"""
example_concepts.py — Capture, inject, blend, and transplant concepts.

Requires an HLM3 checkpoint with tokenizer.
Run: python example_concepts.py --checkpoint model.pt
"""

import argparse
import torch
from qriton_hlm import BasinSurgeon

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', required=True)
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    surgeon = BasinSurgeon.from_checkpoint(args.checkpoint, device=args.device)
    L = args.layer

    # --- Capture 'polite' ---
    polite_examples = [
        'Thank you so much for your help',
        'I really appreciate your patience',
        'That is very kind of you',
        'Would you mind helping me with this',
    ]
    print("Capturing 'polite'...")
    for text in polite_examples:
        r = surgeon.capture(layer=L, text=text, concept_name='polite')
        print(f"  E={r['energy']:.4f} basin={r['is_basin']} \"{text[:40]}\"")

    # --- Capture 'formal' ---
    formal_examples = [
        'Per our previous correspondence',
        'Please find attached the requested documents',
        'I am writing to inform you that',
        'In accordance with the regulations',
    ]
    print("\nCapturing 'formal'...")
    for text in formal_examples:
        surgeon.capture(layer=L, text=text, concept_name='formal')

    print(f"\nConcepts: {surgeon.list_concepts()}")

    # --- Blend ---
    blend = surgeon.blend('polite', 'formal', 'professional', ratio=0.6)
    print(f"\nBlend: {blend['ratio']}")

    # --- Inject ---
    print("\nInjecting 'professional'...")
    result = surgeon.inject_concept(layer=L, concept_name='professional', strength=0.1)
    print(f"  Before: {result['existed_before']} -> After: {result['exists_after']}")

    # --- Apply + benchmark ---
    surgeon.apply(layer=L)
    bench = surgeon.benchmark()
    print(f"\nPerplexity: {bench['perplexity']:.2f}")

    # --- Export ---
    surgeon.export_concept('professional', 'professional.concept')
    print("\nExported to professional.concept")

    # --- Restore ---
    surgeon.restore_all()
    print("Restored all layers.")

if __name__ == '__main__':
    main()
