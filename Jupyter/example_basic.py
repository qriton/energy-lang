"""
example_basic.py — Basic basin surgery with qriton-hlm.

Works without a checkpoint — uses a synthetic W matrix.
Run: python example_basic.py
"""

import torch
from qriton_hlm import BasinSurgeon, compute_energy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Create synthetic Hopfield layer ---
d_model = 128
torch.manual_seed(42)
W = torch.randn(d_model, d_model, device=DEVICE) * 0.01
W = (W + W.T) / 2

surgeon = BasinSurgeon.from_W(W, device=DEVICE)

# --- Survey ---
survey = surgeon.survey(layer=0)
print(f"Basins found: {survey['num_basins']}")
for i in range(min(5, survey['num_basins'])):
    print(f"  B{i} E={survey['energies'][i]:+.4f} pop={survey['populations'][i]}")

# --- Inject ---
result = surgeon.inject(layer=0, seed=42, strength=0.1)
print(f"\nInject: existed={result['existed_before']} -> exists={result['exists_after']}")

# --- Verify ---
v = surgeon.verify(layer=0, seed=42)
print(f"Verify: basin={v['is_basin']} cos={v['cos']:.4f} E={v['energy']:.4f}")

# --- Diff ---
diff = surgeon.diff(layer=0)
if diff:
    print(f"Change: {diff['relative_pct']:.2f}%")

# --- Restore ---
surgeon.restore(layer=0)
print("\nRestored.")
