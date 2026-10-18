**0.12.0 as shipped**

| edit | strength | n | target reached | concept cues changed | concept decisions changed | concepts kept | 0.12 gate accepts | new guard accepts (all basins / concepts) |
|---|---|---|---|---|---|---|---|---|
| inject (seed) | 0.1 | 20 | 0% | 4% | 19% | 97% | 15% | 0% / 65% |
| inject (seed) | 0.5 | 20 | 0% | 37% | 51% | 66% | 0% | 0% / 0% |
| inject (seed) | 1.0 | 20 | 0% | 58% | 66% | 43% | 0% | 0% / 0% |
| inject (pattern) | 0.1 | 20 | 0% | 7% | 21% | 94% | 15% | 0% / 55% |
| inject (pattern) | 0.5 | 20 | 0% | 39% | 51% | 62% | 0% | 0% / 0% |
| inject (pattern) | 1.0 | 20 | 0% | 61% | 66% | 41% | 5% | 0% / 0% |
| remove | 0.1 | 20 | 45% | 1% | 6% | 99% | 0% | 15% / 95% |
| remove | 0.5 | 20 | 100% | 2% | 12% | 99% | 0% | 10% / 90% |
| remove | 1.0 | 20 | 95% | 5% | 21% | 95% | 0% | 10% / 80% |
| move | 0.1 | 20 | 0% | 4% | 17% | 99% | 10% | 0% / 70% |
| move | 0.5 | 20 | 0% | 34% | 48% | 68% | 0% | 0% / 0% |
| move | 1.0 | 20 | 65% | 58% | 68% | 42% | 0% | 0% / 0% |

**Fixed (projected, default)**

| edit | strength | n | target reached | concept cues changed | concept decisions changed | concepts kept | 0.12 gate accepts | new guard accepts (all basins / concepts) |
|---|---|---|---|---|---|---|---|---|
| inject (seed) | 0.1 | 20 | 0% | 0% | 1% | 98% | 100% | 100% / 100% |
| inject (seed) | 0.5 | 20 | 100% | 1% | 16% | 98% | 85% | 5% / 95% |
| inject (seed) | 1.0 | 20 | 100% | 11% | 38% | 88% | 65% | 0% / 0% |
| inject (seed) | auto (≈0.43) | 20 | 100% | 0% | 12% | 98% | 95% | 15% / 95% |
| inject (pattern) | 0.1 | 20 | 0% | 0% | 1% | 98% | 95% | 100% / 100% |
| inject (pattern) | 0.5 | 20 | 100% | 1% | 17% | 97% | 85% | 10% / 95% |
| inject (pattern) | 1.0 | 20 | 100% | 11% | 37% | 88% | 65% | 0% / 0% |
| inject (pattern) | auto (≈0.43) | 20 | 100% | 0% | 12% | 99% | 95% | 15% / 95% |
| remove | 0.1 | 20 | 25% | 0% | 4% | 99% | 50% | 80% / 95% |
| remove | 0.5 | 20 | 100% | 3% | 12% | 97% | 5% | 20% / 95% |
| remove | 1.0 | 20 | 90% | 5% | 19% | 94% | 20% | 5% / 85% |
| remove | auto (≈0.29) | 20 | 100% | 1% | 7% | 98% | 10% | 50% / 90% |
| move | 0.1 | 20 | 0% | 0% | 4% | 98% | 65% | 80% / 95% |
| move | 0.5 | 20 | 95% | 3% | 20% | 97% | 15% | 15% / 95% |
| move | 1.0 | 20 | 100% | 29% | 51% | 68% | 0% | 0% / 0% |
| move | auto (≈0.48) | 20 | 95% | 1% | 13% | 98% | 20% | 35% / 100% |

shipped inject reinforced: Counter({'existing': 45, 'mirror': 42, 'spurious': 33}) hits 0 / 120
shipped inject accepted by 0.12 gate: 7 max collateral among accepted 0.5244444608688354 max decisions 0.46861404180526733
shipped remove accepted: 0 / 60
