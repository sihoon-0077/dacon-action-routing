# Phase 1 Ceiling v3

- min_support: `3`
- alpha: `1.0`

| level | states | coverage | expected argmax | expected bias | empirical bias |
|---|---:|---:|---:|---:|---:|
| `S1` | 62442 | 0.086 | 0.104025 | 0.110849 | 0.135492 |
| `S2` | 67838 | 0.019 | 0.103082 | 0.109738 | 0.135777 |
| `S3` | 68014 | 0.019 | 0.103043 | 0.109699 | 0.135777 |
| `S4` | 69150 | 0.008 | 0.102489 | 0.109106 | 0.135870 |
| `S5` | 69526 | 0.004 | 0.102396 | 0.109011 | 0.135870 |

## Interpretation

- Best expected Macro-F1: `0.110849` at `S1`.
- This is an optimistic policy-recovery ceiling because the distribution table is estimated on the full train set.
