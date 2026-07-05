# Strict Advanced OOF Summary

- finished_at: `2026-07-05 13:51:08`
- Macro-F1: `0.710559`
- accuracy: `0.711229`
- NLL: `0.832679`
- rows: `70000`
- smoke_rows: `0`

## Fold Metrics

| Fold | Rows | Macro-F1 | Accuracy | NLL |
|---:|---:|---:|---:|---:|
| `0` | `13898` | `0.710527` | `0.715427` | `0.826936` |
| `1` | `14078` | `0.713344` | `0.711820` | `0.844428` |
| `2` | `14033` | `0.708044` | `0.705979` | `0.838471` |
| `3` | `13925` | `0.708957` | `0.712101` | `0.812399` |
| `4` | `14066` | `0.711401` | `0.710863` | `0.840895` |

## Decision Note

- This cache is strict: each row is predicted by an advanced router trained without that row's fold.
- Use this directory with `scripts/run_distill_step2.py --advanced-oof-dir ...` for leak-safe distill validation.
