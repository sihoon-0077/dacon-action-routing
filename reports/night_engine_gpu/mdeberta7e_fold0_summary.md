# mDeBERTa 384 7Epoch Fold0 Gate

- timestamp: `2026-07-06`
- run: `mdeberta384_v2_384_7e_gate`
- script: `pipeline_v4/train_fold.py`
- config base: `pipeline_v4/configs/mdeberta_v2_384.yaml`
- fold: `0`
- serializer: `v2`
- max_len: `384`
- epochs: `7`
- backbone: `microsoft/mdeberta-v3-base`

## Result

| epoch | train_loss | NLL | Macro-F1 | Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 1.522657 | 0.858805 | 0.611004 | 0.655346 |
| 2 | 0.961757 | 0.756317 | 0.680056 | 0.700460 |
| 3 | 0.880817 | 0.724476 | 0.698270 | 0.714132 |
| 4 | 0.840341 | 0.691290 | 0.711389 | 0.726507 |
| 5 | 0.800039 | 0.676428 | 0.723628 | 0.736941 |
| 6 | 0.771017 | 0.677653 | 0.722586 | 0.736005 |
| 7 | 0.756123 | 0.676475 | 0.725085 | 0.738308 |

## Comparison

- previous fold0 reference: `mdeberta384_v2_384_5e` best Macro-F1 `0.717801`.
- epoch5 delta vs previous fold0: `+0.005827`.
- epoch7 delta vs previous fold0: `+0.007284`.

## Decision

- Undertraining hypothesis passes.
- Continuing beyond 5 epochs improves fold0 Macro-F1 materially.
- However, the current training script selected and saved checkpoints by minimum NLL. The saved checkpoint is epoch5, while the best Macro-F1 was epoch7.
- Before building a submit candidate from this path, patch/save a final or macro-best checkpoint.

