# OOF Policy-Recovery v4 Progress

Date: 2026-07-02

Spec source: `C:/Users/kiros/Downloads/FINAL_SPEC_v4_OOF_PIPELINE.md`

## Implemented

- `pipeline_v4/common/constants.py`
- `pipeline_v4/common/data_io.py`
- `pipeline_v4/common/make_folds.py`
- `pipeline_v4/serialize.py`
- `pipeline_v4/tests/test_serialize.py`
- `pipeline_v4/train_fold.py`
- `pipeline_v4/calibrate.py`
- `pipeline_v4/build_oof.py`
- `pipeline_v4/optimize_bias.py`
- configs:
  - `pipeline_v4/configs/mdeberta_a.yaml`
  - `pipeline_v4/configs/mdeberta_a_local8gb.yaml`
  - `pipeline_v4/configs/xlmr_a.yaml`

## Q02 Fold Generation

Generated:

- `pipeline_v4/folds/fold_assignments.csv`
- `pipeline_v4/folds/prior.json`

Fold counts:

| fold | samples |
|---:|---:|
| 0 | 13,898 |
| 1 | 14,078 |
| 2 | 14,033 |
| 3 | 13,925 |
| 4 | 14,066 |

Validation:

- session leakage assert passed
- max fold-count deviation: `0.007286`
- all 14 classes are present in every fold

## Q03 Serialization Golden

Generated and verified:

- `pipeline_v4/tests/golden_serialize.txt`

Manual check:

- `[NOW]` is first.
- fixed fields `[META]`, `[OPEN]`, `[MIX]`, `[FLAG]` are always present.
- history order is `[H6]` oldest to `[H1]` newest.

## Q04 Fold0 Gate Training

Run:

- config: `pipeline_v4/configs/mdeberta_a_local8gb.yaml`
- effective batch: `2 * 16 = 32`
- local GPU: RTX 4060 Ti 8GB

First attempt:

- failed with `ValueError: Attempting to unscale FP16 gradients`
- cause: local HF config loaded mDeBERTa weights in fp16
- fix: force trainable model parameters to fp32 before enabling autocast

Second attempt:

- log dir: `pipeline_v4/artifacts/reports/mdeberta_a/fold_0_run2`
- status at start: running
- observed GPU: about `6.9GB` VRAM and active utilization

Gate criteria to evaluate after epoch 1 / best epoch:

- epoch 1 fold-val Macro-F1 >= `0.60`
- NLL should improve until saturation
- best epoch argmax Macro-F1 >= `0.70` before continuing fold1-4

## Notes

- `mdeberta_a_local8gb.yaml` only changes micro-batch sizing for the local 8GB GPU. It keeps the v4 effective batch size at 32.
- Heavy outputs are ignored:
  - `pipeline_v4/artifacts/models/`
  - `pipeline_v4/artifacts/oof/`
  - logs and `pid.txt`
