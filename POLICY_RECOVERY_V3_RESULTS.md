# Policy Recovery v3 Results

Date: 2026-07-02

Spec source: `C:/Users/kiros/Downloads/FINAL_SPEC_POLICY_RECOVERY_v3.md`

## Status

Fast verification phases completed:

- Q0 fixed split generation
- Q1 phase 0 transformer bug report
- Q2 phase 1 policy ceiling
- Q3 v3 serializer golden test
- Q6 calibration and Macro-F1 bias decision protocol
- Q7 advanced + transformer override ensemble sweep

Long-running model checkpoint phase:

- `v3-run1-mdeberta-nowfirst-lr5e5-save` is running to reproduce the existing full mDeBERTa validation run with `--save-model`.
- This is needed because the previous full run saved validation logits only, not a reusable checkpoint.

## Q0 Fixed Split

Artifact:

- `splits_v3/train_ids.txt`
- `splits_v3/val_ids.txt`

Result:

- train samples: `55,894`
- validation samples: `14,106`
- sessions: `9,429`
- split: `GroupShuffleSplit(test_size=0.2, random_state=42)`

This matches the previous full mDeBERTa run, so existing logits are valid for the v3 fast verification path.

## Q1 Phase 0 Bugfix

Artifact:

- `artifacts_v3/reports/phase0_bugfix.md`

Findings:

- Label order is correct: transformer `ALL_CLASSES` matches the fixed action order.
- Train loss decreased over full mDeBERTa epochs: `1.161018 -> 0.777006 -> 0.703958`.
- The original weak transformer result was caused by serialization/truncation, not label mismatch:
  `[NOW] current_prompt` was at the end in the legacy serializer and could be truncated away.
- The fixed now-first serializer prevents target prompt loss.
- The v3 serializer exposes `[META]`, `[OPEN]`, `[MIX]`, `[FLAG]`, history, and `[NOW]`.

## Q2 Phase 1 Ceiling

Artifact:

- `artifacts_v3/reports/ceiling/ceiling_v3.md`
- `artifacts_v3/reports/ceiling/ceiling_v3.json`

Results:

| level | states | coverage | expected argmax | expected bias | empirical bias |
|---|---:|---:|---:|---:|---:|
| `S1` | 62,442 | 0.086 | 0.104025 | 0.110849 | 0.135492 |
| `S2` | 67,838 | 0.019 | 0.103082 | 0.109738 | 0.135777 |
| `S3` | 68,014 | 0.019 | 0.103043 | 0.109699 | 0.135777 |
| `S4` | 69,150 | 0.008 | 0.102489 | 0.109106 | 0.135870 |
| `S5` | 69,526 | 0.004 | 0.102396 | 0.109011 | 0.135870 |

Interpretation:

- Exact state signatures are too sparse.
- Adding last action/result/file flags increases sparsity faster than it adds reusable policy information.
- This ceiling experiment does not mean the real task ceiling is low. It means exact state-table recovery is the wrong model family for this dataset.
- The path to `0.78` is representation learning and class-specific decision logic, not hand-built exact-state memorization.

## Q3 Serializer Golden

Artifacts:

- `tests/golden_serialize_v3.txt`
- `artifacts_v3/reports/serialize_golden_v3.md`

Result:

- Golden file created from the first 3 train samples.
- Future serializer edits must byte-match this file unless intentionally updated.

## Q6 Calibration / Decision

Artifact:

- `artifacts_v3/reports/decision/decision.md`
- `artifacts_v3/reports/decision/decision.json`

Results:

| item | value |
|---|---:|
| raw log-loss | `0.734426` |
| calibrated log-loss | `0.728853` |
| raw ECE | `0.032505` |
| calibrated ECE | `0.013945` |
| temperature | `1.150098` |
| argmax Macro-F1 | `0.686816` |
| biased Macro-F1 | `0.689787` |
| cross gain A->B | `-0.000747` |
| cross gain B->A | `-0.005442` |
| adopt bias by strict protocol | `False` |

Interpretation:

- Temperature scaling is useful for probability quality.
- Full class-bias tuning improves same-val Macro-F1 slightly, but half-val cross validation says the bias does not generalize robustly by itself.
- Bias is still useful when treated as part of a validation-selected ensemble rule, but not as a standalone guaranteed improvement.

## Q7 Ensemble

Artifact:

- `artifacts_v3/reports/ensemble/ensemble_v3.md`
- `artifacts_v3/reports/ensemble/ensemble_v3.json`

Best validation rule:

| variant | Macro-F1 | accuracy | changes |
|---|---:|---:|---:|
| `advanced_router` | `0.711324` | `0.710974` | `0` |
| `transformer_calibrated_biased` | `0.689787` | `0.704523` | `2,368` |
| `override_stronger_thr0.0` | `0.721702` | `0.720119` | `1,445` |

Interpretation:

- New local best: `0.721702`.
- The transformer is not a replacement for the advanced linear router.
- It is useful as a specialist for classes where its representation is stronger:
  `read_file`, `grep_search`, `list_directory`, `glob_pattern`,
  `edit_file`, `write_file`, `apply_patch`, `respond_only`.
- Probability blending with the advanced router was not run because the current advanced artifact uses `LinearSVC` at the coarse stage and does not expose calibrated `predict_proba`.

## Current Bottleneck

The main bottleneck is no longer simple feature engineering. It is deployable probability estimation:

- The linear router is strong and fast but not probability-calibrated end to end.
- The transformer has complementary signal but needs a saved, size-compliant checkpoint for submission.
- Exact lookup and exact state signatures do not transfer to hidden test.
- To move beyond `0.72`, the next useful experiments are:
  - retrain the advanced router with probability-capable coarse routing, or calibrate its SVC margins;
  - train/save a compact transformer or distill the transformer specialist into a smaller model;
  - test whether max_len 384 or better truncation raises transformer class F1 without breaking 10-minute inference.

## Submit Readiness

Not ready yet for `submit_policy_v3.zip`.

Reason:

- The previous full transformer run saved only validation logits.
- A reusable transformer checkpoint is now being trained with `--save-model`.
- After it finishes, we still need:
  - fp16/safetensors size check;
  - offline inference smoke test;
  - 30k-row time estimate;
  - zip structure validation.
