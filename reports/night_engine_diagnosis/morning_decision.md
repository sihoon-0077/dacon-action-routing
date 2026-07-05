# Morning Engine Diagnosis Report

## 1. Backbone Gate

| model | fold_count | avg_best_macro | decision |
|---|---:|---:|---|
| `mdeberta384_v2_384_5e` | `5` | `0.718031` | `baseline_capacity_around_0.718_oof` |

## 2. Training Curve Audit

| fold | best_epoch | best_f1 | ep4_to_ep5_delta | nll_still_decreasing | decision |
|---|---:|---:|---:|---|---|
| `0` | `5` | `0.717801` | `0.000522` | `True` | `near_plateau` |
| `1` | `5` | `0.716547` | `0.001055` | `True` | `undertraining_possible` |
| `2` | `4` | `0.715503` | `0.000830` | `False` | `near_plateau` |
| `3` | `5` | `0.726254` | `0.013106` | `True` | `undertraining_possible` |
| `4` | `5` | `0.714051` | `0.002359` | `True` | `undertraining_possible` |

## 3. Large Preflight

| model | cuda | tokenizer_cached | zip_feasible | role | notes |
|---|---|---|---|---|---|
| `FacebookAI/xlm-roberta-large` | `True` | `False` | `False` | `preflight_failed_or_not_cached` | `large_not_available_locally:OSError` |

## 4. Submission Defense

| zip | smoke | runtime_sec | size_mb | decision |
|---|---|---:|---:|---|
| `cand_distill.zip` | `True` | `9.23` | `420.260` | `keep_public_baseline` |

## 5. Final Morning Decision

- Existing mDeBERTa curves average around `0.718031` OOF; this explains the public `0.71~0.72` shelf.
- Folds with NLL still decreasing at epoch 5: `4/5`.
- Large-model submit path is not proven on this machine because preflight did not find a cached runnable large model/GPU path.
- Keep `cand_distill.zip` as the defense line while v2.3/teacher experiments are validated.
