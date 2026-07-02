# Policy Reconstruction Experiment Results

Date: 2026-07-02

Plan source: `C:/Users/kiros/Downloads/policy_reconstruction_experiment_plan_codex.md`

## Summary

This phase reframed action routing as policy reconstruction:

1. Estimate `P(action | state)`.
2. Check replay/leakage behavior separately.
3. Calibrate probability estimates.
4. Optimize Macro-F1 decision rules.
5. Combine complementary estimators.

The most useful immediate result is not transformer-only replacement. It is a hybrid decision rule:

| Variant | Macro-F1 | Accuracy |
|---|---:|---:|
| advanced router | `0.711324` | `0.710974` |
| full mDeBERTa argmax | `0.686816` | `0.702751` |
| full mDeBERTa + temperature + class bias | `0.689963` | `0.704736` |
| advanced + transformer stronger-class override | `0.719520` | `0.718559` |

## Generated Code

Common utilities:

- `src/constants.py`
- `src/io_utils.py`
- `src/state_features.py`
- `src/metrics.py`
- `src/serialize_policy.py`
- `src/replay_lookup.py`
- `src/calibration.py`
- `src/decision_bias.py`
- `src/transition_prior.py`
- `src/submit_utils.py`

Scripts:

- `scripts/audit_dataset.py`
- `scripts/estimate_policy_ceiling.py`
- `scripts/audit_replay_leak.py`
- `scripts/train_transformer.py`
- `scripts/calibrate_temperature.py`
- `scripts/optimize_class_bias.py`
- `scripts/blend_probabilities.py`

## Dataset Audit

Artifacts:

- `artifacts/reports/dataset_audit_policy_frame.md`
- `artifacts/reports/dataset_audit_policy_frame.json`

Purpose:

- label/group distribution
- history and transition structure
- state-signature ambiguity
- prompt/state entropy checks

Interpretation:

- History and state fields are essential, but exact state signatures do not generalize across session-grouped folds.
- This supports the current direction: learned representations plus decision optimization, not pure memorization.

## Policy Ceiling

Artifacts:

- `artifacts/reports/policy_ceiling/ceiling_summary.md`
- `artifacts/reports/policy_ceiling/ceiling_summary.csv`
- `artifacts/reports/policy_ceiling/classwise_best.csv`

Results:

| Signature | Argmax Macro-F1 | Bias Tuned Macro-F1 |
|---|---:|---:|
| `S0_raw_prompt` | `0.133483` | `0.147650` |
| `S1_template_prompt` | `0.142060` | `0.154695` |
| `S2_tpl_last1` | `0.072289` | `0.090351` |
| `S3_tpl_last2` | `0.048258` | `0.068630` |
| `S4_tpl_last2_result` | `0.045868` | `0.066384` |
| `S5_tpl_last3_result_meta` | `0.021295` | `0.041930` |
| `S6_tpl_last3_result_open_lang` | `0.020312` | `0.041096` |

Interpretation:

- Exact state-signature backoff is not a useful hidden-generalization ceiling.
- The low score is expected because GroupKFold removes session overlap and most detailed signatures become sparse/unseen.
- Use this as an anti-replay signal, not as the true model-capacity ceiling.

## Replay Audit

Artifacts:

- `artifacts/reports/replay_audit/replay_audit.md`
- `artifacts/reports/replay_audit/replay_audit.json`

Session-scoped replay:

- train internal coverage: `60548/70000`
- train internal precision: `1.000000`
- GroupSplit train-to-valid coverage: `0/14106`
- transductive valid coverage: `12215/14106`
- transductive valid precision: `1.000000`
- placeholder test hits: `5/5`

Global prompt replay:

- train internal precision: `0.995432`
- GroupSplit train-to-valid coverage: `1065/14106`
- GroupSplit train-to-valid precision: `0.432864`
- transductive valid precision: `0.995474`

Interpretation:

- Replay is very strong only inside the same session or batch.
- It does not generalize under group-disjoint validation.
- Keep replay variants separate from safe submissions.

## Serialization Audit

Artifacts:

- `artifacts/reports/serialization_preview/serialization_preview.md`
- `artifacts/reports/serialization_preview/token_length_stats.csv`

Key rule:

- `[NOW] current_prompt` must be first.
- Older history can be truncated; the target prompt cannot.

This fixed the earlier transformer bug where `[NOW]` was dropped in many long samples.

## Transformer Probability Estimator

Existing full transformer run reused:

- `reports/transformer/B-full-mdeberta-70k-nowfirst-lr5e5-none-3e`

Results:

| Epoch | Macro-F1 | Accuracy |
|---:|---:|---:|
| 1 | `0.623584` | `0.648873` |
| 2 | `0.676489` | `0.697363` |
| 3 | `0.686816` | `0.702751` |

Class-level interpretation:

- Transformer is competitive or better on `grep_search`, `list_directory`, `glob_pattern`, `edit_file`, `write_file`, `apply_patch`, `respond_only`.
- It is weaker on `run_bash`, `run_tests`, `lint_or_typecheck`, `ask_user`, `plan_task`, `web_search`.

## Calibration

Artifacts:

- `artifacts/calibration/mdeberta_full_nowfirst/calibration_report.md`
- `artifacts/calibration/mdeberta_full_nowfirst/temperature.json`
- `artifacts/calibration/mdeberta_full_nowfirst/calibrated_probs.npy`

Results:

| Metric | Raw | Calibrated |
|---|---:|---:|
| log-loss | `0.734426` | `0.728853` |
| ECE | `0.032505` | `0.013945` |

Temperature:

- `1.150098`

Interpretation:

- Temperature scaling improves probability quality.
- Macro-F1 does not necessarily improve from calibration alone; it enables better downstream decision tuning.

## Class Bias

Artifacts:

- `artifacts/bias/mdeberta_full_nowfirst/bias_tuning_report.md`
- `artifacts/bias/mdeberta_full_nowfirst/class_bias.json`

Results:

| Variant | Macro-F1 | Accuracy |
|---|---:|---:|
| calibrated argmax | `0.686816` | `0.702751` |
| calibrated + class bias | `0.689963` | `0.704736` |

Interpretation:

- Bias tuning helps, but transformer-only still trails advanced router.
- Better use transformer as a complementary expert.

## Blend / Decision Override

Artifacts:

- `artifacts/reports/blend_sweep_mdeberta_full/blend_sweep.md`
- `artifacts/reports/blend_sweep_mdeberta_full/blend_sweep.csv`

Best validation variant:

- `override_transformer_stronger_thr0.0`
- Macro-F1 `0.719520`
- Accuracy `0.718559`
- Changes vs advanced router: `1462`

Interpretation:

- This is currently the best policy-reconstruction decision result.
- It uses advanced router as the main policy and transformer as a specialist where transformer class F1 was stronger.
- Next implementation target: train full-data transformer checkpoint and build a submit variant that reproduces this override rule at inference time.

## Next Steps

1. Train/save a full-data transformer model checkpoint, not just validation logits.
2. Build a submission variant:
   - advanced router prediction
   - transformer prediction/confidence
   - override only for the stronger action set
3. Smoke test inference time on 30k rows.
4. If inference exceeds 10 minutes, distill transformer specialist into a smaller linear/embedding model or restrict transformer to inspect/modify candidates.
