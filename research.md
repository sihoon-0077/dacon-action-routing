# Dacon Action Decision Research Log

## Current Bests

| Track | Best Macro-F1 | Experiment | Note |
|---|---:|---|---|
| Baseline | 0.4369 | current_prompt TF-IDF + LogReg | Uses only current prompt. |
| Linear autoresearch | 0.6332 | compact_union_lr_c2p0 | Compact action history is the biggest jump. |
| Embedding concat | 0.6392 | tfidf_emb_num_lsvc_c1_scale0p5 | MiniLM alone is weak, but helps as an auxiliary signal. |
| Model zoo ensemble | 0.6585 | 10_score_voting_ensemble_2 | Heterogeneous score voting improves over single models. |

## Confirmed Findings

- The task behaves more like agent workflow state transition than plain intent classification.
- `history` is the strongest feature family after `current_prompt`.
- Compact action sequence tokens beat dumping full history text.
- Sparse TF-IDF with linear models is still the strongest single-model baseline.
- RandomForest and ExtraTrees on SVD+dense features are weak as standalone models.
- LightGBM is useful mostly as an ensemble diversity source.
- MiniLM embeddings are weak alone but add useful complementary signal when combined with TF-IDF and numeric/session features.

## Current Hypothesis

The next lift should come from explicitly modeling:

1. `current_prompt`
2. recent `history` user content
3. recent `assistant_action.name` sequence
4. recent `assistant_action.result_summary`
5. recent `assistant_action.args`
6. `workspace.open_files`
7. `workspace.last_ci_status`
8. `turn_index`
9. `language_pref`
10. `workspace.language_mix`

In particular, add transition priors such as `P(next_action | last_action)` and `P(next_action | last2_actions)` on top of the text model scores.


## State Routing Experiments
- Finished: 2026-07-01 22:58:39
- Best Macro-F1: `0.624782` via `state_v3_union_lsvc_c0p7`
- Key idea: richer state serialization + transition priors + group specialists.

Top results:
- `0.624782` `state_v3_union_lsvc_c0p7`
- `0.621464` `state_v2_union_lsvc_c0p7`
- `0.614381` `state_v1_union_lsvc_c0p7`
- `0.591774` `state_v2_logreg_transition_a1_0.05_a2_0.0`
- `0.591305` `state_v3_union_logreg_c2`
- `0.590993` `state_v3_logreg_transition_a1_0.05_a2_0.0`
- `0.590914` `state_v3_logreg_transition_a1_0.05_a2_0.05`
- `0.590769` `state_v2_union_sgd_log`
- `0.590543` `state_v2_union_logreg_c2`
- `0.589976` `state_v2_logreg_group_specialist_w0.05`

## Compact State Score Experiments
- Finished: 2026-07-01 23:18:38
- Best Macro-F1: `0.666351` via `compact_flags_lr_combo_a1_0.06_gw_0.08_rw_0.02`
- Key idea: keep compact text; add small transition/group/rule score adjustments.
- Submission artifact trained on full data: `model/compact_flags_router.pkl`.
- Submission zip: `submit_compact_flags_router.zip`.
- Implementation note: `predict_proba` columns must be aligned from `model.classes_` to `ALL_CLASSES`; otherwise class scores shift to wrong labels.

Top results:
- `0.666351` `compact_flags_lr_combo_a1_0.06_gw_0.08_rw_0.02`
- `0.666205` `compact_flags_lr_combo_a1_0.08_gw_0.08_rw_0.02`
- `0.665449` `compact_flags_lr_combo_a1_0.04_gw_0.08_rw_0.02`
- `0.665082` `compact_flags_lr_combo_a1_0.06_gw_0.08_rw_0.04`
- `0.664754` `compact_flags_lr_combo_a1_0.08_gw_0.08_rw_0.04`
- `0.664289` `compact_flags_lr_combo_a1_0.04_gw_0.08_rw_0.04`
- `0.663991` `compact_flags_lr_prior_a1_0.04_a2_0.06`
- `0.663654` `compact_flags_lr_prior_a1_0.08_a2_0.06`
- `0.663611` `compact_flags_lr_prior_a1_0.1_a2_0.06`
- `0.663192` `compact_flags_lr_prior_a1_0.06_a2_0.06`

## Margin Coarse/Fine Routing Experiments
- Finished: 2026-07-01 23:51:30
- Best Macro-F1: `0.698975` via `stratified_fine_oracle_group`
- Key idea: 4-way coarse group model, margin threshold, fine group specialists, flat fallback.

Top results:
- `0.698975` `stratified_fine_oracle_group`
- `0.694036` `stratified_fine_by_coarse_svc_all`
- `0.694036` `stratified_svc_hard_gating_t0.0`
- `0.693107` `stratified_svc_hard_gating_t0.1`
- `0.691912` `stratified_svc_hard_gating_t0.2`
- `0.690695` `stratified_svc_hard_gating_t0.3`
- `0.689422` `stratified_svc_hard_gating_t0.4`
- `0.687859` `stratified_svc_hard_gating_t0.5`
- `0.687409` `stratified_svc_safety_check_t0.0`
- `0.686456` `stratified_svc_safety_check_t0.1`

## Margin Coarse/Fine Routing Experiments
- Finished: 2026-07-02 00:04:35
- Best Macro-F1: `0.692082` via `group_shuffle_fine_oracle_group`
- Key idea: 4-way coarse group model, margin threshold, fine group specialists, flat fallback.

Top results:
- `0.692082` `group_shuffle_fine_oracle_group`
- `0.686921` `group_shuffle_fine_by_coarse_svc_all`
- `0.686921` `group_shuffle_svc_hard_gating_t0.0`
- `0.685972` `group_shuffle_svc_hard_gating_t0.1`
- `0.685186` `group_shuffle_svc_hard_gating_t0.2`
- `0.684337` `group_shuffle_svc_safety_check_t0.0`
- `0.684211` `group_shuffle_svc_hard_gating_t0.3`
- `0.683561` `group_shuffle_svc_hard_gating_t0.4`
- `0.683398` `group_shuffle_svc_safety_check_t0.1`
- `0.682696` `group_shuffle_svc_safety_check_t0.2`

## Routing Margin Submission
- Full-data artifact: `model/routing_margin_router.pkl`
- Submission zip: `submit_routing_margin_router.zip`
- Zip size: about 16.5 MB.
- Script path: `script.py`; it now loads `routing_margin_router.pkl` first, then falls back to `compact_flags_router.pkl`, then older models.
- Validation basis: deployable `coarse_svc -> fine_logreg` scored `0.694036` on stratified split and `0.686921` on GroupShuffleSplit.
- Main conclusion: coarse group routing is highly reliable (`~99.2%` group accuracy), so hard routing with threshold `0.0` beat the document's more conservative margin `0.4` rule in both splits.
- Remaining gap: oracle group routing reaches `0.698975` stratified and `0.692082` group split, so the next bottleneck is mostly within-group fine action confusion, not coarse group selection.

## Next Steps Margin04 Final Update
- Finished: 2026-07-02 00:44:00
- Tested `codex_next_steps_margin04.md` ideas: doc-style rule serializer, full LinearSVC flat/coarse/fine routing, compact+doc hybrid text, and margin threshold sweep.
- Result: doc-style serializer alone underperformed (`doc_rule8` deployable GroupShuffleSplit `0.659354`; `doc_rule12` `0.647508`).
- Result: fine LinearSVC underperformed fine LogReg on compact flags (`0.682792` vs LogReg `0.687763` GroupShuffleSplit).
- Useful lift: keep compact flags, keep fine LogReg, raise coarse LinearSVC `C` from `0.7` to `2.0`.
- New deployable validation: `0.695150` stratified, `0.687763` GroupShuffleSplit.
- Updated full-data artifact: `model/routing_margin_router.pkl`.
- Updated submission zip: `submit_routing_margin_router.zip`.
- Final routing rule remains threshold `0.0` hard routing: always choose coarse group, then run that group's fine action model. Margin `0.4` was consistently lower.

## Next Steps Margin04 Experiments
- Finished: 2026-07-02 00:29:09
- Run dir: `next_steps_margin04_runs`
- Best Macro-F1: `0.687466` via `compact_flags_svcC0.7_svc_fine_oracle_group`
- Key idea: doc-style rule hints, LinearSVC flat/coarse/fine routing, GroupShuffleSplit-first validation.

Top results:
- `0.687466` `compact_flags_svcC0.7_svc_fine_oracle_group`
- `0.682792` `compact_flags_svcC0.7_svc_fine_by_coarse_all`
- `0.682792` `compact_flags_svcC0.7_svc_hard_t0.0`
- `0.681988` `compact_flags_svcC0.7_svc_hard_t0.1`
- `0.681480` `compact_flags_svcC0.7_svc_hard_t0.2`
- `0.680774` `compact_flags_svcC0.7_svc_hard_t0.3`
- `0.680364` `compact_flags_svcC0.7_svc_hard_t0.4`
- `0.679758` `compact_flags_svcC0.7_svc_hard_t0.5`
- `0.678059` `compact_flags_svcC0.7_svc_hard_t0.6`
- `0.677177` `doc_rule8_svcC0.7_svc_fine_oracle_group`

## Next Steps Margin04 Experiments
- Finished: 2026-07-02 00:37:40
- Run dir: `next_steps_margin04_logreg_runs`
- Best Macro-F1: `0.697359` via `compact_plus_doc8_svcC2.0_logreg_fine_oracle_group`
- Key idea: doc-style rule hints, LinearSVC flat/coarse/fine routing, GroupShuffleSplit-first validation.

Top results:
- `0.697359` `compact_plus_doc8_svcC2.0_logreg_fine_oracle_group`
- `0.692082` `compact_flags_svcC2.0_logreg_fine_oracle_group`
- `0.687763` `compact_flags_svcC2.0_logreg_fine_by_coarse_all`
- `0.687763` `compact_flags_svcC2.0_logreg_hard_t0.0`
- `0.687503` `compact_flags_svcC2.0_logreg_hard_t0.1`
- `0.687436` `compact_flags_svcC2.0_logreg_hard_t0.3`
- `0.687308` `compact_flags_svcC2.0_logreg_hard_t0.2`
- `0.686556` `compact_flags_svcC2.0_logreg_hard_t0.4`
- `0.686315` `compact_plus_doc8_svcC2.0_logreg_fine_by_coarse_all`
- `0.686315` `compact_plus_doc8_svcC2.0_logreg_hard_t0.0`

## Next Steps Margin04 Experiments
- Finished: 2026-07-02 00:40:49
- Run dir: `next_steps_margin04_strat_c2_runs`
- Best Macro-F1: `0.699362` via `compact_flags_svcC2.0_logreg_fine_oracle_group`
- Key idea: doc-style rule hints, LinearSVC flat/coarse/fine routing, GroupShuffleSplit-first validation.

Top results:
- `0.699362` `compact_flags_svcC2.0_logreg_fine_oracle_group`
- `0.695150` `compact_flags_svcC2.0_logreg_fine_by_coarse_all`
- `0.695150` `compact_flags_svcC2.0_logreg_hard_t0.0`
- `0.694994` `compact_flags_svcC2.0_logreg_hard_t0.1`
- `0.694531` `compact_flags_svcC2.0_logreg_hard_t0.2`
- `0.694425` `compact_flags_svcC2.0_logreg_hard_t0.3`
- `0.693461` `compact_flags_svcC2.0_logreg_hard_t0.4`
- `0.693005` `compact_flags_svcC2.0_logreg_hard_t0.5`
- `0.692006` `compact_flags_svcC2.0_logreg_hard_t0.6`
- `0.690209` `compact_flags_svcC2.0_logreg_hard_t0.8`

## Advanced Action Routing Experiments
- Finished: 2026-07-02 03:39:15
- Run dir: `reports\exp_advanced_action_routing`
- Split: `group`
- Baseline reproduced Macro-F1: `0.687763`
- Best Macro-F1: `0.711324` via `phase6_pair_resolver_t0.08`
- Tested: group-specific vectorizers/serializers, fine-margin flat fallback, transition prior, pairwise resolvers, memory lookup.

Top results:
- `0.711324` `phase6_pair_resolver_t0.08`
- `0.711258` `phase6_pair_resolver_t0.1`
- `0.710874` `phase7_memory_prompt_m3_r0.85_b0.15`
- `0.710874` `phase7_memory_prompt_m3_r0.9_b0.15`
- `0.710874` `phase7_memory_prompt_m3_r0.95_b0.15`
- `0.710826` `phase7_memory_prompt_m3_r0.8_b0.15`
- `0.710759` `phase6_pair_resolver_t0.05`
- `0.710709` `phase7_memory_prompt_m3_r0.8_b0.1`
- `0.710709` `phase7_memory_prompt_m3_r0.85_b0.1`
- `0.710709` `phase7_memory_prompt_m3_r0.9_b0.1`

## NEXT_EXPERIMENT_v2 Group Ceiling
- Finished: 2026-07-02 03:54:55
- Run dir: `reports\exp_v2_group_ceiling`
- Isolated macro upper estimate: `0.706192`
- `inspect` best `0.553876` via `inspect_specialized_x2_word`
- `modify` best `0.916111` via `modify_specialized_word_char`
- `execute` best `0.702804` via `execute_specialized_word`
- `communicate` best `0.703610` via `communicate_specialized_x2_word_char_num`

## Advanced Router Submission
- Finished: 2026-07-02 04:04:20
- Full-data artifact: `model/advanced_router.pkl`
- Submission zip: `submit_advanced_router.zip`
- Zip size: about 29.2 MB.
- Validation basis: GroupShuffle Macro-F1 `0.711324` via `phase6_pair_resolver_t0.08`.
- Model structure: compact coarse LinearSVC, group-specific specialized_x2 fine LogReg vectorizers, `last2_action` transition prior (`alpha=0.3`, `smooth=1.0`), pairwise resolvers (`threshold=0.08`).
- `script.py` now loads `advanced_router.pkl` first, then falls back to `routing_margin_router.pkl`, `compact_flags_router.pkl`, and older models.
- v2 ceiling interpretation: inspect remains the largest structural bottleneck (`0.553876` isolated best), while modify is mostly solved and execute/communicate plateau near `0.70`.
- Practical conclusion: current lightweight linear-feature family likely plateaus around low `0.71`; reaching `0.78` likely needs OOF stacking, stronger representation learning/distillation, or a much better inspect/communicate specialist.

## Exact History Lookup Experiment
- Finished: 2026-07-02
- Script: `lookup_memory_experiment.py`
- Output: `reports/lookup_memory_experiment.json`
- Question: if a `current_prompt` exactly appears earlier as a user turn followed by an `assistant_action`, can we override the model prediction with that next action?
- Base validation: advanced router Macro-F1 `0.711324`.

Key results:
- `split_train_history_only__raw__unique`: Macro-F1 `0.690282`, delta `-0.021042`, coverage `1065/14000`, covered accuracy `0.433`.
- `split_train_history_only__raw__majority`: Macro-F1 `0.681010`, delta `-0.030314`, coverage `1407/14000`, covered accuracy `0.416`.
- `split_train_plus_val_history_transductive__raw__last`: Macro-F1 `0.960086`, delta `+0.248762`, coverage `12341/14000`, covered accuracy `0.982`.
- `split_train_plus_val_history_transductive__raw__unique`: Macro-F1 `0.942706`, delta `+0.231383`, coverage `11268/14000`, covered accuracy `0.995`.
- Public sample `test.jsonl` probe: all 5/5 samples hit the train-history lookup.

Interpretation:
- The useful signal is not ordinary train-memory generalization. Train-history-only lookup hurts validation.
- The huge gain appears when validation/test histories are scanned as a batch. This suggests repeated prompt/action traces inside the distributed session data.
- If the rules permit transductive use of the full evaluation `test.jsonl` features, a batch-level exact lookup override could score much higher. If not, keep it out of the official submission.

## Leak And Transformer v3 Experiments
- Finished: 2026-07-02
- Plan: `C:\Users\kiros\Downloads\EXPERIMENT_PLAN_LEAK_AND_TRANSFORMER.md`
- Report: `LEAK_AND_TRANSFORMER_RESULTS.md`

Track A: session-scoped exact lookup:
- Script: `session_lookup_experiment.py`
- Output: `reports/session_lookup_validation.json`
- Base advanced router Macro-F1: `0.711324`.
- `A2-1_val_self`: Macro-F1 `0.973726`, delta `+0.262402`, coverage `12217/14000` (`0.866`), covered accuracy `1.000`.
- `A2-2_train_only`: Macro-F1 `0.711324`, delta `+0.000000`, coverage `0/14000`.
- `A2-3_train_plus_val`: Macro-F1 `0.973726`, delta `+0.262402`.
- Public sample probe: train-history hits `5/5`, train+test-history hits `5/5`.
- Submission artifact: `submit_lookup_probe.zip`, about `51.0 MB`, verified locally. Contents: `script.py`, `requirements.txt`, `model/advanced_router.pkl`, `data/train.jsonl`.
- Interpretation: the strong signal is `(session_id, prompt)` exact matching from later history in the same session. This is a transductive/batch-structure exploit, not ordinary train generalization.

Track B: transformer probe:
- Script: `transformer_action_routing.py`
- Extra dependencies: `requirements-transformer.txt`
- Local GPU: RTX 4060 Ti 8GB.
- mDeBERTa token length report on 70k: mean `306.1`, p50 `314`, p90 `479`, p95 `514`, p99 `580`, max `703`; `34180` samples exceed 320 tokens.
- `B-smoke-mdeberta-1k-stable`: Macro-F1 `0.020147`.
- `B-smoke-xlm-1k`: Macro-F1 `0.020147`.
- `B-probe-mdeberta-10k`: Macro-F1 `0.090248` after 1 epoch, balanced loss, lr `2e-5`.
- `B-probe-mdeberta-10k-lr5e5-none-3e`: Macro-F1 `0.317084` after 3 epochs, no class weight, lr `5e-5`.
- Important fix: force `model_dtype=float32`; otherwise mDeBERTa loaded as FP16 and produced NaN loss.
- Interpretation: transformer code now runs, but the 10k probe is not yet competitive with the `0.711` linear router. Full 70k training is estimated at roughly 90-100 minutes locally with `max_len=320`, batch size 2.

Transformer bug diagnosis follow-up:
- User correctly flagged that Macro-F1 `0.317084` is suspicious because it underperforms a simple TF-IDF baseline.
- Label order check: `LABEL_TO_ID` and evaluation labels match `ALL_CLASSES`; no label-index mismatch found.
- Same 10k split prompt-only TF-IDF+LogReg Macro-F1: `0.384628`.
- Root cause found: the original transformer serializer placed `[NOW] current_prompt` at the end, while tokenizer truncation keeps the beginning.
- On the 10k diagnostic subset, `4937/10000` examples exceeded 320 tokens and `4393/10000` lost `[NOW]` after truncation.
- Fix: move `[NOW]` to the front and serialize recent history first (`layout=now_first`).
- After fix, `[NOW]` missing after truncation: `0/10000`.
- Re-run `B-probe-mdeberta-10k-nowfirst-lr5e5-none-3e`: best Macro-F1 `0.490004` at epoch 2, accuracy `0.587851`.
- Fixed epoch curve: epoch 1 `0.390567`, epoch 2 `0.490004`, epoch 3 `0.483550`.
- Remaining weak classes in 10k probe: `glob_pattern` F1 `0.000`, `list_directory` `0.067`, `web_search` `0.000`, `lint_or_typecheck` `0.118`.

Full transformer run:
- Run: `B-full-mdeberta-70k-nowfirst-lr5e5-none-3e`.
- Settings: full 70k, GroupShuffleSplit seed 42, `microsoft/mdeberta-v3-base`, `layout=now_first`, `max_len=320`, `epochs=3`, `lr=5e-5`, `loss_weight=none`, `batch_size=2`, `grad_accum=16`, `amp=fp16`, `model_dtype=float32`.
- Runtime: `5775s` (`~96.3 min`) on local RTX 4060 Ti 8GB.
- Best: epoch 3 Macro-F1 `0.686816`, accuracy `0.702751`.
- Epoch curve: epoch 1 `0.623584`, epoch 2 `0.676489`, epoch 3 `0.686816`.
- Stronger than advanced router on class F1 for `grep_search`, `list_directory`, `glob_pattern`, `edit_file`, `write_file`, `apply_patch`, `respond_only`; weaker on execute/communicate classes.
- Quick validation hybrid: advanced router `0.711324`; transformer alone `0.686816`; action-set override using transformer for its stronger classes reached Macro-F1 `0.719520`.
- Interpretation: transformer is useful as complementary specialist/logit feature, not as a standalone replacement yet.

## Policy Reconstruction Experiments
- Finished: 2026-07-02
- Plan: `C:\Users\kiros\Downloads\policy_reconstruction_experiment_plan_codex.md`
- Report: `POLICY_RECONSTRUCTION_RESULTS.md`
- New framework code: `src/` common utilities and `scripts/` audit/calibration/bias/blend scripts.

Policy framing:
- Treat the task as reconstructing `P(action | state)`, then separate three questions: ordinary generalization, replay/transductive behavior, and Macro-F1 decision optimization.
- This matters because leaderboard `0.78` is unlikely to come from pure current-prompt TF-IDF. The useful signal is probably a mix of history serialization, class-specific specialists, probability calibration, and controlled transductive/replay behavior if rules allow it.

Replay audit:
- Session-scoped replay is perfect only inside the same session/batch: train internal precision `1.000000`, transductive valid precision `1.000000`.
- GroupSplit train-to-valid session-scoped coverage is `0/14106`, so it does not explain safe hidden generalization.
- Global prompt train-to-valid precision is only `0.432864`, so raw global exact prompt lookup is too risky as a primary rule.
- Public placeholder test hits are `5/5`, but actual hidden submission did not improve, so the previous lookup package likely had no useful hidden hits.

Policy ceiling / memorization audit:
- Best exact-signature GroupKFold result is `S1_template_prompt` with bias tuning Macro-F1 `0.154695`.
- Detailed signatures get worse because group-disjoint folds make exact states sparse/unseen.
- Conclusion: exact state-signature memorization is not the route to `0.78`; it is mainly an anti-leak diagnostic.

Transformer calibration and bias:
- Full mDeBERTa validation Macro-F1: `0.686816`.
- Temperature scaling improved log-loss `0.734426 -> 0.728853` and ECE `0.032505 -> 0.013945`.
- Class-bias tuning improved transformer Macro-F1 `0.686816 -> 0.689963`.
- Useful, but still below the linear advanced router.

Best current validation decision:
- Advanced router alone: Macro-F1 `0.711324`.
- Transformer alone: Macro-F1 `0.686816`.
- Advanced router + transformer stronger-class override: Macro-F1 `0.719520`, accuracy `0.718559`, `1462` predictions changed.
- Current practical next step: train/save a full-data transformer checkpoint and build a submit variant that uses advanced router as the base, then overrides only on the transformer-strong action set.

## Policy Recovery v3
- Finished fast validation phases: 2026-07-02
- Plan: `C:\Users\kiros\Downloads\FINAL_SPEC_POLICY_RECOVERY_v3.md`
- Report: `POLICY_RECOVERY_V3_RESULTS.md`

Completed:
- Q0 fixed `GroupShuffleSplit` files at `splits_v3/`: train `55894`, val `14106`, sessions `9429`.
- Q1 phase 0 bug report: label order is correct; train loss decreases; old transformer weakness came from `[NOW]` prompt being truncated in the legacy tail serializer.
- Q2 v3 ceiling: exact state tables are too sparse. Best expected Macro-F1 is only `0.110849` at `S1`, and S5 coverage is `0.004`.
- Q3 serializer golden file: `tests/golden_serialize_v3.txt`.
- Q6 decision: temperature `1.150098`, log-loss `0.734426 -> 0.728853`, ECE `0.032505 -> 0.013945`, argmax Macro-F1 `0.686816`, same-val biased Macro-F1 `0.689787`; strict valA/valB bias adoption fails (`-0.000747`, `-0.005442`).
- Q7 ensemble: new local best `0.721702` from `advanced_router + calibrated/bias transformer stronger-class override`, accuracy `0.720119`, changes vs advanced `1445`.

Current long run:
- `reports/transformer/v3-run1-mdeberta-nowfirst-lr5e5-save`
- Purpose: reproduce the full mDeBERTa run with `--save-model`, because the previous full run saved logits only.
- Local GPU status at launch: RTX 4060 Ti 8GB, training uses about `7.0GB` VRAM.

Submission package:
- The save-model run completed in `5739.6s`; saved checkpoint best epoch 3 Macro-F1 `0.683043`.
- Recalibrated saved checkpoint: temperature `1.158088`, biased Macro-F1 `0.691702`.
- Saved-checkpoint ensemble best: `advanced_router + saved mDeBERTa stronger-class override`, Macro-F1 `0.721087`, accuracy `0.719410`.
- Built `submit_policy_v3.zip`, size `547,553,522` bytes, unpacked `603,014,931` bytes.
- Offline smoke passed. 1,000-row local benchmark: `21.25s` including model load.
- Session lookup disabled in this v3 package to match the ensemble validation protocol.

Leaderboard debug:
- `submit_policy_v3.zip` scored the same hidden/public value as advanced-only: `0.704342388`.
- Local end-to-end validation execution changed `1882` predictions vs advanced-router predictions, so the local package logic is not identical when transformer inference runs.
- Most likely server behavior: transformer/tokenizer load failed under server `transformers==4.46.3`, then the script fell back to the advanced router via the safety `try/except`.
- Compatibility packages created locally:
  - `submit_policy_v3_spm.zip`: includes `spm.model` plus `model.safetensors`.
  - `submit_policy_v3_bin_spm.zip`: includes `spm.model` plus `pytorch_model.bin`.
- Recommended next submit: `submit_policy_v3_bin_spm.zip`. If it still ties `0.704342388`, use a strict no-fallback debug package to expose the server-side load error.

Interpretation:
- Lookup leak and exact state-table recovery are dead ends for hidden generalization.
- The useful shape is still a two-expert policy: fast advanced linear router as base, transformer representation as specialist.
- The next real blocker after this submit package is improving the transformer specialist without making inference exceed 10 minutes.

## OOF Policy Recovery v4
- Started: 2026-07-02
- Plan: `C:\Users\kiros\Downloads\FINAL_SPEC_v4_OOF_PIPELINE.md`
- Progress report: `OOF_PIPELINE_V4_PROGRESS.md`
- Implemented `pipeline_v4/` scaffold, deterministic session folds, v4 serializer, golden test, multi-task fold trainer, calibration, OOF assembly, and bias optimizer.
- Fold generation complete: counts `13898/14078/14033/13925/14066`, max deviation `0.007286`, all classes present per fold.
- Serializer golden generated and verified: `[NOW]` first, `[H6]` oldest to `[H1]` newest.
- Fold0 gate training started with `mdeberta_a_local8gb` effective batch 32. First attempt hit fp16-gradient unscale error; fixed by forcing trainable weights to fp32 before autocast. Second attempt is running.

## Intuition Validation Protocol v2
- Finished: 2026-07-03
- Plan: `C:\Users\kiros\Downloads\intuition_validation_protocol_v2_codex.md`
- Report: `artifacts/intuition/SUMMARY.md`
- Runtime: full CPU proxy run about `18m`; corrected decision-stage rerun about `24s`.

Baselines:
- Advanced router validation Macro-F1: `0.711324`.
- Static advanced + transformer stronger-class override validation Macro-F1: `0.721702`.
- v4 mDeBERTa fold0 best Macro-F1: `0.693044`.

Validated decisions:
- I1 workflow flags passed Tier B: proxy delta `+0.003935`, execute target delta `+0.011375`, stable on both half splits. Adopt as serializer/state feature candidate.
- I145 combined bundle passed weakly: proxy delta `+0.002803`, but half split B was nearly flat. Use only as a v2 serializer ablation candidate, not directly as final logic.
- I4 numeric result buckets failed: delta `-0.001317`.
- I5 surface flags failed: delta `-0.001257`.
- I6 last3 prior failed after corrected advanced/transformer baseline: transformer-score gain only `+0.000442`.
- I7 turn-bucket bias failed: cross-half average delta `-0.008994`.
- I9 learned override selector failed: cross-half delta vs static override `-0.020104`; it over-selects transformer overrides.
- I10 class-specific thresholds looked good on full validation (`0.724497`, `+0.002795` vs static), but failed strict half-split stability: A->B `+0.000785`, B->A `-0.003547` vs static. Reject for current submit because it overfits validation.
- I3 structural ExtraTrees probe failed badly (`0.317806` on A->B), so structural-only tree member is not useful as a direct ensemble member.

Interpretation:
- The safest current submit logic remains the static stronger-class transformer override, not a learned selector or class-threshold table.
- The only feature hypothesis worth carrying into transformer serializer v2 is workflow-state flags. Numeric result parsing and surface flags should stay out unless redesigned.
- Full transformer replacement remains rejected; transformer should stay a specialist behind the advanced router.

## Research Operating System

Started: 2026-07-03

Purpose:
- Stop naming experiments by vibes.
- Every expensive run must have a pre-written hypothesis, a cheap validation path, an expensive validation path, and an adoption/rejection gate.
- Public leaderboard submissions are validation probes, not the primary model-selection mechanism.
- Negative results are first-class assets because they shrink the search space.

Core rule:
- Adopt only if the pre-declared gate passes.
- If a trick improves the same validation split but fails half-split or group validation, reject it for submit.
- If a model cannot fit the 10-minute inference budget, treat it as a teacher/specialist, not as a direct full-test predictor.

### Hypothesis Ledger

| ID | Hypothesis | Cheap Validation | Expensive Validation | Pass Gate | Result | Decision |
|---|---|---|---|---|---|---|
| H1 | `[NOW]` prompt truncation caused early transformer weakness. | Token audit on 10k/full train. | mDeBERTa fold/full rerun with now-first serializer. | `[NOW]` kept 100%; fold/full improves by at least `+0.005`. | Pass. Missing `[NOW]` dropped from `4393/10000` to `0/10000`; full mDeBERTa reached `0.686816`; hybrid reached `0.719520`. | Adopt. |
| H2 | Workflow-state flags help action routing. | Tier B linear proxy and half split. | Add to transformer serializer. | Overall `+0.002` and execute target class lift. | Pass. I1 proxy delta `+0.003935`, execute delta `+0.011375`. | Adopt as serializer feature. |
| H3 | XLM-R tokenizer/backbone is better for mixed Korean/English agent state. | Token audit and fold0 3epoch. | fold1 confirm, then full train only if gates pass. | fold0 `>=0.720`; strong if `>=0.730`; fold0/fold1 average `>=0.725` for full train. | Fail. Token audit passed, but fold0 best Macro-F1 was only `0.697038`. | Reject as main track. |
| H4 | Numeric result buckets improve inspect classes. | Tier B linear proxy. | Transformer bundle only if proxy passes. | Macro-F1 `+0.002` and inspect `+0.004`. | Fail. I4 delta `-0.001317`. | Reject. |
| H5 | Prompt surface flags help ask/plan/respond routing. | Tier B linear proxy. | Transformer serializer ablation only if proxy passes. | Macro-F1 `+0.002` and communicate lift. | Fail. I5 delta `-0.001257`. | Reject for now. |
| H6 | Last-action transition priors still add lift on top of advanced/transformer. | Corrected decision-stage validation. | OOF stacking only if proxy passes. | Transformer-score gain `>=+0.002`. | Fail. I6 gain only `+0.000442`. | Reject. |
| H7 | Turn-bucket bias fixes stage-dependent actions. | Half-split validation. | Submit only if both split directions improve. | A->B and B->A both positive. | Fail. Cross-half average `-0.008994`. | Reject. |
| H8 | Learned override selector beats static transformer strong-class override. | Half-split selector test. | OOF selector only if stable. | Beats static on both halves. | Fail. I9 cross-half delta `-0.020104`. | Reject. |
| H9 | Class-specific thresholds improve Macro-F1. | Same-val and half-split validation. | Submit only if half-split stable. | A->B and B->A both positive vs static. | Fail. Same-val `0.724497`, but B->A `-0.003547`. | Reject for submit. |
| H10 | Exact replay/lookup is a safe hidden-generalization feature. | Group split train-to-valid lookup. | Public probe only, not final unless rule-safe. | Train-to-valid coverage and precision useful without transductive batch use. | Fail. Train-only lookup delta `0` or worse; transductive self-history huge but rule-risky. | Exclude from official model. |

### Experiment Tiers

| Tier | Cost | Allowed Work | Examples | Time Budget | Promotion Gate |
|---|---|---|---|---:|---|
| A | No training | Statistics, coverage, entropy, token length, conflict rate. | `?` prompt action mix, `0 matches` transition, token audit. | Minutes | Clear directional signal or safety check. |
| B | Cheap proxy | Linear model, small subset, fold0 short epoch, half split. | Add workflow flags to TF-IDF; threshold half-split. | 10-30 min | Macro-F1 lift `>=+0.002` and target class lift. |
| C | GPU fold | Transformer fold0, 2-3 epochs, baseline vs variant. | XLM-R fold0; mDeBERTa serializer ablation. | 1-3 hr | fold0 gate met and no runtime blocker. |
| D | Submit candidate | Full train, best epoch, zip packaging, offline smoke, server probe. | full encoder ep3/ep5, candidate-gated transformer submit. | Hours | Local gate passed, runtime likely under 10 min, zip under 1GB. |

Tier discipline:
- Do not send Tier A/B ideas straight to GPU.
- Do not run full train unless fold gates pass.
- Do not submit a model that has not passed offline smoke.

### Negative Result Ledger

| Experiment | Result | Decision | Why It Matters |
|---|---:|---|---|
| I4 numeric result buckets | `-0.001317` | Reject | Numeric buckets alone do not justify serializer complexity. |
| I5 surface flags | `-0.001257` | Reject | Surface flags are weak as independent features. |
| I6 last3 prior | `+0.000442` | Reject | Transition priors are mostly already captured by advanced router/transformer state. |
| I7 turn-bucket bias | `-0.008994` cross-half average | Reject | Stage bias overfits validation. |
| I9 learned selector | `-0.020104` vs static | Reject | Learned selector over-selects transformer overrides. |
| I10 class thresholds | same-val up, half-split unstable | Reject for submit | Public-style threshold hunting is overfit-prone. |
| Exact train-history lookup | hurts or no lift | Reject | Hidden generalization does not come from train-memory exact matching. |
| Full transformer replacement | below advanced router | Reject as standalone | Transformer should be a specialist or teacher, not the base model yet. |

### Submission Ledger

| Submit | Local Basis | Public/Hidden | Runtime | Changed | Hypothesis | Conclusion |
|---|---|---:|---:|---:|---|---|
| `submit_01_fixed_stable.zip` | advanced router line | `0.704342388` | `1m47s` | 0 | Stable lightweight baseline. | Baseline. |
| `submit_02_fixed_lookup.zip` | lookup probe | `0.704342388` | `1m52s` | unknown | Exact lookup may exploit repeated hidden states. | No hidden lift; reject lookup as main path. |
| `submit_policy_v3.zip` | local hybrid `~0.721` | `0.704342388` | `1m49s` | expected local changes | Transformer override should lift. | Server likely skipped transformer/fallback issue. |
| `submit_policy_v3_spm.zip` | compatibility package | `0.7099979659` | `6m10s` | unknown | Add tokenizer/SPM compatibility. | Real lift, but still far from 0.78. |
| `submit_v4_fold0_debug.zip` | fold0 transformer debug | TLE | `>10m` | transformer all rows | Full hidden test transformer inference at 512. | Runtime failure; not viable direct. |
| `submit_v4_fold0_fast.zip` | fold0 gated debug | pending/diagnostic | under local smoke | `selected=5/5 changed=1` on sample | Verify v4 override activation under time budget. | Debug only, not final performance candidate. |
| `submit_v4_fold0_384_12k.zip` | fold0 transformer gated debug | pending | local smoke passed | `selected=5/5 changed=1` on sample | Test a less-aggressive TLE-safe gate: max_len 384, batch 64, top 12k candidates. | Submit probe. |

Submission interpretation:
- A server score equal to `0.704342388` usually means the transformer path did not run or made no effective hidden changes.
- A TLE means the model may be useful as a teacher/specialist, but direct all-row inference is not deployable.
- Every submit must record purpose, runtime, changed count, and conclusion.

### Submission Engineering Checklist

Required before any code-submit zip:
- `local_files_only=True` for transformer/tokenizer loading.
- `output/submission.csv` is always created.
- `requirements.txt` is empty or minimal.
- Model load failure has an intentional fallback only for non-debug packages.
- Debug packages may disable fallback to expose server errors.
- CUDA OOM has batch-size fallback or a conservative batch size.
- `max_len`, `batch_size`, candidate limit, and threshold are config-driven.
- Inference benchmark is recorded on local sample or 1000-row proxy.
- Zip size is below `1GB`.
- Offline smoke test runs from the extracted submit directory.

Transformer submit rule:
- Full 30k-row inference is allowed only if estimated runtime is comfortably below 10 minutes.
- Otherwise use candidate gating, distillation, or advanced-router base plus transformer specialist.
- If candidate gating reduces model coverage too much, mark the package as diagnostic, not final.

### Current XLM-R State v1 Experiment

Hypothesis:
- XLM-R may be a better encoder for mixed Korean/English agent-state serialization than mDeBERTa.

Pre-declared gates:
- Token audit must keep `[NOW]`, `[LAST]`, `[STATE]` at `100%` for `max_len=512`.
- fold0 Macro-F1 `>=0.720` to continue.
- fold0 Macro-F1 `>=0.730` to prioritize XLM-R over mDeBERTa.
- fold0/fold1 average `>=0.725` to start full-data XLM-R training.

Current status:
- Token audit passed for `max_len=512`: over limit `0%`; `[NOW]`, `[LAST]`, `[STATE]` kept `100%`.
- `max_len=384` also has over limit `0%`, but keeps fewer history pairs on average than 512.
- fold0 3epoch training finished.
- Best fold0 result: epoch `3`, Macro-F1 `0.697038`, NLL `0.704592`, accuracy `0.715499`.
- Training loss curve: `1.3637 -> 0.7925 -> 0.6923`.
- Validation Macro-F1 curve: `0.655583 -> 0.684692 -> 0.697038`.

Decision:
- XLM-R state v1 failed the pre-declared fold0 gate.
- Fold1 and full-data XLM-R training are skipped.
- Keep the token-audit/serializer code as reusable infrastructure, but do not spend more GPU on XLM-R unless a new Tier A/B hypothesis explains why the fold0 result should improve materially.
- Next better use of GPU: mDeBERTa specialist improvement, distillation from transformer logits into a fast student, or inspect-class targeted experiments.

### Current mDeBERTa 384 v2 Gate Experiment

Hypothesis:
- `max_len=384` can preserve most useful v2 state while reducing submit-time TLE risk compared with `512`.

Pre-declared gates:
- Token audit: `[NOW]`, `[LAST]`, `[STATE]` must be kept at `100%`; `[SEQ] >=95%`.
- Strong full-train gate: fold0 Macro-F1 `>=0.716`.
- Recommended full-train gate: fold0 Macro-F1 `>=0.712`.
- Stop gate: fold0 Macro-F1 `<0.705`.

Result:
- Token audit passed: p50 `330`, p90 `374`, p95 `379`, p99 `383`, over-rate `0.0000`, average history pairs kept `1.99`.
- Fold0 best: epoch `5`, Macro-F1 `0.717801`, NLL `0.681687`, accuracy `0.733055`.
- Epoch curve: `0.588438 -> 0.683268 -> 0.709016 -> 0.717279 -> 0.717801`.

Decision:
- Strong full-train gate passed.
- Start full-data `mDeBERTa v2 max_len=384` training.
- Save epoch 3 and epoch 5 checkpoints and build `cand8000` submit probes for both.
