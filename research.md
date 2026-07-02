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
