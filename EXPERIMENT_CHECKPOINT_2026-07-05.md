# Experiment Checkpoint - 2026-07-05

This checkpoint records the current experiment state before pausing work.

## Completed Experiments

### 1. Inspect Bottleneck

- Script: `scripts/run_inspect_bottleneck_experiments.py`
- Output: `reports/inspect_bottleneck/`
- Base proxy: `fast_flat`
- Fold0 Macro-F1: `0.638020`
- N2b state-machine: `FAIL`
- N2c inspect pair resolvers: `FAIL`
- N2d teacher/student: `SKIPPED` because aligned teacher logits/checkpoint were unavailable.
- N4i candidate gating: `CANDIDATE_ONLY`

Decision: inspect remains the biggest bottleneck, but the local standalone/pair fixes did not pass adoption.

### 2. Execute Router Rule + Resolver

- Script: `scripts/run_execute_router_experiment.py`
- Output: `reports/execute_router_experiment/`
- Base proxy: `fast_flat`
- Base Macro-F1: `0.638020`
- Rule-only execute overrides: harmful, rejected.
- Best resolver: `logreg_word_char_strict_base_execute_thr0.45`
- Macro-F1 delta: `+0.001647`
- Execute macro-F1 delta: `+0.007688`
- Net correct delta: `+12`
- Class deltas:
  - `run_bash`: `+0.002052`
  - `run_tests`: `+0.006095`
  - `lint_or_typecheck`: `+0.014917`

Decision: execute 3-way resolver is the most promising execute-side signal so far.

### 3. Micro Execute/Web Search Rules

- Script: `scripts/run_micro_execute_websearch_experiment.py`
- Output: `reports/micro_rules/`
- Base proxy: `fast_flat`
- Base Macro-F1: `0.638020`

Decisions:

- Execute hard rules: `REJECT`
- Execute pair resolver: strict adoption `REJECT`, but candidate-boost signal exists.
- Best pair resolver candidate: `pair_logreg__exact_pair_no_lint__thr0.55`
- Pair resolver Macro-F1 delta: `+0.001246`
- Pair resolver net changed rows: `+16`
- Web hard override: `REJECT`
- Web candidate boost: none
- Submit zip: not built

Decision: web_search micro rules are not useful locally; run_tests/run_bash pair correction is positive but weaker than the 3-way execute resolver.

## Interrupted / Paused Experiment

### 4. run_tests vs run_bash Deep Experiments

- Script: `scripts/run_tests_bash_deep_experiments.py`
- Status: script created and `py_compile` passed.
- Execution was stopped by user request before final metrics were produced.
- Partial cache exists:
  - `reports/run_tests_bash_deep/inner_router/_cache/fast_flat_router_rf35000.joblib`
- Final outputs not yet produced:
  - `reports/run_tests_bash_deep/summary.md`
  - `reports/run_tests_bash_deep/experiment_results.csv`
  - `reports/run_tests_bash_deep/best_config.json`

Planned deep experiment families:

- `E1_prompt_only_pair`
- `E2_rich_context_pair`
- `E3_command_intent_pair`
- `E4_base_aware_pair`
- `E5_oof_flip_keep_meta`
- `E6_pair_plus_flip_consensus`

Note: the first version timed out because it predicted base-router outputs for the full training set and used a heavy serializer. The script was patched to predict only pair-training rows and to remove the extra `fast_router_text` block from the base-aware serializer. It has not yet been rerun to completion after that patch.

## Current Best Execute Direction

Use this priority when resuming:

1. Re-run `scripts/run_tests_bash_deep_experiments.py` after the speed patch.
2. Compare its best result against `logreg_word_char_strict_base_execute_thr0.45` from `reports/execute_router_experiment/`.
3. If the deep pair/flip experiment does not beat `+0.001647`, prefer the 3-way execute resolver.
4. Do not use web_search hard rules from the current micro-rule results.

## Files Added During This Work

- `scripts/run_inspect_bottleneck_experiments.py`
- `scripts/run_execute_router_experiment.py`
- `scripts/run_micro_execute_websearch_experiment.py`
- `scripts/run_tests_bash_deep_experiments.py`
- `EXPERIMENT_CHECKPOINT_2026-07-05.md`
