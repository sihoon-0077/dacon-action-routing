# Dacon Action Decision Autoresearch Notes

## Goal

Improve the 14-class Macro-F1 action classifier under code-submission constraints:

- inference under 10 minutes for hidden 30k test rows
- package install under 10 minutes
- submission under 1GB
- offline inference after package install
- target environment: T4 GPU, 3 vCPU, 12GB RAM

Because inference is the only server-side workload, CPU scikit-learn models are a good first search space:
small artifacts, fast load, fast prediction, no network dependency.

## Baseline

Baseline uses only `current_prompt` with word TF-IDF 1-2 grams and balanced logistic regression.
Local validation Macro-F1: 0.436882.

## Data Signals

- `current_prompt`: direct intent words like run, test, open, create, search, fix.
- `history`: previous assistant actions strongly affect next action.
  Examples observed locally:
  - `read_file -> edit_file`
  - `edit_file -> run_tests`
  - `grep_search -> read_file/edit_file/grep_search`
- `session_meta.workspace`:
  - `open_files` helps choose read/edit targets.
  - `last_ci_status` can signal test/fix/response flow.
  - `git_dirty` can separate inspect-only from modify/verify workflows.
- Class imbalance matters because the metric is Macro-F1. Rare classes such as `web_search`,
  `write_file`, and `lint_or_typecheck` need explicit lexical/history cues.

## Initial Hypotheses

1. Current prompt alone misses workflow state; compact history action sequence should improve.
2. Last assistant action is a strong transition feature.
3. `result_summary` and `args` can distinguish "found file" -> `read_file` from "opened file" -> `edit_file`.
4. Character n-grams should help Korean/English mixed prompts and file/path fragments.
5. LinearSVC may beat logistic regression for sparse TF-IDF Macro-F1.
6. Prompt weighting matters because full context can drown out the current user request.
7. Metadata tokens can improve low-frequency actions like `lint_or_typecheck`, `web_search`, and `ask_user`.

