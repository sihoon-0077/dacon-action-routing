# Dataset Audit: Policy Frame

## Basic

- train rows: `70000`
- test rows: `5`
- unique train sessions: `9429`
- turn_index min/max: `1` / `18`
- history length min/max: `0` / `12`
- missing required key rows: `0`

## Label Distribution

| action | count | ratio |
|---|---:|---:|
| `read_file` | 9257 | 0.1322 |
| `grep_search` | 9912 | 0.1416 |
| `list_directory` | 4329 | 0.0618 |
| `glob_pattern` | 5284 | 0.0755 |
| `edit_file` | 11171 | 0.1596 |
| `write_file` | 1481 | 0.0212 |
| `apply_patch` | 4823 | 0.0689 |
| `run_bash` | 5068 | 0.0724 |
| `run_tests` | 4561 | 0.0652 |
| `lint_or_typecheck` | 2283 | 0.0326 |
| `ask_user` | 2701 | 0.0386 |
| `plan_task` | 2679 | 0.0383 |
| `web_search` | 1273 | 0.0182 |
| `respond_only` | 5178 | 0.0740 |

## Group Distribution

| group | count | ratio |
|---|---:|---:|
| `communicate` | 11831 | 0.1690 |
| `execute` | 11912 | 0.1702 |
| `inspect` | 28782 | 0.4112 |
| `modify` | 17475 | 0.2496 |

## Ambiguity By Signature

| level | keys | mean support | mean entropy | mean top1 ratio | multi-label keys |
|---|---:|---:|---:|---:|---:|
| `S0_raw_prompt` | 63250 | 1.107 | 0.037 | 0.984 | 2221 |
| `S1_template_prompt` | 62644 | 1.117 | 0.042 | 0.981 | 2524 |
| `S2_tpl_last1` | 67880 | 1.031 | 0.007 | 0.997 | 480 |
| `S3_tpl_last2` | 69140 | 1.012 | 0.002 | 0.999 | 156 |
| `S4_tpl_last2_result` | 69227 | 1.011 | 0.002 | 0.999 | 142 |
| `S5_tpl_last3_result_meta` | 69960 | 1.001 | 0.000 | 1.000 | 13 |
| `S6_tpl_last3_result_open_lang` | 69981 | 1.000 | 0.000 | 1.000 | 7 |

## Top Last-Action Transitions

| last_action | action | count |
|---|---|---:|
| `read_file` | `edit_file` | 2577 |
| `edit_file` | `run_tests` | 2444 |
| `grep_search` | `edit_file` | 2065 |
| `grep_search` | `read_file` | 1819 |
| `NONE` | `list_directory` | 1818 |
| `grep_search` | `grep_search` | 1726 |
| `edit_file` | `edit_file` | 1638 |
| `NONE` | `read_file` | 1488 |
| `read_file` | `grep_search` | 1331 |
| `read_file` | `read_file` | 1240 |
| `grep_search` | `glob_pattern` | 1181 |
| `NONE` | `plan_task` | 1121 |
| `glob_pattern` | `grep_search` | 1102 |
| `edit_file` | `apply_patch` | 1076 |
| `list_directory` | `read_file` | 1059 |
| `NONE` | `grep_search` | 1046 |
| `run_tests` | `edit_file` | 1040 |
| `edit_file` | `respond_only` | 1018 |
| `NONE` | `run_bash` | 1011 |
| `edit_file` | `grep_search` | 996 |
| `run_bash` | `run_bash` | 978 |
| `list_directory` | `grep_search` | 907 |
| `run_bash` | `edit_file` | 878 |
| `glob_pattern` | `glob_pattern` | 872 |
| `run_tests` | `respond_only` | 774 |
| `glob_pattern` | `read_file` | 771 |
| `read_file` | `apply_patch` | 770 |
| `apply_patch` | `lint_or_typecheck` | 757 |
| `edit_file` | `run_bash` | 739 |
| `edit_file` | `lint_or_typecheck` | 727 |

## Interpretation

- Lower entropy after adding last actions/result buckets indicates the task is policy-state reconstruction, not prompt-only intent classification.
