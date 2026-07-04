# CPU Tier-A Battery 15 Reproduction

- data: `70000` train rows
- method: conditional label distribution only, no training
- gate: relative class-probability movement of about 30%+ with enough support

## Summary

| ID | Front | Hypothesis | Stat | Final | Best bucket | N | Top | Max shift |
|---|---|---|---|---|---|---:|---|---:|
| I-1 | inspect | prompt slash depth | reject | reject | `1` | 1898 | `grep_search` 0.341 | `list_directory` +0.086 |
| I-2 | inspect | last list/glob count bucket | pass | weak_pass | `glob_pattern:16+` | 1539 | `grep_search` 0.350 | `glob_pattern` +0.699 |
| I-3 | inspect | inspect streak | pass | pass | `3` | 1668 | `glob_pattern` 0.381 | `glob_pattern` +1.074 |
| I-4 | inspect | open file count | pass | pass_observable | `0` | 13056 | `read_file` 0.348 | `list_directory` +0.734 |
| I-5 | inspect | quoted symbol / identifier | reject | reject | `quoted_code` | 278 | `grep_search` 0.381 | `list_directory` -0.235 |
| C-1 | communicate | turn bucket | pass | pass_observable | `t1` | 1936 | `plan_task` 0.579 | `plan_task` +1.557 |
| C-2 | communicate | last group communication chain | pass | pass_observable | `last_comm` | 1289 | `plan_task` 0.396 | `web_search` +0.867 |
| C-3 | communicate | prompt length bucket | pass | weak_pass | `s` | 3379 | `respond_only` 0.916 | `respond_only` +1.093 |
| C-4 | communicate | multi demand marker count | pass | reject | `1` | 1066 | `ask_user` 0.345 | `respond_only` -0.567 |
| C-5 | communicate | last CI status | weak_pass | pass_observable | `failed` | 2856 | `ask_user` 0.322 | `respond_only` -0.447 |
| E-1 | execute | test/lint state split | pass | pass_strong | `test=never|lint=fail|eat=0|eal=1` | 184 | `lint_or_typecheck` 0.641 | `lint_or_typecheck` +2.346 |
| E-2 | execute | last modified extension | pass | pass | `tsx` | 556 | `lint_or_typecheck` 0.446 | `lint_or_typecheck` +1.327 |
| E-3 | execute | last CI status | reject | reject | `none` | 2960 | `run_bash` 0.529 | `run_bash` +0.244 |
| E-4 | execute | last execute self-repeat | pass | pass_observable | `lint_or_typecheck:fail` | 375 | `lint_or_typecheck` 0.629 | `lint_or_typecheck` +2.284 |
| E-5 | execute | dominant workspace language | pass | weak_replaced_by_E2 | `tsx` | 861 | `run_bash` 0.541 | `lint_or_typecheck` +1.394 |

## Adopted Serializer v2.2 Cards

- `test` and `lint` states split, plus `edits_after_test` / `edits_after_lint`.
- `insp_streak` for long inspect chains.
- `last_mod_ext` for execute-channel choice after edits.
- `open_cnt` for inspect routing.
- `count_bucket` for the last `list_directory` / `glob_pattern` result.
- `len_bucket` as a low-cost communicate feature.

Notes:
- `stat` is the mechanical distribution gate from this reproduction.
- `final` follows the supplied experiment ledger: observable or redundant signals are not all serializer features.
- `C-4` and `E-5` show measurable movement here, but the final decision remains reject/replaced because the direction is not a clean new routing card.

## Files

- `I-1`: `i_1.csv`
- `I-2`: `i_2.csv`
- `I-3`: `i_3.csv`
- `I-4`: `i_4.csv`
- `I-5`: `i_5.csv`
- `C-1`: `c_1.csv`
- `C-2`: `c_2.csv`
- `C-3`: `c_3.csv`
- `C-4`: `c_4.csv`
- `C-5`: `c_5.csv`
- `E-1`: `e_1.csv`
- `E-2`: `e_2.csv`
- `E-3`: `e_3.csv`
- `E-4`: `e_4.csv`
- `E-5`: `e_5.csv`
