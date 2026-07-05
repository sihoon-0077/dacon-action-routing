# V2.3 Signal Experiments

## Proxy Model Results

| variant | Macro-F1 | delta | inspect4 | execute3 | communicate4 |
|---|---:|---:|---:|---:|---:|
| `base_v2_2` | `0.514217` | `0.000000` | `0.359723` | `0.455582` | `0.538628` |
| `v23_open` | `0.515951` | `0.001734` | `0.361623` | `0.457353` | `0.539136` |
| `v23_meta` | `0.514168` | `-0.000049` | `0.359461` | `0.454544` | `0.539154` |
| `v23_target_symbol` | `0.512863` | `-0.001354` | `0.357794` | `0.453687` | `0.538658` |
| `v23_all` | `0.515925` | `0.001708` | `0.361606` | `0.454851` | `0.540224` |

## Top Lift Signals

| feature | value | action | support | lift | p(action|value) |
|---|---|---|---:|---:|---:|
| `budget_bucket_v23` | `b0` | `ask_user` | `495` | `5.445` | `0.210` |
| `prompt_file_rel` | `not_open` | `write_file` | `6271` | `4.032` | `0.085` |
| `elapsed_bucket` | `e0` | `write_file` | `580` | `3.830` | `0.081` |
| `budget_bucket_v23` | `b0` | `respond_only` | `495` | `3.823` | `0.283` |
| `open_profile` | `html_only` | `web_search` | `289` | `3.425` | `0.062` |
| `open_profile` | `many3+` | `respond_only` | `369` | `3.261` | `0.241` |
| `open_profile` | `many3+` | `lint_or_typecheck` | `369` | `3.074` | `0.100` |
| `elapsed_bucket` | `e0` | `list_directory` | `580` | `2.927` | `0.181` |
| `open_profile` | `none` | `write_file` | `23498` | `2.832` | `0.060` |
| `open_profile` | `txt_only` | `web_search` | `1096` | `2.810` | `0.051` |
| `open_profile` | `test_only` | `web_search` | `259` | `2.760` | `0.050` |
| `elapsed_bucket` | `e0` | `plan_task` | `580` | `2.658` | `0.102` |
| `open_profile` | `sql_only` | `web_search` | `736` | `2.615` | `0.048` |
| `open_profile` | `js_only` | `lint_or_typecheck` | `6527` | `2.410` | `0.079` |
| `target_symbol_present` | `yes` | `lint_or_typecheck` | `13887` | `2.369` | `0.077` |
| `open_profile` | `none` | `list_directory` | `23498` | `2.343` | `0.145` |
| `prompt_file_rel` | `open` | `grep_search` | `3208` | `2.245` | `0.318` |
| `target_symbol_present` | `yes` | `run_tests` | `13887` | `2.171` | `0.141` |
| `open_profile` | `css_only` | `web_search` | `411` | `2.141` | `0.039` |
| `open_profile` | `mixed` | `apply_patch` | `4143` | `2.088` | `0.144` |

## Decision

- Treat lift signals as feature candidates only; adoption requires proxy or distill validation.
- `v23_all` must beat `base_v2_2` before spending GPU on a v2.3 transformer serializer.
