# Leap Diagnostics

## Model Summary

| model | Macro-F1 | modify3 | inspect4 | execute3 | communicate3 |
|---|---:|---:|---:|---:|---:|
| `strict_blend_bias_pred` | `0.724084` | `0.922817` | `0.564897` | `0.721557` | `0.650279` |
| `strict_blend_05` | `0.721237` | `0.922240` | `0.561203` | `0.720455` | `0.643630` |
| `d2m5_student_oof` | `0.718463` | `0.923631` | `0.560718` | `0.709794` | `0.640507` |
| `teacher_oof` | `0.718193` | `0.963399` | `0.567974` | `0.700337` | `0.597359` |
| `advanced_strict_oof` | `0.710559` | `0.905945` | `0.548670` | `0.711461` | `0.635957` |

## Best Stable Hard Overrides

| base | source | group | thr | delta | new | min_fold | changed | net |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `advanced_strict_oof` | `teacher_oof` | `modify3` | `0.70` | `0.012295` | `0.722854` | `0.009014` | `1348` | `1005` |
| `advanced_strict_oof` | `teacher_oof` | `modify3` | `0.55` | `0.012264` | `0.722823` | `0.007842` | `1484` | `1007` |
| `advanced_strict_oof` | `teacher_oof` | `modify3` | `0.40` | `0.012122` | `0.722680` | `0.007491` | `1538` | `1000` |
| `advanced_strict_oof` | `teacher_oof` | `modify3` | `0.85` | `0.012062` | `0.722620` | `0.009345` | `1201` | `977` |
| `advanced_strict_oof` | `teacher_oof` | `modify3` | `0.00` | `0.012060` | `0.722619` | `0.007431` | `1545` | `997` |
| `advanced_strict_oof` | `teacher_oof` | `modify3` | `0.25` | `0.012046` | `0.722605` | `0.007431` | `1544` | `997` |
| `advanced_strict_oof` | `teacher_oof` | `modify3` | `0.95` | `0.010055` | `0.720613` | `0.007448` | `895` | `801` |
| `strict_blend_05` | `teacher_oof` | `modify3` | `0.85` | `0.009112` | `0.730349` | `0.006142` | `1000` | `776` |
| `strict_blend_05` | `teacher_oof` | `modify3` | `0.70` | `0.009099` | `0.730336` | `0.005052` | `1144` | `782` |
| `d2m5_student_oof` | `teacher_oof` | `modify3` | `0.85` | `0.009026` | `0.727490` | `0.005743` | `1005` | `753` |
| `strict_blend_bias_pred` | `teacher_oof` | `modify3` | `0.85` | `0.009001` | `0.733085` | `0.005976` | `993` | `761` |
| `strict_blend_bias_pred` | `teacher_oof` | `modify3` | `0.70` | `0.009001` | `0.733085` | `0.004835` | `1140` | `768` |

## Best Stable Group Probability Blends

| base | source | group | w_source | delta | new | min_fold | changed | net |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `advanced_strict_oof` | `teacher_oof` | `modify3` | `0.50` | `0.012389` | `0.722947` | `0.010310` | `1184` | `979` |
| `advanced_strict_oof` | `teacher_oof` | `modify3` | `0.65` | `0.012328` | `0.722886` | `0.008919` | `1320` | `986` |
| `advanced_strict_oof` | `teacher_oof` | `modify3` | `0.80` | `0.012136` | `0.722694` | `0.008081` | `1409` | `984` |
| `advanced_strict_oof` | `teacher_oof` | `modify3` | `1.00` | `0.012011` | `0.722570` | `0.007428` | `1563` | `998` |
| `strict_blend_05` | `teacher_oof` | `modify3` | `0.50` | `0.009403` | `0.730640` | `0.006712` | `965` | `775` |
| `strict_blend_05` | `teacher_oof` | `modify3` | `0.65` | `0.009278` | `0.730515` | `0.005433` | `1112` | `768` |
| `d2m5_student_oof` | `teacher_oof` | `modify3` | `0.65` | `0.009015` | `0.727478` | `0.005006` | `1081` | `734` |
| `strict_blend_05` | `teacher_oof` | `modify3` | `0.80` | `0.008965` | `0.730202` | `0.004232` | `1207` | `755` |
| `advanced_strict_oof` | `teacher_oof` | `modify3` | `0.35` | `0.008791` | `0.719350` | `0.007405` | `799` | `692` |
| `d2m5_student_oof` | `teacher_oof` | `modify3` | `0.80` | `0.008715` | `0.727178` | `0.003701` | `1184` | `714` |
| `strict_blend_05` | `teacher_oof` | `modify3` | `1.00` | `0.008593` | `0.729830` | `0.003314` | `1358` | `757` |
| `d2m5_student_oof` | `teacher_oof` | `modify3` | `1.00` | `0.008218` | `0.726681` | `0.002615` | `1355` | `710` |

## Oracle Pool Ceiling

| group | rows | best single | any-model correct rate | missed by all |
|---|---:|---:|---:|---:|
| `inspect4` | `28782` | `0.567974` | `0.673824` | `9388` |
| `modify3` | `17475` | `0.963399` | `0.980200` | `346` |
| `execute3` | `11912` | `0.720455` | `0.821525` | `2126` |
| `communicate4` | `11831` | `0.731105` | `0.862818` | `1623` |
| `communicate3` | `6653` | `0.643630` | `0.756050` | `1623` |
| `all14` | `70000` | `0.721237` | `0.807386` | `13483` |

## Interpretation

- `teacher_oof` is fold-held-out, so its strong modify3 score is not label leakage.
- A public-submit jump requires making that teacher signal available under the 10 minute limit, usually by candidate-gated transformer inference or distillation.
- Stable override rows require positive overall delta and no fold worse than `-0.0005`; looser rows are research-only.
