# Inspect Fast Harness

- base Macro-F1: `0.724084`
- base inspect4 Macro-F1: `0.564897`
- best: `base_strict_distill_bias`
- best Macro-F1: `0.724084`
- delta: `0.000000`
- inspect delta: `0.000000`
- changed: `0`

## Top Variants

| name | Macro-F1 | delta | inspect4 | inspect_delta | changed |
|---|---:|---:|---:|---:|---:|
| `base_strict_distill_bias` | `0.724084` | `0.000000` | `0.564897` | `0.000000` | `0` |
| `template_s3_p0.7` | `0.724084` | `0.000000` | `0.564897` | `0.000000` | `0` |
| `template_s5_p0.7` | `0.724084` | `0.000000` | `0.564897` | `0.000000` | `0` |
| `template_s5_p0.8` | `0.724084` | `0.000000` | `0.564897` | `0.000000` | `0` |
| `template_s10_p0.75` | `0.724084` | `0.000000` | `0.564897` | `0.000000` | `0` |
| `template_s3_p0.6` | `0.724059` | `-0.000025` | `0.564809` | `-0.000088` | `3` |
| `template_s2_p0.6` | `0.724047` | `-0.000037` | `0.564769` | `-0.000128` | `5` |
| `logreg_word_m1.5` | `0.723110` | `-0.000974` | `0.561490` | `-0.003407` | `3753` |
| `svc_word_char_m1.5` | `0.723090` | `-0.000994` | `0.561418` | `-0.003479` | `2425` |
| `logreg_word_m1.0` | `0.720505` | `-0.003579` | `0.552370` | `-0.012527` | `8508` |
| `svc_word_char_m1.0` | `0.719570` | `-0.004514` | `0.549099` | `-0.015799` | `6642` |
| `logreg_word_m0.5` | `0.711791` | `-0.012292` | `0.521874` | `-0.043024` | `16773` |

## Decision Rule

- Adopt only if strict OOF delta is positive, inspect4 delta is positive, and no fold is materially worse.
- If all variants are negative, do not build a public zip for this inspect specialist family.
