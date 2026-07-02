# Phase 5 Ensemble v3

- E3 probability blending with advanced router was skipped because the current advanced artifact does not expose calibrated `predict_proba`.

| name | Macro-F1 | accuracy | changes |
|---|---:|---:|---:|
| `override_stronger_thr0.0` | 0.721702 | 0.720119 | 1445 |
| `override_stronger_thr0.4` | 0.718721 | 0.718205 | 772 |
| `override_stronger_thr0.5` | 0.717564 | 0.717354 | 479 |
| `override_stronger_thr0.6` | 0.717520 | 0.717213 | 348 |
| `override_stronger_thr0.7` | 0.717254 | 0.716645 | 268 |
| `inspect_only_thr0.0` | 0.717166 | 0.716716 | 1114 |
| `override_stronger_thr0.8` | 0.716567 | 0.715582 | 188 |
| `override_stronger_thr0.9` | 0.715926 | 0.714590 | 113 |
| `inspect_only_thr0.4` | 0.714083 | 0.714519 | 459 |
| `inspect_only_thr0.5` | 0.713033 | 0.713455 | 181 |
| `inspect_only_thr0.6` | 0.712713 | 0.712959 | 73 |
| `inspect_only_thr0.7` | 0.712588 | 0.712817 | 44 |
| `inspect_only_thr0.8` | 0.711839 | 0.711825 | 14 |
| `advanced_router` | 0.711324 | 0.710974 | 0 |
| `inspect_only_thr0.9` | 0.711324 | 0.710974 | 0 |
| `transformer_calibrated_biased` | 0.689787 | 0.704523 | 2368 |
