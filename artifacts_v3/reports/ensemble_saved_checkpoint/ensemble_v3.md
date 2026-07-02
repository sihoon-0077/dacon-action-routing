# Phase 5 Ensemble v3

- E3 probability blending with advanced router was skipped because the current advanced artifact does not expose calibrated `predict_proba`.

| name | Macro-F1 | accuracy | changes |
|---|---:|---:|---:|
| `override_stronger_thr0.0` | 0.721087 | 0.719410 | 1457 |
| `override_stronger_thr0.4` | 0.718241 | 0.717780 | 808 |
| `inspect_only_thr0.0` | 0.717217 | 0.716433 | 1123 |
| `override_stronger_thr0.6` | 0.717008 | 0.716716 | 352 |
| `override_stronger_thr0.7` | 0.716658 | 0.716433 | 277 |
| `override_stronger_thr0.5` | 0.716615 | 0.716078 | 481 |
| `override_stronger_thr0.8` | 0.716287 | 0.715866 | 209 |
| `override_stronger_thr0.9` | 0.715674 | 0.714731 | 122 |
| `inspect_only_thr0.4` | 0.714385 | 0.714731 | 495 |
| `inspect_only_thr0.6` | 0.712869 | 0.713243 | 78 |
| `inspect_only_thr0.5` | 0.712778 | 0.712959 | 184 |
| `inspect_only_thr0.7` | 0.712102 | 0.712250 | 32 |
| `inspect_only_thr0.8` | 0.711668 | 0.711541 | 12 |
| `advanced_router` | 0.711324 | 0.710974 | 0 |
| `inspect_only_thr0.9` | 0.711324 | 0.710974 | 0 |
| `transformer_calibrated_biased` | 0.691702 | 0.704736 | 2376 |
