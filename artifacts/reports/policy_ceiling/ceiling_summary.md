# Policy Ceiling Estimate

- folds: `5` GroupKFold by session
- smoothing alpha: `1.0`

## Summary

| level | argmax Macro-F1 | bias tuned Macro-F1 |
|---|---:|---:|
| `S0_raw_prompt` | 0.133483 | 0.147650 |
| `S1_template_prompt` | 0.142060 | 0.154695 |
| `S2_tpl_last1` | 0.072289 | 0.090351 |
| `S3_tpl_last2` | 0.048258 | 0.068630 |
| `S4_tpl_last2_result` | 0.045868 | 0.066384 |
| `S5_tpl_last3_result_meta` | 0.021295 | 0.041930 |
| `S6_tpl_last3_result_open_lang` | 0.020312 | 0.041096 |

## Conclusion

- Best argmax Macro-F1: `0.142060`
- Best bias-tuned Macro-F1: `0.154695`
- Best signature level: `S1_template_prompt`
- Interpretation: this is a memorized observed-state ceiling with train-fold backoff; if it is low, hidden state or representation capacity matters more than lookup-style signatures.
