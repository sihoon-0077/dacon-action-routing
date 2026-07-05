# Distill Step2 Summary

- finished_at: `2026-07-05 14:59:50`
- teacher OOF Macro-F1: `0.718193`
- advanced strict-OOF feature baseline Macro-F1: `0.710559`
- advanced feature source: `strict_oof`
- best blend Macro-F1: `0.721237`
- bias adopted: `True`
- final adopted: `True`
- submit zip: `not_built`

## Fast Students

| Name | Macro-F1 | Accuracy | NLL |
|---|---:|---:|---:|
| `D2-G1_hard_noadv` | `0.427698` | `0.419200` | `2.767911` |
| `D2-G2_hard_adv` | `0.652948` | `0.667657` | `1.531091` |
| `D2-G3_pseudo_t0.55_b0.4` | `0.663437` | `0.678486` | `2.492789` |
| `D2-G3_pseudo_t0.65_b0.4` | `0.664209` | `0.680871` | `2.342456` |
| `D2-G3_pseudo_t0.75_b0.6` | `0.661663` | `0.679329` | `2.228254` |
| `D2-G4_hybrid_imitation` | `0.691677` | `0.689429` | `3.096526` |

## MLP OOF

| Name | Macro-F1 | Accuracy | NLL |
|---|---:|---:|---:|
| `D2-M1` | `0.715189` | `0.716743` | `0.772184` |
| `D2-M2` | `0.716455` | `0.719343` | `0.805393` |
| `D2-M3` | `0.715242` | `0.718314` | `0.835477` |
| `D2-M4` | `0.715263` | `0.719543` | `0.985394` |
| `D2-M5` | `0.718463` | `0.721857` | `0.815825` |
| `D2-M6` | `0.715460` | `0.717229` | `0.808185` |

## Decision

- Teacher probabilities were used only as training targets, not inference features.
- TF-IDF/SVD was fit on full train text for the quick full battery; record this as unsupervised feature-cache shortcut.
- Advanced router probabilities came from strict fold-held-out OOF features.
