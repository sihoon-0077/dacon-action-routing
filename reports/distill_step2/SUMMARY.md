# Distill Step2 Summary

- finished_at: `2026-07-05 12:31:02`
- teacher OOF Macro-F1: `0.718193`
- advanced full-fit feature baseline Macro-F1: `0.819835`
- best blend Macro-F1: `0.834668`
- bias adopted: `False`
- final adopted: `False`
- submit zip: `not_built`
- caution: `advanced features were generated from the full-fit advanced_router artifact; the headline blend score is not strict OOF and must not be used as a submit decision by itself`

## Fast Students

| Name | Macro-F1 | Accuracy | NLL |
|---|---:|---:|---:|
| `D2-G1_hard_noadv` | `0.427698` | `0.419200` | `2.767911` |
| `D2-G2_hard_adv` | `0.789265` | `0.797557` | `1.200223` |
| `D2-G3_pseudo_t0.55_b0.4` | `0.747133` | `0.763000` | `2.279683` |
| `D2-G3_pseudo_t0.65_b0.4` | `0.760429` | `0.772200` | `2.120066` |
| `D2-G3_pseudo_t0.75_b0.6` | `0.766425` | `0.777071` | `2.090717` |
| `D2-G4_hybrid_imitation` | `0.757477` | `0.734457` | `2.517619` |

## MLP OOF

| Name | Macro-F1 | Accuracy | NLL |
|---|---:|---:|---:|
| `D2-M1` | `0.816098` | `0.812743` | `0.545791` |
| `D2-M2` | `0.812753` | `0.810400` | `0.561945` |
| `D2-M3` | `0.806879` | `0.804729` | `0.583068` |
| `D2-M4` | `0.808124` | `0.806271` | `0.676115` |
| `D2-M5` | `0.818678` | `0.816729` | `0.540516` |
| `D2-M6` | `0.811826` | `0.809086` | `0.569885` |

## Decision

- Teacher probabilities were used only as training targets, not inference features.
- TF-IDF/SVD was fit on full train text for the quick full battery; record this as unsupervised feature-cache shortcut.
- Advanced router probabilities were recomputed from the existing full-fit artifact because they are available at inference time.
- The high `0.834668` blend score is therefore an optimistic diagnostic score, not a leaderboard-ready validation score.
- Do not submit the generated distill package from this run. The placeholder zip was deleted.
- Next action: build strict advanced OOF features, then rerun the final blend/adoption gate.
