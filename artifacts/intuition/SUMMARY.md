# Intuition Validation Summary v2

## Baselines

- advanced router validation Macro-F1: `0.7113236414043568`
- static hybrid validation Macro-F1: `0.7217016318970499`
- v4 mDeBERTa fold0 best Macro-F1: `0.6930435622271415`

## Matrix

| ID | intuition | Tier A | Tier B | Tier C | final | delta | note |
|---|---|---|---|---|---|---:|---|
| I1 | workflow flags | not_run | pass | not_run | adopt | 0.003935057635254702 | Adopt into serializer state if the execute loop gain is stable. |
| I4 | numeric result buckets | not_run | fail | not_run | reject | -0.0013171336441258896 | Reject for now: it did not clear the Tier B delta/stability rule. |
| I5 | surface flags | not_run | fail | not_run | reject | -0.001257222130098934 | Reject for now: it did not clear the Tier B delta/stability rule. |
| I145 | I1+I4+I5 bundle | not_run | pass | not_run | adopt | 0.0028025019952017116 | Use as candidate serializer-v2 bundle only if it beats individual features. |
| I3 | structural GBDT | not_applicable | fail | not_applicable | analysis_only | -0.39092229434951176 | Use as diversity probe only; not a submit member unless probability blend later proves useful. |
| I6 | last3 prior | pass | fail | not_applicable | reject | 0.00044207427739251237 | Reject: last3 prior did not add enough over calibrated transformer scores. |
| I7 | turn bias | pass | fail | not_applicable | reject | -0.008993616513466252 | Reject: cross-half turn bias gain is too small or unstable. |
| I9 | override selector | not_applicable | fail | not_applicable | reject | -0.02010356223656956 | Reject for current submit: selector does not beat static override under strict half validation. |
| I10 | class thresholds | not_applicable | fail | not_applicable | reject | 0.0027948770831316416 | Reject: class thresholds overfit validation halves. |

## Practical Decision

- Submit-facing gain is still concentrated in the existing static transformer override, not in a new selector/threshold yet.
- Serializer candidates should be adopted only if their Tier B proxy is positive and then tested in a short transformer-v2 ablation.
- v4 fold0 reached a healthy but not submit-winning `0.6930`, so full replacement by transformer remains rejected for now.
