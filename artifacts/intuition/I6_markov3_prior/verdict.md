# I6. Markov Order 3 Prior Verdict

## Result
- Tier A: pass
- Tier B: fail
- Tier C: not_applicable
- Final: reject

## Metrics
- Base Macro-F1: 0.6897865805656586
- New Macro-F1: 0.6902286548430511
- Delta: 0.00044207427739251237
- Target class/group delta: None
- Worst class drop: None
- Half split stability: not_run

## Feature Availability
- Uses current_prompt: no
- Uses history assistant_action: yes
- Uses result_summary: no
- Uses train labels at inference: no
- Uses future steps: no
- Uses full test batch: no

## Decision
Reject: last3 prior did not add enough over calibrated transformer scores.
