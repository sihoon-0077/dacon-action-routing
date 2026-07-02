# I4. Numeric Result Buckets Verdict

## Result
- Tier A: not_run
- Tier B: fail
- Tier C: not_run
- Final: reject

## Metrics
- Base Macro-F1: 0.6599809062162016
- New Macro-F1: 0.6586637725720758
- Delta: -0.0013171336441258896
- Target class/group delta: -0.0030049532918922234
- Worst class drop: -0.005861971913002195
- Half split stability: {'valA': {'base_macro_f1': 0.6609971189335544, 'new_macro_f1': 0.6606689846044711, 'delta': -0.0003281343290832961}, 'valB': {'base_macro_f1': 0.6583522515680095, 'new_macro_f1': 0.656042242111368, 'delta': -0.002310009456641504}}

## Feature Availability
- Uses current_prompt: yes
- Uses history assistant_action: yes
- Uses result_summary: yes
- Uses train labels at inference: no
- Uses future steps: no
- Uses full test batch: no

## Decision
Reject for now: it did not clear the Tier B delta/stability rule.
