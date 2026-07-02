# I1. Workflow Flags Verdict

## Result
- Tier A: not_run
- Tier B: pass
- Tier C: not_run
- Final: adopt

## Metrics
- Base Macro-F1: 0.6599809062162016
- New Macro-F1: 0.6639159638514563
- Delta: 0.003935057635254702
- Target class/group delta: 0.011374562691761477
- Worst class drop: -0.004239574722145645
- Half split stability: {'valA': {'base_macro_f1': 0.6609971189335544, 'new_macro_f1': 0.6652814151069285, 'delta': 0.004284296173374114}, 'valB': {'base_macro_f1': 0.6583522515680095, 'new_macro_f1': 0.6620028553178253, 'delta': 0.0036506037498158506}}

## Feature Availability
- Uses current_prompt: yes
- Uses history assistant_action: yes
- Uses result_summary: yes
- Uses train labels at inference: no
- Uses future steps: no
- Uses full test batch: no

## Decision
Adopt into serializer state if the execute loop gain is stable.
