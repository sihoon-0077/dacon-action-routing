# I145. I1+I4+I5 Bundle Verdict

## Result
- Tier A: not_run
- Tier B: pass
- Tier C: not_run
- Final: adopt

## Metrics
- Base Macro-F1: 0.6599809062162016
- New Macro-F1: 0.6627834082114034
- Delta: 0.0028025019952017116
- Target class/group delta: 0.002802501995201858
- Worst class drop: -0.008624843173827834
- Half split stability: {'valA': {'base_macro_f1': 0.6609971189335544, 'new_macro_f1': 0.6664805215165607, 'delta': 0.0054834025830062805}, 'valB': {'base_macro_f1': 0.6583522515680095, 'new_macro_f1': 0.6585472764851504, 'delta': 0.00019502491714096237}}

## Feature Availability
- Uses current_prompt: yes
- Uses history assistant_action: yes
- Uses result_summary: yes
- Uses train labels at inference: no
- Uses future steps: no
- Uses full test batch: no

## Decision
Use as candidate serializer-v2 bundle only if it beats individual features.
