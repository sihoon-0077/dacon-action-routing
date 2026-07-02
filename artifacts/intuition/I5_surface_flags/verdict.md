# I5. Surface/Punctuation Flags Verdict

## Result
- Tier A: not_run
- Tier B: fail
- Tier C: not_run
- Final: reject

## Metrics
- Base Macro-F1: 0.6599809062162016
- New Macro-F1: 0.6587236840861027
- Delta: -0.001257222130098934
- Target class/group delta: -0.002195989018094313
- Worst class drop: -0.009320263977595111
- Half split stability: {'valA': {'base_macro_f1': 0.6609971189335544, 'new_macro_f1': 0.6605856847161939, 'delta': -0.00041143421736045127}, 'valB': {'base_macro_f1': 0.6583522515680095, 'new_macro_f1': 0.6563214079314623, 'delta': -0.0020308436365471527}}

## Feature Availability
- Uses current_prompt: yes
- Uses history assistant_action: yes
- Uses result_summary: yes
- Uses train labels at inference: no
- Uses future steps: no
- Uses full test batch: no

## Decision
Reject for now: it did not clear the Tier B delta/stability rule.
