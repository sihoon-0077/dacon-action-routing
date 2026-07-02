# I10. Class-Specific Override Threshold Verdict

## Result
- Tier A: not_applicable
- Tier B: fail
- Tier C: not_applicable
- Final: reject

## Metrics
- Base Macro-F1: 0.7113236414043568
- New Macro-F1: 0.7244965089801816
- Delta: 0.0027948770831316416
- Target class/group delta: 0.01317286757582481
- Worst class drop: None
- Half split stability: [{'direction': 'A_to_B', 'macro_f1': 0.7220995651039799, 'accuracy': 0.7166456670866582, 'advanced_macro_f1': 0.7087287637290434, 'static_macro_f1': 0.7213146091318968, 'delta_vs_advanced': 0.013370801374936514, 'delta_vs_static': 0.0007849559720830568, 'changes_vs_advanced': 640}, {'direction': 'B_to_A', 'macro_f1': 0.7181706460482183, 'accuracy': 0.7226770070371966, 'advanced_macro_f1': 0.7136336390240811, 'static_macro_f1': 0.7217174620428055, 'delta_vs_advanced': 0.004537007024137285, 'delta_vs_static': -0.0035468159945871713, 'changes_vs_advanced': 735}]

## Feature Availability
- Uses current_prompt: yes
- Uses history assistant_action: yes
- Uses result_summary: yes
- Uses train labels at inference: no
- Uses future steps: no
- Uses full test batch: no

## Decision
Reject: class thresholds overfit validation halves.
