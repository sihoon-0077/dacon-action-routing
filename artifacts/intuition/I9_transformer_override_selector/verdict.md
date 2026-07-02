# I9. Transformer Override Selector Verdict

## Result
- Tier A: not_applicable
- Tier B: fail
- Tier C: not_applicable
- Final: reject

## Metrics
- Base Macro-F1: 0.7113236414043568
- New Macro-F1: None
- Delta: -0.02010356223656956
- Target class/group delta: -0.009768728025780604
- Worst class drop: None
- Half split stability: [{'direction': 'A_to_B', 'macro_f1': 0.7025748761443642, 'accuracy': 0.7061458770824584, 'advanced_macro_f1': 0.7087287637290434, 'static_macro_f1': 0.7213146091318968, 'delta_vs_advanced': -0.006153887584679163, 'delta_vs_static': -0.01873973298753262, 'changes_vs_advanced': 1160, 'selector_precision': 0.3905172413793103, 'selector_recall': 0.9847826086956522}, {'direction': 'B_to_A', 'macro_f1': 0.700250070557199, 'accuracy': 0.713772799080856, 'advanced_macro_f1': 0.7136336390240811, 'static_macro_f1': 0.7217174620428055, 'delta_vs_advanced': -0.013383568466882045, 'delta_vs_static': -0.021467391485606502, 'changes_vs_advanced': 1046, 'selector_precision': 0.381453154875717, 'selector_recall': 0.9755501222493888}]

## Feature Availability
- Uses current_prompt: yes
- Uses history assistant_action: yes
- Uses result_summary: yes
- Uses train labels at inference: no
- Uses future steps: no
- Uses full test batch: no

## Decision
Reject for current submit: selector does not beat static override under strict half validation.
