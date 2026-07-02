# I7. Turn-Bucket Bias Verdict

## Result
- Tier A: pass
- Tier B: fail
- Tier C: not_applicable
- Final: reject

## Metrics
- Base Macro-F1: 0.6897865805656586
- New Macro-F1: None
- Delta: -0.008993616513466252
- Target class/group delta: None
- Worst class drop: None
- Half split stability: [{'direction': 'A_to_B', 'macro_f1': 0.6784513350589041, 'accuracy': 0.6907461850762985, 'base_macro_f1': 0.6909069499794646, 'delta': -0.012455614920560465}, {'direction': 'B_to_A', 'macro_f1': 0.6826685733390466, 'accuracy': 0.7063047536981186, 'base_macro_f1': 0.6882001914454187, 'delta': -0.005531618106372038}]

## Feature Availability
- Uses current_prompt: no
- Uses history assistant_action: no
- Uses result_summary: no
- Uses train labels at inference: no
- Uses future steps: no
- Uses full test batch: no

## Decision
Reject: cross-half turn bias gain is too small or unstable.
