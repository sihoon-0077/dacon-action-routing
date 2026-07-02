# I3. Structural GBDT Verdict

## Result
- Tier A: not_applicable
- Tier B: fail
- Tier C: not_applicable
- Final: analysis_only

## Metrics
- Base Macro-F1: 0.7087287637290434
- New Macro-F1: 0.3178064693795316
- Delta: -0.39092229434951176
- Target class/group delta: None
- Worst class drop: None
- Half split stability: A_to_B only

## Feature Availability
- Uses current_prompt: yes
- Uses history assistant_action: yes
- Uses result_summary: yes
- Uses train labels at inference: no
- Uses future steps: no
- Uses full test batch: no

## Decision
Use as diversity probe only; not a submit member unless probability blend later proves useful.
