# Granite EP2 Submit Candidate

- created_at: `2026-07-09`
- zip: `granite_ep2.zip`
- source checkpoint: `pipeline_v4/artifacts/models/granite311_v2_384_3e_gate/fold_0`
- checkpoint epoch: `2`
- fold0 Macro-F1: `0.7267799199659711`
- fold0 NLL: `0.6799960617048597`
- fold0 accuracy: `0.7409699237300331`
- zip bytes: `567181967`
- unpacked bytes: `686134549`

## Inference Config

- base router: `model/advanced_router.pkl`
- transformer: Granite-311M multilingual R2
- serializer: `v2_2` patched into submit `script.py`
- max_len: `384`
- batch_size: `32`
- max_transformer_samples: `12000`
- prefilter actions: inspect + modify + execute + respond_only
- override threshold: `0.0`
- session lookup: disabled
- requirements: `transformers>=4.48.0`

## Smoke

- command: `CUDA_VISIBLE_DEVICES='' python granite_ep2/script.py`
- result: pass
- sample rows: `5`
- transformer selected: `4/5`
- changed: `2`

## Public Evaluation

- submitted_at: `2026-07-09 23:00:40`
- public submission id: `30338`
- public Macro-F1: `0.7093429258`
- server runtime: `4m 33s`

Comparison:

| Candidate | Public Macro-F1 | Runtime | Note |
|---|---:|---:|---|
| `cand_distill.zip` | `0.7174979343` | `2m 58s` | current defense |
| `v4ep5_384_20k.zip` | `0.7127296320` | `6m 05s` | stronger mDeBERTa public submit |
| `granite_ep2.zip` | `0.7093429258` | `4m 33s` | this submit |

## Risk

- This is a fold0 checkpoint trained on 80% of train, not a full-data model.
- The public score can differ because the transformer is candidate-capped at 12000 rows for runtime.
- The evaluation server default `transformers==4.46.3` may not support `modernbert`, so the zip includes `transformers>=4.48.0`.

## Decision

- Reject as a primary submit candidate.
- The local fold0 score `0.726780` did not transfer to public.
- Use this as evidence that larger encoder swaps need 5-fold OOF or full-data confirmation before spending more submissions.
