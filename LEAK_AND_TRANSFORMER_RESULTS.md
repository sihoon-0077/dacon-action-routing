# Leak And Transformer Experiment Results

Date: 2026-07-02

Base model: `advanced_router.pkl`

Base GroupShuffle Macro-F1: `0.711324`

## Track A: Session-Scoped Lookup

Script: `session_lookup_experiment.py`

Output: `reports/session_lookup_validation.json`

Validation results:

| Experiment | Macro-F1 | Delta | Coverage | Covered Accuracy | Changed Good / Bad |
|---|---:|---:|---:|---:|---:|
| `A2-1_val_self` | `0.973726` | `+0.262402` | `12217/14000` (`0.866`) | `1.000` | `3786 / 1` |
| `A2-2_train_only` | `0.711324` | `+0.000000` | `0/14000` | `0.000` | `0 / 0` |
| `A2-3_train_plus_val` | `0.973726` | `+0.262402` | `12217/14000` (`0.866`) | `1.000` | `3786 / 1` |
| `A2-4_all_train_optimistic` | `0.973726` | `+0.262402` | `12217/14000` (`0.866`) | `1.000` | `3786 / 1` |

Public sample probe:

- `train_hits`: `5/5`
- `train_plus_test_hits`: `5/5`

Interpretation:

- Session-scoped lookup is the real signal. Cross-session prompt lookup is noisy, but `(session_id, prompt)` exact match is essentially deterministic in validation.
- `train_only` has zero validation coverage under GroupShuffle because validation sessions are disjoint.
- If hidden test contains multiple steps from the same sessions, scanning the full evaluation `test.jsonl` history can dramatically improve score.
- If hidden test contains exactly one step per session, this safely falls back to `advanced_router`.

Submission artifact:

- `submit_lookup_probe.zip`
- Size: about `51.0 MB`
- Contents:
  - `script.py`
  - `requirements.txt`
  - `model/advanced_router.pkl`
  - `data/train.jsonl`
- Local verification: passed; public sample logged `hits=5/5`, `changed=2`.

## Track B: Transformer Probe

Script: `transformer_action_routing.py`

Extra dependencies: `requirements-transformer.txt`

Local GPU:

- NVIDIA GeForce RTX 4060 Ti
- 8GB VRAM

Tokenizer / length report using `microsoft/mdeberta-v3-base`:

| Metric | Value |
|---|---:|
| samples | `70000` |
| mean tokens | `306.1` |
| p50 | `314` |
| p90 | `479` |
| p95 | `514` |
| p99 | `580` |
| max | `703` |
| over 320 | `34180` |
| over 384 | `25816` |
| over 512 | `3566` |

Finding:

- `max_len=320` truncates nearly half the data.
- `max_len=384` is likely better for T4 16GB, but local 8GB is already near the limit with batch size 2.

Transformer smoke/probe results:

| Run | Model | Data | Epochs | Config | Macro-F1 | Accuracy | Time |
|---|---|---:|---:|---|---:|---:|---:|
| `B-smoke-mdeberta-1k-stable` | mDeBERTa | `1k` | 1 | `max_len=256`, balanced | `0.020147` | `0.1642` | `48s` |
| `B-smoke-xlm-1k` | XLM-R | `1k` | 1 | `max_len=256`, balanced | `0.020147` | `0.1642` | `34s` |
| `B-probe-mdeberta-10k` | mDeBERTa | `10k` | 1 | `max_len=320`, balanced, lr `2e-5` | `0.090248` | `0.2098` | `276s` |
| `B-probe-mdeberta-10k-lr5e5-none-3e` | mDeBERTa | `10k` | 3 | `max_len=320`, no weight, lr `5e-5` | `0.317084` | `0.3916` | `823s` |
| `B-probe-mdeberta-10k-nowfirst-lr5e5-none-3e` | mDeBERTa | `10k` | 3 | `[NOW]` first, `max_len=320`, no weight, lr `5e-5` | `0.490004` | `0.5879` | `548s` best epoch |

Epoch curve for the best 10k probe:

- epoch 1: Macro-F1 `0.089734`
- epoch 2: Macro-F1 `0.277246`
- epoch 3: Macro-F1 `0.317084`

Bug diagnosis and fix:

- The original serializer placed `[NOW] {current_prompt}` at the end.
- Hugging Face tokenizers truncate from the right by default, so long examples kept old history and dropped the target prompt.
- On the same 10k diagnostic subset, `4937/10000` examples exceeded 320 tokens and `4393/10000` lost `[NOW]` after truncation.
- Label index order was checked and matched `ALL_CLASSES`; the issue was not a label-order bug.
- After moving `[NOW]` to the front and placing recent history first, `[NOW]` missing count became `0/10000`.

Fixed-layout epoch curve:

- epoch 1: Macro-F1 `0.390567`, accuracy `0.5156`
- epoch 2: Macro-F1 `0.490004`, accuracy `0.5879`
- epoch 3: Macro-F1 `0.483550`, accuracy `0.5863`

Implementation notes:

- mDeBERTa initially loaded as FP16 in this environment and produced NaN loss. Fix: force `model_dtype=float32`, use AMP only for forward, and use GradScaler.
- Full 70k / 3 epoch run is feasible but expensive on local 8GB. Runtime estimate from 10k probe is roughly 90-100 minutes at `max_len=320`, batch size 2.
- The fixed-layout transformer is still not yet competitive on the 10k probe, but the previous `0.317` result was invalid because current prompts were often truncated away.

## Recommendation

Use `submit_lookup_probe.zip` as the immediate diagnostic submission if the competition rules allow full test-batch feature scanning.

Keep `submit_advanced_router.zip` as the conservative no-transductive fallback.

Transformer is technically running now, but the current probe does not justify replacing the router. The next useful transformer step is a full 70k run with:

- `microsoft/mdeberta-v3-base`
- `max_len=384` on T4 16GB if memory allows
- `loss_weight=none`
- `lr=5e-5`
- `epochs=3`
- `model_dtype=float32`
- `amp=fp16`
