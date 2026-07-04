# SupCon/LCL + INTENT v2.1 Cheap Probe Result

Date: 2026-07-04

## Purpose

This experiment checked whether the proposed SupCon/LCL and INTENT v2.1 directions are worth GPU training.

No new transformer was trained. The goal was to run the cheap validation steps first:

- S1: confusion-pair metrics and group Macro-F1
- S2: confidence/margin profile for major confusion pairs
- S3: INTENT v2.1 Tier-B LogReg feature test

## Inputs

- Base transformer diagnostics: `mdeberta384_v2_384_5e` fold0 logits/probs
- Fold: `fold0`
- INTENT baseline: `compact_flags_text` + LogisticRegression
- INTENT experiment: `compact_flags_text + [INTENT]` + LogisticRegression
- Report directory: `reports/supcon_intent_probe/`

## S1 Metrics

| Metric | Value |
|---|---:|
| inspect pair error mean | `0.140393` |
| communicate pair error mean | `0.190631` |
| inspect4 Macro-F1 | `0.581960` |
| communicate4 Macro-F1 | `0.685132` |
| execute3 Macro-F1 | `0.698268` |
| modify3 Macro-F1 | `0.962013` |

The main bottlenecks are still inspect and communication/execute boundary classes. Modify classes are near ceiling.

## S2 Margin Profile

| Pair | Error Rate | Error Mean Margin | Low Margin `<0.1` | High Margin `>0.3` |
|---|---:|---:|---:|---:|
| `read_file <-> grep_search` | `0.209332` | `0.273700` | `0.333744` | `0.296798` |
| `read_file <-> list_directory` | `0.205269` | `0.146146` | `0.440285` | `0.076649` |
| `ask_user <-> plan_task` | `0.229323` | `0.467365` | `0.090164` | `0.700820` |
| `run_tests <-> lint_or_typecheck` | `0.164248` | `0.470043` | `0.093023` | `0.744186` |
| `run_bash <-> run_tests` | `0.154976` | `0.582753` | `0.056140` | `0.807018` |

Interpretation:

- `read_file <-> list_directory` has many low-margin errors, so pair-bias or calibration may help.
- `ask_user <-> plan_task`, `run_tests <-> lint_or_typecheck`, and `run_bash <-> run_tests` are mostly high-margin errors. These look like representation/loss problems, not simple threshold problems.
- This keeps the SupCon/LCL hypothesis alive.

## M2 Centroid Proxy

Saved pooled embeddings were not available, so this run used fold0 logits as a cheap centroid proxy.

| Group | Avg Logit-Centroid Cosine Distance |
|---|---:|
| inspect4 | `0.031538` |
| communicate4 | `0.576690` |
| execute3 | `0.037018` |
| modify3 | `0.878548` |

Caveat:

- `communicate4` average is inflated because `respond_only` is far from the triad.
- Inside the triad, `ask_user`, `plan_task`, and `web_search` remain close.
- True centroid separation needs pooled encoder embeddings from a separate forward pass.

## S3 INTENT Tier-B

| Model | Overall Macro-F1 | Communicate4 Macro-F1 | Web F1 | Ask F1 | Plan F1 | Respond F1 |
|---|---:|---:|---:|---:|---:|---:|
| base compact LogReg | `0.658805` | `0.650895` | `0.521295` | `0.548094` | `0.538000` | `0.996190` |
| INTENT v2.1 LogReg | `0.658710` | `0.649840` | `0.522034` | `0.545951` | `0.535183` | `0.996190` |
| delta | `-0.000095` | `-0.001055` | `+0.000739` | `-0.002144` | `-0.002817` | `0.000000` |

Decision:

- INTENT v2.1 fails the Tier-B gate.
- Target gate was communicate4 `+0.004`; actual result was `-0.001055`.
- Do not spend a standalone GPU run on INTENT tags.
- INTENT may only be reused as a cheap side feature inside a stronger adopted serializer later.

## Final Decision

| Track | Decision | Reason |
|---|---|---|
| INTENT v2.1 | Reject | No Tier-B lift; communicate4 decreased. |
| Pair-bias | Maybe later | Useful mainly for low-margin pairs like `read_file <-> list_directory`. |
| SupCon/LCL | Keep as candidate | High-margin confusion pairs suggest representation/loss issues. |

Recommended next step:

1. Run **A0** first: class-balanced sampler only, no SupCon.
2. Continue to A1/A2/A3 only if A0 is not worse than the current 3epoch baseline by more than `0.003`.
3. If A0 passes, test:
   - A1: weak SupCon
   - A2: medium SupCon
   - A3: LCL weighted negative loss

Do not jump straight to the full 4-run SupCon/LCL matrix before A0 confirms that the sampler itself is safe.
