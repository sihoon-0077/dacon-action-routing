import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.constants import ALL_CLASSES, LABEL2ID, session_of
from pipeline_v4.common.data_io import load_labels


RUN = "mdeberta384_v2_384_5e"
OOF_DIR = ROOT / "pipeline_v4" / "artifacts" / "oof" / RUN
OUT = ROOT / "reports" / "cycle3_oof_rcheck_supcon"
TARGET_PAIRS = [
    ("ask_user", "plan_task"),
    ("run_tests", "lint_or_typecheck"),
    ("run_bash", "run_tests"),
    ("read_file", "grep_search"),
    ("read_file", "list_directory"),
]
CONF_BINS = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.85), (0.85, 1.01)]


def macro(y, pred):
    return float(f1_score(y, pred, labels=list(range(len(ALL_CLASSES))), average="macro", zero_division=0))


def fit_temperature(logits, y):
    def obj(temp):
        return log_loss(y, softmax(logits / temp, axis=1), labels=list(range(len(ALL_CLASSES))))

    before = obj(1.0)
    result = minimize_scalar(obj, bounds=(0.5, 5.0), method="bounded")
    temp = float(result.x)
    after = obj(temp)
    return temp, before, after


def load_fold0():
    logits = np.load(OOF_DIR / "fold_0_logits.npy")
    y = np.load(OOF_DIR / "fold_0_y.npy").astype(int)
    ids = (OOF_DIR / "fold_0_ids.txt").read_text(encoding="utf-8").splitlines()
    return ids, logits, y


def top_arrays(probs):
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    p1 = probs[np.arange(len(probs)), top1]
    p2 = probs[np.arange(len(probs)), top2]
    return top1, top2, p1, p2, p1 - p2


def same_pair_mask(top1, top2, pair):
    a, b = LABEL2ID[pair[0]], LABEL2ID[pair[1]]
    return ((top1 == a) & (top2 == b)) | ((top1 == b) & (top2 == a))


def r_check(ids, probs, y):
    top1, top2, p1, _, margin = top_arrays(probs)
    rows = []
    summary = []
    for pair in TARGET_PAIRS:
        pair_mask = same_pair_mask(top1, top2, pair)
        pair_rows = []
        for lo, hi in CONF_BINS:
            mask = pair_mask & (p1 >= lo) & (p1 < hi)
            n = int(mask.sum())
            if n:
                observed = float((top1[mask] == y[mask]).mean())
                mean_p = float(p1[mask].mean())
                mean_margin = float(margin[mask].mean())
                correct = int((top1[mask] == y[mask]).sum())
            else:
                observed = mean_p = mean_margin = 0.0
                correct = 0
            midpoint = (lo + hi) / 2
            row = {
                "pair": f"{pair[0]}<->{pair[1]}",
                "bin": f"{lo:.2f}-{hi:.2f}",
                "n": n,
                "correct": correct,
                "expected_midpoint": midpoint,
                "mean_p1": mean_p,
                "observed_acc": observed,
                "gap_midpoint_minus_observed": midpoint - observed if n else 0.0,
                "gap_meanp_minus_observed": mean_p - observed if n else 0.0,
                "mean_margin": mean_margin,
            }
            rows.append(row)
            pair_rows.append(row)
        valid = [r for r in pair_rows if r["n"] >= 20]
        weighted_gap = (
            sum(r["gap_meanp_minus_observed"] * r["n"] for r in valid) / max(sum(r["n"] for r in valid), 1)
            if valid
            else 0.0
        )
        high_gap_bins = sum(1 for r in valid if r["gap_meanp_minus_observed"] >= 0.15)
        low_gap_bins = sum(1 for r in valid if abs(r["gap_meanp_minus_observed"]) <= 0.05)
        pair_mask_count = int(pair_mask.sum())
        summary.append(
            {
                "pair": f"{pair[0]}<->{pair[1]}",
                "decision_region_n": pair_mask_count,
                "weighted_gap_meanp_minus_observed": weighted_gap,
                "valid_bins": len(valid),
                "high_gap_bins_ge_0_15": high_gap_bins,
                "low_gap_bins_abs_le_0_05": low_gap_bins,
                "mean_margin": float(margin[pair_mask].mean()) if pair_mask.any() else 0.0,
                "top1_acc_in_region": float((top1[pair_mask] == y[pair_mask]).mean()) if pair_mask.any() else 0.0,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary)


def predict_with_pair_bias(probs, pair=("read_file", "list_directory"), d=0.0, margin_thr=0.1):
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    top1, top2, p1, p2, margin = top_arrays(probs)
    mask = same_pair_mask(top1, top2, pair) & (margin < margin_thr)
    biased = logp.copy()
    biased[mask, LABEL2ID[pair[0]]] += d
    return biased.argmax(axis=1), mask


def pair_bias_grid(ids, probs, y):
    groups = np.array([session_of(x) for x in ids], dtype=object)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    a_idx, b_idx = next(splitter.split(np.arange(len(y)), y, groups=groups))
    base_pred = probs.argmax(axis=1)
    base_all = macro(y, base_pred)
    base_a = macro(y[a_idx], base_pred[a_idx])
    base_b = macro(y[b_idx], base_pred[b_idx])
    rows = []
    for margin_thr in [0.05, 0.10, 0.15, 0.20]:
        for d in np.round(np.arange(-0.30, 0.301, 0.05), 2):
            pred_all, mask_all = predict_with_pair_bias(probs, d=float(d), margin_thr=float(margin_thr))
            pred_a, _ = predict_with_pair_bias(probs[a_idx], d=float(d), margin_thr=float(margin_thr))
            pred_b, _ = predict_with_pair_bias(probs[b_idx], d=float(d), margin_thr=float(margin_thr))
            rows.append(
                {
                    "d_to_read_file": float(d),
                    "margin_thr": float(margin_thr),
                    "macro_all": macro(y, pred_all),
                    "delta_all": macro(y, pred_all) - base_all,
                    "macro_A": macro(y[a_idx], pred_a),
                    "delta_A": macro(y[a_idx], pred_a) - base_a,
                    "macro_B": macro(y[b_idx], pred_b),
                    "delta_B": macro(y[b_idx], pred_b) - base_b,
                    "changed_all": int((pred_all != base_pred).sum()),
                    "eligible_all": int(mask_all.sum()),
                }
            )
    df = pd.DataFrame(rows)
    df["half_min_delta"] = df[["delta_A", "delta_B"]].min(axis=1)
    df["half_avg_delta"] = df[["delta_A", "delta_B"]].mean(axis=1)
    return df.sort_values(["half_min_delta", "delta_all"], ascending=[False, False]), base_all


def oof_status():
    rows = []
    root = ROOT / "pipeline_v4" / "artifacts" / "oof"
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        folds = []
        total = 0
        for k in range(5):
            ids_path = run_dir / f"fold_{k}_ids.txt"
            logits_path = run_dir / f"fold_{k}_logits.npy"
            if ids_path.exists() and logits_path.exists():
                n = len(ids_path.read_text(encoding="utf-8").splitlines())
                folds.append(k)
                total += n
        rows.append({"run": run_dir.name, "folds_present": ",".join(map(str, folds)), "n_rows": total, "complete_5fold": len(folds) == 5})
    return pd.DataFrame(rows)


def write_summary(temp, nll_before, nll_after, r_summary, bias_df, oof_df):
    supcon_pairs = ["ask_user<->plan_task", "run_tests<->lint_or_typecheck", "run_bash<->run_tests"]
    supcon_trigger = r_summary[(r_summary["pair"].isin(supcon_pairs)) & (r_summary["weighted_gap_meanp_minus_observed"] >= 0.15)]
    best_bias = bias_df.iloc[0].to_dict()
    lines = [
        "# Cycle3 OOF / R-Check / SupCon Gate Result",
        "",
        "## Temperature Calibration",
        f"- fold0 temperature: `{temp:.6f}`",
        f"- NLL before: `{nll_before:.6f}`",
        f"- NLL after: `{nll_after:.6f}`",
        "",
        "## R-Check Summary",
        "",
        "| Pair | N | Weighted Gap | Valid Bins | High Gap Bins | Top1 Acc | Mean Margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in r_summary.to_dict("records"):
        lines.append(
            f"| `{row['pair']}` | `{int(row['decision_region_n'])}` | `{row['weighted_gap_meanp_minus_observed']:.6f}` | "
            f"`{int(row['valid_bins'])}` | `{int(row['high_gap_bins_ge_0_15'])}` | `{row['top1_acc_in_region']:.6f}` | `{row['mean_margin']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- Gap is `mean predicted confidence - observed accuracy` inside the pair decision region.",
            "- Large positive gap means the model is confidently wrong or miscalibrated on that pair.",
            "",
            "## Pair-Bias Read/List",
            f"- best d_to_read_file: `{best_bias['d_to_read_file']}`",
            f"- best margin_thr: `{best_bias['margin_thr']}`",
            f"- delta_all: `{best_bias['delta_all']:.6f}`",
            f"- delta_A: `{best_bias['delta_A']:.6f}`",
            f"- delta_B: `{best_bias['delta_B']:.6f}`",
            f"- changed_all: `{int(best_bias['changed_all'])}`",
            "",
            "## OOF Status",
            "",
            "| Run | Folds Present | Rows | Complete 5-Fold |",
            "|---|---|---:|---|",
        ]
    )
    for row in oof_df.to_dict("records"):
        lines.append(f"| `{row['run']}` | `{row['folds_present']}` | `{int(row['n_rows'])}` | `{bool(row['complete_5fold'])}` |")
    lines.extend(["", "## Decision"])
    if len(supcon_trigger) >= 1:
        lines.append("- SupCon/LCL gate is open: at least one high-margin target pair has weighted gap >= `0.15`.")
    else:
        lines.append("- SupCon/LCL gate is not strongly opened by R-check: no target high-margin pair reached weighted gap >= `0.15`.")
    if best_bias["half_min_delta"] >= 0.001:
        lines.append("- Pair-bias read/list passes the strict half-split gate.")
    elif best_bias["delta_all"] > 0:
        lines.append("- Pair-bias read/list is analysis-only: full-fold delta is positive but half-split gate is weak.")
    else:
        lines.append("- Pair-bias read/list should be rejected for submit.")
    if not any(oof_df["complete_5fold"]):
        lines.append("- No complete 5-fold transformer OOF exists yet; OOF submission-line work requires additional fold training.")
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids, logits, y = load_fold0()
    temp, before, after = fit_temperature(logits, y)
    probs = softmax(logits / temp, axis=1)
    r_bins, r_summary = r_check(ids, probs, y)
    bias_df, base_all = pair_bias_grid(ids, probs, y)
    oof_df = oof_status()
    r_bins.to_csv(OUT / "r_check_bins.csv", index=False)
    r_summary.to_csv(OUT / "r_check_summary.csv", index=False)
    bias_df.to_csv(OUT / "pair_bias_read_list.csv", index=False)
    oof_df.to_csv(OUT / "oof_status.csv", index=False)
    payload = {"run": RUN, "temperature": temp, "nll_before": before, "nll_after": after, "base_macro_f1": base_all}
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = write_summary(temp, before, after, r_summary, bias_df, oof_df)
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
