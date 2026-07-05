import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {c: i for i, c in enumerate(ALL_CLASSES)}
GROUPS = {
    "inspect4": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "modify3": ["edit_file", "write_file", "apply_patch"],
    "execute3": ["run_bash", "run_tests", "lint_or_typecheck"],
    "communicate4": ["ask_user", "plan_task", "web_search", "respond_only"],
    "communicate3": ["ask_user", "plan_task", "web_search"],
}
GROUP_IDS = {name: np.array([LABEL2ID[c] for c in classes], dtype=np.int64) for name, classes in GROUPS.items()}
OUT_DIR = ROOT / "reports" / "leap_diagnostics"


def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_y_and_ids():
    labels = {}
    with open(ROOT / "data" / "train_labels.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]
    ids = []
    with open(ROOT / "data" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["id"])
    y = np.asarray([LABEL2ID[labels[sid]] for sid in ids], dtype=np.int64)
    return ids, y


def load_folds(ids):
    fold_map = {}
    with open(ROOT / "pipeline_v4" / "folds" / "fold_assignments.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fold_map[row["id"]] = int(row["fold"])
    return np.asarray([fold_map[sid] for sid in ids], dtype=np.int64)


def metrics(y, pred):
    return {
        "macro_f1": float(f1_score(y, pred, labels=list(range(len(ALL_CLASSES))), average="macro", zero_division=0)),
        "accuracy": float((y == pred).mean()),
    }


def class_rows(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(len(ALL_CLASSES))), zero_division=0)
    return [
        {"class": cls, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, cls in enumerate(ALL_CLASSES)
    ]


def group_macro(y, pred, group):
    ids = GROUP_IDS[group]
    return float(f1_score(y, pred, labels=list(ids), average="macro", zero_division=0))


def group_binary_f1(y, pred, group):
    ids = GROUP_IDS[group]
    return float(f1_score(np.isin(y, ids), np.isin(pred, ids), zero_division=0))


def fold_deltas(y, base_pred, new_pred, folds):
    rows = []
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        b = metrics(y[mask], base_pred[mask])["macro_f1"]
        n = metrics(y[mask], new_pred[mask])["macro_f1"]
        rows.append(n - b)
    return rows


def net_correct(y, base_pred, new_pred):
    before = base_pred == y
    after = new_pred == y
    return int((after & ~before).sum() - (before & ~after).sum())


def apply_override(base_pred, source_pred, source_conf, action_ids, threshold):
    mask = np.isin(source_pred, action_ids) & (source_conf >= threshold)
    out = base_pred.copy()
    out[mask] = source_pred[mask]
    return out, mask


def score_override(y, folds, base_name, base_pred, source_name, source_pred, source_conf, group_name, threshold):
    action_ids = GROUP_IDS[group_name]
    pred, mask = apply_override(base_pred, source_pred, source_conf, action_ids, threshold)
    base_m = metrics(y, base_pred)
    new_m = metrics(y, pred)
    fds = fold_deltas(y, base_pred, pred, folds)
    row = {
        "base": base_name,
        "source": source_name,
        "group": group_name,
        "threshold": float(threshold),
        "base_macro_f1": base_m["macro_f1"],
        "new_macro_f1": new_m["macro_f1"],
        "macro_delta": new_m["macro_f1"] - base_m["macro_f1"],
        "changed": int((pred != base_pred).sum()),
        "override_count": int(mask.sum()),
        "net_correct": net_correct(y, base_pred, pred),
        "min_fold_delta": float(min(fds)),
        "max_fold_delta": float(max(fds)),
        "mean_fold_delta": float(np.mean(fds)),
    }
    for group in ["inspect4", "modify3", "execute3", "communicate3"]:
        row[f"{group}_delta"] = group_macro(y, pred, group) - group_macro(y, base_pred, group)
    return row, pred


def prob_blend_rows(y, folds, base_name, base_probs, source_name, source_probs):
    rows = []
    base_pred = base_probs.argmax(axis=1)
    for group_name, ids in GROUP_IDS.items():
        if group_name == "communicate3":
            continue
        for w_source in [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]:
            probs = base_probs.copy()
            probs[:, ids] = (1.0 - w_source) * base_probs[:, ids] + w_source * source_probs[:, ids]
            pred = probs.argmax(axis=1)
            fds = fold_deltas(y, base_pred, pred, folds)
            rows.append({
                "base": base_name,
                "source": source_name,
                "group": group_name,
                "w_source": float(w_source),
                "base_macro_f1": metrics(y, base_pred)["macro_f1"],
                "new_macro_f1": metrics(y, pred)["macro_f1"],
                "macro_delta": metrics(y, pred)["macro_f1"] - metrics(y, base_pred)["macro_f1"],
                "changed": int((pred != base_pred).sum()),
                "net_correct": net_correct(y, base_pred, pred),
                "min_fold_delta": float(min(fds)),
                "mean_fold_delta": float(np.mean(fds)),
            })
    return rows


def model_summary_rows(y, models):
    rows = []
    for name, probs in models.items():
        pred = probs.argmax(axis=1)
        row = {"model": name, **metrics(y, pred)}
        for group in GROUPS:
            row[f"{group}_macro_f1"] = group_macro(y, pred, group)
            row[f"{group}_binary_f1"] = group_binary_f1(y, pred, group)
        rows.append(row)
    return rows


def oracle_rows(y, models):
    rows = []
    preds = {name: probs.argmax(axis=1) for name, probs in models.items()}
    names = list(preds.keys())
    for group_name, ids in GROUP_IDS.items():
        mask = np.isin(y, ids)
        if not mask.any():
            continue
        any_correct = np.zeros(mask.sum(), dtype=bool)
        for name in names:
            any_correct |= preds[name][mask] == y[mask]
        rows.append({
            "group": group_name,
            "rows": int(mask.sum()),
            "best_single_group_macro": max(group_macro(y, pred, group_name) for pred in preds.values()),
            "any_model_correct_rate": float(any_correct.mean()),
            "missed_by_all": int((~any_correct).sum()),
        })
    any_correct_all = np.zeros(len(y), dtype=bool)
    for pred in preds.values():
        any_correct_all |= pred == y
    rows.append({
        "group": "all14",
        "rows": int(len(y)),
        "best_single_group_macro": max(metrics(y, pred)["macro_f1"] for pred in preds.values()),
        "any_model_correct_rate": float(any_correct_all.mean()),
        "missed_by_all": int((~any_correct_all).sum()),
    })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids, y = load_y_and_ids()
    folds = load_folds(ids)
    advanced = np.load(ROOT / "artifacts" / "advanced_oof_strict" / "advanced_oof_probs.npy").astype(np.float32)
    teacher = np.load(ROOT / "artifacts" / "distill_step2_strict" / "teacher_oof" / "teacher_oof_probs.npy").astype(np.float32)
    d2m5 = np.load(ROOT / "reports" / "distill_step2_strict" / "mlp_oof" / "D2-M5" / "oof_probs.npy").astype(np.float32)
    bias_payload = read_json(ROOT / "reports" / "distill_step2_strict" / "blends" / "best_config.json")
    bias = np.asarray(
        [bias_payload["bias"]["bias_by_class"].get(cls, 0.0) for cls in ALL_CLASSES],
        dtype=np.float32,
    )
    strict_blend = 0.5 * d2m5 + 0.5 * advanced
    strict_blend_bias_pred = (np.log(np.clip(strict_blend, 1e-12, 1.0)) + bias[None, :]).argmax(axis=1)
    strict_blend_bias_probs = np.zeros_like(strict_blend)
    strict_blend_bias_probs[np.arange(len(y)), strict_blend_bias_pred] = 1.0
    models = {
        "advanced_strict_oof": advanced,
        "teacher_oof": teacher,
        "d2m5_student_oof": d2m5,
        "strict_blend_05": strict_blend,
        "strict_blend_bias_pred": strict_blend_bias_probs,
    }

    model_rows = model_summary_rows(y, models)
    write_csv(OUT_DIR / "model_summary.csv", model_rows, list(model_rows[0].keys()))
    for name, probs in models.items():
        write_csv(OUT_DIR / f"classwise_{name}.csv", class_rows(y, probs.argmax(axis=1)), ["class", "precision", "recall", "f1", "support"])

    override_rows = []
    best_preds = {}
    thresholds = [0.0, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95]
    for base_name in ["advanced_strict_oof", "d2m5_student_oof", "strict_blend_05", "strict_blend_bias_pred"]:
        base_pred = models[base_name].argmax(axis=1)
        for source_name in ["teacher_oof", "advanced_strict_oof", "d2m5_student_oof"]:
            if source_name == base_name:
                continue
            source_probs = models[source_name]
            source_pred = source_probs.argmax(axis=1)
            source_conf = source_probs.max(axis=1)
            for group_name in ["modify3", "execute3", "inspect4", "communicate3", "communicate4"]:
                for thr in thresholds:
                    row, pred = score_override(
                        y, folds, base_name, base_pred, source_name, source_pred, source_conf, group_name, thr
                    )
                    override_rows.append(row)
                    key = (row["base"], row["source"], row["group"], row["threshold"])
                    best_preds[key] = pred
    write_csv(OUT_DIR / "override_sweep.csv", override_rows, list(override_rows[0].keys()))

    blend_rows = []
    for base_name in ["advanced_strict_oof", "d2m5_student_oof", "strict_blend_05"]:
        for source_name in ["teacher_oof", "advanced_strict_oof", "d2m5_student_oof"]:
            if source_name == base_name:
                continue
            blend_rows.extend(prob_blend_rows(y, folds, base_name, models[base_name], source_name, models[source_name]))
    write_csv(OUT_DIR / "group_prob_blend_sweep.csv", blend_rows, list(blend_rows[0].keys()))

    oracle = oracle_rows(y, {k: v for k, v in models.items() if k != "strict_blend_bias_pred"})
    write_csv(OUT_DIR / "oracle_pool_ceiling.csv", oracle, list(oracle[0].keys()))

    positive = [
        row for row in override_rows
        if row["macro_delta"] > 0 and row["min_fold_delta"] >= -0.0005 and row["changed"] > 0
    ]
    positive = sorted(positive, key=lambda r: (r["macro_delta"], r["min_fold_delta"], r["net_correct"]), reverse=True)
    positive_blend = [
        row for row in blend_rows
        if row["macro_delta"] > 0 and row["min_fold_delta"] >= -0.0005 and row["changed"] > 0
    ]
    positive_blend = sorted(positive_blend, key=lambda r: (r["macro_delta"], r["min_fold_delta"], r["net_correct"]), reverse=True)

    lines = [
        "# Leap Diagnostics",
        "",
        "## Model Summary",
        "",
        "| model | Macro-F1 | modify3 | inspect4 | execute3 | communicate3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(model_rows, key=lambda r: r["macro_f1"], reverse=True):
        lines.append(
            f"| `{row['model']}` | `{row['macro_f1']:.6f}` | `{row['modify3_macro_f1']:.6f}` | "
            f"`{row['inspect4_macro_f1']:.6f}` | `{row['execute3_macro_f1']:.6f}` | `{row['communicate3_macro_f1']:.6f}` |"
        )
    lines.extend(["", "## Best Stable Hard Overrides", ""])
    if positive:
        lines.append("| base | source | group | thr | delta | new | min_fold | changed | net |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for row in positive[:12]:
            lines.append(
                f"| `{row['base']}` | `{row['source']}` | `{row['group']}` | `{row['threshold']:.2f}` | "
                f"`{row['macro_delta']:.6f}` | `{row['new_macro_f1']:.6f}` | `{row['min_fold_delta']:.6f}` | "
                f"`{row['changed']}` | `{row['net_correct']}` |"
            )
    else:
        lines.append("- No stable positive hard override under current thresholds.")
    lines.extend(["", "## Best Stable Group Probability Blends", ""])
    if positive_blend:
        lines.append("| base | source | group | w_source | delta | new | min_fold | changed | net |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for row in positive_blend[:12]:
            lines.append(
                f"| `{row['base']}` | `{row['source']}` | `{row['group']}` | `{row['w_source']:.2f}` | "
                f"`{row['macro_delta']:.6f}` | `{row['new_macro_f1']:.6f}` | `{row['min_fold_delta']:.6f}` | "
                f"`{row['changed']}` | `{row['net_correct']}` |"
            )
    else:
        lines.append("- No stable positive group probability blend under current grid.")
    lines.extend(["", "## Oracle Pool Ceiling", ""])
    lines.append("| group | rows | best single | any-model correct rate | missed by all |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in oracle:
        lines.append(
            f"| `{row['group']}` | `{row['rows']}` | `{row['best_single_group_macro']:.6f}` | "
            f"`{row['any_model_correct_rate']:.6f}` | `{row['missed_by_all']}` |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `teacher_oof` is fold-held-out, so its strong modify3 score is not label leakage.",
        "- A public-submit jump requires making that teacher signal available under the 10 minute limit, usually by candidate-gated transformer inference or distillation.",
        "- Stable override rows require positive overall delta and no fold worse than `-0.0005`; looser rows are research-only.",
    ])
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT_DIR / "summary.md").resolve())


if __name__ == "__main__":
    main()
