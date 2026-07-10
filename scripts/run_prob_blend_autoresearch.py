import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import ACTIONS
from src.io_utils import load_train, write_csv

OUT = ROOT / "reports" / "prob_blend_autoresearch"
EPS = 1e-12


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def macro(y, pred):
    return float(f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0))


def fold_values(y, pred, folds):
    vals = []
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        vals.append(macro(y[mask], pred[mask]))
    return vals


def evaluate(name, scores, y, folds, base_score, w_adv="", w_teacher="", w_d2="", bias_scale=""):
    probs = softmax(scores, axis=1)
    pred = np.array([ACTIONS[i] for i in probs.argmax(axis=1)], dtype=object)
    fvals = fold_values(y, pred, folds)
    score = macro(y, pred)
    return {
        "name": name,
        "macro_f1": score,
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, labels=ACTIONS, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, labels=ACTIONS, average="macro", zero_division=0)),
        "delta": score - base_score,
        "w_adv": w_adv,
        "w_teacher": w_teacher,
        "w_d2": w_d2,
        "bias_scale": bias_scale,
        "changed": int((pred != y).sum()),
        "min_fold": min(fvals),
        "folds": ";".join(f"{v:.6f}" for v in fvals),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = load_train(ROOT / "data")
    y = np.array([s["action"] for s in samples], dtype=object)
    folds = np.load(ROOT / "artifacts" / "distill_step2_strict" / "fold_ids.npy")

    adv = np.load(ROOT / "artifacts" / "advanced_oof_strict" / "advanced_oof_probs.npy").astype(np.float32)
    teacher = np.load(ROOT / "artifacts" / "distill_step2_strict" / "teacher_oof" / "teacher_oof_probs.npy").astype(np.float32)
    d2 = np.load(ROOT / "reports" / "distill_step2_strict" / "mlp_oof" / "D2-M5" / "oof_probs.npy").astype(np.float32)
    cfg = read_json(ROOT / "reports" / "distill_step2_strict" / "blends" / "best_config.json")
    bias = np.array([float(cfg["bias"]["bias_by_class"].get(a, 0.0)) for a in ACTIONS], dtype=np.float32)

    base_blend = 0.5 * adv + 0.5 * d2
    base_scores = np.log(np.clip(base_blend, EPS, 1.0)) + bias[None, :]
    base_pred = np.array([ACTIONS[i] for i in softmax(base_scores, axis=1).argmax(axis=1)], dtype=object)
    base_score = macro(y, base_pred)

    rows = []
    rows.append(
        {
            "name": "base_strict_distill_bias",
            "macro_f1": base_score,
            "accuracy": float(accuracy_score(y, base_pred)),
            "precision_macro": float(precision_score(y, base_pred, labels=ACTIONS, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, base_pred, labels=ACTIONS, average="macro", zero_division=0)),
            "delta": 0.0,
            "w_adv": "",
            "w_teacher": "",
            "w_d2": "",
            "bias_scale": "",
            "changed": 0,
            "min_fold": min(fold_values(y, base_pred, folds)),
            "folds": ";".join(f"{v:.6f}" for v in fold_values(y, base_pred, folds)),
        }
    )

    logs = {
        "adv": np.log(np.clip(adv, EPS, 1.0)),
        "teacher": np.log(np.clip(teacher, EPS, 1.0)),
        "d2": np.log(np.clip(d2, EPS, 1.0)),
    }
    weight_grid = set()
    for wa in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for wt in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            wd = 1.0 - wa - wt
            if wd < -1e-9 or wd > 1.0:
                continue
            weight_grid.add((round(wa, 2), round(wt, 2), round(wd, 2)))

    # Local refinement around the current best a=0.40, teacher=0.50, d2=0.10.
    for wa in [0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46]:
        for wt in [0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56]:
            wd = 1.0 - wa - wt
            if wd < -1e-9 or wd > 1.0:
                continue
            weight_grid.add((round(wa, 2), round(wt, 2), round(wd, 2)))

    for wa, wt, wd in sorted(weight_grid):
        base = wa * logs["adv"] + wt * logs["teacher"] + wd * logs["d2"]
        for bs in [0.0, 0.5, 0.65, 0.70, 0.75, 0.80, 0.875, 1.0, 1.25, 1.5]:
            scores = base + bs * bias[None, :]
            rows.append(
                evaluate(
                    f"logblend_a{wa:.2f}_t{wt:.2f}_d{wd:.2f}_b{bs:.2f}",
                    scores,
                    y,
                    folds,
                    base_score,
                    w_adv=wa,
                    w_teacher=wt,
                    w_d2=wd,
                    bias_scale=bs,
                )
            )

    rows = sorted(rows, key=lambda r: (r["macro_f1"], r["min_fold"]), reverse=True)
    write_csv(
        OUT / "results.csv",
        rows,
        [
            "name",
            "macro_f1",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "delta",
            "w_adv",
            "w_teacher",
            "w_d2",
            "bias_scale",
            "changed",
            "min_fold",
            "folds",
        ],
    )
    best = rows[0]
    (OUT / "best_config.json").write_text(
        json.dumps(
            {
                "name": best["name"],
                "macro_f1": best["macro_f1"],
                "delta": best["delta"],
                "w_adv": best["w_adv"],
                "w_teacher": best["w_teacher"],
                "w_d2": best["w_d2"],
                "bias_scale": best["bias_scale"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Probability Blend Autoresearch",
        "",
        f"- base Macro-F1: `{base_score:.6f}`",
        f"- best: `{best['name']}`",
        f"- best Macro-F1: `{best['macro_f1']:.6f}`",
        f"- best delta: `{best['delta']:.6f}`",
        "",
        "## Top Variants",
        "",
        "| name | Macro-F1 | Acc | Prec | Rec | delta | min_fold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:20]:
        lines.append(
            f"| `{row['name']}` | `{row['macro_f1']:.6f}` | `{row['accuracy']:.6f}` | "
            f"`{row['precision_macro']:.6f}` | `{row['recall_macro']:.6f}` | "
            f"`{row['delta']:.6f}` | `{row['min_fold']:.6f}` |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]), flush=True)


if __name__ == "__main__":
    main()
