import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import ACTIONS
from src.io_utils import write_csv, write_json


def score(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=ACTIONS, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--advanced-predictions", required=True)
    parser.add_argument("--out-dir", default="artifacts_v3/reports/ensemble")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logits = np.load(args.logits)
    label_ids = np.load(args.labels).astype(np.int64)
    y_true = [ACTIONS[i] for i in label_ids]
    decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
    temp = float(decision["temperature"])
    bias = np.asarray(decision["bias"], dtype=np.float64)
    probs = softmax(logits / temp, axis=1)
    log_scores = np.log(np.clip(probs, 1e-12, 1.0)) + bias[None, :]
    tf_ids = log_scores.argmax(axis=1)
    tf_pred = [ACTIONS[i] for i in tf_ids]
    tf_conf = probs.max(axis=1)

    rows = list(csv.DictReader(open(args.advanced_predictions, encoding="utf-8")))
    adv_pred = [row["y_pred"] for row in rows]
    adv_true = [row["y_true"] for row in rows]
    if adv_true != y_true:
        raise ValueError("advanced prediction validation order does not match transformer labels")

    stronger = {
        "read_file",
        "grep_search",
        "list_directory",
        "glob_pattern",
        "edit_file",
        "write_file",
        "apply_patch",
        "respond_only",
    }
    inspect = {"read_file", "grep_search", "list_directory", "glob_pattern"}
    result_rows = []

    def add(name, pred, note=""):
        payload = score(y_true, pred)
        payload.update(
            {
                "name": name,
                "changes_vs_advanced": int(sum(a != b for a, b in zip(adv_pred, pred))),
                "note": note,
            }
        )
        result_rows.append(payload)

    add("advanced_router", adv_pred, "safe submitted linear router")
    add("transformer_calibrated_biased", tf_pred, "mDeBERTa + T + bias")
    for thr in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = [tp if tp in stronger and conf >= thr else ap for ap, tp, conf in zip(adv_pred, tf_pred, tf_conf)]
        add(f"override_stronger_thr{thr}", pred, "override advanced when transformer predicts stronger-class action")
        pred = [
            tp if ap in inspect and tp in inspect and conf >= thr else ap
            for ap, tp, conf in zip(adv_pred, tf_pred, tf_conf)
        ]
        add(f"inspect_only_thr{thr}", pred, "override only inside inspect group")

    result_rows.sort(key=lambda row: row["macro_f1"], reverse=True)
    write_csv(out_dir / "ensemble_v3.csv", result_rows, ["name", "macro_f1", "accuracy", "changes_vs_advanced", "note"])
    write_json(
        out_dir / "ensemble_v3.json",
        {
            "best": result_rows[0],
            "rows": result_rows,
            "advanced_proba_available": False,
            "advanced_proba_note": "advanced_router.pkl uses LinearSVC for coarse routing and does not expose calibrated predict_proba; E3 probability blending was not run.",
        },
    )
    lines = [
        "# Phase 5 Ensemble v3",
        "",
        "- E3 probability blending with advanced router was skipped because the current advanced artifact does not expose calibrated `predict_proba`.",
        "",
        "| name | Macro-F1 | accuracy | changes |",
        "|---|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(f"| `{row['name']}` | {row['macro_f1']:.6f} | {row['accuracy']:.6f} | {row['changes_vs_advanced']} |")
    (out_dir / "ensemble_v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {out_dir / 'ensemble_v3.md'}")


if __name__ == "__main__":
    main()
