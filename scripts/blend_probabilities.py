import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import ACTIONS
from src.io_utils import load_train, write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-logits", required=True)
    parser.add_argument("--transformer-labels", required=True)
    parser.add_argument("--advanced-predictions", required=True)
    parser.add_argument("--out-dir", default="artifacts/reports/blend_sweep")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logits = np.load(args.transformer_logits)
    probs = softmax(logits, axis=1)
    labels = np.load(args.transformer_labels).astype(int)
    rows = list(csv.DictReader(open(args.advanced_predictions, encoding="utf-8")))
    adv_pred = [row["y_pred"] for row in rows]
    y_true = [row["y_true"] for row in rows]
    tr_pred = [ACTIONS[i] for i in probs.argmax(axis=1)]
    conf = probs.max(axis=1)
    if len(adv_pred) != len(tr_pred):
        raise ValueError("advanced predictions and transformer logits length mismatch")

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
    result_rows = []

    def add_result(name, pred):
        result_rows.append(
            {
                "name": name,
                "macro_f1": f1_score(y_true, pred, labels=ACTIONS, average="macro", zero_division=0),
                "accuracy": accuracy_score(y_true, pred),
                "changes_vs_advanced": sum(a != b for a, b in zip(adv_pred, pred)),
            }
        )

    add_result("advanced", adv_pred)
    add_result("transformer", tr_pred)
    for thr in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = [tp if tp in stronger and c >= thr else ap for ap, tp, c in zip(adv_pred, tr_pred, conf)]
        add_result(f"override_transformer_stronger_thr{thr}", pred)
        pred = [
            tp
            if ap in {"read_file", "grep_search", "list_directory", "glob_pattern"}
            and tp in {"read_file", "grep_search", "list_directory", "glob_pattern"}
            and c >= thr
            else ap
            for ap, tp, c in zip(adv_pred, tr_pred, conf)
        ]
        add_result(f"inspect_only_thr{thr}", pred)
    result_rows.sort(key=lambda x: x["macro_f1"], reverse=True)
    write_csv(out_dir / "blend_sweep.csv", result_rows, ["name", "macro_f1", "accuracy", "changes_vs_advanced"])
    lines = ["# Blend Sweep", "", "| name | Macro-F1 | accuracy | changes |", "|---|---:|---:|---:|"]
    for row in result_rows:
        lines.append(f"| `{row['name']}` | {row['macro_f1']:.6f} | {row['accuracy']:.6f} | {row['changes_vs_advanced']} |")
    (out_dir / "blend_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {out_dir / 'blend_sweep.md'}")


if __name__ == "__main__":
    main()
