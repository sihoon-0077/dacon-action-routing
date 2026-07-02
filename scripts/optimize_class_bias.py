import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import ACTIONS
from src.decision_bias import optimize_class_bias, predict_with_bias, save_bias
from src.io_utils import write_csv, write_json
from src.metrics import classwise_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0.002)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = np.load(args.probs)
    labels = np.load(args.labels).astype(int)
    raw_pred = probs.argmax(axis=1)
    bias, history = optimize_class_bias(probs, labels, labels=list(range(len(ACTIONS))), max_iter=args.max_iter, l2=args.l2)
    tuned_pred = predict_with_bias(probs, bias)
    save_bias(out_dir / "class_bias.json", bias, ACTIONS)
    write_json(out_dir / "bias_history.json", history)
    write_csv(
        out_dir / "classwise_before.csv",
        classwise_rows(labels, raw_pred, list(range(len(ACTIONS)))),
        ["label", "precision", "recall", "f1", "support"],
    )
    write_csv(
        out_dir / "classwise_after.csv",
        classwise_rows(labels, tuned_pred, list(range(len(ACTIONS)))),
        ["label", "precision", "recall", "f1", "support"],
    )
    payload = {
        "raw_macro_f1": float(f1_score(labels, raw_pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)),
        "bias_macro_f1": float(f1_score(labels, tuned_pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)),
        "raw_accuracy": float(accuracy_score(labels, raw_pred)),
        "bias_accuracy": float(accuracy_score(labels, tuned_pred)),
    }
    write_json(out_dir / "bias_metrics.json", payload)
    lines = [
        "# Class Bias Tuning Report",
        "",
        f"- raw Macro-F1: `{payload['raw_macro_f1']:.6f}`",
        f"- bias Macro-F1: `{payload['bias_macro_f1']:.6f}`",
        f"- raw accuracy: `{payload['raw_accuracy']:.6f}`",
        f"- bias accuracy: `{payload['bias_accuracy']:.6f}`",
    ]
    (out_dir / "bias_tuning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {out_dir / 'bias_tuning_report.md'}")


if __name__ == "__main__":
    main()
