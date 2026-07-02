import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import ACTIONS, ACTION_TO_ID
from src.decision_bias import optimize_class_bias, predict_with_bias, save_bias
from src.io_utils import get_session_id, load_train, write_csv, write_json
from src.metrics import classwise_rows, confusion_rows
from src.state_features import SIGNATURE_LEVELS, build_signature


def counts_to_probs(counter, global_counts, alpha):
    total = sum(counter.values())
    global_total = sum(global_counts.values())
    if total == 0:
        return np.array([global_counts[action] / max(global_total, 1) for action in ACTIONS], dtype=np.float64)
    probs = []
    for action in ACTIONS:
        prior = global_counts[action] / max(global_total, 1)
        probs.append((counter[action] + alpha * prior) / (total + alpha))
    return np.array(probs, dtype=np.float64)


def oof_for_level(samples, labels, groups, level, n_folds=5, alpha=1.0):
    y_ids = np.array([ACTION_TO_ID[y] for y in labels], dtype=np.int64)
    probs = np.zeros((len(samples), len(ACTIONS)), dtype=np.float64)
    pred = np.empty(len(samples), dtype=object)
    splitter = GroupKFold(n_splits=n_folds)
    for train_idx, val_idx in splitter.split(np.arange(len(samples)), y_ids, groups):
        global_counts = Counter(labels[i] for i in train_idx)
        table = defaultdict(Counter)
        for i in train_idx:
            table[build_signature(samples[i], level)][labels[i]] += 1
        global_probs = counts_to_probs(Counter(), global_counts, alpha=0.0)
        for i in val_idx:
            key = build_signature(samples[i], level)
            if key in table:
                row = counts_to_probs(table[key], global_counts, alpha)
            else:
                row = global_probs
            probs[i] = row
            pred[i] = ACTIONS[int(row.argmax())]
    return probs, pred, y_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out-dir", default="artifacts/reports/policy_ceiling")
    args = parser.parse_args()

    samples = load_train(args.data_dir)
    labels = [sample["action"] for sample in samples]
    groups = np.array([get_session_id(sample["id"]) for sample in samples])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_payload = None
    for level in SIGNATURE_LEVELS:
        probs, pred, y_ids = oof_for_level(samples, labels, groups, level, args.n_folds, args.alpha)
        raw_macro = f1_score(labels, pred, labels=ACTIONS, average="macro", zero_division=0)
        raw_acc = accuracy_score(labels, pred)
        bias, history = optimize_class_bias(probs, y_ids, labels=list(range(len(ACTIONS))), max_iter=4, l2=0.002)
        bias_pred_ids = predict_with_bias(probs, bias)
        bias_pred = [ACTIONS[i] for i in bias_pred_ids]
        bias_macro = f1_score(labels, bias_pred, labels=ACTIONS, average="macro", zero_division=0)
        row = {
            "level": level,
            "argmax_macro_f1": raw_macro,
            "argmax_accuracy": raw_acc,
            "bias_tuned_macro_f1": bias_macro,
            "bias_tuned_accuracy": accuracy_score(labels, bias_pred),
            "bias_history_last": history[-1]["macro_f1"],
        }
        rows.append(row)
        if best_payload is None or bias_macro > best_payload["row"]["bias_tuned_macro_f1"]:
            best_payload = {"row": row, "probs": probs, "pred": bias_pred, "bias": bias, "y_ids": y_ids}

    write_csv(out_dir / "ceiling_summary.csv", rows, list(rows[0].keys()))
    write_csv(out_dir / "classwise_best.csv", classwise_rows(labels, best_payload["pred"], ACTIONS), ["label", "precision", "recall", "f1", "support"])
    write_csv(out_dir / "confusion_best.csv", confusion_rows(labels, best_payload["pred"], ACTIONS), ["true"] + ACTIONS)
    save_bias(out_dir / "best_class_bias.json", best_payload["bias"], ACTIONS)
    write_json(out_dir / "ceiling_summary.json", {"rows": rows, "best": best_payload["row"]})

    lines = [
        "# Policy Ceiling Estimate",
        "",
        f"- folds: `{args.n_folds}` GroupKFold by session",
        f"- smoothing alpha: `{args.alpha}`",
        "",
        "## Summary",
        "",
        "| level | argmax Macro-F1 | bias tuned Macro-F1 |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| `{row['level']}` | {row['argmax_macro_f1']:.6f} | {row['bias_tuned_macro_f1']:.6f} |")
    best = best_payload["row"]
    lines += [
        "",
        "## Conclusion",
        "",
        f"- Best argmax Macro-F1: `{max(r['argmax_macro_f1'] for r in rows):.6f}`",
        f"- Best bias-tuned Macro-F1: `{best['bias_tuned_macro_f1']:.6f}`",
        f"- Best signature level: `{best['level']}`",
        "- Interpretation: this is a memorized observed-state ceiling with train-fold backoff; if it is low, hidden state or representation capacity matters more than lookup-style signatures.",
    ]
    (out_dir / "ceiling_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {out_dir / 'ceiling_summary.md'}")


if __name__ == "__main__":
    main()
