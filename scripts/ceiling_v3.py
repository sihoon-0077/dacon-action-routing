import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import ACTIONS, ACTION_TO_ID
from src.io_utils import load_train, write_csv, write_json
from src.metrics import classwise_rows
from src.state_features import SIGNATURE_LEVELS_V3, build_signature_v3


def counter_to_probs(counter, global_counts, alpha=1.0):
    total = sum(counter.values())
    global_total = max(sum(global_counts.values()), 1)
    if total == 0:
        return np.asarray([global_counts[action] / global_total for action in ACTIONS], dtype=np.float64)
    probs = []
    for action in ACTIONS:
        prior = global_counts[action] / global_total
        probs.append((counter[action] + alpha * prior) / (total + alpha))
    return np.asarray(probs, dtype=np.float64)


def expected_macro_f1(probs, pred_ids):
    scores = []
    for cls in range(len(ACTIONS)):
        p = probs[:, cls]
        mask = pred_ids == cls
        tp = float(p[mask].sum())
        fp = float((1.0 - p[mask]).sum())
        fn = float(p[~mask].sum())
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom <= 0 else 2.0 * tp / denom)
    return float(np.mean(scores))


def predict_with_bias(probs, bias):
    scores = np.log(np.clip(probs, 1e-12, 1.0)) + bias[None, :]
    return scores.argmax(axis=1)


def optimize_expected_bias(probs, max_sweeps=3, seed=42):
    rng = np.random.default_rng(seed)
    grid = np.arange(-1.5, 1.5001, 0.05)
    bias = np.zeros(probs.shape[1], dtype=np.float64)
    pred = predict_with_bias(probs, bias)
    best = expected_macro_f1(probs, pred)
    history = [{"sweep": 0, "expected_macro_f1": best}]
    order = list(range(probs.shape[1]))
    for sweep in range(1, max_sweeps + 1):
        rng.shuffle(order)
        for cls in order:
            local_best = best
            local_value = bias[cls]
            for value in grid:
                candidate = bias.copy()
                candidate[cls] = value
                score = expected_macro_f1(probs, predict_with_bias(probs, candidate))
                if score > local_best + 1e-12:
                    local_best = score
                    local_value = value
            bias[cls] = local_value
            best = local_best
        history.append({"sweep": sweep, "expected_macro_f1": best})
    return bias, history


def build_level_probs(samples, labels, min_support=3, alpha=1.0):
    global_counts = Counter(labels)
    level_tables = {}
    for level in SIGNATURE_LEVELS_V3:
        table = defaultdict(Counter)
        for sample, label in zip(samples, labels):
            table[build_signature_v3(sample, level)][label] += 1
        level_tables[level] = table

    prev_probs = None
    outputs = {}
    global_probs = counter_to_probs(Counter(), global_counts, alpha=0.0)
    for level in SIGNATURE_LEVELS_V3:
        table = level_tables[level]
        probs = np.zeros((len(samples), len(ACTIONS)), dtype=np.float64)
        supports = np.zeros(len(samples), dtype=np.int32)
        used_backoff = np.zeros(len(samples), dtype=bool)
        for i, sample in enumerate(samples):
            counter = table[build_signature_v3(sample, level)]
            support = sum(counter.values())
            supports[i] = support
            if support < min_support:
                used_backoff[i] = True
                probs[i] = prev_probs[i] if prev_probs is not None else global_probs
            else:
                probs[i] = counter_to_probs(counter, global_counts, alpha)
        outputs[level] = {
            "probs": probs,
            "support": supports,
            "used_backoff": used_backoff,
            "n_states": len(table),
        }
        prev_probs = probs
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="artifacts_v3/reports/ceiling")
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    samples = load_train(args.data_dir)
    labels = [sample["action"] for sample in samples]
    y_ids = np.asarray([ACTION_TO_ID[label] for label in labels], dtype=np.int64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = build_level_probs(samples, labels, args.min_support, args.alpha)
    rows = []
    best = None
    for level, payload in outputs.items():
        probs = payload["probs"]
        argmax_ids = probs.argmax(axis=1)
        bias, history = optimize_expected_bias(probs)
        bias_ids = predict_with_bias(probs, bias)
        row = {
            "level": level,
            "n_states": int(payload["n_states"]),
            "coverage_support_ge_min": float((~payload["used_backoff"]).mean()),
            "median_support": float(np.median(payload["support"])),
            "expected_macro_f1_argmax": expected_macro_f1(probs, argmax_ids),
            "expected_macro_f1_bias": expected_macro_f1(probs, bias_ids),
            "empirical_macro_f1_argmax": f1_score(y_ids, argmax_ids, labels=list(range(len(ACTIONS))), average="macro", zero_division=0),
            "empirical_macro_f1_bias": f1_score(y_ids, bias_ids, labels=list(range(len(ACTIONS))), average="macro", zero_division=0),
            "empirical_accuracy_bias": accuracy_score(y_ids, bias_ids),
        }
        rows.append(row)
        if best is None or row["expected_macro_f1_bias"] > best["row"]["expected_macro_f1_bias"]:
            best = {"row": row, "pred_ids": bias_ids, "bias": bias, "history": history}

    write_csv(out_dir / "ceiling_v3_summary.csv", rows, list(rows[0].keys()))
    write_csv(
        out_dir / "ceiling_v3_classwise_best.csv",
        classwise_rows(y_ids, best["pred_ids"], list(range(len(ACTIONS)))),
        ["label", "precision", "recall", "f1", "support"],
    )
    payload = {"rows": rows, "best": best["row"], "best_bias": best["bias"].tolist(), "bias_history": best["history"]}
    write_json(out_dir / "ceiling_v3.json", payload)

    lines = [
        "# Phase 1 Ceiling v3",
        "",
        f"- min_support: `{args.min_support}`",
        f"- alpha: `{args.alpha}`",
        "",
        "| level | states | coverage | expected argmax | expected bias | empirical bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['level']}` | {row['n_states']} | {row['coverage_support_ge_min']:.3f} | "
            f"{row['expected_macro_f1_argmax']:.6f} | {row['expected_macro_f1_bias']:.6f} | "
            f"{row['empirical_macro_f1_bias']:.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Best expected Macro-F1: `{best['row']['expected_macro_f1_bias']:.6f}` at `{best['row']['level']}`.",
        "- This is an optimistic policy-recovery ceiling because the distribution table is estimated on the full train set.",
    ]
    (out_dir / "ceiling_v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {out_dir / 'ceiling_v3.md'}")


if __name__ == "__main__":
    main()
