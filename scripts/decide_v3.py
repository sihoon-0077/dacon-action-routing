import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import fit_temperature
from src.constants import ACTIONS, ACTION_TO_ID, SEED
from src.io_utils import get_session_id, load_train, write_csv, write_json
from src.metrics import classwise_rows, expected_calibration_error


def reconstruct_val_ids(data_dir, val_labels):
    samples = load_train(data_dir)
    y = np.asarray([ACTION_TO_ID[sample["action"]] for sample in samples], dtype=np.int64)
    groups = np.asarray([get_session_id(sample["id"]) for sample in samples], dtype=object)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    _, val_idx = next(splitter.split(np.arange(len(samples)), y, groups=groups))
    if len(val_idx) != len(val_labels):
        raise ValueError(f"val size mismatch split={len(val_idx)} labels={len(val_labels)}")
    if not np.array_equal(y[val_idx], val_labels):
        raise ValueError("reconstructed split labels do not match val_labels.npy")
    return [samples[i]["id"] for i in val_idx], groups[val_idx]


def pred_with_bias_from_log_probs(log_probs, bias):
    return (log_probs + bias[None, :]).argmax(axis=1)


def macro_ids(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)


def optimize_bias_spec(log_probs, y_true, seed=SEED):
    rng = np.random.default_rng(seed)
    grid = np.arange(-1.5, 1.5001, 0.05)
    bias = np.zeros(log_probs.shape[1], dtype=np.float64)
    best = macro_ids(y_true, pred_with_bias_from_log_probs(log_probs, bias))
    history = [{"sweep": 0, "macro_f1": best}]
    order = list(range(log_probs.shape[1]))
    for sweep in range(1, 4):
        rng.shuffle(order)
        for cls in order:
            best_value = bias[cls]
            best_local = best
            for value in grid:
                cand = bias.copy()
                cand[cls] = value
                score = macro_ids(y_true, pred_with_bias_from_log_probs(log_probs, cand))
                if score > best_local + 1e-12:
                    best_local = score
                    best_value = value
            bias[cls] = best_value
            best = best_local
        history.append({"sweep": sweep, "macro_f1": best})
    return bias, history


def evaluate_bias(log_probs, y_true, bias):
    pred = pred_with_bias_from_log_probs(log_probs, bias)
    return {
        "macro_f1": macro_ids(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--logits", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", default="artifacts_v3/reports/decision")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logits = np.load(args.logits)
    y = np.load(args.labels).astype(np.int64)
    val_ids, val_groups = reconstruct_val_ids(args.data_dir, y)

    raw_probs = softmax(logits, axis=1)
    raw_ll = log_loss(y, raw_probs, labels=list(range(len(ACTIONS))))
    raw_ece, _ = expected_calibration_error(y, raw_probs)
    temp, probs, cal_ll = fit_temperature(logits, y, labels=list(range(len(ACTIONS))))
    cal_ece, _ = expected_calibration_error(y, probs)
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))

    argmax_pred = probs.argmax(axis=1)
    full_bias, full_history = optimize_bias_spec(log_probs, y, seed=SEED)
    full_pred = pred_with_bias_from_log_probs(log_probs, full_bias)

    half_splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    a_idx, b_idx = next(half_splitter.split(np.arange(len(y)), y, groups=val_groups))
    bias_a, hist_a = optimize_bias_spec(log_probs[a_idx], y[a_idx], seed=SEED)
    bias_b, hist_b = optimize_bias_spec(log_probs[b_idx], y[b_idx], seed=SEED + 1)
    base_b = evaluate_bias(log_probs[b_idx], y[b_idx], np.zeros(len(ACTIONS)))
    base_a = evaluate_bias(log_probs[a_idx], y[a_idx], np.zeros(len(ACTIONS)))
    a2b = evaluate_bias(log_probs[b_idx], y[b_idx], bias_a)
    b2a = evaluate_bias(log_probs[a_idx], y[a_idx], bias_b)

    cross_gain_a2b = a2b["macro_f1"] - base_b["macro_f1"]
    cross_gain_b2a = b2a["macro_f1"] - base_a["macro_f1"]
    cross_gain_avg = (cross_gain_a2b + cross_gain_b2a) / 2.0
    adopt_bias = cross_gain_avg >= 0.003

    pred_names = [ACTIONS[i] for i in full_pred]
    true_names = [ACTIONS[i] for i in y]
    write_csv(
        out_dir / "classwise_biased.csv",
        classwise_rows(true_names, pred_names, ACTIONS),
        ["label", "precision", "recall", "f1", "support"],
    )
    write_csv(
        out_dir / "predictions_biased.csv",
        [
            {"id": sample_id, "y_true": ACTIONS[int(t)], "y_pred": ACTIONS[int(p)]}
            for sample_id, t, p in zip(val_ids, y, full_pred)
        ],
        ["id", "y_true", "y_pred"],
    )
    payload = {
        "temperature": float(temp),
        "raw_log_loss": float(raw_ll),
        "calibrated_log_loss": float(cal_ll),
        "raw_ece": float(raw_ece),
        "calibrated_ece": float(cal_ece),
        "f1_argmax": macro_ids(y, argmax_pred),
        "acc_argmax": accuracy_score(y, argmax_pred),
        "f1_biased": macro_ids(y, full_pred),
        "acc_biased": accuracy_score(y, full_pred),
        "bias": full_bias.tolist(),
        "bias_by_class": {action: float(value) for action, value in zip(ACTIONS, full_bias)},
        "bias_history": full_history,
        "crossval_gain_A2B": float(cross_gain_a2b),
        "crossval_gain_B2A": float(cross_gain_b2a),
        "crossval_gain_avg": float(cross_gain_avg),
        "adopt_bias_by_protocol": bool(adopt_bias),
        "valA_size": int(len(a_idx)),
        "valB_size": int(len(b_idx)),
        "classes": ACTIONS,
    }
    write_json(out_dir / "decision.json", payload)

    lines = [
        "# Phase 3/4 Decision v3",
        "",
        f"- temperature: `{temp:.6f}`",
        f"- raw log-loss: `{raw_ll:.6f}`",
        f"- calibrated log-loss: `{cal_ll:.6f}`",
        f"- raw ECE: `{raw_ece:.6f}`",
        f"- calibrated ECE: `{cal_ece:.6f}`",
        f"- argmax Macro-F1: `{payload['f1_argmax']:.6f}`",
        f"- biased Macro-F1: `{payload['f1_biased']:.6f}`",
        f"- cross gain A->B: `{cross_gain_a2b:.6f}`",
        f"- cross gain B->A: `{cross_gain_b2a:.6f}`",
        f"- adopt bias by protocol: `{adopt_bias}`",
    ]
    (out_dir / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {out_dir / 'decision.md'}")


if __name__ == "__main__":
    main()
