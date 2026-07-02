import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_recall_fscore_support


def macro_f1(y_true, y_pred, labels=None):
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def classwise_rows(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return [
        {
            "label": label,
            "precision": float(pi),
            "recall": float(ri),
            "f1": float(fi),
            "support": int(si),
        }
        for label, pi, ri, fi, si in zip(labels, p, r, f, s)
    ]


def confusion_rows(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, true_label in enumerate(labels):
        row = {"true": true_label}
        for j, pred_label in enumerate(labels):
            row[pred_label] = int(cm[i, j])
        rows.append(row)
    return rows


def safe_log_loss(y_true_ids, probs, labels):
    return float(log_loss(y_true_ids, probs, labels=labels))


def expected_calibration_error(y_true_ids, probs, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true_ids).astype(float)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-12)
        if not mask.any():
            bucket_rows.append({"lo": float(lo), "hi": float(hi), "count": 0, "accuracy": None, "confidence": None})
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(acc - avg_conf)
        bucket_rows.append({"lo": float(lo), "hi": float(hi), "count": int(mask.sum()), "accuracy": acc, "confidence": avg_conf})
    return float(ece), bucket_rows
