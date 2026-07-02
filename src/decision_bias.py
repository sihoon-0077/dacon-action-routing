import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def predict_with_bias(probs, bias):
    scores = np.log(np.clip(probs, 1e-12, 1.0)) + np.asarray(bias)[None, :]
    return scores.argmax(axis=1)


def optimize_class_bias(
    probs,
    y_true_ids,
    labels,
    max_iter=4,
    grid=None,
    l2=0.002,
):
    probs = np.asarray(probs, dtype=np.float64)
    y_true_ids = np.asarray(y_true_ids, dtype=np.int64)
    if grid is None:
        grid = np.linspace(-1.0, 1.0, 41)
    bias = np.zeros(probs.shape[1], dtype=np.float64)

    def score(candidate):
        pred = predict_with_bias(probs, candidate)
        f1 = f1_score(y_true_ids, pred, labels=labels, average="macro", zero_division=0)
        return float(f1 - l2 * np.square(candidate).mean())

    best_score = score(bias)
    history = [{"iter": 0, "score": best_score, "macro_f1": float(f1_score(y_true_ids, predict_with_bias(probs, bias), labels=labels, average="macro", zero_division=0))}]
    for it in range(1, max_iter + 1):
        improved = False
        for cls in range(probs.shape[1]):
            current = bias[cls]
            local_best = (best_score, current)
            for value in grid:
                cand = bias.copy()
                cand[cls] = value
                cand_score = score(cand)
                if cand_score > local_best[0] + 1e-12:
                    local_best = (cand_score, value)
            if local_best[1] != current:
                bias[cls] = local_best[1]
                best_score = local_best[0]
                improved = True
        pred = predict_with_bias(probs, bias)
        history.append({"iter": it, "score": best_score, "macro_f1": float(f1_score(y_true_ids, pred, labels=labels, average="macro", zero_division=0))})
        if not improved:
            break
    return bias, history


def save_bias(path, bias, action_names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {action: float(value) for action, value in zip(action_names, bias)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
