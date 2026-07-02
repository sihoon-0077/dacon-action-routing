import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.constants import ALL_CLASSES, SEED, session_of


def predict(log_probs, bias):
    return (log_probs + bias[None, :]).argmax(axis=1)


def macro(y, pred):
    return float(f1_score(y, pred, labels=list(range(len(ALL_CLASSES))), average="macro", zero_division=0))


def optimize_bias(oof_probs, y, prior, seed=SEED, grid=None, n_sweeps=3):
    if grid is None:
        grid = np.arange(-1.5, 1.51, 0.05)
    log_probs = np.log(np.clip(oof_probs, 1e-12, 1.0))
    best_tau, best_f1 = 0.0, -1.0
    for tau in [0.0, 0.25, 0.5, 0.75, 1.0]:
        b0 = -tau * np.log(prior)
        b0 = b0 - b0.mean()
        f1 = macro(y, predict(log_probs, b0))
        if f1 > best_f1:
            best_tau, best_f1 = tau, f1
    bias = -best_tau * np.log(prior)
    bias -= bias.mean()

    rng = np.random.default_rng(seed)
    order = np.arange(len(ALL_CLASSES))
    for _ in range(n_sweeps):
        rng.shuffle(order)
        for c in order:
            best_value, best_score = bias[c], macro(y, predict(log_probs, bias))
            for g in grid:
                cand = bias.copy()
                cand[c] = g
                score = macro(y, predict(log_probs, cand))
                if score > best_score + 1e-12:
                    best_value, best_score = g, score
            bias[c] = best_value
    return best_tau, bias, macro(y, predict(log_probs, bias))


def classwise(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(len(ALL_CLASSES))), zero_division=0)
    return [
        {"class": cls, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, cls in enumerate(ALL_CLASSES)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--artifact-dir", default="pipeline_v4/artifacts")
    args = parser.parse_args()
    artifact = Path(args.artifact_dir)
    oof_dir = artifact / "oof" / args.run
    out_dir = artifact / "decision" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    probs = np.load(oof_dir / "oof_probs.npy")
    y = np.load(oof_dir / "oof_y.npy").astype(int)
    ids = (oof_dir / "oof_ids.txt").read_text(encoding="utf-8").splitlines()
    prior = np.bincount(y, minlength=len(ALL_CLASSES)).astype(float) + 1.0
    prior /= prior.sum()
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    base_pred = probs.argmax(axis=1)
    f1_before = macro(y, base_pred)

    groups = np.asarray([session_of(sample_id) for sample_id in ids], dtype=object)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    a_idx, b_idx = next(splitter.split(np.arange(len(y)), y, groups=groups))
    tau_a, bias_a, _ = optimize_bias(probs[a_idx], y[a_idx], prior, seed=SEED)
    tau_b, bias_b, _ = optimize_bias(probs[b_idx], y[b_idx], prior, seed=SEED + 1)
    base_a = macro(y[a_idx], base_pred[a_idx])
    base_b = macro(y[b_idx], base_pred[b_idx])
    a2b = macro(y[b_idx], predict(log_probs[b_idx], bias_a))
    b2a = macro(y[a_idx], predict(log_probs[a_idx], bias_b))
    gain_a2b = a2b - base_b
    gain_b2a = b2a - base_a
    adopted = gain_a2b > 0 and gain_b2a > 0 and (gain_a2b + gain_b2a) / 2 >= 0.003

    tau, bias, f1_after = optimize_bias(probs, y, prior, seed=SEED)
    pred_after = predict(log_probs, bias)
    payload = {
        "run": args.run,
        "tau_init": tau,
        "bias": [float(x) for x in bias],
        "bias_by_class": {cls: float(bias[i]) for i, cls in enumerate(ALL_CLASSES)},
        "f1_before": f1_before,
        "f1_after": f1_after,
        "crossval_A2B": gain_a2b,
        "crossval_B2A": gain_b2a,
        "adopted": bool(adopted),
        "classes": ALL_CLASSES,
        "classwise_before": classwise(y, base_pred),
        "classwise_after": classwise(y, pred_after),
    }
    (out_dir / "bias.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
