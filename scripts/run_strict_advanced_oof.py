import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.constants import ALL_CLASSES, LABEL2ID
from pipeline_v4.common.data_io import load_fold_rows, load_train_samples
from script import ADVANCED_ACTION_TO_GROUP, ADVANCED_GROUP_TO_ACTIONS, advanced_group_text, compact_flags_text
from scripts.run_distill_step2 import (
    Status,
    advanced_feature_matrix,
    classwise_rows,
    metrics_from_probs,
    predict_advanced_with_scores,
    write_csv,
    write_json,
)
from train_advanced_router import build_vectorizer, train_pair_resolvers, transition_counts


SEED = 42


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_fold_map(path):
    return {row["id"]: int(row["fold"]) for row in load_fold_rows(path)}


def train_advanced_artifact(samples, labels, args, fold):
    y = np.asarray(labels, dtype=object)
    y_group = np.asarray([ADVANCED_ACTION_TO_GROUP[label] for label in y], dtype=object)
    print(f"[{now()}] fold={fold} train advanced rows={len(samples)}", flush=True)

    start = time.time()
    coarse_vectorizer = build_vectorizer(max_features=args.coarse_features, min_df=args.min_df)
    x_coarse = coarse_vectorizer.fit_transform([compact_flags_text(sample) for sample in samples])
    coarse_model = LinearSVC(C=args.coarse_c, class_weight="balanced", random_state=SEED, dual="auto", max_iter=2500)
    coarse_model.fit(x_coarse, y_group)
    print(f"[{now()}] fold={fold} coarse shape={x_coarse.shape} sec={time.time() - start:.1f}", flush=True)

    group_vectorizers = {}
    group_models = {}
    for group, actions in ADVANCED_GROUP_TO_ACTIONS.items():
        start = time.time()
        idx = np.where(np.isin(y, actions))[0]
        texts = [advanced_group_text(samples[i], group) for i in idx]
        vectorizer = build_vectorizer(max_features=args.group_features, min_df=args.min_df)
        x = vectorizer.fit_transform(texts)
        model = LogisticRegression(
            max_iter=args.group_max_iter,
            C=args.group_c,
            class_weight="balanced",
            random_state=SEED,
        )
        model.fit(x, y[idx])
        group_vectorizers[group] = vectorizer
        group_models[group] = model
        print(
            f"[{now()}] fold={fold} group={group} rows={len(idx)} shape={x.shape} sec={time.time() - start:.1f}",
            flush=True,
        )

    pair_resolvers = train_pair_resolvers(samples, y, max_features=args.pair_features)
    return {
        "kind": "advanced_action_router_v1_strict_oof",
        "classes": ALL_CLASSES,
        "group_to_actions": ADVANCED_GROUP_TO_ACTIONS,
        "action_to_group": ADVANCED_ACTION_TO_GROUP,
        "coarse_vectorizer": coarse_vectorizer,
        "coarse_model": coarse_model,
        "group_vectorizers": group_vectorizers,
        "group_models": group_models,
        "transition_last2": transition_counts(samples, y),
        "global_counts": dict(Counter(y)),
        "pair_resolvers": pair_resolvers,
        "config": {
            "group_text_variant": "specialized_x2",
            "prior_key": "last2_action",
            "prior_alpha": args.prior_alpha,
            "prior_smooth": args.prior_smooth,
            "pair_threshold": args.pair_threshold,
        },
        "validation_split": "pipeline_v4/folds fold-held-out strict OOF",
        "experiment": "strict_advanced_oof",
        "created_at": now(),
        "fold": fold,
    }


def write_summary(path, metrics, args, fold_metrics):
    lines = [
        "# Strict Advanced OOF Summary",
        "",
        f"- finished_at: `{now()}`",
        f"- Macro-F1: `{metrics['macro_f1']:.6f}`",
        f"- accuracy: `{metrics['accuracy']:.6f}`",
        f"- NLL: `{metrics['nll']:.6f}`",
        f"- rows: `{metrics['rows']}`",
        f"- smoke_rows: `{args.smoke_rows}`",
        "",
        "## Fold Metrics",
        "",
        "| Fold | Rows | Macro-F1 | Accuracy | NLL |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in fold_metrics:
        lines.append(
            f"| `{row['fold']}` | `{row['rows']}` | `{row['macro_f1']:.6f}` | `{row['accuracy']:.6f}` | `{row['nll']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision Note",
            "",
            "- This cache is strict: each row is predicted by an advanced router trained without that row's fold.",
            "- Use this directory with `scripts/run_distill_step2.py --advanced-oof-dir ...` for leak-safe distill validation.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--fold-file", default="pipeline_v4/folds/fold_assignments.csv")
    parser.add_argument("--out-dir", default="artifacts/advanced_oof_strict")
    parser.add_argument("--report-dir", default="reports/advanced_oof_strict")
    parser.add_argument("--coarse-features", type=int, default=220_000)
    parser.add_argument("--group-features", type=int, default=180_000)
    parser.add_argument("--pair-features", type=int, default=80_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--coarse-c", type=float, default=2.0)
    parser.add_argument("--group-c", type=float, default=2.0)
    parser.add_argument("--group-max-iter", type=int, default=800)
    parser.add_argument("--prior-alpha", type=float, default=0.3)
    parser.add_argument("--prior-smooth", type=float, default=1.0)
    parser.add_argument("--pair-threshold", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--smoke-rows", type=int, default=0)
    parser.add_argument("--save-fold-models", action="store_true")
    args = parser.parse_args()

    np.random.seed(SEED)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runner_pid.txt").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    status = Status(report_dir / "status.md", total_units=5.2)
    status.update("start", "loading train and folds", done_units=0.05)

    samples = load_train_samples(args.data_dir)
    if args.smoke_rows:
        samples = samples[: args.smoke_rows]
    labels = [sample["action"] for sample in samples]
    y = np.asarray([LABEL2ID[label] for label in labels], dtype=np.int64)
    ids = [sample["id"] for sample in samples]
    fold_map = load_fold_map(args.fold_file)
    folds = np.asarray([fold_map[sample["id"]] for sample in samples], dtype=np.int64)

    n = len(samples)
    scores = np.zeros((n, len(ALL_CLASSES)), dtype=np.float32)
    probs = np.zeros((n, len(ALL_CLASSES)), dtype=np.float32)
    pred = np.zeros(n, dtype=np.int64)
    group = np.zeros(n, dtype=np.int64)
    fold_metrics = []

    for fold in range(5):
        val_idx = np.where(folds == fold)[0]
        train_idx = np.where(folds != fold)[0]
        if len(val_idx) == 0:
            continue
        status.update(
            "train_fold",
            f"fold={fold} train={len(train_idx)} val={len(val_idx)}",
            done_units=fold,
            current_fold=fold,
        )
        fold_train_samples = [samples[i] for i in train_idx]
        fold_labels = [labels[i] for i in train_idx]
        artifact = train_advanced_artifact(fold_train_samples, fold_labels, args, fold)
        if args.save_fold_models:
            model_path = out_dir / f"fold_{fold}" / "advanced_router.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(artifact, model_path, compress=3)

        status.update("predict_fold", f"fold={fold} val={len(val_idx)}", done_units=fold + 0.75, current_fold=fold)
        val_samples = [samples[i] for i in val_idx]
        fold_scores, fold_probs, fold_pred, fold_group = predict_advanced_with_scores(
            val_samples, artifact, batch_size=args.batch_size
        )
        scores[val_idx] = fold_scores
        probs[val_idx] = fold_probs
        pred[val_idx] = fold_pred
        group[val_idx] = fold_group
        fm = metrics_from_probs(y[val_idx], fold_probs)
        fm["fold"] = fold
        fm["rows"] = int(len(val_idx))
        fold_metrics.append(fm)
        write_json(report_dir / f"fold_{fold}_metrics.json", fm)
        print(f"[{now()}] fold={fold} strict advanced f1={fm['macro_f1']:.6f}", flush=True)

    features = advanced_feature_matrix(probs, pred, group)
    np.save(out_dir / "advanced_oof_scores.npy", scores)
    np.save(out_dir / "advanced_oof_probs.npy", probs)
    np.save(out_dir / "advanced_oof_pred.npy", pred)
    np.save(out_dir / "advanced_oof_group.npy", group)
    np.save(out_dir / "advanced_features.npy", features)
    write_json(out_dir / "class_order.json", ALL_CLASSES)
    write_json(out_dir / "ids.json", ids)
    metrics = metrics_from_probs(y, probs)
    metrics["rows"] = int(n)
    write_json(report_dir / "metrics.json", metrics)
    write_csv(
        report_dir / "classwise_f1.csv",
        classwise_rows(y, pred),
        ["class", "precision", "recall", "f1", "support"],
    )
    write_summary(report_dir / "SUMMARY.md", metrics, args, fold_metrics)
    status.update(
        "finished",
        f"strict advanced f1={metrics['macro_f1']:.6f}",
        done_units=5.2,
        macro_f1=f"{metrics['macro_f1']:.6f}",
    )


if __name__ == "__main__":
    main()
