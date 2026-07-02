import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from compact_state_experiments import ALL_CLASSES, compact_text


ACTION_TO_GROUP = {
    "read_file": "inspect",
    "grep_search": "inspect",
    "list_directory": "inspect",
    "glob_pattern": "inspect",
    "edit_file": "modify",
    "write_file": "modify",
    "apply_patch": "modify",
    "run_bash": "execute",
    "run_tests": "execute",
    "lint_or_typecheck": "execute",
    "ask_user": "communicate",
    "plan_task": "communicate",
    "web_search": "communicate",
    "respond_only": "communicate",
}

GROUP_TO_ACTIONS = {
    "inspect": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "modify": ["edit_file", "write_file", "apply_patch"],
    "execute": ["run_bash", "run_tests", "lint_or_typecheck"],
    "communicate": ["ask_user", "plan_task", "web_search", "respond_only"],
}
GROUPS = list(GROUP_TO_ACTIONS)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels(path):
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def session_id(sample_id):
    return sample_id.rsplit("-step_", 1)[0]


def build_vectorizer(max_features=220_000):
    half = max_features // 2
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=half,
                    sublinear_tf=True,
                    lowercase=True,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=half,
                    sublinear_tf=True,
                    lowercase=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def align_scores(classes, scores, target_classes):
    out = np.full((scores.shape[0], len(target_classes)), -1e9, dtype=np.float32)
    for i, cls in enumerate(classes):
        out[:, target_classes.index(str(cls))] = scores[:, i]
    return out


def log_proba_scores(model, x, target_classes):
    return align_scores(model.classes_, np.log(model.predict_proba(x) + 1e-9), target_classes)


def decision_scores(model, x, target_classes):
    scores = model.decision_function(x)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return align_scores(model.classes_, scores, target_classes)


def normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    scores -= scores.mean(axis=1, keepdims=True)
    return scores / (scores.std(axis=1, keepdims=True) + 1e-6)


def top_margin(scores, classes):
    order = np.argsort(scores, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    margins = scores[np.arange(scores.shape[0]), top1] - scores[np.arange(scores.shape[0]), top2]
    preds = np.array([classes[i] for i in top1], dtype=object)
    return preds, margins


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=ALL_CLASSES, average="macro", zero_division=0)


def eval_pred(name, y_true, y_pred, results_path, extra=None):
    row = {
        "name": name,
        "status": "ok",
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "seconds": 0.0,
    }
    if extra:
        row.update(extra)
    append_jsonl(results_path, row)
    print(f"{name}: macro_f1={row['macro_f1']:.6f} acc={row['accuracy']:.6f}", flush=True)
    return row


def group_accuracy(y_true_action, group_pred, mask=None):
    true_group = np.array([ACTION_TO_GROUP[a] for a in y_true_action], dtype=object)
    if mask is None:
        mask = np.ones(len(true_group), dtype=bool)
    if mask.sum() == 0:
        return 0.0
    return float((true_group[mask] == group_pred[mask]).mean())


def fine_predict(fine_models, group_pred, x_val):
    preds = np.empty(x_val.shape[0], dtype=object)
    for group in GROUPS:
        idx = np.where(group_pred == group)[0]
        if len(idx):
            preds[idx] = fine_models[group].predict(x_val[idx])
    return preds


def fine_scores_full(fine_models, x_val):
    out = np.full((x_val.shape[0], len(ALL_CLASSES)), -1e9, dtype=np.float32)
    for group, actions in GROUP_TO_ACTIONS.items():
        scores = log_proba_scores(fine_models[group], x_val, actions)
        for i, action in enumerate(actions):
            out[:, ALL_CLASSES.index(action)] = scores[:, i]
    return out


def update_research(summary):
    with open("research.md", "a", encoding="utf-8") as f:
        f.write("\n## Margin Coarse/Fine Routing Experiments\n")
        f.write(f"- Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if summary.get("best"):
            b = summary["best"]
            f.write(f"- Best Macro-F1: `{b['macro_f1']:.6f}` via `{b['name']}`\n")
        f.write("- Key idea: 4-way coarse group model, margin threshold, fine group specialists, flat fallback.\n")
        f.write("\nTop results:\n")
        for row in summary.get("top10", [])[:10]:
            f.write(f"- `{row['macro_f1']:.6f}` `{row['name']}`\n")


def run_experiment(args):
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
    label_map = load_labels(Path(args.data_dir) / "train_labels.csv")
    y = np.array([label_map[s["id"]] for s in samples], dtype=object)
    y_group = np.array([ACTION_TO_GROUP[a] for a in y], dtype=object)
    texts = [compact_text(s, "flags") for s in samples]
    indices = np.arange(len(samples))

    if args.group_split:
        groups = np.array([session_id(s["id"]) for s in samples], dtype=object)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(indices, y, groups=groups))
        split_name = "group_shuffle"
    else:
        train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
        split_name = "stratified"

    x_train_text = [texts[i] for i in train_idx]
    x_val_text = [texts[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_group_train = y_group[train_idx]
    y_group_val = y_group[val_idx]

    print(f"split={split_name} train={len(train_idx)} val={len(val_idx)}", flush=True)
    vectorizer = build_vectorizer(args.max_features)
    start = time.time()
    x_train = vectorizer.fit_transform(x_train_text)
    x_val = vectorizer.transform(x_val_text)
    print(f"vectorized shape={x_train.shape} sec={time.time()-start:.1f}", flush=True)

    start = time.time()
    flat = LogisticRegression(max_iter=900, C=2.0, class_weight="balanced", random_state=42)
    flat.fit(x_train, y_train)
    flat_scores = log_proba_scores(flat, x_val, ALL_CLASSES)
    flat_pred = np.array([ALL_CLASSES[i] for i in flat_scores.argmax(axis=1)], dtype=object)
    eval_pred(f"{split_name}_flat14_logreg", y_val, flat_pred, results_path, {"seconds": time.time() - start})

    start = time.time()
    coarse_lr = LogisticRegression(max_iter=700, C=2.0, class_weight="balanced", random_state=42)
    coarse_lr.fit(x_train, y_group_train)
    coarse_lr_scores = log_proba_scores(coarse_lr, x_val, GROUPS)
    coarse_lr_pred, coarse_lr_margin = top_margin(coarse_lr_scores, GROUPS)
    append_jsonl(
        results_path,
        {
            "name": f"{split_name}_coarse_lr_diagnostics",
            "status": "ok",
            "seconds": time.time() - start,
            "group_accuracy": group_accuracy(y_val, coarse_lr_pred),
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "auxiliary": True,
        },
    )
    print(f"coarse_lr group_acc={group_accuracy(y_val, coarse_lr_pred):.6f}", flush=True)

    start = time.time()
    coarse_svc = LinearSVC(C=0.7, class_weight="balanced", random_state=42, dual="auto", max_iter=2500)
    coarse_svc.fit(x_train, y_group_train)
    coarse_svc_scores = decision_scores(coarse_svc, x_val, GROUPS)
    coarse_svc_pred, coarse_svc_margin = top_margin(coarse_svc_scores, GROUPS)
    append_jsonl(
        results_path,
        {
            "name": f"{split_name}_coarse_svc_diagnostics",
            "status": "ok",
            "seconds": time.time() - start,
            "group_accuracy": group_accuracy(y_val, coarse_svc_pred),
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "auxiliary": True,
        },
    )
    print(f"coarse_svc group_acc={group_accuracy(y_val, coarse_svc_pred):.6f}", flush=True)

    fine_models = {}
    for group, actions in GROUP_TO_ACTIONS.items():
        mask = np.array([label in actions for label in y_train])
        model = LogisticRegression(max_iter=700, C=2.0, class_weight="balanced", random_state=42)
        start = time.time()
        model.fit(x_train[mask], y_train[mask])
        fine_models[group] = model
        print(f"fine {group} rows={mask.sum()} sec={time.time()-start:.1f}", flush=True)

    fine_pred_by_true_group = fine_predict(fine_models, y_group_val, x_val)
    eval_pred(f"{split_name}_fine_oracle_group", y_val, fine_pred_by_true_group, results_path)

    fine_scores = fine_scores_full(fine_models, x_val)
    fine_by_coarse_lr = fine_predict(fine_models, coarse_lr_pred, x_val)
    fine_by_coarse_svc = fine_predict(fine_models, coarse_svc_pred, x_val)
    eval_pred(f"{split_name}_fine_by_coarse_lr_all", y_val, fine_by_coarse_lr, results_path)
    eval_pred(f"{split_name}_fine_by_coarse_svc_all", y_val, fine_by_coarse_svc, results_path)

    for coarse_name, group_pred, margin, group_scores in [
        ("lr", coarse_lr_pred, coarse_lr_margin, coarse_lr_scores),
        ("svc", coarse_svc_pred, coarse_svc_margin, coarse_svc_scores),
    ]:
        flat_group = np.array([ACTION_TO_GROUP[p] for p in flat_pred], dtype=object)
        for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
            high = margin >= threshold
            coverage = float(high.mean())
            gacc = group_accuracy(y_val, group_pred, high)

            hard = flat_pred.copy()
            hard[high] = fine_predict(fine_models, group_pred, x_val)[high]
            eval_pred(
                f"{split_name}_{coarse_name}_hard_gating_t{threshold}",
                y_val,
                hard,
                results_path,
                {"threshold": threshold, "coverage": coverage, "group_acc_covered": gacc},
            )

            safety = flat_pred.copy()
            disagree = high & (flat_group != group_pred)
            if disagree.any():
                safety[disagree] = fine_predict(fine_models, group_pred, x_val)[disagree]
            eval_pred(
                f"{split_name}_{coarse_name}_safety_check_t{threshold}",
                y_val,
                safety,
                results_path,
                {
                    "threshold": threshold,
                    "coverage": coverage,
                    "group_acc_covered": gacc,
                    "changed": int(disagree.sum()),
                },
            )

        group_norm = normalize(group_scores)
        flat_norm = normalize(flat_scores)
        fine_norm = normalize(fine_scores)
        for gw in [0.05, 0.10, 0.15, 0.25, 0.40]:
            group_boost = np.zeros_like(flat_norm)
            for gi, group in enumerate(GROUPS):
                for action in GROUP_TO_ACTIONS[group]:
                    group_boost[:, ALL_CLASSES.index(action)] = group_norm[:, gi]
            scores = flat_norm + gw * group_boost
            pred = np.array([ALL_CLASSES[i] for i in scores.argmax(axis=1)], dtype=object)
            eval_pred(f"{split_name}_{coarse_name}_soft_group_boost_w{gw}", y_val, pred, results_path, {"group_weight": gw})

        for fw in [0.05, 0.10, 0.15, 0.25]:
            scores = flat_norm + fw * fine_norm
            pred = np.array([ALL_CLASSES[i] for i in scores.argmax(axis=1)], dtype=object)
            eval_pred(f"{split_name}_{coarse_name}_soft_fine_boost_w{fw}", y_val, pred, results_path, {"fine_weight": fw})

    ok_rows = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") == "ok" and not row.get("auxiliary") and "macro_f1" in row:
                ok_rows.append(row)
    ok_rows = sorted(ok_rows, key=lambda r: r["macro_f1"], reverse=True)
    summary = {
        "best": ok_rows[0] if ok_rows else None,
        "top10": ok_rows[:10],
        "split": split_name,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    update_research(summary)
    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--run-dir", default="./routing_margin_runs")
    parser.add_argument("--max-features", type=int, default=220_000)
    parser.add_argument("--group-split", action="store_true")
    args = parser.parse_args()
    try:
        run_experiment(args)
    except Exception:
        Path(args.run_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.run_dir) / "error.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
