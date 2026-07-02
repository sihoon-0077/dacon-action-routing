import csv
import json
import os
import time
from collections import Counter, defaultdict

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from script import (
    ADVANCED_ACTION_TO_GROUP,
    ADVANCED_GROUP_TO_ACTIONS,
    ALL_CLASSES,
    advanced_group_text,
    advanced_last2_action,
    advanced_pair_text,
    compact_flags_text,
)


PAIR_PRIORITY = [
    ("read_file", "grep_search"),
    ("grep_search", "glob_pattern"),
    ("list_directory", "glob_pattern"),
    ("edit_file", "apply_patch"),
    ("edit_file", "write_file"),
    ("run_bash", "run_tests"),
    ("run_tests", "lint_or_typecheck"),
    ("ask_user", "plan_task"),
    ("plan_task", "web_search"),
    ("respond_only", "plan_task"),
]


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


def build_vectorizer(max_features=180_000, min_df=2):
    half = max_features // 2
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=min_df,
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
                    min_df=min_df,
                    max_features=half,
                    sublinear_tf=True,
                    lowercase=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def transition_counts(samples, y):
    counts = defaultdict(Counter)
    for sample, label in zip(samples, y):
        counts[advanced_last2_action(sample)][label] += 1
    return {key: dict(value) for key, value in counts.items()}


def train_pair_resolvers(samples, y, max_features):
    resolvers = {}
    y = np.asarray(y, dtype=object)
    for raw_pair in PAIR_PRIORITY:
        pair = tuple(sorted(raw_pair))
        if ADVANCED_ACTION_TO_GROUP[pair[0]] != ADVANCED_ACTION_TO_GROUP[pair[1]]:
            continue
        idx = np.where(np.isin(y, pair))[0]
        if len(idx) < 50:
            continue
        start = time.time()
        texts = [advanced_pair_text(samples[i], pair) for i in idx]
        vectorizer = build_vectorizer(max_features=max_features, min_df=2)
        x = vectorizer.fit_transform(texts)
        model = LogisticRegression(max_iter=700, C=2.0, class_weight="balanced", random_state=42)
        model.fit(x, y[idx])
        resolvers[pair] = {"vectorizer": vectorizer, "model": model}
        print(f"pair {pair[0]} vs {pair[1]} rows={len(idx)} sec={time.time() - start:.1f}")
    return resolvers


def main():
    data_dir = "./data"
    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)

    samples = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    labels = load_labels(os.path.join(data_dir, "train_labels.csv"))
    y = np.array([labels[s["id"]] for s in samples], dtype=object)
    y_group = np.array([ADVANCED_ACTION_TO_GROUP[label] for label in y], dtype=object)
    print(f"samples={len(samples)}")

    start = time.time()
    coarse_vectorizer = build_vectorizer(max_features=220_000, min_df=2)
    x_coarse = coarse_vectorizer.fit_transform([compact_flags_text(sample) for sample in samples])
    print(f"coarse vectorized shape={x_coarse.shape} sec={time.time() - start:.1f}")

    start = time.time()
    coarse_model = LinearSVC(C=2.0, class_weight="balanced", random_state=42, dual="auto", max_iter=2500)
    coarse_model.fit(x_coarse, y_group)
    print(f"coarse fit sec={time.time() - start:.1f}")

    group_vectorizers = {}
    group_models = {}
    for group, actions in ADVANCED_GROUP_TO_ACTIONS.items():
        start = time.time()
        idx = np.where(np.isin(y, actions))[0]
        texts = [advanced_group_text(samples[i], group) for i in idx]
        vectorizer = build_vectorizer(max_features=180_000, min_df=2)
        x = vectorizer.fit_transform(texts)
        model = LogisticRegression(max_iter=800, C=2.0, class_weight="balanced", random_state=42)
        model.fit(x, y[idx])
        group_vectorizers[group] = vectorizer
        group_models[group] = model
        print(f"group {group} rows={len(idx)} shape={x.shape} sec={time.time() - start:.1f}")

    pair_resolvers = train_pair_resolvers(samples, y, max_features=80_000)

    artifact = {
        "kind": "advanced_action_router_v1",
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
            "prior_alpha": 0.3,
            "prior_smooth": 1.0,
            "pair_threshold": 0.08,
        },
        "validation_macro_f1": 0.7113236414043568,
        "validation_split": "GroupShuffleSplit session_id seed=42",
        "experiment": "phase6_pair_resolver_t0.08",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(model_dir, "advanced_router.pkl")
    joblib.dump(artifact, path, compress=3)
    print(f"saved={path}")


if __name__ == "__main__":
    main()
