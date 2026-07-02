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

from compact_state_experiments import ALL_CLASSES, CLASS_TO_GROUP, compact_text, last_actions


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


def build_vectorizer(max_features=220_000, min_df=2):
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


def transition_counts(samples, y, index):
    counts = defaultdict(Counter)
    for sample, label in zip(samples, y):
        counts[last_actions(sample)[index]][label] += 1
    return {key: dict(value) for key, value in counts.items()}


def main():
    data_dir = "./data"
    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)

    samples = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    labels = load_labels(os.path.join(data_dir, "train_labels.csv"))
    y = np.array([labels[s["id"]] for s in samples], dtype=object)
    group_y = np.array([CLASS_TO_GROUP[label] for label in y], dtype=object)
    texts = [compact_text(sample, "flags") for sample in samples]

    print(f"samples={len(samples)}")
    start = time.time()
    vectorizer = build_vectorizer(max_features=220_000, min_df=2)
    x = vectorizer.fit_transform(texts)
    print(f"vectorized shape={x.shape} sec={time.time() - start:.1f}")

    start = time.time()
    clf = LogisticRegression(max_iter=900, C=2.0, class_weight="balanced", random_state=42)
    clf.fit(x, y)
    print(f"fit clf sec={time.time() - start:.1f}")

    start = time.time()
    group_clf = LogisticRegression(max_iter=500, C=2.0, class_weight="balanced", random_state=42)
    group_clf.fit(x, group_y)
    print(f"fit group_clf sec={time.time() - start:.1f}")

    artifact = {
        "kind": "compact_flags_router",
        "classes": ALL_CLASSES,
        "vectorizer": vectorizer,
        "clf": clf,
        "group_clf": group_clf,
        "transition1": transition_counts(samples, y, 0),
        "transition2": transition_counts(samples, y, 1),
        "global_counts": dict(Counter(y)),
        "weights": {
            "prior1": 0.06,
            "prior2": 0.03,
            "group": 0.08,
            "rules": 0.02,
            "smooth1": 1.0,
            "smooth2": 2.0,
        },
        "validation_macro_f1": 0.6663507272653266,
        "experiment": "compact_flags_lr_combo_a1_0.06_gw_0.08_rw_0.02",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(model_dir, "compact_flags_router.pkl")
    joblib.dump(artifact, path, compress=3)
    print(f"saved={path}")


if __name__ == "__main__":
    main()
