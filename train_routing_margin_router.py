import csv
import json
import os
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from compact_state_experiments import ALL_CLASSES, compact_text
from routing_margin_experiments import ACTION_TO_GROUP, GROUP_TO_ACTIONS


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


def main():
    data_dir = "./data"
    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)

    samples = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    labels = load_labels(os.path.join(data_dir, "train_labels.csv"))
    y = np.array([labels[s["id"]] for s in samples], dtype=object)
    y_group = np.array([ACTION_TO_GROUP[label] for label in y], dtype=object)
    texts = [compact_text(sample, "flags") for sample in samples]

    print(f"samples={len(samples)}")
    start = time.time()
    vectorizer = build_vectorizer(max_features=220_000, min_df=2)
    x = vectorizer.fit_transform(texts)
    print(f"vectorized shape={x.shape} sec={time.time() - start:.1f}")

    start = time.time()
    coarse_svc = LinearSVC(
        C=2.0,
        class_weight="balanced",
        random_state=42,
        dual="auto",
        max_iter=2500,
    )
    coarse_svc.fit(x, y_group)
    print(f"fit coarse_svc sec={time.time() - start:.1f}")

    fine_models = {}
    for group, actions in GROUP_TO_ACTIONS.items():
        start = time.time()
        mask = np.isin(y, actions)
        model = LogisticRegression(
            max_iter=700,
            C=2.0,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(x[mask], y[mask])
        fine_models[group] = model
        print(f"fit fine {group} rows={int(mask.sum())} sec={time.time() - start:.1f}")

    artifact = {
        "kind": "routing_margin_router",
        "classes": ALL_CLASSES,
        "groups": list(GROUP_TO_ACTIONS.keys()),
        "group_to_actions": GROUP_TO_ACTIONS,
        "action_to_group": ACTION_TO_GROUP,
        "vectorizer": vectorizer,
        "coarse_svc": coarse_svc,
        "fine_models": fine_models,
        "strategy": "coarse_svc_then_fine_logreg_always",
        "validation_macro_f1": 0.6951503047995322,
        "group_shuffle_macro_f1": 0.6877626094295418,
        "oracle_group_macro_f1": 0.6993616214576527,
        "experiment": "compact_flags_svcC2.0_logreg_fine_by_coarse_all",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(model_dir, "routing_margin_router.pkl")
    joblib.dump(artifact, path, compress=3)
    print(f"saved={path}")


if __name__ == "__main__":
    main()
