import csv
import json
import os
from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ALL_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]


def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_labels(path):
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def extract_current_prompt(sample):
    text = sample.get("current_prompt", "")
    if not isinstance(text, str):
        return "" if text is None else str(text)
    return text


def main():
    data_dir = "./data"
    model_dir = "./model"
    metrics_path = os.path.join(model_dir, "baseline_metrics.json")
    model_path = os.path.join(model_dir, "tfidf_logreg.pkl")

    samples = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    labels = load_labels(os.path.join(data_dir, "train_labels.csv"))

    missing = [s["id"] for s in samples if s["id"] not in labels]
    if missing:
        raise ValueError(f"Missing labels: {len(missing)}")

    x = [extract_current_prompt(s) for s in samples]
    y = [labels[s["id"]] for s in samples]

    print(f"samples={len(x)} classes={len(set(y))}")
    print("label_distribution=")
    for cls, count in Counter(y).most_common():
        print(f"  {cls:18s} {count:6d} {count / len(y):.4%}")

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    C=2.0,
                    random_state=42,
                ),
            ),
        ]
    )

    print("fit validation model...")
    pipe.fit(x_train, y_train)
    val_pred = pipe.predict(x_val)
    macro_f1 = f1_score(
        y_val,
        val_pred,
        labels=ALL_CLASSES,
        average="macro",
        zero_division=0,
    )
    print(f"Validation Macro-F1: {macro_f1:.6f}")
    print(
        classification_report(
            y_val,
            val_pred,
            labels=ALL_CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    print("refit on full train...")
    pipe.fit(x, y)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipe, model_path, compress=3)

    metrics = {
        "validation_macro_f1": macro_f1,
        "train_samples": len(x),
        "classes": ALL_CLASSES,
        "label_distribution": dict(Counter(y)),
        "features": "current_prompt_tfidf_word_1_2gram",
        "model": "LogisticRegression(class_weight=balanced,C=2.0)",
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"saved_model={model_path}")
    print(f"saved_metrics={metrics_path}")


if __name__ == "__main__":
    main()
