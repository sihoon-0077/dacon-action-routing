import csv
import json
from pathlib import Path

from .constants import FOLD_FILE


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_labels(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def load_fold_rows(path=FOLD_FILE):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_fold_ids(fold_k, path=FOLD_FILE):
    return {row["id"] for row in load_fold_rows(path) if int(row["fold"]) == int(fold_k)}


def load_train_samples(data_dir="./data"):
    data_dir = Path(data_dir)
    labels = load_labels(data_dir / "train_labels.csv")
    samples = []
    for sample in iter_jsonl(data_dir / "train.jsonl"):
        sample["action"] = labels[sample["id"]]
        samples.append(sample)
    return samples
