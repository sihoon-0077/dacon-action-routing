import csv
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def load_train(data_dir="data"):
    data_dir = Path(data_dir)
    samples = read_jsonl(data_dir / "train.jsonl")
    labels = load_labels(data_dir / "train_labels.csv")
    for sample in samples:
        sample["action"] = labels[sample["id"]]
    return samples


def load_test(data_dir="data"):
    return read_jsonl(Path(data_dir) / "test.jsonl")


def get_session_id(sample_id):
    return sample_id.rsplit("-step_", 1)[0] if "-step_" in sample_id else sample_id


def get_step(sample_id):
    if "-step_" not in sample_id:
        return None
    try:
        return int(sample_id.rsplit("-step_", 1)[1])
    except ValueError:
        return None


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
