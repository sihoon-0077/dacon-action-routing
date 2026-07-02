import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.constants import ALL_CLASSES, DATA_DIR, FOLD_FILE, N_FOLDS, SEED, session_of
from pipeline_v4.common.data_io import iter_jsonl, load_labels


def make_folds(data_dir=DATA_DIR, out_path=FOLD_FILE, seed=SEED, n_folds=N_FOLDS, force=False):
    out_path = Path(out_path)
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists; fold regeneration is forbidden")

    data_dir = Path(data_dir)
    labels = load_labels(data_dir / "train_labels.csv")
    rows = []
    sessions = set()
    for sample in iter_jsonl(data_dir / "train.jsonl"):
        sid = session_of(sample["id"])
        rows.append({"id": sample["id"], "session": sid, "action": labels[sample["id"]]})
        sessions.add(sid)

    sessions = sorted(sessions)
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)
    fold_of_session = {sid: i % n_folds for i, sid in enumerate(sessions)}
    for row in rows:
        row["fold"] = fold_of_session[row["session"]]

    session_folds = defaultdict(set)
    fold_counts = Counter()
    fold_class_counts = defaultdict(Counter)
    for row in rows:
        fold = row["fold"]
        session_folds[row["session"]].add(fold)
        fold_counts[fold] += 1
        fold_class_counts[fold][row["action"]] += 1
    assert all(len(v) == 1 for v in session_folds.values())

    counts = np.array([fold_counts[i] for i in range(n_folds)], dtype=float)
    max_dev = float((np.abs(counts - counts.mean()) / counts.mean()).max())
    assert max_dev < 0.05, f"fold sample count deviation too large: {max_dev:.4f}"
    for fold in range(n_folds):
        missing = [c for c in ALL_CLASSES if fold_class_counts[fold][c] == 0]
        assert not missing, f"fold {fold} missing classes: {missing}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "session", "fold"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "session": row["session"], "fold": row["fold"]})

    total = len(rows)
    class_counts = Counter(row["action"] for row in rows)
    prior = {c: (class_counts[c] + 1) / (total + len(ALL_CLASSES)) for c in ALL_CLASSES}
    prior_payload = {
        "classes": ALL_CLASSES,
        "counts": {c: int(class_counts[c]) for c in ALL_CLASSES},
        "prior": prior,
        "n": total,
        "seed": seed,
        "n_folds": n_folds,
        "fold_counts": {str(i): int(fold_counts[i]) for i in range(n_folds)},
        "max_fold_count_deviation": max_dev,
    }
    with open(out_path.parent / "prior.json", "w", encoding="utf-8") as f:
        json.dump(prior_payload, f, ensure_ascii=False, indent=2)
    return prior_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out", default=FOLD_FILE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload = make_folds(args.data_dir, args.out, force=args.force)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
