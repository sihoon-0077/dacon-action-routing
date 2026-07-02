import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import SEED, VAL_RATIO
from src.io_utils import get_session_id, load_train


def make_split(data_dir="data", out_dir="splits", seed=SEED, val_ratio=VAL_RATIO):
    samples = load_train(data_dir)
    ids = np.array([sample["id"] for sample in samples], dtype=object)
    y = np.array([sample["action"] for sample in samples], dtype=object)
    groups = np.array([get_session_id(sample_id) for sample_id in ids], dtype=object)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.arange(len(samples)), y, groups=groups))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "train_ids.txt").write_text("\n".join(ids[train_idx]) + "\n", encoding="utf-8")
    (out_path / "val_ids.txt").write_text("\n".join(ids[val_idx]) + "\n", encoding="utf-8")
    return {
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "n_sessions": int(len(set(groups))),
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "train_path": str(out_path / "train_ids.txt"),
        "val_path": str(out_path / "val_ids.txt"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="splits")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    args = parser.parse_args()
    info = make_split(args.data_dir, args.out_dir, args.seed, args.val_ratio)
    for key, value in info.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
