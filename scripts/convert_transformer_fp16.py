import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(src, local_files_only=True)
    model = model.half()
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(src, use_fast=False, local_files_only=True)
    tokenizer.save_pretrained(dst)

    classes = src / "classes.json"
    if classes.exists():
        shutil.copy2(classes, dst / "classes.json")
    total = sum(path.stat().st_size for path in dst.rglob("*") if path.is_file())
    print(f"saved={dst} bytes={total}")


if __name__ == "__main__":
    main()
