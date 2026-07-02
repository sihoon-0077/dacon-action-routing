import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_train
from src.serialize_policy import serialize_policy_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-name", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--max-len", type=int, default=320)
    parser.add_argument("--out-dir", default="artifacts/reports/serialization_preview")
    parser.add_argument("--dry-run-serialization", action="store_true")
    parser.add_argument("--n-preview", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_train(args.data_dir)
    texts = [serialize_policy_state(sample, layout="now_first") for sample in samples]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    lengths = []
    now_missing = 0
    for text in texts:
        full = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        clipped = tokenizer(text, max_length=args.max_len, truncation=True, padding="max_length")["input_ids"]
        decoded = tokenizer.decode([i for i in clipped if i != tokenizer.pad_token_id], skip_special_tokens=False)
        lengths.append(len(full))
        now_missing += int("NOW" not in decoded)
    arr = np.array(lengths)
    stats = {
        "n": len(texts),
        "mean": float(arr.mean()),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "over_max_len": int((arr > args.max_len).sum()),
        "now_missing_after_trunc": int(now_missing),
    }
    (out_dir / "token_length_stats.csv").write_text(
        "metric,value\n" + "\n".join(f"{k},{v}" for k, v in stats.items()) + "\n",
        encoding="utf-8",
    )
    preview_lines = ["# Serialization Preview", "", f"- model: `{args.model_name}`", f"- max_len: `{args.max_len}`", ""]
    for i, (sample, text) in enumerate(zip(samples[: args.n_preview], texts[: args.n_preview]), 1):
        preview_lines += [f"## Sample {i}: `{sample['id']}`", "", "```text", text[:3000], "```", ""]
    (out_dir / "serialization_preview.md").write_text("\n".join(preview_lines), encoding="utf-8")
    print(f"saved: {out_dir / 'serialization_preview.md'}")
    print(f"saved: {out_dir / 'token_length_stats.csv'}")


if __name__ == "__main__":
    main()
