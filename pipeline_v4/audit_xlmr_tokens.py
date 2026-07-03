import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.data_io import load_train_samples
from pipeline_v4.serialize import serialize_for_tokenizer


MARKERS = ["[NOW]", "[LAST]", "[STATE]", "[SEQ]", "[FILES]", "[FLAG]", "[META]"]


def audit_lengths(samples, tokenizer, serializer, max_len):
    lengths = []
    history_counts = []
    marker_counts = {marker: 0 for marker in MARKERS}
    for idx, sample in enumerate(samples, 1):
        text = serialize_for_tokenizer(sample, tokenizer, max_len, serializer)
        ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        lengths.append(len(ids))
        history_counts.append(sum(1 for marker in ["[H1]", "[H2]", "[H3]", "[H4]", "[H5]"] if marker in text))
        for marker in MARKERS:
            marker_counts[marker] += int(marker in text)
        if idx % 10_000 == 0:
            print(json.dumps({"event": "audit_progress", "max_len": max_len, "rows": idx}, ensure_ascii=False), flush=True)
    arr = np.asarray(lengths, dtype=np.int64)
    return {
        "max_len": int(max_len),
        "n": int(len(samples)),
        "mean_len": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max_observed": int(arr.max()),
        "over_limit": int((arr > max_len).sum()),
        "over_limit_rate": float((arr > max_len).mean()),
        "avg_history_pairs_kept": float(np.mean(history_counts)),
        "marker_kept_rate": {marker: marker_counts[marker] / max(len(samples), 1) for marker in MARKERS},
    }


def write_markdown(path, payload):
    lines = ["# Transformer Token Audit", ""]
    lines.append(f"- model: `{payload['model_name']}`")
    lines.append(f"- serializer: `{payload['serializer']}`")
    lines.append(f"- samples: `{payload['n']}`")
    lines.append("")
    lines.append("| max_len | p50 | p90 | p95 | p99 | max | over_rate | avg H kept | NOW | LAST | STATE |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["rows"]:
        kept = row["marker_kept_rate"]
        lines.append(
            f"| {row['max_len']} | {row['p50']:.0f} | {row['p90']:.0f} | {row['p95']:.0f} | "
            f"{row['p99']:.0f} | {row['max_observed']} | {row['over_limit_rate']:.4f} | "
            f"{row['avg_history_pairs_kept']:.2f} | {kept['[NOW]']:.3f} | {kept['[LAST]']:.3f} | {kept['[STATE]']:.3f} |"
        )
    lines.append("")
    lines.append("## Decision")
    best_512 = next((row for row in payload["rows"] if row["max_len"] == 512), None)
    if best_512 and all(best_512["marker_kept_rate"].get(marker, 0.0) >= 1.0 for marker in ["[NOW]", "[LAST]", "[STATE]"]):
        lines.append("- gate: `pass`")
        lines.append("- reason: `[NOW]`, `[LAST]`, `[STATE]` are kept at max_len 512.")
    else:
        lines.append("- gate: `fail`")
        lines.append("- reason: required state markers are not fully preserved at max_len 512.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--serializer", default="xlmr_state_v1")
    parser.add_argument("--max-lens", nargs="+", type=int, default=[256, 320, 384, 512])
    parser.add_argument("--out-dir", default="pipeline_v4/artifacts/reports/xlmr_state_v1_512")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    samples = load_train_samples(args.data_dir)
    rows = [audit_lengths(samples, tokenizer, args.serializer, max_len) for max_len in args.max_lens]
    payload = {
        "model_name": args.model_name,
        "serializer": args.serializer,
        "n": len(samples),
        "rows": rows,
    }
    (out_dir / "token_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_dir / "token_audit.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
