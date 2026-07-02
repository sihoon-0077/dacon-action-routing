import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.data_io import iter_jsonl
from pipeline_v4.serialize import serialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--golden", default="pipeline_v4/tests/golden_serialize.txt")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    samples = []
    for sample in iter_jsonl(Path(args.data_dir) / "train.jsonl"):
        samples.append(sample)
        if len(samples) == 3:
            break
    text = "\n\n--- SAMPLE ---\n\n".join(serialize(sample) for sample in samples) + "\n"
    golden = Path(args.golden)
    golden.parent.mkdir(parents=True, exist_ok=True)
    if args.update or not golden.exists():
        golden.write_text(text, encoding="utf-8")
        print(f"updated: {golden}")
        return
    expected = golden.read_text(encoding="utf-8")
    if expected != text:
        raise SystemExit(f"golden mismatch: {golden}")
    print(f"golden ok: {golden}")


if __name__ == "__main__":
    main()
