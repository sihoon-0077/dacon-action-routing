import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_train
from src.serialize_policy import serialize_policy_v3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--golden", default="tests/golden_serialize_v3.txt")
    parser.add_argument("--out-md", default="artifacts_v3/reports/serialize_golden_v3.md")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    samples = load_train(args.data_dir)[:3]
    text = "\n\n--- SAMPLE ---\n\n".join(serialize_policy_v3(sample) for sample in samples) + "\n"
    golden = Path(args.golden)
    golden.parent.mkdir(parents=True, exist_ok=True)
    created = False
    matched = False
    if not golden.exists() or args.update:
        golden.write_text(text, encoding="utf-8")
        created = True
        matched = True
    else:
        matched = golden.read_text(encoding="utf-8") == text
        if not matched:
            raise SystemExit(f"golden mismatch: {golden}")

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                "# Serialize Golden v3",
                "",
                f"- golden path: `{golden}`",
                f"- created_or_updated: `{created}`",
                f"- byte_exact_match: `{matched}`",
                f"- sample_count: `{len(samples)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
