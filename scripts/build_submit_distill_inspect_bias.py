import argparse
import json
import shutil
import zipfile
from pathlib import Path

from build_submit_distill import DISTILL_SCRIPT


INSPECT_BIAS_DELTA = {
    "read_file": 0.05,
    "grep_search": -0.05,
    "list_directory": 0.05,
    "glob_pattern": -0.10,
}


def zip_dir(src_dir, zip_path):
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir).as_posix())
    return zip_path.stat().st_size


def patched_config(student_dir):
    config_path = Path(student_dir) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bias = dict(config.get("bias_by_class", {}))
    for action, delta in INSPECT_BIAS_DELTA.items():
        bias[action] = float(bias.get(action, 0.0)) + float(delta)
    config["bias_by_class"] = bias
    config["inspect_bias_delta"] = INSPECT_BIAS_DELTA
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-dir", default="model/distill_student_strict")
    parser.add_argument("--out-dir", default="distill_ib")
    parser.add_argument("--zip-path", default="distill_ib.zip")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "model").mkdir(parents=True)
    (out_dir / "pipeline_v4").mkdir(parents=True)
    (out_dir / "pipeline_v4" / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2("pipeline_v4/serialize.py", out_dir / "pipeline_v4" / "serialize.py")
    shutil.copy2("script.py", out_dir / "router_base.py")
    (out_dir / "script.py").write_text(DISTILL_SCRIPT.strip() + "\n", encoding="utf-8")
    (out_dir / "requirements.txt").write_text("", encoding="utf-8")

    student_dir = Path(args.student_dir)
    for name in [
        "advanced_router.pkl",
        "dense_encoder.pkl",
        "scaler.pkl",
        "student.pt",
        "svd.pkl",
        "vectorizer.pkl",
    ]:
        shutil.copy2(student_dir / name, out_dir / "model" / name)
    (out_dir / "model" / "config.json").write_text(
        json.dumps(patched_config(student_dir), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    size = zip_dir(out_dir, args.zip_path)
    unpacked = sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())
    print(f"out_dir={out_dir}")
    print(f"zip={args.zip_path} zip_bytes={size} unpacked_bytes={unpacked}")
    print(f"inspect_bias_delta={INSPECT_BIAS_DELTA}")


if __name__ == "__main__":
    main()
