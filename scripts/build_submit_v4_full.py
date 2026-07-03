import argparse
import json
import shutil
import zipfile
from pathlib import Path


ALL_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

DEFAULT_OVERRIDE_ACTIONS = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "respond_only",
]


def copytree_clean(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns(
        "optimizer*",
        "scheduler*",
        "trainer_state.json",
        "*.bin.index.json",
        "__pycache__",
    )
    shutil.copytree(src, dst, ignore=ignore)


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


def load_optional_decision(path):
    path = Path(path) if path else None
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--advanced-router", default="model/advanced_router.pkl")
    parser.add_argument("--v4-main", required=True)
    parser.add_argument("--fold-decision", default="")
    parser.add_argument("--script", default="script.py")
    parser.add_argument("--out-dir", default="submit_v4_full_512")
    parser.add_argument("--zip-path", default="submit_v4_full_512.zip")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "model").mkdir(parents=True)

    shutil.copy2(args.script, out_dir / "script.py")
    (out_dir / "requirements.txt").write_text("", encoding="utf-8")
    shutil.copy2(args.advanced_router, out_dir / "model" / "advanced_router.pkl")
    copytree_clean(args.v4_main, out_dir / "model" / "v4_main")

    learned = load_optional_decision(args.fold_decision)
    use_bias = bool(learned.get("adopted", False))
    bias_by_class = learned.get("bias_by_class", {}) if use_bias else {}
    temperatures = learned.get("temperatures", {})
    temp_values = [float(v) for v in temperatures.values()] if isinstance(temperatures, dict) else []
    temperature = sum(temp_values) / len(temp_values) if temp_values else float(learned.get("temperature", 1.0))

    decision = {
        "classes": ALL_CLASSES,
        "temperature": float(temperature),
        "bias_by_class": {name: float(bias_by_class.get(name, 0.0)) for name in ALL_CLASSES},
        "bias_source_adopted": use_bias,
        "override_actions": DEFAULT_OVERRIDE_ACTIONS,
        "override_threshold": float(args.threshold),
        "max_len": int(args.max_len),
        "batch_size": int(args.batch_size),
        "direct": bool(args.direct),
        "disable_session_lookup": True,
    }
    (out_dir / "model" / "v4_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "model" / "decision.json").write_text(
        json.dumps({"disable_session_lookup": True}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    zip_size = zip_dir(out_dir, args.zip_path)
    unpacked_size = sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())
    print(f"out_dir={out_dir}")
    print(f"zip={args.zip_path} zip_bytes={zip_size} unpacked_bytes={unpacked_size}")
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
