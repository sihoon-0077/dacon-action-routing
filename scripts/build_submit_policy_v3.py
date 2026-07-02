import argparse
import json
import shutil
import zipfile
from pathlib import Path


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
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns("*.bin.index.json", "optimizer*", "scheduler*", "trainer_state.json")
    shutil.copytree(src, dst, ignore=ignore)


def zip_dir(src_dir, zip_path):
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(Path(src_dir).rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir).as_posix())
    return zip_path.stat().st_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--advanced-router", default="model/advanced_router.pkl")
    parser.add_argument("--tf-main", required=True)
    parser.add_argument("--decision", default="artifacts_v3/reports/decision/decision.json")
    parser.add_argument("--script", default="script.py")
    parser.add_argument("--out-dir", default="submit_policy_v3")
    parser.add_argument("--zip-path", default="submit_policy_v3.zip")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "model").mkdir(parents=True)

    shutil.copy2(args.script, out_dir / "script.py")
    (out_dir / "requirements.txt").write_text("", encoding="utf-8")
    shutil.copy2(args.advanced_router, out_dir / "model" / "advanced_router.pkl")
    copytree_clean(Path(args.tf_main), out_dir / "model" / "tf_main")

    decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
    decision["override_actions"] = DEFAULT_OVERRIDE_ACTIONS
    decision["override_threshold"] = args.threshold
    decision["max_len"] = 320
    decision["history_pairs"] = 6
    decision["batch_size"] = args.batch_size
    decision["disable_session_lookup"] = True
    decision["classes"] = decision.get("classes") or [
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
    (out_dir / "model" / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_size = zip_dir(out_dir, args.zip_path)
    unpacked_size = sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())
    print(f"out_dir={out_dir}")
    print(f"zip={args.zip_path} zip_bytes={zip_size} unpacked_bytes={unpacked_size}")


if __name__ == "__main__":
    main()
