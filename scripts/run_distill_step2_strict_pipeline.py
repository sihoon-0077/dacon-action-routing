import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_status(path, stage, detail, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Distill Step2 Strict Pipeline Status",
        "",
        f"- updated_at: `{now()}`",
        f"- stage: `{stage}`",
        f"- detail: `{detail}`",
        "",
        "## Extra",
    ]
    for key, value in sorted(extra.items()):
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{now()}] {stage}: {detail}", flush=True)


def run_step(name, cmd, status_path):
    write_status(status_path, name, "starting", command=" ".join(cmd))
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    elapsed = (time.time() - started) / 3600
    if proc.returncode != 0:
        write_status(status_path, "failed", f"{name} returncode={proc.returncode}", elapsed_hours=f"{elapsed:.2f}")
        raise SystemExit(proc.returncode)
    write_status(status_path, name, "finished", elapsed_hours=f"{elapsed:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--advanced-out-dir", default="artifacts/advanced_oof_strict")
    parser.add_argument("--advanced-report-dir", default="reports/advanced_oof_strict")
    parser.add_argument("--distill-artifact-dir", default="artifacts/distill_step2_strict")
    parser.add_argument("--distill-report-dir", default="reports/distill_step2_strict")
    parser.add_argument("--distill-model-dir", default="model/distill_student_strict")
    parser.add_argument("--max-features", type=int, default=160_000)
    parser.add_argument("--svd-dim", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--reuse-advanced", action="store_true")
    args = parser.parse_args()

    status_path = Path(args.distill_report_dir) / "PIPELINE_STATUS.md"
    runner_dir = Path(args.distill_report_dir) / "runner"
    runner_dir.mkdir(parents=True, exist_ok=True)
    (runner_dir / "pipeline_pid.txt").write_text(str(os.getpid()) + "\n", encoding="utf-8")

    write_status(status_path, "start", "strict OOF advanced + strict distill pipeline")
    advanced_done = Path(args.advanced_out_dir) / "advanced_oof_probs.npy"
    if args.reuse_advanced and advanced_done.exists():
        write_status(status_path, "strict_advanced_oof", "reusing existing strict advanced cache")
    else:
        run_step(
            "strict_advanced_oof",
            [
                sys.executable,
                "scripts/run_strict_advanced_oof.py",
                "--out-dir",
                args.advanced_out_dir,
                "--report-dir",
                args.advanced_report_dir,
            ],
            status_path,
        )

    run_step(
        "strict_distill",
        [
            sys.executable,
            "scripts/run_distill_step2.py",
            "--advanced-oof-dir",
            args.advanced_out_dir,
            "--artifact-dir",
            args.distill_artifact_dir,
            "--report-dir",
            args.distill_report_dir,
            "--model-dir",
            args.distill_model_dir,
            "--max-features",
            str(args.max_features),
            "--svd-dim",
            str(args.svd_dim),
            "--epochs",
            str(args.epochs),
        ],
        status_path,
    )
    write_status(status_path, "finished", "strict pipeline completed")


if __name__ == "__main__":
    main()
