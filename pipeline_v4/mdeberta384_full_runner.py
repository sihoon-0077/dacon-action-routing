import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RUN_NAME = "mdeberta384_v2_384_full_5e"
ARTIFACT = ROOT / "pipeline_v4" / "artifacts"
REPORT_DIR = ARTIFACT / "mdeberta384_full"
LOG_DIR = REPORT_DIR / "logs"
STATUS = REPORT_DIR / "status.md"


def append(line=""):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_cmd(name, cmd, allow_fail=False, cwd=None):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"{name}.stdout.log"
    err_path = LOG_DIR / f"{name}.stderr.log"
    append(f"## {name}")
    append("```text")
    append(" ".join(str(x) for x in cmd))
    append("```")
    start = time.time()
    with out_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run([str(x) for x in cmd], cwd=cwd or ROOT, stdout=out, stderr=err, text=True)
    elapsed = time.time() - start
    append(f"- exit_code: `{proc.returncode}`")
    append(f"- elapsed_sec: `{elapsed:.1f}`")
    append(f"- stdout: `{out_path}`")
    append(f"- stderr: `{err_path}`")
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(proc.returncode)
    return proc.returncode


def smoke_submit(submit_dir):
    submit_dir = Path(submit_dir)
    data_src = ROOT / "data"
    data_dst = submit_dir / "data"
    output_dir = submit_dir / "output"
    if data_dst.exists():
        shutil.rmtree(data_dst)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    data_dst.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    shutil.copy2(data_src / "test.jsonl", data_dst / "test.jsonl")
    shutil.copy2(data_src / "sample_submission.csv", data_dst / "sample_submission.csv")
    return run_cmd(f"smoke_{submit_dir.name}", [PYTHON, "script.py"], cwd=submit_dir, allow_fail=False)


def build_submit(epoch):
    model_dir = ARTIFACT / "models" / RUN_NAME / f"full_epoch_{epoch}"
    if not model_dir.exists():
        raise FileNotFoundError(f"missing epoch checkpoint: {model_dir}")
    out_dir = ROOT / f"submit_v4_full384_ep{epoch}_cand8000"
    zip_path = ROOT / f"submit_v4_full384_ep{epoch}_cand8000.zip"
    run_cmd(
        f"build_ep{epoch}",
        [
            PYTHON,
            "scripts/build_submit_v4_full.py",
            "--v4-main",
            str(model_dir),
            "--out-dir",
            str(out_dir),
            "--zip-path",
            str(zip_path),
            "--max-len",
            "384",
            "--batch-size",
            "64",
            "--max-transformer-samples",
            "8000",
            "--threshold",
            "0.0",
        ],
    )
    smoke_submit(out_dir)
    size = zip_path.stat().st_size if zip_path.exists() else None
    append(f"- submit_ep{epoch}: `{zip_path}` size=`{size}`")


def main():
    STATUS.write_text("# mDeBERTa 384 Full Status\n\n", encoding="utf-8")
    append(f"- started_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    run_cmd(
        "train_full384",
        [
            PYTHON,
            "pipeline_v4/train_full.py",
            "--config",
            "pipeline_v4/configs/mdeberta_v2_384.yaml",
            "--override",
            f"run_name={RUN_NAME}",
            "--override",
            "epochs=5",
            "--override",
            "max_len=384",
            "--override",
            "serializer=v2",
            "--override",
            "save_epochs=3,5",
        ],
    )
    for epoch in [3, 5]:
        build_submit(epoch)
    payload = {
        "run": RUN_NAME,
        "zips": [
            str(ROOT / "submit_v4_full384_ep3_cand8000.zip"),
            str(ROOT / "submit_v4_full384_ep5_cand8000.zip"),
        ],
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (REPORT_DIR / "done.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append(f"- finished_at: `{payload['finished_at']}`")


if __name__ == "__main__":
    main()
