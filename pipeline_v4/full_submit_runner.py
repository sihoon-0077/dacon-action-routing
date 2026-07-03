import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RUN_NAME = "v2full_512_5e"
FOLD_RUN = "v2bundle_512_5e"
ARTIFACT = ROOT / "pipeline_v4" / "artifacts"
REPORT_DIR = ARTIFACT / "full_submit" / RUN_NAME
ZIP_PATH = ROOT / "submit_v4_full_512.zip"
OUT_DIR = ROOT / "submit_v4_full_512"


def run_step(name, cmd, allow_fail=False, cwd=None):
    cwd = Path(cwd) if cwd else ROOT
    cmd = [str(x) for x in cmd]
    print(json.dumps({"event": "step_start", "name": name, "cmd": cmd}, ensure_ascii=False), flush=True)
    start = time.time()
    proc = subprocess.run(cmd, cwd=cwd, text=True)
    payload = {
        "event": "step_done",
        "name": name,
        "returncode": proc.returncode,
        "elapsed_sec": time.time() - start,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(proc.returncode)
    return proc.returncode


def existing_bias_or_empty():
    bias_path = ARTIFACT / "decision" / FOLD_RUN / "bias.json"
    temp_path = ARTIFACT / "calib" / FOLD_RUN / "temperatures.json"
    out_path = REPORT_DIR / "fold_decision_for_submit.json"
    payload = {}
    if bias_path.exists():
        payload.update(json.loads(bias_path.read_text(encoding="utf-8")))
    if temp_path.exists():
        payload.update(json.loads(temp_path.read_text(encoding="utf-8")))
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def smoke_test_submit():
    data_src = ROOT / "data"
    data_dst = OUT_DIR / "data"
    output_dir = OUT_DIR / "output"
    if data_dst.exists():
        shutil.rmtree(data_dst)
    shutil.copytree(data_src, data_dst, ignore=shutil.ignore_patterns("train.jsonl", "train_labels.csv"))
    output_dir.mkdir(exist_ok=True)
    run_step("smoke_test_submit", [PYTHON, "script.py"], allow_fail=False, cwd=OUT_DIR)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "runner_pid.txt").write_text(str(__import__("os").getpid()) + "\n", encoding="utf-8")

    train_cmd = [
        PYTHON,
        "pipeline_v4/train_full.py",
        "--config",
        "pipeline_v4/configs/mdeberta_a_local8gb.yaml",
        "--override",
        f"run_name={RUN_NAME}",
        "--override",
        "max_len=512",
        "--override",
        "epochs=5",
        "--override",
        "serializer=v2",
        "--override",
        "class_weight=none",
    ]
    run_step("train_full", train_cmd)

    run_step("calibrate_fold_run", [PYTHON, "pipeline_v4/calibrate.py", "--run", FOLD_RUN, "--folds", "0,1"], allow_fail=True)
    run_step("build_oof_fold_run", [PYTHON, "pipeline_v4/build_oof.py", "--run", FOLD_RUN, "--folds", "0,1"], allow_fail=True)
    run_step("optimize_bias_fold_run", [PYTHON, "pipeline_v4/optimize_bias.py", "--run", FOLD_RUN], allow_fail=True)

    decision_path = existing_bias_or_empty()
    build_cmd = [
        PYTHON,
        "scripts/build_submit_v4_full.py",
        "--v4-main",
        str(ARTIFACT / "models" / RUN_NAME / "full"),
        "--fold-decision",
        str(decision_path),
        "--out-dir",
        str(OUT_DIR),
        "--zip-path",
        str(ZIP_PATH),
        "--max-len",
        "256",
        "--batch-size",
        "64",
        "--max-transformer-samples",
        "8000",
        "--threshold",
        "0.0",
    ]
    run_step("build_submit", build_cmd)

    smoke_test_submit()
    payload = {
        "event": "done",
        "zip_path": str(ZIP_PATH),
        "out_dir": str(OUT_DIR),
        "model_dir": str(ARTIFACT / "models" / RUN_NAME / "full"),
    }
    (REPORT_DIR / "done.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
