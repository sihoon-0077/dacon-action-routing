import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RUN = "mdeberta384_v2_384_5e"
CONFIG = "pipeline_v4/configs/mdeberta_v2_384.yaml"
ARTIFACT = ROOT / "pipeline_v4" / "artifacts"
STATE = ARTIFACT / "cycle3_oof_runner"
LOG_DIR = STATE / "logs"
STATUS = STATE / "status.md"
FOLDS = [1, 2, 3, 4]


def append(text):
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def run_step(name, cmd, cwd=ROOT, allow_fail=False):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"{name}.stdout.log"
    stderr_path = LOG_DIR / f"{name}.stderr.log"
    append(f"## {name}")
    append("```text")
    append(" ".join(str(x) for x in cmd))
    append("```")
    start = time.time()
    with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=err, text=True)
    elapsed = time.time() - start
    append(f"- exit_code: `{proc.returncode}`")
    append(f"- elapsed_sec: `{elapsed:.1f}`")
    append(f"- stdout: `{stdout_path}`")
    append(f"- stderr: `{stderr_path}`")
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")
    return proc.returncode


def fold_done(fold):
    oof = ARTIFACT / "oof" / RUN
    report = ARTIFACT / "reports" / RUN / f"fold_{fold}_metrics.json"
    return (oof / f"fold_{fold}_logits.npy").exists() and (oof / f"fold_{fold}_ids.txt").exists() and report.exists()


def summarize_fold(fold):
    path = ARTIFACT / "reports" / RUN / f"fold_{fold}_metrics.json"
    if not path.exists():
        append(f"- fold{fold}_metrics: missing")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best", {})
    append(
        f"- fold{fold}_best: epoch=`{best.get('epoch')}` "
        f"macro_f1=`{float(best.get('fold_val_macro_f1', 0.0)):.6f}` "
        f"nll=`{float(best.get('fold_val_nll', 0.0)):.6f}` "
        f"accuracy=`{float(best.get('fold_val_accuracy', 0.0)):.6f}`"
    )


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (STATE / "pid.txt").write_text(str(__import__("os").getpid()) + "\n", encoding="utf-8")
    STATUS.write_text(
        "# Cycle3 OOF Runner\n\n"
        f"- started_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n"
        f"- run: `{RUN}`\n"
        f"- config: `{CONFIG}`\n",
        encoding="utf-8",
    )

    for fold in FOLDS:
        if fold_done(fold):
            append(f"## train_fold{fold}")
            append("- skipped: existing fold artifacts found")
            summarize_fold(fold)
            continue
        cmd = [
            PYTHON,
            "pipeline_v4/train_fold.py",
            "--fold",
            str(fold),
            "--config",
            CONFIG,
            "--override",
            f"run_name={RUN}",
            "--override",
            "epochs=5",
            "--override",
            "max_len=384",
            "--override",
            "serializer=v2",
        ]
        run_step(f"train_fold{fold}", cmd)
        summarize_fold(fold)

    run_step("calibrate_5fold", [PYTHON, "pipeline_v4/calibrate.py", "--run", RUN, "--folds", "0,1,2,3,4"])
    run_step("build_oof_5fold", [PYTHON, "pipeline_v4/build_oof.py", "--run", RUN, "--folds", "0,1,2,3,4"])
    run_step("optimize_bias_5fold", [PYTHON, "pipeline_v4/optimize_bias.py", "--run", RUN])

    oof_report = ARTIFACT / "oof" / RUN / "oof_report.json"
    decision = ARTIFACT / "decision" / RUN / "bias.json"
    if oof_report.exists():
        append("## oof_report")
        append("```json")
        append(oof_report.read_text(encoding="utf-8"))
        append("```")
    if decision.exists():
        payload = json.loads(decision.read_text(encoding="utf-8"))
        append("## bias_decision")
        append(f"- f1_before: `{float(payload.get('f1_before', 0.0)):.6f}`")
        append(f"- f1_after: `{float(payload.get('f1_after', 0.0)):.6f}`")
        append(f"- crossval_A2B: `{float(payload.get('crossval_A2B', 0.0)):.6f}`")
        append(f"- crossval_B2A: `{float(payload.get('crossval_B2A', 0.0)):.6f}`")
        append(f"- adopted: `{payload.get('adopted')}`")

    append(f"- finished_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")


if __name__ == "__main__":
    main()
