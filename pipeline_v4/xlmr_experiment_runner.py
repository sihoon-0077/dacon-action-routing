import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RUN_NAME = "xlmr_state_v1_512"
REPORT_DIR = ROOT / "pipeline_v4" / "artifacts" / "reports" / RUN_NAME
LOG_DIR = ROOT / "pipeline_v4" / "artifacts" / "xlmr_state_v1" / "logs"
STATUS = ROOT / "pipeline_v4" / "artifacts" / "xlmr_state_v1" / "status.md"


def append(line=""):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_cmd(name, cmd):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"{name}.stdout.log"
    err_path = LOG_DIR / f"{name}.stderr.log"
    append(f"## {name}")
    append("```text")
    append(" ".join(str(x) for x in cmd))
    append("```")
    start = time.time()
    with out_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run([str(x) for x in cmd], cwd=ROOT, stdout=out, stderr=err, text=True)
    elapsed = time.time() - start
    append(f"- exit_code: `{proc.returncode}`")
    append(f"- elapsed_sec: `{elapsed:.1f}`")
    append(f"- stdout: `{out_path}`")
    append(f"- stderr: `{err_path}`")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def fold_metrics(fold):
    path = REPORT_DIR / f"fold_{fold}_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("best")


def train_fold(fold):
    run_cmd(
        f"train_fold{fold}",
        [
            PYTHON,
            "pipeline_v4/train_fold.py",
            "--config",
            "pipeline_v4/configs/xlmr_state_v1_512.yaml",
            "--fold",
            str(fold),
        ],
    )
    metrics = fold_metrics(fold)
    append(f"- fold{fold}_best: `{json.dumps(metrics, ensure_ascii=False)}`")
    return metrics


def write_decision(fold0, fold1=None):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# XLM-R State v1 Run Report", ""]
    lines.append("## Config")
    lines.append("- model: `FacebookAI/xlm-roberta-base`")
    lines.append("- serializer: `xlmr_state_v1`")
    lines.append("- max_len: `512`")
    lines.append("- epochs: `3`")
    lines.append("- batch_size: `4`")
    lines.append("- grad_accum: `8`")
    lines.append("")
    lines.append("## Fold Results")
    lines.append("| fold | best_epoch | Macro-F1 | NLL | Accuracy |")
    lines.append("|---:|---:|---:|---:|---:|")
    for fold, metrics in [(0, fold0), (1, fold1)]:
        if metrics:
            lines.append(
                f"| {fold} | {metrics.get('epoch')} | {metrics.get('fold_val_macro_f1'):.6f} | "
                f"{metrics.get('fold_val_nll'):.6f} | {metrics.get('fold_val_accuracy'):.6f} |"
            )
    lines.append("")
    lines.append("## Decision")
    f0 = float(fold0.get("fold_val_macro_f1", 0.0)) if fold0 else 0.0
    f1 = float(fold1.get("fold_val_macro_f1", 0.0)) if fold1 else None
    if f1 is not None and (f0 + f1) / 2 >= 0.725:
        lines.append("- decision: `full-train-candidate`")
        lines.append("- reason: fold0/fold1 average cleared 0.725 gate.")
    elif f0 >= 0.720 and f1 is None:
        lines.append("- decision: `run-fold1`")
        lines.append("- reason: fold0 cleared 0.720 gate.")
    elif f0 >= 0.710:
        lines.append("- decision: `compare-or-tune`")
        lines.append("- reason: XLM-R is near the mDeBERTa/advanced-router band but has not clearly won.")
    else:
        lines.append("- decision: `stop-or-debug`")
        lines.append("- reason: fold0 did not clear 0.710 gate.")
    (REPORT_DIR / "decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    STATUS.write_text("# XLM-R State v1 Status\n\n", encoding="utf-8")
    append(f"- started_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    run_cmd(
        "token_audit",
        [
            PYTHON,
            "pipeline_v4/audit_xlmr_tokens.py",
            "--model-name",
            "FacebookAI/xlm-roberta-base",
            "--serializer",
            "xlmr_state_v1",
            "--max-lens",
            "256",
            "320",
            "384",
            "512",
            "--out-dir",
            str(REPORT_DIR),
        ],
    )
    fold0 = train_fold(0)
    fold1 = None
    if fold0 and float(fold0.get("fold_val_macro_f1", 0.0)) >= 0.720:
        append("- fold0_gate: `pass`; launching fold1")
        fold1 = train_fold(1)
    else:
        append("- fold0_gate: `stop`; fold1 skipped")
    write_decision(fold0, fold1)
    append(f"- decision_report: `{REPORT_DIR / 'decision_report.md'}`")
    append(f"- finished_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")


if __name__ == "__main__":
    main()
