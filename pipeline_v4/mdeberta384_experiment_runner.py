import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RUN_NAME = "mdeberta384_v2_384_5e"
ARTIFACT = ROOT / "pipeline_v4" / "artifacts"
REPORT_DIR = ARTIFACT / "reports" / RUN_NAME
LOG_DIR = ARTIFACT / "mdeberta384" / "logs"
STATUS = ARTIFACT / "mdeberta384" / "status.md"


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


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_passed():
    path = REPORT_DIR / "token_audit.json"
    if not path.exists():
        return False, "missing token_audit.json"
    payload = load_json(path)
    row = next((row for row in payload.get("rows", []) if int(row.get("max_len")) == 384), None)
    if not row:
        return False, "missing max_len=384 audit row"
    kept = row.get("marker_kept_rate", {})
    required = ["[NOW]", "[LAST]", "[STATE]"]
    if not all(float(kept.get(marker, 0.0)) >= 1.0 for marker in required):
        return False, f"required marker kept rates failed: {kept}"
    if float(kept.get("[SEQ]", 0.0)) < 0.95:
        return False, f"[SEQ] kept rate below 0.95: {kept.get('[SEQ]')}"
    return True, (
        f"p50={row['p50']:.0f}, p90={row['p90']:.0f}, p95={row['p95']:.0f}, "
        f"p99={row['p99']:.0f}, over_rate={row['over_limit_rate']:.4f}, "
        f"avg_history={row['avg_history_pairs_kept']:.2f}"
    )


def fold_metrics(fold):
    path = REPORT_DIR / f"fold_{fold}_metrics.json"
    if not path.exists():
        return None
    return load_json(path).get("best")


def write_decision(metrics):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# mDeBERTa 384 v2 Run Report", ""]
    lines.append("## Config")
    lines.append("- model: `microsoft/mdeberta-v3-base`")
    lines.append("- serializer: `v2`")
    lines.append("- max_len: `384`")
    lines.append("- epochs: `5`")
    lines.append("- batch_size: `2`")
    lines.append("- grad_accum: `16`")
    lines.append("")
    lines.append("## Gate")
    lines.append("- full-train strong gate: `fold0 Macro-F1 >= 0.716`")
    lines.append("- full-train recommended gate: `fold0 Macro-F1 >= 0.712`")
    lines.append("- uncertain band: `0.705 <= fold0 Macro-F1 < 0.712`")
    lines.append("- stop gate: `< 0.705`")
    lines.append("")
    if metrics:
        f1 = float(metrics.get("fold_val_macro_f1", 0.0))
        lines.append("## Fold0 Result")
        lines.append(f"- best_epoch: `{metrics.get('epoch')}`")
        lines.append(f"- Macro-F1: `{f1:.6f}`")
        lines.append(f"- NLL: `{float(metrics.get('fold_val_nll')):.6f}`")
        lines.append(f"- Accuracy: `{float(metrics.get('fold_val_accuracy')):.6f}`")
        lines.append("")
        lines.append("## Decision")
        if f1 >= 0.716:
            lines.append("- decision: `full-train-strong`")
            lines.append("- reason: 384 is nearly equal to 512 fold0 performance.")
        elif f1 >= 0.712:
            lines.append("- decision: `full-train-recommended`")
            lines.append("- reason: performance loss may be worth runtime gains.")
        elif f1 >= 0.705:
            lines.append("- decision: `manual-review`")
            lines.append("- reason: uncertain band; inspect class-wise F1/NLL before full train.")
        else:
            lines.append("- decision: `stop`")
            lines.append("- reason: fold0 failed the minimum 384 gate.")
    else:
        lines.append("## Decision")
        lines.append("- decision: `pending`")
    (REPORT_DIR / "decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    STATUS.write_text("# mDeBERTa 384 v2 Status\n\n", encoding="utf-8")
    append(f"- started_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    append("## Phase 0 full512")
    full512_ep3 = ARTIFACT / "models" / "v2full_512_5e" / "full" / "model.pt"
    if full512_ep3.exists():
        append(f"- full512_checkpoint: `{full512_ep3}`")
    else:
        append("- full512_checkpoint: `missing`; previous full512 run was stopped before epoch checkpoint.")

    run_cmd(
        "token_audit_mdeberta_v2_384",
        [
            PYTHON,
            "pipeline_v4/audit_xlmr_tokens.py",
            "--model-name",
            "microsoft/mdeberta-v3-base",
            "--serializer",
            "v2",
            "--max-lens",
            "384",
            "--out-dir",
            str(REPORT_DIR),
        ],
    )
    ok, reason = audit_passed()
    append(f"- token_audit_gate: `{'pass' if ok else 'fail'}`")
    append(f"- token_audit_summary: `{reason}`")
    if not ok:
        write_decision(None)
        append("- stopped before fold0 because token audit failed.")
        return

    run_cmd(
        "train_fold0",
        [
            PYTHON,
            "pipeline_v4/train_fold.py",
            "--config",
            "pipeline_v4/configs/mdeberta_v2_384.yaml",
            "--fold",
            "0",
        ],
    )
    metrics = fold_metrics(0)
    append(f"- fold0_best: `{json.dumps(metrics, ensure_ascii=False)}`")
    write_decision(metrics)
    append(f"- decision_report: `{REPORT_DIR / 'decision_report.md'}`")
    append(f"- finished_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")


if __name__ == "__main__":
    main()
