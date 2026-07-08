import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "overnight_encoder_gates"
ARTIFACT = ROOT / "pipeline_v4" / "artifacts"
STATUS = REPORT / "status.md"
SUMMARY = REPORT / "SUMMARY.md"
PY = sys.executable


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_status(text):
    REPORT.mkdir(parents=True, exist_ok=True)
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    print(text, flush=True)


def run_cmd(name, cmd):
    REPORT.mkdir(parents=True, exist_ok=True)
    out = REPORT / f"{name}.stdout.log"
    err = REPORT / f"{name}.stderr.log"
    append_status(f"\n## {name}\n- started: `{now()}`\n- cmd: `{' '.join(map(str, cmd))}`")
    started = time.time()
    with open(out, "w", encoding="utf-8") as fo, open(err, "w", encoding="utf-8") as fe:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fo, stderr=fe)
    elapsed = time.time() - started
    append_status(f"- finished: `{now()}`")
    append_status(f"- returncode: `{proc.returncode}`")
    append_status(f"- elapsed_sec: `{elapsed:.1f}`")
    append_status(f"- stdout: `{out}`")
    append_status(f"- stderr: `{err}`")
    return proc.returncode


def metric_path(run_name):
    return ARTIFACT / "reports" / run_name / "fold_0_metrics.json"


def read_best(run_name):
    path = metric_path(run_name)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best", {})
    history = payload.get("history", [])
    return {
        "run": run_name,
        "best_epoch": best.get("epoch"),
        "best_macro_f1": float(best.get("fold_val_macro_f1", 0.0)),
        "best_nll": float(best.get("fold_val_nll", 0.0)),
        "best_accuracy": float(best.get("fold_val_accuracy", 0.0)),
        "history": history,
        "path": str(path),
    }


def smoke(name, model, serializer, max_len, trust=False):
    args = [
        PY,
        "scripts/encoder_smoke.py",
        "--model",
        model,
        "--serializer",
        serializer,
        "--max-len",
        str(max_len),
        "--audit-limit",
        "2000",
        "--out",
        str(REPORT / f"{name}.json"),
    ]
    if trust:
        args.append("--trust-remote-code")
    return run_cmd(f"{name}_smoke", args)


def train(name, config, epochs, run_name):
    return run_cmd(
        f"{name}_train_{epochs}e",
        [
            PY,
            "pipeline_v4/train_fold.py",
            "--fold",
            "0",
            "--config",
            config,
            "--epochs",
            str(epochs),
            "--override",
            f"run_name={run_name}",
        ],
    )


def write_summary(results):
    lines = [
        "# Overnight Encoder Gates",
        "",
        f"- updated_at: `{now()}`",
        "",
        "| stage | run | best_epoch | Macro-F1 | NLL | Accuracy | decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in results:
        best = row.get("best") or {}
        lines.append(
            f"| `{row['stage']}` | `{best.get('run', row.get('run', 'none'))}` | "
            f"`{best.get('best_epoch', '')}` | `{best.get('best_macro_f1', 0.0):.6f}` | "
            f"`{best.get('best_nll', 0.0):.6f}` | `{best.get('best_accuracy', 0.0):.6f}` | "
            f"{row.get('decision', '')} |"
        )
    lines.extend([
        "",
        "## Gates",
        "",
        "- Granite 1epoch pass: Macro-F1 >= `0.600`.",
        "- Granite 3epoch serious candidate: Macro-F1 >= `0.715`.",
        "- XLM-R-large 1epoch pass: Macro-F1 >= `0.580`.",
        "- XLM-R-large 3epoch serious candidate: Macro-F1 >= `0.715`.",
    ])
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(f"# Overnight Encoder Gates Status\n\n- started: `{now()}`\n", encoding="utf-8")
    (REPORT / "runner_pid.txt").write_text(str(__import__("os").getpid()) + "\n", encoding="ascii")
    results = []

    granite_model = "ibm-granite/granite-embedding-311m-multilingual-r2"
    granite_config = "pipeline_v4/configs/granite311_v2_384_gate.yaml"
    if smoke("granite311", granite_model, "v2_2", 384) == 0:
        run1 = "granite311_v2_384_1e_gate"
        rc = train("granite311", granite_config, 1, run1)
        best = read_best(run1)
        decision = "fail_or_missing"
        if rc == 0 and best:
            decision = "pass_1e_run_3e" if best["best_macro_f1"] >= 0.600 else "stop_granite_below_1e_gate"
        results.append({"stage": "granite_1e", "run": run1, "best": best, "decision": decision})
        write_summary(results)
        if best and best["best_macro_f1"] >= 0.600:
            run3 = "granite311_v2_384_3e_gate"
            rc = train("granite311", granite_config, 3, run3)
            best3 = read_best(run3)
            decision3 = "serious_candidate" if best3 and best3["best_macro_f1"] >= 0.715 else "diagnostic_only"
            if rc != 0:
                decision3 = "failed"
            results.append({"stage": "granite_3e", "run": run3, "best": best3, "decision": decision3})
            write_summary(results)
    else:
        results.append({"stage": "granite_smoke", "run": "none", "best": None, "decision": "smoke_failed"})
        write_summary(results)

    xlmr_model = "FacebookAI/xlm-roberta-large"
    xlmr_config = "pipeline_v4/configs/xlmr_large_v2_320_gate.yaml"
    if smoke("xlmr_large", xlmr_model, "v2_2", 320) == 0:
        run1 = "xlmr_large_v2_320_1e_gate"
        rc = train("xlmr_large", xlmr_config, 1, run1)
        best = read_best(run1)
        decision = "fail_or_missing"
        if rc == 0 and best:
            decision = "pass_1e_run_3e" if best["best_macro_f1"] >= 0.580 else "stop_xlmr_below_1e_gate"
        results.append({"stage": "xlmr_large_1e", "run": run1, "best": best, "decision": decision})
        write_summary(results)
        if best and best["best_macro_f1"] >= 0.580:
            run3 = "xlmr_large_v2_320_3e_gate"
            rc = train("xlmr_large", xlmr_config, 3, run3)
            best3 = read_best(run3)
            decision3 = "serious_candidate" if best3 and best3["best_macro_f1"] >= 0.715 else "diagnostic_only"
            if rc != 0:
                decision3 = "failed"
            results.append({"stage": "xlmr_large_3e", "run": run3, "best": best3, "decision": decision3})
            write_summary(results)
    else:
        results.append({"stage": "xlmr_large_smoke", "run": "none", "best": None, "decision": "smoke_failed"})
        write_summary(results)

    append_status(f"\n# Finished\n- finished_at: `{now()}`")
    write_summary(results)


if __name__ == "__main__":
    main()

