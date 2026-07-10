import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "autoresearch_loop"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(name, args, timeout):
    started = time.time()
    out_dir = REPORT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{name}.stdout.log"
    stderr_path = out_dir / f"{name}.stderr.log"
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        stdout.write(f"\n\n===== {now()} START {name} =====\n")
        stdout.flush()
        proc = subprocess.run(
            [sys.executable, *args],
            cwd=str(ROOT),
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            check=False,
        )
        stdout.write(f"\n===== {now()} END {name} code={proc.returncode} elapsed={time.time() - started:.1f}s =====\n")
    return {
        "name": name,
        "code": proc.returncode,
        "elapsed_sec": round(time.time() - started, 1),
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
    }


def best_from_csv(path):
    path = ROOT / path
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    def key(row):
        try:
            return float(row.get("delta", row.get("macro_f1", 0.0)))
        except Exception:
            return -999.0
    row = max(rows, key=key)
    return {k: row.get(k) for k in ["name", "macro_f1", "delta", "inspect_f1", "inspect_delta", "changed", "fixed_target_errors", "min_fold_delta"] if k in row}


def write_status(payload):
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Autoresearch Loop Status",
        "",
        f"- updated_at: `{now()}`",
        f"- started_at: `{payload['started_at']}`",
        f"- cycle: `{payload['cycle']}`",
        f"- stage: `{payload['stage']}`",
        f"- elapsed_hours: `{payload['elapsed_hours']:.2f}`",
        f"- remaining_hours: `{payload['remaining_hours']:.2f}`",
        f"- stop_after_hours: `{payload['stop_after_hours']}`",
        "",
        "## Current Best",
        "",
    ]
    for name, row in payload.get("best", {}).items():
        lines.append(f"- {name}: `{row}`")
    lines += ["", "## Last Commands", ""]
    for cmd in payload.get("last_commands", [])[-8:]:
        lines.append(f"- `{cmd['name']}` code={cmd['code']} elapsed={cmd['elapsed_sec']}s")
    lines += [
        "",
        "## Gate",
        "",
        f"- plus_0_03_gate: `{payload.get('plus_0_03_gate')}`",
        "- If gate opens, the next heartbeat should expand the same pairwise/meta-router pattern to execute and communicate bottlenecks.",
    ]
    (REPORT / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--sleep-min", type=float, default=10.0)
    args = parser.parse_args()

    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "pid.txt").write_text(str(os_getpid()), encoding="utf-8")
    started = time.time()
    payload = {
        "started_at": now(),
        "cycle": 0,
        "stage": "init",
        "elapsed_hours": 0.0,
        "remaining_hours": args.hours,
        "stop_after_hours": args.hours,
        "last_commands": [],
        "best": {},
        "plus_0_03_gate": False,
    }
    write_status(payload)

    while True:
        elapsed_h = (time.time() - started) / 3600.0
        if elapsed_h >= args.hours:
            payload["stage"] = "finished_time_budget"
            payload["elapsed_hours"] = elapsed_h
            payload["remaining_hours"] = 0.0
            write_status(payload)
            break
        payload["cycle"] += 1
        payload["stage"] = "running_cycle"
        payload["elapsed_hours"] = elapsed_h
        payload["remaining_hours"] = max(args.hours - elapsed_h, 0.0)
        write_status(payload)

        commands = [
            ("replay_audit", ["scripts/audit_replay_leak.py", "--data-dir", "data", "--out-dir", "artifacts/reports/replay_audit"], 600),
            ("inspect_autoresearch", ["scripts/run_inspect_autoresearch.py"], 1800),
            ("prob_blend_autoresearch", ["scripts/run_prob_blend_autoresearch.py"], 600),
            ("meta_router_autoresearch", ["scripts/run_meta_router_autoresearch.py"], 3600),
        ]
        for name, cmd, timeout in commands:
            payload["stage"] = f"running_{name}"
            write_status(payload)
            try:
                result = run_cmd(name, cmd, timeout)
            except subprocess.TimeoutExpired:
                result = {"name": name, "code": "timeout", "elapsed_sec": timeout, "stdout": "", "stderr": ""}
            payload["last_commands"].append(result)
            payload["best"] = {
                "inspect": best_from_csv("reports/inspect_autoresearch/results.csv"),
                "prob_blend": best_from_csv("reports/prob_blend_autoresearch/results.csv"),
                "meta_router": best_from_csv("reports/meta_router_autoresearch/results.csv"),
            }
            best_deltas = []
            for row in payload["best"].values():
                if row and row.get("delta") not in (None, ""):
                    try:
                        best_deltas.append(float(row["delta"]))
                    except Exception:
                        pass
            payload["plus_0_03_gate"] = bool(best_deltas and max(best_deltas) >= 0.03)
            write_status(payload)

        payload["stage"] = "sleeping"
        payload["elapsed_hours"] = (time.time() - started) / 3600.0
        payload["remaining_hours"] = max(args.hours - payload["elapsed_hours"], 0.0)
        write_status(payload)
        time.sleep(max(args.sleep_min, 0.1) * 60.0)


def os_getpid():
    import os

    return os.getpid()


if __name__ == "__main__":
    main()
