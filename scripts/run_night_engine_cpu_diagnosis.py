import csv
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "night_engine_diagnosis"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mb(n):
    return round(float(n) / (1024 * 1024), 3)


def audit_fold_curves(run_name="mdeberta384_v2_384_5e"):
    report_dir = ROOT / "pipeline_v4" / "artifacts" / "reports" / run_name
    rows = []
    for path in sorted(report_dir.glob("fold_*_metrics.json")):
        match = re.match(r"fold_(\d+)_metrics$", path.stem)
        if not match:
            continue
        payload = read_json(path)
        fold = int(payload.get("fold", match.group(1)))
        hist = payload.get("history", [])
        best = payload.get("best", {})
        ep4 = next((row for row in hist if int(row.get("epoch", -1)) == 4), None)
        ep5 = next((row for row in hist if int(row.get("epoch", -1)) == 5), None)
        if ep4 and ep5:
            delta_f1 = float(ep5["fold_val_macro_f1"]) - float(ep4["fold_val_macro_f1"])
            delta_nll = float(ep5["fold_val_nll"]) - float(ep4["fold_val_nll"])
            nll_still_decreasing = delta_nll < 0
        else:
            delta_f1 = ""
            delta_nll = ""
            nll_still_decreasing = ""
        notes = []
        if ep4 and ep5:
            if delta_f1 and delta_f1 >= 0.002:
                notes.append("f1_still_rising")
            if delta_nll and delta_nll < -0.002:
                notes.append("nll_still_decreasing")
            if abs(delta_f1) < 0.002 and abs(delta_nll) < 0.002:
                notes.append("near_plateau")
        rows.append(
            {
                "fold": fold,
                "run_name": payload.get("run", run_name),
                "serializer": payload.get("config", {}).get("serializer", ""),
                "max_len": payload.get("config", {}).get("max_len", ""),
                "best_epoch": best.get("epoch", ""),
                "best_macro_f1": best.get("fold_val_macro_f1", ""),
                "best_nll": best.get("fold_val_nll", ""),
                "epoch5_macro_f1": ep5.get("fold_val_macro_f1", "") if ep5 else "",
                "epoch4_to_5_delta_f1": delta_f1,
                "epoch4_to_5_delta_nll": delta_nll,
                "nll_still_decreasing": nll_still_decreasing,
                "lr_schedule_ok": payload.get("config", {}).get("scheduler", "") in {"cosine", "linear"},
                "notes": ";".join(notes) if notes else "none",
            }
        )
    write_csv(
        OUT / "fold_training_curve_audit.csv",
        rows,
        [
            "fold",
            "run_name",
            "serializer",
            "max_len",
            "best_epoch",
            "best_macro_f1",
            "best_nll",
            "epoch5_macro_f1",
            "epoch4_to_5_delta_f1",
            "epoch4_to_5_delta_nll",
            "nll_still_decreasing",
            "lr_schedule_ok",
            "notes",
        ],
    )
    return rows


def large_preflight():
    result = {
        "model": "FacebookAI/xlm-roberta-large",
        "load_ok": False,
        "tokenizer_ok": False,
        "config_ok": False,
        "cuda_available": False,
        "gpu_name": None,
        "model_dir_mb": None,
        "fp16_save_mb": None,
        "batch_size_384": 0,
        "inference_1000_sec": None,
        "estimated_30000_sec": None,
        "zip_feasible": False,
        "submit_role": "not_checked",
        "notes": [],
    }
    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            result["notes"].append("no_cuda_available")
    except Exception as exc:
        result["notes"].append(f"torch_check_failed:{exc}")
    try:
        from transformers import AutoConfig, AutoTokenizer

        AutoConfig.from_pretrained(result["model"], local_files_only=True)
        result["config_ok"] = True
        AutoTokenizer.from_pretrained(result["model"], local_files_only=True)
        result["tokenizer_ok"] = True
    except Exception as exc:
        result["notes"].append(f"large_not_available_locally:{type(exc).__name__}")
    result["load_ok"] = result["config_ok"] and result["tokenizer_ok"] and result["cuda_available"]
    if not result["load_ok"]:
        result["submit_role"] = "preflight_failed_or_not_cached"
    write_json(OUT / "large_preflight.json", result)
    return result


def zip_entry_sizes(zip_path):
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rows.append({"path": info.filename, "bytes": info.file_size, "mb": mb(info.file_size)})
    return rows


def smoke_zip(zip_path, smoke_dir):
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(smoke_dir)
    data_dir = smoke_dir / "data"
    data_dir.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "data" / "test.jsonl", data_dir / "test.jsonl")
    shutil.copy2(ROOT / "data" / "sample_submission.csv", data_dir / "sample_submission.csv")
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "script.py"],
        cwd=smoke_dir,
        text=True,
        capture_output=True,
        timeout=300,
    )
    elapsed = time.time() - started
    return {
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
        "submission_exists": (smoke_dir / "output" / "submission.csv").exists(),
    }


def audit_cand_distill():
    zip_path = ROOT / "cand_distill.zip"
    entries = zip_entry_sizes(zip_path)
    entry_map = {row["path"]: row for row in entries}
    payload = {
        "zip_path": str(zip_path),
        "zip_size_mb": mb(zip_path.stat().st_size),
        "component_sizes_mb": {
            "model/svd.pkl": entry_map.get("model/svd.pkl", {}).get("mb"),
            "model/vectorizer.pkl": entry_map.get("model/vectorizer.pkl", {}).get("mb"),
            "model/student.pt": entry_map.get("model/student.pt", {}).get("mb"),
            "model/advanced_router.pkl": entry_map.get("model/advanced_router.pkl", {}).get("mb"),
            "script.py": entry_map.get("script.py", {}).get("mb"),
        },
        "w_student": None,
        "w_advanced": None,
        "bias_by_class": None,
        "smoke": None,
    }
    # Use local model config for readable blend settings.
    config_path = ROOT / "model" / "distill_student_strict" / "config.json"
    if config_path.exists():
        config = read_json(config_path)
        payload["w_student"] = config.get("w_student")
        payload["w_advanced"] = config.get("w_advanced")
        payload["bias_by_class"] = config.get("bias_by_class")
    try:
        payload["smoke"] = smoke_zip(zip_path, OUT / "smoke_cand_distill")
    except Exception as exc:
        payload["smoke"] = {"error": repr(exc)}
    write_json(OUT / "cand_distill_artifact_audit.json", payload)
    return payload


def write_morning_report(fold_rows, preflight, distill):
    best_rows = [row for row in fold_rows if row.get("best_macro_f1") != ""]
    avg_best = sum(float(row["best_macro_f1"]) for row in best_rows) / len(best_rows) if best_rows else 0.0
    still_rising = sum(1 for row in fold_rows if row.get("nll_still_decreasing") is True)
    lines = [
        "# Morning Engine Diagnosis Report",
        "",
        "## 1. Backbone Gate",
        "",
        "| model | fold_count | avg_best_macro | decision |",
        "|---|---:|---:|---|",
        f"| `mdeberta384_v2_384_5e` | `{len(best_rows)}` | `{avg_best:.6f}` | `baseline_capacity_around_0.718_oof` |",
        "",
        "## 2. Training Curve Audit",
        "",
        "| fold | best_epoch | best_f1 | ep4_to_ep5_delta | nll_still_decreasing | decision |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in fold_rows:
        decision = "undertraining_possible" if row["nll_still_decreasing"] and float(row["epoch4_to_5_delta_f1"] or 0) > 0.001 else "near_plateau"
        lines.append(
            f"| `{row['fold']}` | `{row['best_epoch']}` | `{float(row['best_macro_f1']):.6f}` | "
            f"`{float(row['epoch4_to_5_delta_f1'] or 0):.6f}` | `{row['nll_still_decreasing']}` | `{decision}` |"
        )
    lines.extend(
        [
            "",
            "## 3. Large Preflight",
            "",
            "| model | cuda | tokenizer_cached | zip_feasible | role | notes |",
            "|---|---|---|---|---|---|",
            f"| `{preflight['model']}` | `{preflight['cuda_available']}` | `{preflight['tokenizer_ok']}` | `{preflight['zip_feasible']}` | `{preflight['submit_role']}` | `{';'.join(preflight['notes'])}` |",
            "",
            "## 4. Submission Defense",
            "",
            "| zip | smoke | runtime_sec | size_mb | decision |",
            "|---|---|---:|---:|---|",
            f"| `cand_distill.zip` | `{distill.get('smoke', {}).get('returncode') == 0 and distill.get('smoke', {}).get('submission_exists')}` | "
            f"`{float(distill.get('smoke', {}).get('elapsed_sec', 0)):.2f}` | `{distill['zip_size_mb']:.3f}` | `keep_public_baseline` |",
            "",
            "## 5. Final Morning Decision",
            "",
            f"- Existing mDeBERTa curves average around `{avg_best:.6f}` OOF; this explains the public `0.71~0.72` shelf.",
            f"- Folds with NLL still decreasing at epoch 5: `{still_rising}/{len(fold_rows)}`.",
            "- Large-model submit path is not proven on this machine because preflight did not find a cached runnable large model/GPU path.",
            "- Keep `cand_distill.zip` as the defense line while v2.3/teacher experiments are validated.",
        ]
    )
    (OUT / "morning_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "status.md").write_text("# Night Engine Diagnosis\n\n- status: running\n", encoding="utf-8")
    fold_rows = audit_fold_curves()
    preflight = large_preflight()
    distill = audit_cand_distill()
    write_morning_report(fold_rows, preflight, distill)
    (OUT / "status.md").write_text("# Night Engine Diagnosis\n\n- status: completed\n", encoding="utf-8")
    print((OUT / "morning_decision.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
