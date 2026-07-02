import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.data_io import load_train_samples
from pipeline_v4.serialize import serialize_for_tokenizer


ARTIFACT = Path("pipeline_v4/artifacts/night_ops")
STATUS = ARTIFACT / "status.md"
LOG_DIR = ARTIFACT / "logs"
BASE_CONFIG = Path("pipeline_v4/configs/mdeberta_a_local8gb.yaml")
RUN3_EPOCH3 = {"fold_val_nll": 0.7440481854642883, "fold_val_macro_f1": 0.6817017357844227}


def append(text=""):
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text, flush=True)


def run_cmd(name, args):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"{name}.stdout.log"
    err_path = LOG_DIR / f"{name}.stderr.log"
    append(f"## RUN {name}")
    append("```text")
    append(" ".join(args))
    append("```")
    started = time.time()
    with out_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(args, cwd=ROOT, stdout=out, stderr=err, text=True)
    elapsed = time.time() - started
    append(f"- exit_code: `{proc.returncode}`")
    append(f"- elapsed_sec: `{elapsed:.1f}`")
    append(f"- stdout: `{out_path}`")
    append(f"- stderr: `{err_path}`")
    if proc.returncode != 0:
        append(f"- status: FAILED")
    return proc.returncode


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def c0_check_accum():
    cfg = load_yaml(BASE_CONFIG)
    batch = int(cfg["batch_size"])
    accum = int(cfg["grad_accum"])
    append("# NIGHT OPS v2 Status")
    append("")
    append("## C0 Accum Check")
    append(f"- config: `{BASE_CONFIG}`")
    append(f"- batch_size: `{batch}`")
    append(f"- grad_accum: `{accum}`")
    append(f"- effective_batch: `{batch * accum}`")
    append(f"- verdict: `{'pass' if batch * accum == 32 else 'fail'}`")
    return batch * accum == 32


def c1_decision_stack():
    append("")
    append("## C1 fold0 Calibration/Bias")
    if run_cmd("c1_calibrate_mdeberta_a_fold0", [sys.executable, "pipeline_v4/calibrate.py", "--run", "mdeberta_a", "--folds", "0"]) != 0:
        return
    if run_cmd("c1_build_oof_mdeberta_a_fold0", [sys.executable, "pipeline_v4/build_oof.py", "--run", "mdeberta_a", "--folds", "0"]) != 0:
        return
    if run_cmd("c1_optimize_bias_mdeberta_a_fold0", [sys.executable, "pipeline_v4/optimize_bias.py", "--run", "mdeberta_a"]) != 0:
        return
    bias_path = Path("pipeline_v4/artifacts/decision/mdeberta_a/bias.json")
    if bias_path.exists():
        payload = json.loads(bias_path.read_text(encoding="utf-8"))
        append(f"- f1_before: `{payload.get('f1_before')}`")
        append(f"- f1_after: `{payload.get('f1_after')}`")
        append(f"- crossval_A2B: `{payload.get('crossval_A2B')}`")
        append(f"- crossval_B2A: `{payload.get('crossval_B2A')}`")
        focus = {"glob_pattern", "list_directory", "web_search", "lint_or_typecheck"}
        before = {row["class"]: row["f1"] for row in payload.get("classwise_before", [])}
        after = {row["class"]: row["f1"] for row in payload.get("classwise_after", [])}
        for cls in sorted(focus):
            append(f"- {cls}: `{before.get(cls)}` -> `{after.get(cls)}`")


def length_stats(tokenizer, samples, variant):
    lengths = []
    now_missing = 0
    last_missing = 0
    for sample in samples:
        text = serialize_for_tokenizer(sample, tokenizer, 10_000, variant)
        ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        lengths.append(len(ids))
        if "[NOW]" not in text:
            now_missing += 1
        if variant == "v2" and "[LAST]" not in text:
            last_missing += 1
    arr = np.asarray(lengths, dtype=np.int64)
    report = {
        "variant": variant,
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p97": float(np.percentile(arr, 97)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "over_320": int((arr > 320).sum()),
        "over_384": int((arr > 384).sum()),
        "over_448": int((arr > 448).sum()),
        "over_512": int((arr > 512).sum()),
        "pct_over_320": float((arr > 320).mean()),
        "pct_over_384": float((arr > 384).mean()),
        "pct_over_448": float((arr > 448).mean()),
        "pct_over_512": float((arr > 512).mean()),
        "now_missing": now_missing,
        "last_missing": last_missing,
    }
    return report


def c2_token_lengths():
    append("")
    append("## C2 Token Length")
    cfg = load_yaml(BASE_CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"], use_fast=True)
    samples = load_train_samples("data")
    reports = {
        "v1": length_stats(tokenizer, samples, "v1"),
        "v2": length_stats(tokenizer, samples, "v2"),
    }
    out_path = ARTIFACT / "token_length_report.json"
    out_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append(f"- report: `{out_path}`")
    for variant, report in reports.items():
        append(f"- {variant}: p50={report['p50']:.0f}, p90={report['p90']:.0f}, p97={report['p97']:.0f}, p99={report['p99']:.0f}, >512={report['pct_over_512']:.3f}")
    p97 = max(reports["v1"]["p97"], reports["v2"]["p97"])
    for candidate in [320, 384, 448, 512]:
        if p97 <= candidate:
            chosen = candidate
            break
    else:
        chosen = 512
    append(f"- chosen_max_len: `{chosen}`")
    return chosen


def c3_router_blend_note():
    append("")
    append("## C3 Router Blend")
    append("- skipped: current advanced router artifact is LinearSVC-based and does not expose aligned calibrated probabilities for fold0 v4 logits.")


def c4_intuition_note():
    append("")
    append("## C4 Intuition Proxy")
    summary = Path("artifacts/intuition/SUMMARY.md")
    if summary.exists():
        append(f"- existing report: `{summary}`")
        append("- usable signal: `I1 workflow flags`; rejected: `I4`, `I5`, `I6`, `I7`, `I9`, `I10`, `I3`.")
    else:
        append("- missing `artifacts/intuition/SUMMARY.md`; run `scripts/run_intuition_validation.py` separately.")


def metric_path(run_name):
    return Path("pipeline_v4/artifacts/reports") / run_name / "fold_0_metrics.json"


def best_metrics(run_name):
    path = metric_path(run_name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("best")


def train_fold(run_name, fold, max_len, epochs, serializer="v1", class_weight="none"):
    name = f"{run_name}_fold{fold}"
    args = [
        sys.executable, "pipeline_v4/train_fold.py",
        "--fold", str(fold),
        "--config", str(BASE_CONFIG),
        "--override", f"run_name={run_name}",
        "--override", f"max_len={max_len}",
        "--override", f"epochs={epochs}",
        "--override", f"serializer={serializer}",
        "--override", f"class_weight={class_weight}",
    ]
    return run_cmd(name, args)


def gpu_queue(max_len):
    append("")
    append("## GPU Queue")
    ga = f"diag_maxlen_{max_len}_1e"
    train_fold(ga, 0, max_len, 1, "v1")
    ga_best = best_metrics(ga)
    append(f"- G-A best: `{ga_best}`")

    gb = f"diag_v2bundle_{max_len}_3e"
    train_fold(gb, 0, max_len, 3, "v2")
    gb_best = best_metrics(gb)
    append(f"- G-B best: `{gb_best}`")
    v2_pass = False
    if gb_best:
        v2_pass = gb_best["fold_val_macro_f1"] >= RUN3_EPOCH3["fold_val_macro_f1"] + 0.005 and gb_best["fold_val_nll"] < RUN3_EPOCH3["fold_val_nll"]
    append(f"- G-B pass_rule: `{v2_pass}`")

    if v2_pass:
        append("## Full Queue Decision")
        append("- v2 passed fold0 3e gate; launching v2 5e full folds 0-4.")
        full_run = f"v2bundle_{max_len}_5e"
        folds = [0, 1, 2, 3, 4]
        serializer = "v2"
    else:
        append("## Full Queue Decision")
        append("- v2 did not clear gate; launching stable current serializer folds 1-4 to complete OOF.")
        full_run = "mdeberta_a"
        folds = [1, 2, 3, 4]
        serializer = "v1"
    for fold in folds:
        train_fold(full_run, fold, max_len if serializer == "v2" else 320, 5, serializer)
    append("- full queue finished")
    if full_run == "mdeberta_a":
        run_cmd("final_calibrate_mdeberta_a_all_available", [sys.executable, "pipeline_v4/calibrate.py", "--run", "mdeberta_a"])
        run_cmd("final_build_oof_mdeberta_a_all_available", [sys.executable, "pipeline_v4/build_oof.py", "--run", "mdeberta_a"])
        run_cmd("final_optimize_bias_mdeberta_a_all_available", [sys.executable, "pipeline_v4/optimize_bias.py", "--run", "mdeberta_a"])


def main():
    ok = c0_check_accum()
    if not ok:
        append("C0 failed; stopping before GPU queue.")
        return
    c1_decision_stack()
    max_len = c2_token_lengths()
    c3_router_blend_note()
    c4_intuition_note()
    gpu_queue(max_len)


if __name__ == "__main__":
    main()
