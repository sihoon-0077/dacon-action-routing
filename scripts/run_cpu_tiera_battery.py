import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.constants import ALL_CLASSES
from pipeline_v4.common.data_io import load_train_samples
from pipeline_v4.serialize import (
    count_bucket_from_result,
    last_modified_ext,
    open_count_bucket,
    prompt_len_bucket,
    result_bucket_detail,
    workflow_state_v22,
)


GROUPS = {
    "inspect": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "communicate": ["ask_user", "plan_task", "web_search", "respond_only"],
    "execute": ["run_bash", "run_tests", "lint_or_typecheck"],
}
ACTION_TO_GROUP = {action: group for group, actions in GROUPS.items() for action in actions}
INSPECT = set(GROUPS["inspect"])
MODIFY = {"edit_file", "write_file", "apply_patch"}
EXECUTE = set(GROUPS["execute"])
COMMUNICATE = set(GROUPS["communicate"])
MULTI_RE = re.compile(r"\b(and then|also|first|then|next|both)\b|그리고|먼저|다음|또|둘 다|함께", re.I)
QUOTE_RE = re.compile(r"`[^`]+`|['\"][A-Za-z_][\w.:-]+['\"]")
DOC_FINAL_DECISION = {
    "I-1": "reject",
    "I-2": "weak_pass",
    "I-3": "pass",
    "I-4": "pass_observable",
    "I-5": "reject",
    "C-1": "pass_observable",
    "C-2": "pass_observable",
    "C-3": "weak_pass",
    "C-4": "reject",
    "C-5": "pass_observable",
    "E-1": "pass_strong",
    "E-2": "pass",
    "E-3": "reject",
    "E-4": "pass_observable",
    "E-5": "weak_replaced_by_E2",
}


def actions(sample):
    return [
        turn for turn in sample.get("history", []) or []
        if turn.get("role") == "assistant_action" and turn.get("name")
    ]


def last_action(sample):
    acts = actions(sample)
    return acts[-1] if acts else None


def last_action_name(sample):
    turn = last_action(sample)
    return turn.get("name") if turn else "none"


def last_group(sample):
    return ACTION_TO_GROUP.get(last_action_name(sample), "none")


def inspect_streak(sample):
    n = 0
    for turn in reversed(actions(sample)):
        if turn.get("name") in INSPECT:
            n += 1
        else:
            break
    if n >= 4:
        return "4+"
    return str(n)


def last_list_glob_count(sample):
    turn = last_action(sample)
    if not turn or turn.get("name") not in {"list_directory", "glob_pattern"}:
        return "none"
    return f"{turn.get('name')}:{count_bucket_from_result(turn.get('result_summary', ''))}"


def slash_bucket(sample):
    prompt = sample.get("current_prompt") or ""
    n = prompt.count("/") + prompt.count("\\")
    if n >= 2:
        return "2+"
    return str(n)


def symbol_bucket(sample):
    prompt = sample.get("current_prompt") or ""
    if QUOTE_RE.search(prompt):
        return "quoted_code"
    if re.search(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b|\b[a-z]+[A-Z][A-Za-z0-9]*\b", prompt):
        return "identifier"
    return "none"


def turn_bucket(sample):
    meta = sample.get("session_meta", {}) or {}
    try:
        turn = int(meta.get("turn_index", 99))
    except Exception:
        turn = 99
    if turn <= 1:
        return "t1"
    if turn <= 3:
        return "t2_3"
    if turn <= 8:
        return "t4_8"
    return "t9+"


def last_comm_bucket(sample):
    group = last_group(sample)
    if group == "communicate":
        return "last_comm"
    if group == "none":
        return "none"
    return "last_work"


def multi_bucket(sample):
    n = len(MULTI_RE.findall(sample.get("current_prompt") or ""))
    if n >= 2:
        return "2+"
    return str(n)


def ci_status(sample):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    return str(ws.get("last_ci_status") or "none")


def execute_state_bucket(sample):
    wf = workflow_state_v22(sample)
    return (
        f"test={wf['test']}|lint={wf['lint']}|"
        f"eat={wf['edits_after_test']}|eal={wf['edits_after_lint']}"
    )


def last_execute_bucket(sample):
    for turn in reversed(actions(sample)):
        if turn.get("name") in EXECUTE:
            name = turn.get("name")
            result = "fail" if result_bucket_detail(turn.get("result_summary", "")) == "fail" else "pass"
            return f"{name}:{result}"
    return "none"


def dominant_language(sample):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    mix = ws.get("language_mix", {}) or {}
    if not mix:
        return "none"
    return max(mix.items(), key=lambda item: float(item[1]))[0]


def distribution(samples, labels, target_actions, bucket_fn):
    target = set(target_actions)
    base_counts = Counter(label for label in labels if label in target)
    base_n = sum(base_counts.values())
    base_probs = {action: base_counts[action] / base_n for action in target_actions}
    bucket_counts = defaultdict(Counter)
    for sample, label in zip(samples, labels):
        if label not in target:
            continue
        bucket_counts[bucket_fn(sample)][label] += 1
    rows = []
    for bucket in sorted(bucket_counts):
        counts = bucket_counts[bucket]
        n = sum(counts.values())
        top_action, top_count = counts.most_common(1)[0]
        max_rel = 0.0
        max_abs = 0.0
        max_action = top_action
        row = {
            "bucket": bucket,
            "n": n,
            "top_action": top_action,
            "purity": top_count / n,
        }
        for action in target_actions:
            prob = counts[action] / n if n else 0.0
            base = base_probs[action]
            rel = (prob - base) / base if base > 0 else 0.0
            row[f"p_{action}"] = prob
            row[f"rel_{action}"] = rel
            if abs(rel) > abs(max_rel):
                max_rel = rel
                max_abs = prob - base
                max_action = action
        row["max_shift_action"] = max_action
        row["max_rel_shift"] = max_rel
        row["max_abs_shift"] = max_abs
        rows.append(row)
    return rows, base_probs


def decision(rows):
    eligible = [row for row in rows if row["n"] >= 100]
    if not eligible:
        return "reject"
    max_rel = max(abs(row["max_rel_shift"]) for row in eligible)
    max_abs = max(abs(row["max_abs_shift"]) for row in eligible)
    max_purity = max(row["purity"] for row in eligible)
    if max_rel >= 0.5 and (max_abs >= 0.08 or max_purity >= 0.55):
        return "pass"
    if max_rel >= 0.3 and (max_abs >= 0.04 or max_purity >= 0.45):
        return "weak_pass"
    return "reject"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_dist(value):
    return f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="artifacts/cpu_tiera_battery_15")
    args = parser.parse_args()

    samples = load_train_samples(args.data_dir)
    labels = [sample["action"] for sample in samples]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = [
        ("I-1", "inspect", "prompt slash depth", GROUPS["inspect"], slash_bucket),
        ("I-2", "inspect", "last list/glob count bucket", GROUPS["inspect"], last_list_glob_count),
        ("I-3", "inspect", "inspect streak", GROUPS["inspect"], inspect_streak),
        ("I-4", "inspect", "open file count", GROUPS["inspect"], open_count_bucket),
        ("I-5", "inspect", "quoted symbol / identifier", GROUPS["inspect"], symbol_bucket),
        ("C-1", "communicate", "turn bucket", GROUPS["communicate"], turn_bucket),
        ("C-2", "communicate", "last group communication chain", GROUPS["communicate"], last_comm_bucket),
        ("C-3", "communicate", "prompt length bucket", GROUPS["communicate"], prompt_len_bucket),
        ("C-4", "communicate", "multi demand marker count", GROUPS["communicate"], multi_bucket),
        ("C-5", "communicate", "last CI status", GROUPS["communicate"], ci_status),
        ("E-1", "execute", "test/lint state split", GROUPS["execute"], execute_state_bucket),
        ("E-2", "execute", "last modified extension", GROUPS["execute"], last_modified_ext),
        ("E-3", "execute", "last CI status", GROUPS["execute"], ci_status),
        ("E-4", "execute", "last execute self-repeat", GROUPS["execute"], last_execute_bucket),
        ("E-5", "execute", "dominant workspace language", GROUPS["execute"], dominant_language),
    ]

    summary = []
    payload = {"n": len(samples), "experiments": {}}
    for ident, front, name, target_actions, fn in specs:
        rows, base = distribution(samples, labels, target_actions, fn)
        verdict = decision(rows)
        csv_name = ident.lower().replace("-", "_") + ".csv"
        write_csv(out / csv_name, rows)
        eligible = [row for row in rows if row["n"] >= 100]
        best = max(eligible or rows, key=lambda row: abs(row["max_rel_shift"]))
        item = {
            "id": ident,
            "front": front,
            "name": name,
            "stat_decision": verdict,
            "final_decision": DOC_FINAL_DECISION[ident],
            "base": base,
            "best_bucket": best,
            "csv": csv_name,
        }
        payload["experiments"][ident] = item
        summary.append(item)

    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CPU Tier-A Battery 15 Reproduction",
        "",
        f"- data: `{len(samples)}` train rows",
        "- method: conditional label distribution only, no training",
        "- gate: relative class-probability movement of about 30%+ with enough support",
        "",
        "## Summary",
        "",
        "| ID | Front | Hypothesis | Stat | Final | Best bucket | N | Top | Max shift |",
        "|---|---|---|---|---|---|---:|---|---:|",
    ]
    for item in summary:
        best = item["best_bucket"]
        lines.append(
            f"| {item['id']} | {item['front']} | {item['name']} | {item['stat_decision']} | {item['final_decision']} | "
            f"`{best['bucket']}` | {best['n']} | `{best['top_action']}` {format_dist(best['purity'])} | "
            f"`{best['max_shift_action']}` {best['max_rel_shift']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Adopted Serializer v2.2 Cards",
            "",
            "- `test` and `lint` states split, plus `edits_after_test` / `edits_after_lint`.",
            "- `insp_streak` for long inspect chains.",
            "- `last_mod_ext` for execute-channel choice after edits.",
            "- `open_cnt` for inspect routing.",
            "- `count_bucket` for the last `list_directory` / `glob_pattern` result.",
            "- `len_bucket` as a low-cost communicate feature.",
            "",
            "Notes:",
            "- `stat` is the mechanical distribution gate from this reproduction.",
            "- `final` follows the supplied experiment ledger: observable or redundant signals are not all serializer features.",
            "- `C-4` and `E-5` show measurable movement here, but the final decision remains reject/replaced because the direction is not a clean new routing card.",
            "",
            "## Files",
            "",
        ]
    )
    for item in summary:
        lines.append(f"- `{item['id']}`: `{item['csv']}`")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "summary.md")


if __name__ == "__main__":
    main()
