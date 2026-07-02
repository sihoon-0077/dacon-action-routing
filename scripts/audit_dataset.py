import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import ACTIONS, ACTION_TO_GROUP4
from src.io_utils import get_session_id, load_test, load_train, write_json
from src.state_features import (
    SIGNATURE_LEVELS,
    build_signature,
    bucket_result_summary,
    get_last_action,
    iter_history_pairs,
    last_result_bucket,
)


def entropy(counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count)


def ambiguity_stats(samples, level):
    table = defaultdict(Counter)
    for sample in samples:
        table[build_signature(sample, level)][sample["action"]] += 1
    supports = [sum(counter.values()) for counter in table.values()]
    top_ratios = [max(counter.values()) / sum(counter.values()) for counter in table.values()]
    entropies = [entropy(counter) for counter in table.values()]
    return {
        "level": level,
        "num_keys": len(table),
        "covered_rows": int(sum(supports)),
        "mean_support": float(sum(supports) / max(len(supports), 1)),
        "median_support": float(sorted(supports)[len(supports) // 2]) if supports else 0.0,
        "mean_entropy": float(sum(entropies) / max(len(entropies), 1)),
        "mean_top1_ratio": float(sum(top_ratios) / max(len(top_ratios), 1)),
        "keys_top1_ge_0_9": int(sum(r >= 0.9 for r in top_ratios)),
        "keys_multiple_labels": int(sum(len(counter) > 1 for counter in table.values())),
    }


def transition_rows(samples):
    action_transitions = Counter()
    group_transitions = Counter()
    result_to_action = Counter()
    last_result_to_action = Counter()
    for sample in samples:
        action = sample["action"]
        last = get_last_action(sample)
        result = last_result_bucket(sample)
        action_transitions[(last, action)] += 1
        group_transitions[(ACTION_TO_GROUP4.get(last, "NONE"), ACTION_TO_GROUP4[action])] += 1
        result_to_action[(result, action)] += 1
        last_result_to_action[(last, result, action)] += 1
    return {
        "last_action_to_action_top": [
            {"last_action": k[0], "action": k[1], "count": v}
            for k, v in action_transitions.most_common(80)
        ],
        "last_group_to_group_top": [
            {"last_group": k[0], "group": k[1], "count": v}
            for k, v in group_transitions.most_common(40)
        ],
        "result_to_action_top": [
            {"result_bucket": k[0], "action": k[1], "count": v}
            for k, v in result_to_action.most_common(80)
        ],
        "last_action_result_to_action_top": [
            {"last_action": k[0], "result_bucket": k[1], "action": k[2], "count": v}
            for k, v in last_result_to_action.most_common(80)
        ],
    }


def markdown_report(payload):
    lines = ["# Dataset Audit: Policy Frame", ""]
    basic = payload["basic"]
    lines += [
        "## Basic",
        "",
        f"- train rows: `{basic['train_rows']}`",
        f"- test rows: `{basic['test_rows']}`",
        f"- unique train sessions: `{basic['unique_train_sessions']}`",
        f"- turn_index min/max: `{basic['turn_min']}` / `{basic['turn_max']}`",
        f"- history length min/max: `{basic['history_len_min']}` / `{basic['history_len_max']}`",
        f"- missing required key rows: `{basic['missing_required_keys']}`",
        "",
        "## Label Distribution",
        "",
        "| action | count | ratio |",
        "|---|---:|---:|",
    ]
    for row in payload["label_distribution"]:
        lines.append(f"| `{row['action']}` | {row['count']} | {row['ratio']:.4f} |")
    lines += ["", "## Group Distribution", "", "| group | count | ratio |", "|---|---:|---:|"]
    for row in payload["group_distribution"]:
        lines.append(f"| `{row['group']}` | {row['count']} | {row['ratio']:.4f} |")
    lines += ["", "## Ambiguity By Signature", "", "| level | keys | mean support | mean entropy | mean top1 ratio | multi-label keys |", "|---|---:|---:|---:|---:|---:|"]
    for row in payload["ambiguity"]:
        lines.append(
            f"| `{row['level']}` | {row['num_keys']} | {row['mean_support']:.3f} | {row['mean_entropy']:.3f} | {row['mean_top1_ratio']:.3f} | {row['keys_multiple_labels']} |"
        )
    lines += ["", "## Top Last-Action Transitions", "", "| last_action | action | count |", "|---|---|---:|"]
    for row in payload["transitions"]["last_action_to_action_top"][:30]:
        lines.append(f"| `{row['last_action']}` | `{row['action']}` | {row['count']} |")
    lines += ["", "## Interpretation", "", "- Lower entropy after adding last actions/result buckets indicates the task is policy-state reconstruction, not prompt-only intent classification."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-md", default="artifacts/reports/dataset_audit_policy_frame.md")
    parser.add_argument("--out-json", default="artifacts/reports/dataset_audit_policy_frame.json")
    args = parser.parse_args()

    samples = load_train(args.data_dir)
    try:
        test_samples = load_test(args.data_dir)
    except FileNotFoundError:
        test_samples = []

    required = {"id", "session_meta", "history", "current_prompt", "action"}
    history_lens = [len(sample.get("history", []) or []) for sample in samples]
    turns = [
        (sample.get("session_meta", {}) or {}).get("turn_index")
        for sample in samples
        if (sample.get("session_meta", {}) or {}).get("turn_index") is not None
    ]
    label_counts = Counter(sample["action"] for sample in samples)
    group_counts = Counter(ACTION_TO_GROUP4[sample["action"]] for sample in samples)
    result_counts = Counter()
    for sample in samples:
        for _, action in iter_history_pairs(sample):
            if action:
                result_counts[bucket_result_summary(action.get("result_summary", ""))] += 1

    payload = {
        "basic": {
            "train_rows": len(samples),
            "test_rows": len(test_samples),
            "unique_train_sessions": len({get_session_id(sample["id"]) for sample in samples}),
            "turn_min": min(turns) if turns else None,
            "turn_max": max(turns) if turns else None,
            "history_len_min": min(history_lens),
            "history_len_max": max(history_lens),
            "missing_required_keys": sum(1 for sample in samples if not required.issubset(sample.keys())),
            "duplicated_current_prompt": len(samples) - len({sample.get("current_prompt", "") for sample in samples}),
        },
        "label_distribution": [
            {"action": action, "count": label_counts[action], "ratio": label_counts[action] / len(samples)}
            for action in ACTIONS
        ],
        "group_distribution": [
            {"group": group, "count": count, "ratio": count / len(samples)}
            for group, count in sorted(group_counts.items())
        ],
        "result_bucket_distribution": result_counts.most_common(),
        "ambiguity": [ambiguity_stats(samples, level) for level in SIGNATURE_LEVELS],
        "transitions": transition_rows(samples),
    }

    write_json(args.out_json, payload)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown_report(payload), encoding="utf-8")
    print(f"saved: {args.out_md}")
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
