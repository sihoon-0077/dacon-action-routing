import argparse
import csv
import json
import os
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score


ALL_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def session_id(sample_id):
    return sample_id.rsplit("-step_", 1)[0] if "-step_" in sample_id else sample_id


def iter_session_prompt_action_pairs(sample):
    sid = session_id(sample.get("id", ""))
    history = sample.get("history") or []
    for i, turn in enumerate(history):
        if turn.get("role") != "user":
            continue
        prompt = turn.get("content", "")
        if not prompt:
            continue
        next_action = None
        for later in history[i + 1 :]:
            if later.get("role") == "assistant_action":
                next_action = later.get("name")
                break
            if later.get("role") == "user":
                break
        if next_action:
            yield (sid, prompt), next_action


def build_session_lookup(samples):
    lookup = {}
    pair_count = 0
    collision_count = 0
    collision_examples = []
    for sample in samples:
        for key, action in iter_session_prompt_action_pairs(sample):
            pair_count += 1
            old = lookup.get(key)
            if old is not None and old != action:
                collision_count += 1
                if len(collision_examples) < 10:
                    collision_examples.append(
                        {
                            "session": key[0],
                            "prompt": key[1][:180],
                            "old": old,
                            "new": action,
                            "source_id": sample.get("id"),
                        }
                    )
            lookup[key] = action
    return lookup, {
        "source_samples": len(samples),
        "history_pairs": pair_count,
        "lookup_keys": len(lookup),
        "collision_count": collision_count,
        "collision_examples": collision_examples,
    }


def load_predictions(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def score(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=ALL_CLASSES, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def evaluate_lookup(name, val_rows, sample_by_id, lookup):
    y_true = [row["y_true"] for row in val_rows]
    base_pred = [row["y_pred"] for row in val_rows]
    final_pred = []
    covered = 0
    covered_correct = 0
    changed = 0
    changed_good = 0
    changed_bad = 0
    hit_by_class = Counter()
    correct_by_class = Counter()
    good_examples = []
    bad_examples = []

    for row, base in zip(val_rows, base_pred):
        sample = sample_by_id[row["id"]]
        key = (session_id(row["id"]), sample.get("current_prompt", ""))
        override = lookup.get(key)
        pred = override or base
        final_pred.append(pred)

        if override:
            covered += 1
            hit_by_class[row["y_true"]] += 1
            if override == row["y_true"]:
                covered_correct += 1
                correct_by_class[row["y_true"]] += 1
            if override != base:
                changed += 1
                item = {
                    "id": row["id"],
                    "true": row["y_true"],
                    "base": base,
                    "override": override,
                    "prompt": sample.get("current_prompt", "")[:180],
                }
                if override == row["y_true"] and base != row["y_true"]:
                    changed_good += 1
                    if len(good_examples) < 8:
                        good_examples.append(item)
                elif override != row["y_true"] and base == row["y_true"]:
                    changed_bad += 1
                    if len(bad_examples) < 8:
                        bad_examples.append(item)

    before = score(y_true, base_pred)
    after = score(y_true, final_pred)
    return {
        "name": name,
        "macro_f1_before": before["macro_f1"],
        "macro_f1_after": after["macro_f1"],
        "delta_macro_f1": after["macro_f1"] - before["macro_f1"],
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "delta_accuracy": after["accuracy"] - before["accuracy"],
        "coverage": covered,
        "coverage_rate": covered / max(len(val_rows), 1),
        "covered_accuracy": covered_correct / max(covered, 1),
        "changed": changed,
        "changed_good": changed_good,
        "changed_bad": changed_bad,
        "hit_by_class": dict(hit_by_class),
        "correct_by_class": dict(correct_by_class),
        "good_examples": good_examples,
        "bad_examples": bad_examples,
    }


def public_test_probe(train_samples, test_samples):
    lookup_train, train_stats = build_session_lookup(train_samples)
    lookup_all, all_stats = build_session_lookup(train_samples + test_samples)
    rows = []
    for sample in test_samples:
        key = (session_id(sample.get("id", "")), sample.get("current_prompt", ""))
        rows.append(
            {
                "id": sample.get("id"),
                "train_hit": lookup_train.get(key),
                "train_plus_test_hit": lookup_all.get(key),
                "prompt": sample.get("current_prompt", "")[:160],
            }
        )
    return {
        "test_samples": len(test_samples),
        "train_stats": train_stats,
        "all_stats": all_stats,
        "train_hits": sum(1 for row in rows if row["train_hit"]),
        "train_plus_test_hits": sum(1 for row in rows if row["train_plus_test_hit"]),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--pred-csv",
        default=os.path.join("reports", "exp_advanced_action_routing", "predictions_valid_best.csv"),
    )
    parser.add_argument("--out", default=os.path.join("reports", "session_lookup_validation.json"))
    args = parser.parse_args()

    samples = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    sample_by_id = {sample["id"]: sample for sample in samples}
    val_rows = load_predictions(args.pred_csv)
    val_ids = {row["id"] for row in val_rows}
    train_split_samples = [sample for sample in samples if sample["id"] not in val_ids]
    val_samples = [sample_by_id[row["id"]] for row in val_rows]

    experiments = []
    for name, source_samples in [
        ("A2-1_val_self", val_samples),
        ("A2-2_train_only", train_split_samples),
        ("A2-3_train_plus_val", train_split_samples + val_samples),
        ("A2-4_all_train_optimistic", samples),
    ]:
        lookup, stats = build_session_lookup(source_samples)
        row = evaluate_lookup(name, val_rows, sample_by_id, lookup)
        row["lookup_stats"] = stats
        experiments.append(row)

    public_probe = None
    test_path = os.path.join(args.data_dir, "test.jsonl")
    if os.path.exists(test_path):
        test_samples = load_jsonl(test_path)
        public_probe = public_test_probe(samples, test_samples)

    result = {
        "n_samples": len(samples),
        "n_validation": len(val_rows),
        "n_train_split": len(train_split_samples),
        "experiments": experiments,
        "public_test_probe": public_probe,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"saved: {args.out}")
    for row in experiments:
        print(
            f"{row['name']}: macro={row['macro_f1_after']:.6f} "
            f"delta={row['delta_macro_f1']:+.6f} coverage={row['coverage']} "
            f"cov_rate={row['coverage_rate']:.3f} cov_acc={row['covered_accuracy']:.3f} "
            f"changed={row['changed']} good/bad={row['changed_good']}/{row['changed_bad']}"
        )
    if public_probe:
        print(
            "public_test_probe: "
            f"train_hits={public_probe['train_hits']}/{public_probe['test_samples']} "
            f"train_plus_test_hits={public_probe['train_plus_test_hits']}/{public_probe['test_samples']}"
        )


if __name__ == "__main__":
    main()
