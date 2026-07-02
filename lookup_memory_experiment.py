import argparse
import csv
import json
import os
from collections import Counter, defaultdict

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


def normalize_ws(text):
    return " ".join((text or "").split())


def key_of(text, mode):
    if mode == "raw":
        return text or ""
    if mode == "norm_ws":
        return normalize_ws(text)
    raise ValueError(f"unknown key mode: {mode}")


def iter_user_to_next_action(sample, key_mode):
    history = sample.get("history") or []
    for i, turn in enumerate(history):
        if turn.get("role") != "user":
            continue
        content = turn.get("content")
        if not content:
            continue
        for later in history[i + 1 :]:
            if later.get("role") == "assistant_action" and later.get("name"):
                yield key_of(content, key_mode), later["name"]
                break


def collect_counts(samples, key_mode):
    counts = defaultdict(Counter)
    occurrence_count = 0
    for sample in samples:
        for key, action in iter_user_to_next_action(sample, key_mode):
            counts[key][action] += 1
            occurrence_count += 1
    return counts, occurrence_count


def build_lookup(samples, key_mode="raw", policy="majority"):
    counts, occurrence_count = collect_counts(samples, key_mode)
    lookup = {}
    conflict_keys = 0
    for key, counter in counts.items():
        if not key:
            continue
        if len(counter) > 1:
            conflict_keys += 1
        if policy == "last":
            # Reproduce the proposed overwrite behavior.
            continue
        if policy == "majority":
            lookup[key] = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0][0]
        elif policy == "unique":
            if len(counter) == 1:
                lookup[key] = next(iter(counter))
        else:
            raise ValueError(f"unknown policy: {policy}")

    if policy == "last":
        for sample in samples:
            for key, action in iter_user_to_next_action(sample, key_mode):
                if key:
                    lookup[key] = action

    stats = {
        "source_samples": len(samples),
        "history_pairs": occurrence_count,
        "unique_keys": len(counts),
        "lookup_keys": len(lookup),
        "conflict_keys": conflict_keys,
        "conflict_rate": conflict_keys / max(len(counts), 1),
    }
    return lookup, counts, stats


def load_predictions(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def score(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=ALL_CLASSES, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def evaluate_override(name, val_rows, sample_by_id, lookup, key_mode):
    y_true = [row["y_true"] for row in val_rows]
    base_pred = [row["y_pred"] for row in val_rows]
    pred = []
    covered = 0
    covered_correct = 0
    changed = 0
    changed_good = 0
    changed_bad = 0
    per_class_hits = Counter()
    per_class_correct = Counter()
    examples = []

    for row, base in zip(val_rows, base_pred):
        sample = sample_by_id[row["id"]]
        key = key_of(sample.get("current_prompt"), key_mode)
        override = lookup.get(key)
        final = override or base
        pred.append(final)
        if override:
            covered += 1
            per_class_hits[row["y_true"]] += 1
            if override == row["y_true"]:
                covered_correct += 1
                per_class_correct[row["y_true"]] += 1
            if override != base:
                changed += 1
                if override == row["y_true"] and base != row["y_true"]:
                    changed_good += 1
                    if len(examples) < 8:
                        examples.append(
                            {
                                "id": row["id"],
                                "y_true": row["y_true"],
                                "base": base,
                                "override": override,
                                "current_prompt": sample.get("current_prompt", "")[:140],
                            }
                        )
                elif override != row["y_true"] and base == row["y_true"]:
                    changed_bad += 1

    base_scores = score(y_true, base_pred)
    final_scores = score(y_true, pred)
    return {
        "name": name,
        "base_macro_f1": base_scores["macro_f1"],
        "macro_f1": final_scores["macro_f1"],
        "delta_macro_f1": final_scores["macro_f1"] - base_scores["macro_f1"],
        "base_accuracy": base_scores["accuracy"],
        "accuracy": final_scores["accuracy"],
        "delta_accuracy": final_scores["accuracy"] - base_scores["accuracy"],
        "coverage": covered,
        "coverage_rate": covered / max(len(val_rows), 1),
        "covered_accuracy": covered_correct / max(covered, 1),
        "changed": changed,
        "changed_good": changed_good,
        "changed_bad": changed_bad,
        "per_class_hits": dict(per_class_hits),
        "per_class_correct": dict(per_class_correct),
        "good_examples": examples,
    }


def evaluate_self_history(samples, labels, key_mode, policy):
    y_true = []
    pred = []
    covered = 0
    covered_correct = 0
    for sample in samples:
        lookup, _, _ = build_lookup([sample], key_mode=key_mode, policy=policy)
        key = key_of(sample.get("current_prompt"), key_mode)
        override = lookup.get(key)
        if not override:
            continue
        covered += 1
        true = labels[sample["id"]]
        y_true.append(true)
        pred.append(override)
        if override == true:
            covered_correct += 1
    return {
        "covered": covered,
        "coverage_rate": covered / max(len(samples), 1),
        "covered_accuracy": covered_correct / max(covered, 1),
        "covered_macro_f1": f1_score(y_true, pred, labels=ALL_CLASSES, average="macro", zero_division=0)
        if covered
        else 0.0,
    }


def public_test_probe(train_samples, key_mode, policy, data_dir):
    test_path = os.path.join(data_dir, "test.jsonl")
    if not os.path.exists(test_path):
        return None
    test_samples = load_jsonl(test_path)
    lookup_train, _, train_stats = build_lookup(train_samples, key_mode=key_mode, policy=policy)
    lookup_all, _, all_stats = build_lookup(train_samples + test_samples, key_mode=key_mode, policy=policy)

    rows = []
    for sample in test_samples:
        key = key_of(sample.get("current_prompt"), key_mode)
        rows.append(
            {
                "id": sample["id"],
                "train_lookup": lookup_train.get(key),
                "train_plus_test_lookup": lookup_all.get(key),
                "prompt": sample.get("current_prompt", "")[:140],
            }
        )
    return {
        "test_samples": len(test_samples),
        "train_stats": train_stats,
        "all_stats": all_stats,
        "train_hits": sum(1 for row in rows if row["train_lookup"]),
        "train_plus_test_hits": sum(1 for row in rows if row["train_plus_test_lookup"]),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--pred-csv",
        default=os.path.join("reports", "exp_advanced_action_routing", "predictions_valid_best.csv"),
    )
    parser.add_argument("--out", default=os.path.join("reports", "lookup_memory_experiment.json"))
    args = parser.parse_args()

    samples = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    sample_by_id = {sample["id"]: sample for sample in samples}
    with open(os.path.join(args.data_dir, "train_labels.csv"), encoding="utf-8", newline="") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}

    val_rows = load_predictions(args.pred_csv)
    val_ids = {row["id"] for row in val_rows}
    train_split_samples = [sample for sample in samples if sample["id"] not in val_ids]
    val_samples = [sample_by_id[row["id"]] for row in val_rows]

    results = {
        "n_train_all": len(samples),
        "n_train_split": len(train_split_samples),
        "n_val": len(val_rows),
        "experiments": [],
        "self_history": [],
        "public_test": [],
    }

    for key_mode in ["raw", "norm_ws"]:
        for policy in ["last", "majority", "unique"]:
            for source_name, source_samples in [
                ("split_train_history_only", train_split_samples),
                ("split_train_plus_val_history_transductive", train_split_samples + val_samples),
                ("all_train_history_optimistic", samples),
            ]:
                lookup, _, stats = build_lookup(source_samples, key_mode=key_mode, policy=policy)
                row = evaluate_override(
                    f"{source_name}__{key_mode}__{policy}",
                    val_rows,
                    sample_by_id,
                    lookup,
                    key_mode,
                )
                row["lookup_stats"] = stats
                results["experiments"].append(row)

            self_row = evaluate_self_history(samples, labels, key_mode=key_mode, policy=policy)
            self_row["name"] = f"self_history__{key_mode}__{policy}"
            results["self_history"].append(self_row)

            probe = public_test_probe(samples, key_mode=key_mode, policy=policy, data_dir=args.data_dir)
            if probe:
                probe["name"] = f"public_test__{key_mode}__{policy}"
                results["public_test"].append(probe)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"saved: {args.out}")
    print("\nTop override experiments:")
    for row in sorted(results["experiments"], key=lambda x: x["macro_f1"], reverse=True)[:12]:
        print(
            f"{row['name']}: macro={row['macro_f1']:.6f} "
            f"delta={row['delta_macro_f1']:+.6f} cov={row['coverage']} "
            f"cov_acc={row['covered_accuracy']:.3f} changed={row['changed']} "
            f"good/bad={row['changed_good']}/{row['changed_bad']}"
        )

    print("\nSelf-history exact repeat:")
    for row in results["self_history"]:
        print(
            f"{row['name']}: covered={row['covered']} "
            f"coverage={row['coverage_rate']:.4f} acc={row['covered_accuracy']:.3f}"
        )

    print("\nPublic test probe:")
    for row in results["public_test"]:
        print(
            f"{row['name']}: test={row['test_samples']} train_hits={row['train_hits']} "
            f"train_plus_test_hits={row['train_plus_test_hits']}"
        )


if __name__ == "__main__":
    main()
