import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script import ALL_CLASSES, advanced_open_files, safe_text  # noqa: E402
from scripts.run_inspect_bottleneck_experiments import (  # noqa: E402
    build_vectorizer,
    fast_router_text,
    load_fold_split,
    load_labels,
    predict_fast_router,
    read_jsonl,
    score_macro,
    session_of,
    train_fast_router,
)
from scripts.run_micro_execute_websearch_experiment import (  # noqa: E402
    execute_flags,
    execute_serializer,
    last_action_name,
    result_bucket,
)


PAIR = ("run_bash", "run_tests")
PAIR_SET = set(PAIR)
EXECUTE = ["run_bash", "run_tests", "lint_or_typecheck"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def stable_half(sample_id):
    sid = session_of(sample_id)
    digest = hashlib.md5(sid.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2


def classwise_f1(y_true, pred):
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=ALL_CLASSES, zero_division=0)
    return {
        label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, label in enumerate(ALL_CLASSES)
    }


def class_macro(y_true, pred, labels):
    return float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0))


def half_delta(y_true, base_pred, new_pred, mask):
    if int(mask.sum()) == 0:
        return 0.0
    return score_macro(y_true[mask], new_pred[mask]) - score_macro(y_true[mask], base_pred[mask])


def info_margin_bucket(info):
    margin = float(info.get("margin", 0.0))
    if margin <= 0.01:
        return "m_le_001"
    if margin <= 0.05:
        return "m_le_005"
    if margin <= 0.10:
        return "m_le_010"
    if margin <= 0.25:
        return "m_le_025"
    return "m_hi"


def flag_text(sample):
    flags = execute_flags(sample)
    return " ".join(f"{key}={int(bool(value))}" for key, value in sorted(flags.items()))


def command_hint_text(sample):
    prompt = safe_text(sample.get("current_prompt"), 1400)
    open_files = " ".join(advanced_open_files(sample)[:10])
    flags = execute_flags(sample)
    positives = [key for key, value in flags.items() if value]
    return "\n".join(
        [
            "[NOW] " + prompt,
            "[FLAGS] " + " ".join(positives),
            f"[LAST] action={last_action_name(sample)} result_bucket={result_bucket(sample)}",
            "[OPEN] " + open_files,
        ]
    )


def prompt_only_serializer(sample, info=None, pred=None):
    return safe_text(sample.get("current_prompt"), 1800)


def rich_serializer(sample, info=None, pred=None):
    return execute_serializer(sample)


def command_intent_serializer(sample, info=None, pred=None):
    return command_hint_text(sample)


def base_aware_serializer(sample, info=None, pred=None):
    info = info or {}
    pred = str(pred or info.get("pred") or "unknown")
    return "\n".join(
        [
            execute_serializer(sample),
            (
                f"[BASE_PAIR] pred={pred} top1={info.get('top1')} top2={info.get('top2')} "
                f"margin_bucket={info_margin_bucket(info)} margin={float(info.get('margin', 0.0)):.6f}"
            ),
            "[PAIR_HINT] " + flag_text(sample),
        ]
    )


SERIALIZERS = {
    "E1_prompt_only_pair": prompt_only_serializer,
    "E2_rich_context_pair": rich_serializer,
    "E3_command_intent_pair": command_intent_serializer,
    "E4_base_aware_pair": base_aware_serializer,
}


def train_binary_text_model(train_samples, train_y, train_infos, train_pred, train_indices, serializer, max_features, c=2.0):
    texts = [serializer(train_samples[i], train_infos[i] if train_infos else None, train_pred[i] if train_pred is not None else None) for i in train_indices]
    vectorizer = build_vectorizer(max_features=max_features, min_df=2)
    x = vectorizer.fit_transform(texts)
    model = LogisticRegression(
        C=c,
        max_iter=600,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(x, train_y[train_indices])
    return vectorizer, model


def predict_binary_text_model(vectorizer, model, samples, infos, base_pred, serializer):
    texts = [serializer(sample, info, pred) for sample, info, pred in zip(samples, infos, base_pred)]
    x = vectorizer.transform(texts)
    probs = model.predict_proba(x)
    idx = probs.argmax(axis=1)
    pred = np.asarray([str(model.classes_[i]) for i in idx], dtype=object)
    conf = probs.max(axis=1)
    return pred, conf


def pair_scopes(samples, base_pred, infos):
    no_lint = np.asarray([not execute_flags(sample)["EX_LINT_PROTECT"] for sample in samples], dtype=bool)
    top1 = np.asarray([str(info.get("top1")) for info in infos], dtype=object)
    top2 = np.asarray([str(info.get("top2")) for info in infos], dtype=object)
    margins = np.asarray([float(info.get("margin", 0.0)) for info in infos], dtype=np.float32)
    pred_pair = np.isin(base_pred, list(PAIR_SET))
    top2_pair = pred_pair | np.isin(top2, list(PAIR_SET))
    exact_pair = np.asarray([set([a, b]) == PAIR_SET for a, b in zip(top1, top2)], dtype=bool)
    return {
        "pred_pair_no_lint": pred_pair & no_lint,
        "top2_pair_no_lint": top2_pair & no_lint,
        "exact_pair_no_lint": exact_pair & no_lint,
        "exact_pair_margin025_no_lint": exact_pair & no_lint & (margins <= 0.25),
        "exact_pair_margin010_no_lint": exact_pair & no_lint & (margins <= 0.10),
    }


def opposite(label):
    if label == "run_bash":
        return "run_tests"
    if label == "run_tests":
        return "run_bash"
    return label


def evaluate_override(experiment, scope_name, threshold, y_true, base_pred, candidate_pred, candidate_conf, scope, half_a, half_b):
    apply_mask = scope & (candidate_conf >= threshold)
    new_pred = np.asarray(base_pred, dtype=object).copy()
    new_pred[apply_mask] = candidate_pred[apply_mask]
    changed = apply_mask & (new_pred != base_pred)

    before_support = base_pred[apply_mask] == y_true[apply_mask]
    after_support = new_pred[apply_mask] == y_true[apply_mask]
    before_changed = base_pred[changed] == y_true[changed]
    after_changed = new_pred[changed] == y_true[changed]

    base_macro = score_macro(y_true, base_pred)
    new_macro = score_macro(y_true, new_pred)
    base_cw = classwise_f1(y_true, base_pred)
    new_cw = classwise_f1(y_true, new_pred)
    pair_true = np.isin(y_true, list(PAIR_SET))
    pair_apply = apply_mask & pair_true
    row = {
        "experiment": experiment,
        "scope": scope_name,
        "threshold": threshold,
        "support": int(apply_mask.sum()),
        "override_count": int(changed.sum()),
        "support_accuracy_after": float(after_support.mean()) if len(after_support) else 0.0,
        "support_accuracy_base": float(before_support.mean()) if len(before_support) else 0.0,
        "changed_precision_after": float(after_changed.mean()) if len(after_changed) else 0.0,
        "changed_precision_base": float(before_changed.mean()) if len(before_changed) else 0.0,
        "net_gain_support": int(after_support.sum() - before_support.sum()) if len(after_support) else 0,
        "net_gain_changed": int(after_changed.sum() - before_changed.sum()) if len(after_changed) else 0,
        "base_macro_f1": base_macro,
        "new_macro_f1": new_macro,
        "macro_f1_delta": new_macro - base_macro,
        "pair_macro_f1_before": class_macro(y_true, base_pred, list(PAIR_SET)),
        "pair_macro_f1_after": class_macro(y_true, new_pred, list(PAIR_SET)),
        "pair_macro_f1_delta": class_macro(y_true, new_pred, list(PAIR_SET)) - class_macro(y_true, base_pred, list(PAIR_SET)),
        "run_bash_f1_delta": new_cw["run_bash"]["f1"] - base_cw["run_bash"]["f1"],
        "run_tests_f1_delta": new_cw["run_tests"]["f1"] - base_cw["run_tests"]["f1"],
        "lint_or_typecheck_f1_delta": new_cw["lint_or_typecheck"]["f1"] - base_cw["lint_or_typecheck"]["f1"],
        "pair_true_coverage": float(pair_apply.sum() / max(int(pair_true.sum()), 1)),
        "halfA_delta": half_delta(y_true, base_pred, new_pred, half_a),
        "halfB_delta": half_delta(y_true, base_pred, new_pred, half_b),
        "false_positive_top_classes": json.dumps(Counter(y_true[changed & (new_pred != y_true)]).most_common(8), ensure_ascii=False),
    }
    return row, new_pred, changed


def best_row(rows):
    if not rows:
        return None
    return max(rows, key=lambda row: (row["macro_f1_delta"], row["net_gain_changed"], row["changed_precision_after"]))


def is_adoptable(row):
    if not row:
        return False
    return (
        row["macro_f1_delta"] >= 0.0015
        and row["net_gain_changed"] >= 15
        and row["changed_precision_after"] >= row["changed_precision_base"] + 0.25
        and row["lint_or_typecheck_f1_delta"] >= -0.003
        and row["halfA_delta"] > 0
        and row["halfB_delta"] > 0
    )


def train_flip_keep_model(inner_calib_samples, y_calib, inner_pred, inner_infos, max_features):
    scopes = pair_scopes(inner_calib_samples, inner_pred, inner_infos)
    scope = scopes["pred_pair_no_lint"]
    idx = np.where(scope)[0]
    labels = []
    keep_idx = []
    for i in idx:
        pred = str(inner_pred[i])
        truth = str(y_calib[i])
        if pred not in PAIR_SET:
            continue
        keep_idx.append(i)
        labels.append("flip" if truth == opposite(pred) else "keep")
    if len(set(labels)) < 2:
        raise RuntimeError("flip/keep calibration split does not contain both classes")
    texts = [base_aware_serializer(inner_calib_samples[i], inner_infos[i], inner_pred[i]) for i in keep_idx]
    vectorizer = build_vectorizer(max_features=max_features, min_df=2)
    x = vectorizer.fit_transform(texts)
    model = LogisticRegression(
        C=1.2,
        max_iter=600,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(x, np.asarray(labels, dtype=object))
    return vectorizer, model, {"rows": len(keep_idx), "label_counts": dict(Counter(labels))}


def predict_flip_keep(vectorizer, model, samples, infos, base_pred):
    texts = [base_aware_serializer(sample, info, pred) for sample, info, pred in zip(samples, infos, base_pred)]
    x = vectorizer.transform(texts)
    probs = model.predict_proba(x)
    flip_col = list(model.classes_).index("flip")
    p_flip = probs[:, flip_col]
    candidate = np.asarray([opposite(str(pred)) if str(pred) in PAIR_SET else str(pred) for pred in base_pred], dtype=object)
    return candidate, p_flip


def make_inner_split(samples, train_idx):
    groups = np.asarray([session_of(samples[i]["id"]) for i in train_idx], dtype=object)
    y_dummy = np.zeros(len(train_idx), dtype=np.int64)
    inner_a, inner_b = next(GroupShuffleSplit(n_splits=1, test_size=0.24, random_state=77).split(train_idx, y_dummy, groups))
    return np.asarray(train_idx[inner_a], dtype=np.int64), np.asarray(train_idx[inner_b], dtype=np.int64)


def collect_changed_examples(samples, y_true, base_pred, final_pred, conf, limit=80):
    rows = []
    changed = final_pred != base_pred
    for i in np.where(changed)[0][:limit]:
        rows.append(
            {
                "id": samples[i]["id"],
                "from_base": str(base_pred[i]),
                "to_candidate": str(final_pred[i]),
                "true": str(y_true[i]),
                "correct_before": int(base_pred[i] == y_true[i]),
                "correct_after": int(final_pred[i] == y_true[i]),
                "confidence": float(conf[i]),
                "prompt": safe_text(samples[i].get("current_prompt"), 420),
                "last_action": last_action_name(samples[i]),
                "result_bucket": result_bucket(samples[i]),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../open/data")
    parser.add_argument("--fold-file", default="pipeline_v4/folds/fold_assignments.csv")
    parser.add_argument("--out-root", default="reports/run_tests_bash_deep")
    parser.add_argument("--router-cache-root", default="reports/inspect_bottleneck")
    parser.add_argument("--router-features", type=int, default=50_000)
    parser.add_argument("--pair-features", type=int, default=40_000)
    parser.add_argument("--inner-router-features", type=int, default=35_000)
    parser.add_argument("--refresh-router", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    data_dir = Path(args.data_dir)
    samples = read_jsonl(data_dir / "train.jsonl")
    labels = load_labels(data_dir / "train_labels.csv")
    y = np.asarray([labels[sample["id"]] for sample in samples], dtype=object)
    train_idx, val_idx, split_name = load_fold_split(samples, args.fold_file, fold=0)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    router = train_fast_router(samples, y, train_idx, Path(args.router_cache_root), args.router_features, refresh=args.refresh_router)
    base_pred, _, val_infos = predict_fast_router(val_samples, router)
    base_macro = score_macro(y_val, base_pred)
    halves = np.asarray([stable_half(sample["id"]) for sample in val_samples], dtype=np.int64)
    half_a = halves == 0
    half_b = halves == 1
    scopes = pair_scopes(val_samples, base_pred, val_infos)
    print(
        f"loaded split={split_name} train={len(train_idx)} val={len(val_idx)} base_macro={base_macro:.6f}",
        flush=True,
    )

    results = []
    best_predictions = {}
    experiment_best = []
    pair_train_idx = np.where(np.isin(y_train, list(PAIR_SET)))[0]
    pair_train_samples = [train_samples[i] for i in pair_train_idx]
    pair_train_pred, _, pair_train_infos = predict_fast_router(pair_train_samples, router)
    train_pred = np.full(len(train_samples), "unknown", dtype=object)
    train_infos = [None] * len(train_samples)
    for local_i, train_pos in enumerate(pair_train_idx):
        train_pred[train_pos] = pair_train_pred[local_i]
        train_infos[train_pos] = pair_train_infos[local_i]
    trained_pair_outputs = {}

    for experiment, serializer in SERIALIZERS.items():
        print(f"training {experiment} rows={len(pair_train_idx)}", flush=True)
        vectorizer, model = train_binary_text_model(
            train_samples,
            y_train,
            train_infos,
            train_pred,
            pair_train_idx,
            serializer,
            args.pair_features,
        )
        pred_pair, conf_pair = predict_binary_text_model(vectorizer, model, val_samples, val_infos, base_pred, serializer)
        trained_pair_outputs[experiment] = (vectorizer, model, pred_pair, conf_pair)
        rows_this = []
        for scope_name, scope in scopes.items():
            for threshold in THRESHOLDS:
                row, new_pred, _ = evaluate_override(
                    experiment,
                    scope_name,
                    threshold,
                    y_val,
                    base_pred,
                    pred_pair,
                    conf_pair,
                    scope,
                    half_a,
                    half_b,
                )
                row["model_note"] = "true_pair_binary_logreg"
                results.append(row)
                rows_this.append(row)
                best_predictions[(row["experiment"], row["scope"], row["threshold"])] = (new_pred, conf_pair)
        best = best_row(rows_this)
        experiment_best.append(best)
        print(f"{experiment} best delta={best['macro_f1_delta']:.6f} net={best['net_gain_changed']}", flush=True)

    inner_train_idx, inner_calib_idx = make_inner_split(samples, train_idx)
    inner_router = train_fast_router(
        samples,
        y,
        inner_train_idx,
        out_root / "inner_router",
        args.inner_router_features,
        refresh=args.refresh_router,
    )
    inner_calib_samples = [samples[i] for i in inner_calib_idx]
    y_inner_calib = y[inner_calib_idx]
    inner_pred, _, inner_infos = predict_fast_router(inner_calib_samples, inner_router)
    flip_vectorizer, flip_model, flip_info = train_flip_keep_model(
        inner_calib_samples,
        y_inner_calib,
        inner_pred,
        inner_infos,
        args.pair_features,
    )
    flip_candidate, flip_conf = predict_flip_keep(flip_vectorizer, flip_model, val_samples, val_infos, base_pred)
    flip_rows = []
    for scope_name, scope in scopes.items():
        for threshold in THRESHOLDS:
            row, new_pred, _ = evaluate_override(
                "E5_oof_flip_keep_meta",
                scope_name,
                threshold,
                y_val,
                base_pred,
                flip_candidate,
                flip_conf,
                scope,
                half_a,
                half_b,
            )
            row["model_note"] = "inner_split_flip_keep_logreg"
            results.append(row)
            flip_rows.append(row)
            best_predictions[(row["experiment"], row["scope"], row["threshold"])] = (new_pred, flip_conf)
    flip_best = best_row(flip_rows)
    experiment_best.append(flip_best)
    print(f"E5_oof_flip_keep_meta best delta={flip_best['macro_f1_delta']:.6f} net={flip_best['net_gain_changed']}", flush=True)

    # Consensus is evaluated as a derived high-precision setting using the best rich pair signal plus OOF flip signal.
    consensus_rows = []
    _, _, rich_pred, rich_conf = trained_pair_outputs["E2_rich_context_pair"]
    for scope_name, scope in scopes.items():
        for pair_threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            for flip_threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
                consensus_mask = (rich_pred != base_pred) & (rich_conf >= pair_threshold) & (flip_conf >= flip_threshold)
                candidate = np.asarray(base_pred, dtype=object).copy()
                candidate[consensus_mask] = rich_pred[consensus_mask]
                row, new_pred, _ = evaluate_override(
                    "E6_pair_plus_flip_consensus",
                    scope_name,
                    pair_threshold + flip_threshold / 100.0,
                    y_val,
                    base_pred,
                    candidate,
                    np.where(consensus_mask, 1.0, 0.0),
                    scope,
                    half_a,
                    half_b,
                )
                row["pair_threshold"] = pair_threshold
                row["flip_threshold"] = flip_threshold
                row["model_note"] = "rich_pair_and_oof_flip_agree"
                results.append(row)
                consensus_rows.append(row)
                best_predictions[(row["experiment"], row["scope"], row["threshold"])] = (new_pred, np.where(consensus_mask, np.minimum(rich_conf, flip_conf), 0.0))
    consensus_best = best_row(consensus_rows)
    experiment_best.append(consensus_best)
    print(
        f"E6_pair_plus_flip_consensus best delta={consensus_best['macro_f1_delta']:.6f} "
        f"net={consensus_best['net_gain_changed']}",
        flush=True,
    )

    fieldnames = [
        "experiment",
        "scope",
        "threshold",
        "pair_threshold",
        "flip_threshold",
        "support",
        "override_count",
        "support_accuracy_after",
        "support_accuracy_base",
        "changed_precision_after",
        "changed_precision_base",
        "net_gain_support",
        "net_gain_changed",
        "base_macro_f1",
        "new_macro_f1",
        "macro_f1_delta",
        "pair_macro_f1_before",
        "pair_macro_f1_after",
        "pair_macro_f1_delta",
        "run_bash_f1_delta",
        "run_tests_f1_delta",
        "lint_or_typecheck_f1_delta",
        "pair_true_coverage",
        "halfA_delta",
        "halfB_delta",
        "false_positive_top_classes",
        "model_note",
    ]
    write_csv(out_root / "experiment_results.csv", results, fieldnames)
    write_csv(out_root / "best_by_experiment.csv", experiment_best, fieldnames)

    best = best_row(results)
    final_pred, final_conf = best_predictions[(best["experiment"], best["scope"], best["threshold"])]
    example_rows = collect_changed_examples(val_samples, y_val, base_pred, final_pred, final_conf)
    write_csv(
        out_root / "best_changed_examples.csv",
        example_rows,
        [
            "id",
            "from_base",
            "to_candidate",
            "true",
            "correct_before",
            "correct_after",
            "confidence",
            "prompt",
            "last_action",
            "result_bucket",
        ],
    )

    base_cw = classwise_f1(y_val, base_pred)
    best_cw = classwise_f1(y_val, final_pred)
    class_rows = []
    for cls in ALL_CLASSES:
        class_rows.append(
            {
                "class": cls,
                "base_f1": base_cw[cls]["f1"],
                "best_f1": best_cw[cls]["f1"],
                "f1_delta": best_cw[cls]["f1"] - base_cw[cls]["f1"],
                "support": base_cw[cls]["support"],
            }
        )
    write_csv(out_root / "best_classwise_delta.csv", class_rows, ["class", "base_f1", "best_f1", "f1_delta", "support"])

    confusion_rows = []
    for true_label in PAIR:
        for pred_label in PAIR:
            confusion_rows.append(
                {
                    "true": true_label,
                    "base_pred": pred_label,
                    "count": int(((y_val == true_label) & (base_pred == pred_label)).sum()),
                }
            )
    write_csv(out_root / "base_pair_confusion.csv", confusion_rows, ["true", "base_pred", "count"])
    write_json(
        out_root / "best_config.json",
        {
            "split": split_name,
            "base_macro_f1": base_macro,
            "base_pair_macro_f1": class_macro(y_val, base_pred, list(PAIR_SET)),
            "best": best,
            "adoptable": is_adoptable(best),
            "inner_flip_info": flip_info,
            "inner_train_rows": int(len(inner_train_idx)),
            "inner_calib_rows": int(len(inner_calib_idx)),
        },
    )

    lines = [
        "# run_tests vs run_bash Deep Experiments",
        "",
        f"- timestamp: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- split: `{split_name}`",
        f"- base: `fast_flat local proxy`",
        f"- base Macro-F1: `{base_macro:.6f}`",
        f"- base pair Macro-F1: `{class_macro(y_val, base_pred, list(PAIR_SET)):.6f}`",
        f"- pair train rows: `{len(pair_train_idx)}`",
        f"- inner flip train/calib rows: `{len(inner_train_idx)}` / `{len(inner_calib_idx)}`",
        "",
        "## Best By Experiment",
        "",
    ]
    for row in experiment_best:
        lines.append(
            f"- `{row['experiment']}` scope=`{row['scope']}` thr=`{row['threshold']}` "
            f"delta=`{row['macro_f1_delta']:.6f}` pair_delta=`{row['pair_macro_f1_delta']:.6f}` "
            f"net=`{row['net_gain_changed']}` overrides=`{row['override_count']}` "
            f"changed_precision=`{row['changed_precision_after']:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Overall Best",
            "",
            f"- experiment: `{best['experiment']}`",
            f"- scope: `{best['scope']}`",
            f"- threshold: `{best['threshold']}`",
            f"- Macro-F1 delta: `{best['macro_f1_delta']:.6f}`",
            f"- pair Macro-F1 delta: `{best['pair_macro_f1_delta']:.6f}`",
            f"- net changed rows: `{best['net_gain_changed']}`",
            f"- overrides: `{best['override_count']}`",
            f"- run_bash F1 delta: `{best['run_bash_f1_delta']:.6f}`",
            f"- run_tests F1 delta: `{best['run_tests_f1_delta']:.6f}`",
            f"- lint_or_typecheck F1 delta: `{best['lint_or_typecheck_f1_delta']:.6f}`",
            f"- halfA/halfB delta: `{best['halfA_delta']:.6f}` / `{best['halfB_delta']:.6f}`",
            f"- adoptable by strict criteria: `{'YES' if is_adoptable(best) else 'NO'}`",
        ]
    )
    write_md(out_root / "summary.md", lines)

    with open(ROOT / "research.md", "a", encoding="utf-8") as f:
        f.write(
            "\n## run_tests vs run_bash Deep Experiments\n\n"
            f"- timestamp: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n"
            f"- base Macro-F1: `{base_macro:.6f}`; pair Macro-F1: `{class_macro(y_val, base_pred, list(PAIR_SET)):.6f}`.\n"
            f"- best: `{best['experiment']}` scope=`{best['scope']}` threshold=`{best['threshold']}` "
            f"macro_delta=`{best['macro_f1_delta']:.6f}` pair_delta=`{best['pair_macro_f1_delta']:.6f}` "
            f"net=`{best['net_gain_changed']}` overrides=`{best['override_count']}`.\n"
            f"- strict adoptable: `{'YES' if is_adoptable(best) else 'NO'}`.\n"
        )
    print(f"wrote {out_root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
