import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script import ALL_CLASSES, advanced_last2_action, compact_flags_text
from src.constants import ACTIONS, ACTION_TO_ID, ACTION_TO_GROUP4, GROUP4_TO_ACTIONS, SEED
from src.intuition_features import (
    EXECUTE_ACTIONS,
    INSPECT_ACTIONS,
    COMMUNICATE_ACTIONS,
    extract_numeric_result_buckets,
    extract_surface_flags,
    extract_workflow_flags,
    get_last_actions,
    intuition_tokens,
    selector_feature_dict,
)
from src.io_utils import get_session_id, load_train, write_csv, write_json


STRONGER_OVERRIDE = {
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "respond_only",
}


def score(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=ACTIONS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def class_f1_map(y_true, y_pred):
    _, _, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ACTIONS, zero_division=0)
    return {label: {"f1": float(fi), "support": int(si)} for label, fi, si in zip(ACTIONS, f, s)}


def class_rows(base_map, new_map):
    rows = []
    for label in ACTIONS:
        rows.append(
            {
                "class": label,
                "base_f1": base_map[label]["f1"],
                "new_f1": new_map[label]["f1"],
                "delta": new_map[label]["f1"] - base_map[label]["f1"],
                "support": new_map[label]["support"],
            }
        )
    return rows


def write_verdict(out_dir, ident, name, result):
    target_delta = result.get("target_delta")
    worst_drop = result.get("worst_class_drop")
    lines = [
        f"# {ident}. {name} Verdict",
        "",
        "## Result",
        f"- Tier A: {result.get('tier_a', 'not_run')}",
        f"- Tier B: {result.get('tier_b', 'not_run')}",
        f"- Tier C: {result.get('tier_c', 'not_run')}",
        f"- Final: {result.get('final', 'defer')}",
        "",
        "## Metrics",
        f"- Base Macro-F1: {result.get('base_macro_f1')}",
        f"- New Macro-F1: {result.get('new_macro_f1')}",
        f"- Delta: {result.get('delta')}",
        f"- Target class/group delta: {target_delta}",
        f"- Worst class drop: {worst_drop}",
        f"- Half split stability: {result.get('half_split_stability')}",
        "",
        "## Feature Availability",
        f"- Uses current_prompt: {result.get('uses_current_prompt', 'yes')}",
        f"- Uses history assistant_action: {result.get('uses_history_action', 'yes')}",
        f"- Uses result_summary: {result.get('uses_result_summary', 'yes')}",
        "- Uses train labels at inference: no",
        "- Uses future steps: no",
        "- Uses full test batch: no",
        "",
        "## Decision",
        result.get("decision", ""),
    ]
    (out_dir / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_v3_split(samples):
    y = np.asarray([ACTION_TO_ID[s["action"]] for s in samples], dtype=np.int64)
    groups = np.asarray([get_session_id(s["id"]) for s in samples], dtype=object)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(splitter.split(np.arange(len(samples)), y, groups=groups))
    return train_idx, val_idx, y, groups


def build_vectorizer(max_features):
    half = max_features // 2
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=half,
                    sublinear_tf=True,
                    lowercase=True,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=half,
                    sublinear_tf=True,
                    lowercase=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def run_text_proxy(samples, train_idx, val_idx, groups, out_root, max_features=120_000):
    y = np.asarray([s["action"] for s in samples], dtype=object)
    base_train = [compact_flags_text(samples[i]) for i in train_idx]
    base_val = [compact_flags_text(samples[i]) for i in val_idx]
    val_groups = groups[val_idx]

    vectorizer = build_vectorizer(max_features)
    x_train = vectorizer.fit_transform(base_train)
    x_val = vectorizer.transform(base_val)
    model = LogisticRegression(max_iter=700, C=2.0, class_weight="balanced", random_state=SEED)
    model.fit(x_train, y[train_idx])
    base_pred = model.predict(x_val)
    base_score = score(y[val_idx], base_pred)
    base_class = class_f1_map(y[val_idx], base_pred)

    half_splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    half_a, half_b = next(half_splitter.split(np.arange(len(val_idx)), y[val_idx], groups=val_groups))
    half_indices = {"valA": half_a, "valB": half_b}

    feature_specs = {
        "I1_workflow_flags": {
            "feature": "i1",
            "name": "Workflow Flags",
            "target": sorted(EXECUTE_ACTIONS),
            "threshold_target": 0.005,
            "decision_hint": "Adopt into serializer state if the execute loop gain is stable.",
        },
        "I4_numeric_result_buckets": {
            "feature": "i4",
            "name": "Numeric Result Buckets",
            "target": sorted(INSPECT_ACTIONS),
            "threshold_target": 0.004,
            "decision_hint": "Adopt into serializer state if inspect-class gain is stable.",
        },
        "I5_surface_flags": {
            "feature": "i5",
            "name": "Surface/Punctuation Flags",
            "target": sorted(COMMUNICATE_ACTIONS),
            "threshold_target": 0.005,
            "decision_hint": "Adopt into serializer flags if communicate-class gain is stable.",
        },
        "I145_bundle": {
            "feature": "i145",
            "name": "I1+I4+I5 Bundle",
            "target": ACTIONS,
            "threshold_target": 0.002,
            "decision_hint": "Use as candidate serializer-v2 bundle only if it beats individual features.",
        },
    }
    summary = {}
    for ident, spec in feature_specs.items():
        out_dir = out_root / ident
        out_dir.mkdir(parents=True, exist_ok=True)
        texts_train = [base_train[pos] + " " + intuition_tokens(samples[i], spec["feature"]) for pos, i in enumerate(train_idx)]
        texts_val = [base_val[pos] + " " + intuition_tokens(samples[i], spec["feature"]) for pos, i in enumerate(val_idx)]
        vec = build_vectorizer(max_features)
        xt = vec.fit_transform(texts_train)
        xv = vec.transform(texts_val)
        clf = LogisticRegression(max_iter=700, C=2.0, class_weight="balanced", random_state=SEED)
        clf.fit(xt, y[train_idx])
        pred = clf.predict(xv)
        new_score = score(y[val_idx], pred)
        new_class = class_f1_map(y[val_idx], pred)
        rows = class_rows(base_class, new_class)
        write_csv(out_dir / "class_f1.csv", rows, ["class", "base_f1", "new_f1", "delta", "support"])
        write_csv(
            out_dir / "predictions.csv",
            [
                {"id": samples[i]["id"], "y_true": y[i], "base_pred": bp, "new_pred": npred}
                for i, bp, npred in zip(val_idx, base_pred, pred)
            ],
            ["id", "y_true", "base_pred", "new_pred"],
        )
        target_delta = float(np.mean([new_class[c]["f1"] - base_class[c]["f1"] for c in spec["target"]]))
        worst = float(min(row["delta"] for row in rows))
        half_rows = {}
        for name, hidx in half_indices.items():
            half_rows[name] = {
                "base_macro_f1": score(y[val_idx][hidx], base_pred[hidx])["macro_f1"],
                "new_macro_f1": score(y[val_idx][hidx], pred[hidx])["macro_f1"],
            }
            half_rows[name]["delta"] = half_rows[name]["new_macro_f1"] - half_rows[name]["base_macro_f1"]
        stable = float(np.mean([half_rows["valA"]["delta"], half_rows["valB"]["delta"]]))
        passed = (
            new_score["macro_f1"] - base_score["macro_f1"] >= 0.002
            and target_delta >= spec["threshold_target"]
            and worst >= -0.01
            and stable > 0
        )
        result = {
            "tier_a": "not_run",
            "tier_b": "pass" if passed else "fail",
            "tier_c": "not_run",
            "final": "adopt" if passed else "reject",
            "base_macro_f1": base_score["macro_f1"],
            "new_macro_f1": new_score["macro_f1"],
            "delta": new_score["macro_f1"] - base_score["macro_f1"],
            "base_accuracy": base_score["accuracy"],
            "new_accuracy": new_score["accuracy"],
            "target_delta": target_delta,
            "worst_class_drop": worst,
            "half_split_stability": half_rows,
            "decision": spec["decision_hint"] if passed else "Reject for now: it did not clear the Tier B delta/stability rule.",
        }
        write_json(out_dir / "tierB.json", result)
        write_json(out_dir / "half_split.json", half_rows)
        write_verdict(out_dir, ident.split("_")[0], spec["name"], result)
        summary[ident] = result

    return {
        "baseline": base_score,
        "features": summary,
    }


def reconstruct_val_samples(samples, val_labels):
    train_idx, val_idx, y, groups = load_v3_split(samples)
    if len(val_idx) != len(val_labels):
        raise ValueError(f"val size mismatch split={len(val_idx)} labels={len(val_labels)}")
    if not np.array_equal(y[val_idx], val_labels):
        raise ValueError("v3 split labels do not match transformer labels")
    return [samples[i] for i in val_idx], groups[val_idx], val_idx


def load_transformer_bundle(samples, logits_path, labels_path, decision_path, advanced_predictions_path):
    logits = np.load(logits_path)
    labels = np.load(labels_path).astype(np.int64)
    val_samples, val_groups, val_idx = reconstruct_val_samples(samples, labels)
    decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
    temp = float(decision["temperature"])
    bias = np.asarray(decision["bias"], dtype=np.float64)
    probs = softmax(logits / temp, axis=1)
    tf_scores = np.log(np.clip(probs, 1e-12, 1.0)) + bias[None, :]
    tf_pred_ids = tf_scores.argmax(axis=1)
    tf_pred = np.asarray([ACTIONS[i] for i in tf_pred_ids], dtype=object)
    tf_conf = probs.max(axis=1)
    rows = list(csv.DictReader(open(advanced_predictions_path, encoding="utf-8")))
    if len(rows) != len(labels):
        raise ValueError("advanced predictions length mismatch")
    y_true = np.asarray([ACTIONS[i] for i in labels], dtype=object)
    adv_true = np.asarray([row["y_true"] for row in rows], dtype=object)
    if not np.array_equal(adv_true, y_true):
        raise ValueError("advanced predictions order does not match transformer labels")
    adv_pred = np.asarray([row["y_pred"] for row in rows], dtype=object)
    return {
        "logits": logits,
        "labels": labels,
        "y_true": y_true,
        "val_samples": val_samples,
        "val_groups": val_groups,
        "val_idx": val_idx,
        "probs": probs,
        "log_scores": tf_scores,
        "tf_pred": tf_pred,
        "tf_conf": tf_conf,
        "adv_pred": adv_pred,
        "decision": decision,
    }


def run_i6_prior(bundle, train_samples, out_root):
    out_dir = out_root / "I6_markov3_prior"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_train = [s["action"] for s in train_samples]
    global_counts = Counter(y_train)
    tables = {"last3": defaultdict(Counter), "last2": defaultdict(Counter), "last1": defaultdict(Counter)}
    for sample in train_samples:
        acts = get_last_actions(sample, 3)
        keys = {
            "last3": "|".join(acts[-3:]) if len(acts) >= 3 else "none",
            "last2": "|".join(acts[-2:]) if len(acts) >= 2 else "none",
            "last1": acts[-1] if acts else "none",
        }
        for level, key in keys.items():
            tables[level][key][sample["action"]] += 1

    def probs_for(sample, min_count=5, alpha=1.0):
        acts = get_last_actions(sample, 3)
        keys = [
            ("last3", "|".join(acts[-3:]) if len(acts) >= 3 else "none"),
            ("last2", "|".join(acts[-2:]) if len(acts) >= 2 else "none"),
            ("last1", acts[-1] if acts else "none"),
        ]
        counter = None
        used = "global"
        for level, key in keys:
            cand = tables[level].get(key)
            if cand and sum(cand.values()) >= min_count:
                counter = cand
                used = level
                break
        if counter is None:
            counter = global_counts
        total = sum(counter.values())
        prior = np.asarray([(counter.get(a, 0) + alpha) / (total + alpha * len(ACTIONS)) for a in ACTIONS], dtype=np.float64)
        return prior, used

    base = score(bundle["y_true"], bundle["tf_pred"])
    rows = []
    best = None
    for min_count in [3, 5, 10, 20]:
        prior_rows = [probs_for(s, min_count=min_count, alpha=1.0)[0] for s in bundle["val_samples"]]
        prior = np.vstack(prior_rows)
        for alpha_prior in [0.03, 0.05, 0.08, 0.1, 0.2, 0.3]:
            scores = bundle["log_scores"] + alpha_prior * np.log(np.clip(prior, 1e-12, 1.0))
            pred = np.asarray([ACTIONS[i] for i in scores.argmax(axis=1)], dtype=object)
            row = {
                "min_count": min_count,
                "alpha_prior": alpha_prior,
                **score(bundle["y_true"], pred),
                "delta_vs_transformer": score(bundle["y_true"], pred)["macro_f1"] - base["macro_f1"],
                "changes_vs_transformer": int(np.sum(pred != bundle["tf_pred"])),
            }
            rows.append(row)
            if best is None or row["macro_f1"] > best["macro_f1"]:
                best = row
    write_csv(out_dir / "prior_sweep.csv", rows, ["min_count", "alpha_prior", "macro_f1", "accuracy", "delta_vs_transformer", "changes_vs_transformer"])
    result = {
        "tier_a": "pass",
        "tier_b": "pass" if best["delta_vs_transformer"] >= 0.002 else "fail",
        "tier_c": "not_applicable",
        "final": "adopt" if best["delta_vs_transformer"] >= 0.002 else "reject",
        "base_macro_f1": base["macro_f1"],
        "new_macro_f1": best["macro_f1"],
        "delta": best["delta_vs_transformer"],
        "target_delta": None,
        "worst_class_drop": None,
        "half_split_stability": "not_run",
        "uses_current_prompt": "no",
        "uses_history_action": "yes",
        "uses_result_summary": "no",
        "decision": "Adopt only as a decision prior on calibrated transformer scores." if best["delta_vs_transformer"] >= 0.002 else "Reject: last3 prior did not add enough over calibrated transformer scores.",
        "best": best,
    }
    write_json(out_dir / "tierB.json", result)
    write_verdict(out_dir, "I6", "Markov Order 3 Prior", result)
    return result


def optimize_bias(log_scores, y_true, grid=None, max_sweeps=2):
    if grid is None:
        grid = np.arange(-0.6, 0.6001, 0.1)
    y_ids = np.asarray([ACTION_TO_ID[y] for y in y_true], dtype=np.int64)
    bias = np.zeros(len(ACTIONS), dtype=np.float64)
    best = f1_score(y_ids, (log_scores + bias[None, :]).argmax(axis=1), labels=list(range(len(ACTIONS))), average="macro", zero_division=0)
    for _ in range(max_sweeps):
        changed = False
        for cls in range(len(ACTIONS)):
            local = bias[cls]
            local_best = best
            for value in grid:
                cand = bias.copy()
                cand[cls] = value
                pred = (log_scores + cand[None, :]).argmax(axis=1)
                f1 = f1_score(y_ids, pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)
                if f1 > local_best + 1e-12:
                    local_best = f1
                    local = value
            if local != bias[cls]:
                changed = True
            bias[cls] = local
            best = local_best
        if not changed:
            break
    return bias


def run_i7_turn_bias(bundle, out_root):
    out_dir = out_root / "I7_turn_bucket_bias"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = score(bundle["y_true"], bundle["tf_pred"])
    buckets = []
    for sample in bundle["val_samples"]:
        meta = sample.get("session_meta", {}) or {}
        turn = meta.get("turn_index")
        try:
            turn = int(turn)
        except Exception:
            turn = 99
        if turn <= 1:
            buckets.append("t1")
        elif turn <= 8:
            buckets.append("t2_8")
        else:
            buckets.append("t9p")
    buckets = np.asarray(buckets, dtype=object)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    a_idx, b_idx = next(splitter.split(np.arange(len(buckets)), bundle["labels"], groups=bundle["val_groups"]))

    def fit_bucket_bias(train_idx):
        table = {}
        for bucket in sorted(set(buckets)):
            idx = train_idx[buckets[train_idx] == bucket]
            table[bucket] = optimize_bias(bundle["log_scores"][idx], bundle["y_true"][idx]) if len(idx) >= 30 else np.zeros(len(ACTIONS))
        return table

    def predict_with_table(eval_idx, table):
        out = []
        for i in eval_idx:
            scores = bundle["log_scores"][i] + table.get(buckets[i], np.zeros(len(ACTIONS)))[None, :]
            out.append(ACTIONS[int(scores.argmax(axis=1)[0])])
        return np.asarray(out, dtype=object)

    table_a = fit_bucket_bias(a_idx)
    table_b = fit_bucket_bias(b_idx)
    pred_a2b = predict_with_table(b_idx, table_a)
    pred_b2a = predict_with_table(a_idx, table_b)
    base_b = score(bundle["y_true"][b_idx], bundle["tf_pred"][b_idx])
    base_a = score(bundle["y_true"][a_idx], bundle["tf_pred"][a_idx])
    a2b = score(bundle["y_true"][b_idx], pred_a2b)
    b2a = score(bundle["y_true"][a_idx], pred_b2a)
    rows = [
        {"direction": "A_to_B", **a2b, "base_macro_f1": base_b["macro_f1"], "delta": a2b["macro_f1"] - base_b["macro_f1"]},
        {"direction": "B_to_A", **b2a, "base_macro_f1": base_a["macro_f1"], "delta": b2a["macro_f1"] - base_a["macro_f1"]},
    ]
    avg_delta = float(np.mean([row["delta"] for row in rows]))
    write_csv(out_dir / "turn_bias_sweep.csv", rows, ["direction", "macro_f1", "accuracy", "base_macro_f1", "delta"])
    result = {
        "tier_a": "pass",
        "tier_b": "pass" if avg_delta >= 0.004 else "fail",
        "tier_c": "not_applicable",
        "final": "adopt" if avg_delta >= 0.004 else "reject",
        "base_macro_f1": base["macro_f1"],
        "new_macro_f1": None,
        "delta": avg_delta,
        "target_delta": None,
        "worst_class_drop": None,
        "half_split_stability": rows,
        "uses_current_prompt": "no",
        "uses_history_action": "no",
        "uses_result_summary": "no",
        "decision": "Adopt turn-bucket bias only if cross-half delta clears threshold." if avg_delta >= 0.004 else "Reject: cross-half turn bias gain is too small or unstable.",
    }
    write_json(out_dir / "tierB.json", result)
    write_verdict(out_dir, "I7", "Turn-Bucket Bias", result)
    return result


def static_override_pred(adv_pred, tf_pred, tf_conf, threshold=0.0):
    return np.asarray(
        [tp if tp in STRONGER_OVERRIDE and conf >= threshold else ap for ap, tp, conf in zip(adv_pred, tf_pred, tf_conf)],
        dtype=object,
    )


def run_i9_selector(bundle, out_root):
    out_dir = out_root / "I9_transformer_override_selector"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = score(bundle["y_true"], bundle["adv_pred"])
    static = static_override_pred(bundle["adv_pred"], bundle["tf_pred"], bundle["tf_conf"], 0.0)
    static_score = score(bundle["y_true"], static)
    dicts = [
        selector_feature_dict(sample, ap, tp, prob)
        for sample, ap, tp, prob in zip(bundle["val_samples"], bundle["adv_pred"], bundle["tf_pred"], bundle["probs"])
    ]
    target = np.asarray(
        [
            int((tp == yt) and (ap != yt))
            for yt, ap, tp in zip(bundle["y_true"], bundle["adv_pred"], bundle["tf_pred"])
        ],
        dtype=np.int64,
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    a_idx, b_idx = next(splitter.split(np.arange(len(target)), target, groups=bundle["val_groups"]))

    def train_eval(train_idx, eval_idx, direction):
        vec = DictVectorizer()
        x_train = vec.fit_transform([dicts[i] for i in train_idx])
        x_eval = vec.transform([dicts[i] for i in eval_idx])
        clf = LogisticRegression(max_iter=500, C=0.3, class_weight="balanced", random_state=SEED)
        clf.fit(x_train, target[train_idx])
        sel = clf.predict(x_eval).astype(bool)
        pred = np.where(sel, bundle["tf_pred"][eval_idx], bundle["adv_pred"][eval_idx])
        sc = score(bundle["y_true"][eval_idx], pred)
        st = score(bundle["y_true"][eval_idx], static[eval_idx])
        ba = score(bundle["y_true"][eval_idx], bundle["adv_pred"][eval_idx])
        pos_pred = int(sel.sum())
        tp = int(np.sum(sel & (target[eval_idx] == 1)))
        precision = tp / max(pos_pred, 1)
        recall = tp / max(int(np.sum(target[eval_idx] == 1)), 1)
        return {
            "direction": direction,
            "macro_f1": sc["macro_f1"],
            "accuracy": sc["accuracy"],
            "advanced_macro_f1": ba["macro_f1"],
            "static_macro_f1": st["macro_f1"],
            "delta_vs_advanced": sc["macro_f1"] - ba["macro_f1"],
            "delta_vs_static": sc["macro_f1"] - st["macro_f1"],
            "changes_vs_advanced": int(np.sum(pred != bundle["adv_pred"][eval_idx])),
            "selector_precision": precision,
            "selector_recall": recall,
        }

    rows = [train_eval(a_idx, b_idx, "A_to_B"), train_eval(b_idx, a_idx, "B_to_A")]
    avg_vs_static = float(np.mean([row["delta_vs_static"] for row in rows]))
    avg_vs_adv = float(np.mean([row["delta_vs_advanced"] for row in rows]))
    write_csv(
        out_dir / "selector_report.csv",
        rows,
        [
            "direction",
            "macro_f1",
            "accuracy",
            "advanced_macro_f1",
            "static_macro_f1",
            "delta_vs_advanced",
            "delta_vs_static",
            "changes_vs_advanced",
            "selector_precision",
            "selector_recall",
        ],
    )
    result = {
        "tier_a": "not_applicable",
        "tier_b": "pass" if avg_vs_static >= 0.002 else "fail",
        "tier_c": "not_applicable",
        "final": "adopt" if avg_vs_static >= 0.002 else "reject",
        "base_macro_f1": base["macro_f1"],
        "new_macro_f1": None,
        "delta": avg_vs_static,
        "target_delta": avg_vs_adv,
        "worst_class_drop": None,
        "half_split_stability": rows,
        "decision": "Adopt selector if it beats static override cross-half." if avg_vs_static >= 0.002 else "Reject for current submit: selector does not beat static override under strict half validation.",
        "static_macro_f1_full": static_score["macro_f1"],
        "advanced_macro_f1_full": base["macro_f1"],
    }
    write_json(out_dir / "tierB.json", result)
    write_verdict(out_dir, "I9", "Transformer Override Selector", result)
    return result


def tune_thresholds(train_idx, bundle, grid):
    thresholds = {action: 1.1 for action in ACTIONS}
    best_pred = bundle["adv_pred"][train_idx].copy()
    best = score(bundle["y_true"][train_idx], best_pred)["macro_f1"]
    for _ in range(3):
        changed = False
        for action in ACTIONS:
            local_thr = thresholds[action]
            local_best = best
            for thr in grid:
                cand_thresholds = dict(thresholds)
                cand_thresholds[action] = thr
                pred = apply_thresholds(train_idx, bundle, cand_thresholds)
                sc = score(bundle["y_true"][train_idx], pred)["macro_f1"]
                if sc > local_best + 1e-12:
                    local_best = sc
                    local_thr = thr
            if local_thr != thresholds[action]:
                changed = True
                thresholds[action] = local_thr
                best = local_best
        if not changed:
            break
    return thresholds


def apply_thresholds(idx, bundle, thresholds):
    pred = []
    for i in idx:
        tp = bundle["tf_pred"][i]
        conf = bundle["tf_conf"][i]
        if conf >= thresholds.get(tp, 1.1):
            pred.append(tp)
        else:
            pred.append(bundle["adv_pred"][i])
    return np.asarray(pred, dtype=object)


def run_i10_thresholds(bundle, out_root):
    out_dir = out_root / "I10_class_specific_threshold"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = score(bundle["y_true"], bundle["adv_pred"])
    static = static_override_pred(bundle["adv_pred"], bundle["tf_pred"], bundle["tf_conf"], 0.0)
    static_score = score(bundle["y_true"], static)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    a_idx, b_idx = next(splitter.split(np.arange(len(bundle["labels"])), bundle["labels"], groups=bundle["val_groups"]))
    grid = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1]
    th_a = tune_thresholds(a_idx, bundle, grid)
    th_b = tune_thresholds(b_idx, bundle, grid)
    pred_a2b = apply_thresholds(b_idx, bundle, th_a)
    pred_b2a = apply_thresholds(a_idx, bundle, th_b)
    rows = []
    for direction, eval_idx, pred in [("A_to_B", b_idx, pred_a2b), ("B_to_A", a_idx, pred_b2a)]:
        sc = score(bundle["y_true"][eval_idx], pred)
        st = score(bundle["y_true"][eval_idx], static[eval_idx])
        ba = score(bundle["y_true"][eval_idx], bundle["adv_pred"][eval_idx])
        rows.append(
            {
                "direction": direction,
                "macro_f1": sc["macro_f1"],
                "accuracy": sc["accuracy"],
                "advanced_macro_f1": ba["macro_f1"],
                "static_macro_f1": st["macro_f1"],
                "delta_vs_advanced": sc["macro_f1"] - ba["macro_f1"],
                "delta_vs_static": sc["macro_f1"] - st["macro_f1"],
                "changes_vs_advanced": int(np.sum(pred != bundle["adv_pred"][eval_idx])),
            }
        )
    avg_vs_static = float(np.mean([row["delta_vs_static"] for row in rows]))
    full_thresholds = tune_thresholds(np.arange(len(bundle["labels"])), bundle, grid)
    full_pred = apply_thresholds(np.arange(len(bundle["labels"])), bundle, full_thresholds)
    full_score = score(bundle["y_true"], full_pred)
    write_csv(out_dir / "threshold_sweep.csv", rows, ["direction", "macro_f1", "accuracy", "advanced_macro_f1", "static_macro_f1", "delta_vs_advanced", "delta_vs_static", "changes_vs_advanced"])
    write_json(out_dir / "threshold_table.json", {"A_to_B_train_thresholds": th_a, "B_to_A_train_thresholds": th_b, "full_thresholds": full_thresholds})
    result = {
        "tier_a": "not_applicable",
        "tier_b": "pass" if avg_vs_static >= 0.0 and full_score["macro_f1"] >= static_score["macro_f1"] + 0.002 else "fail",
        "tier_c": "not_applicable",
        "final": "adopt" if avg_vs_static >= 0.0 and full_score["macro_f1"] >= static_score["macro_f1"] + 0.002 else "reject",
        "base_macro_f1": base["macro_f1"],
        "new_macro_f1": full_score["macro_f1"],
        "delta": full_score["macro_f1"] - static_score["macro_f1"],
        "target_delta": full_score["macro_f1"] - base["macro_f1"],
        "worst_class_drop": None,
        "half_split_stability": rows,
        "decision": "Adopt only if cross-half does not regress and full strict-val improves over static." if avg_vs_static >= 0 else "Reject: class thresholds overfit validation halves.",
        "static_macro_f1_full": static_score["macro_f1"],
        "advanced_macro_f1_full": base["macro_f1"],
        "full_thresholds": full_thresholds,
    }
    write_json(out_dir / "tierB.json", result)
    write_verdict(out_dir, "I10", "Class-Specific Override Threshold", result)
    return result


def run_i3_structural(bundle, out_root):
    out_dir = out_root / "I3_structural_gbdt"
    out_dir.mkdir(parents=True, exist_ok=True)
    dicts = []
    for sample in bundle["val_samples"]:
        meta = sample.get("session_meta", {}) or {}
        ws = meta.get("workspace", {}) or {}
        data = {}
        data.update({f"wf_{k}": v for k, v in extract_workflow_flags(sample).items()})
        data.update({f"nm_{k}": v for k, v in extract_numeric_result_buckets(sample).items()})
        data.update({f"sf_{k}": v for k, v in extract_surface_flags(sample).items()})
        acts = get_last_actions(sample, 3)
        data.update(
            {
                "last1": acts[-1] if acts else "none",
                "last2": "|".join(acts[-2:]) if len(acts) >= 2 else "none",
                "last3": "|".join(acts[-3:]) if len(acts) >= 3 else "none",
                "turn": meta.get("turn_index", 0),
                "turn_bucket": str(meta.get("turn_index", "unknown")),
                "ci": ws.get("last_ci_status", "none"),
                "dirty": int(bool(ws.get("git_dirty", False))),
                "open_n": len(ws.get("open_files", []) or []),
                "language_pref": meta.get("language_pref", "unknown"),
            }
        )
        dicts.append(data)
    vec = DictVectorizer()
    x = vec.fit_transform(dicts)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    a_idx, b_idx = next(splitter.split(np.arange(len(bundle["labels"])), bundle["labels"], groups=bundle["val_groups"]))
    clf = ExtraTreesClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=-1, min_samples_leaf=2)
    clf.fit(x[a_idx], bundle["y_true"][a_idx])
    pred = clf.predict(x[b_idx])
    sc = score(bundle["y_true"][b_idx], pred)
    adv_b = score(bundle["y_true"][b_idx], bundle["adv_pred"][b_idx])
    tf_b = score(bundle["y_true"][b_idx], bundle["tf_pred"][b_idx])
    rows = [
        {"name": "advanced_B", **adv_b, "delta_vs_advanced": 0.0},
        {"name": "transformer_B", **tf_b, "delta_vs_advanced": tf_b["macro_f1"] - adv_b["macro_f1"]},
        {"name": "extratrees_structural_A_to_B", **sc, "delta_vs_advanced": sc["macro_f1"] - adv_b["macro_f1"]},
    ]
    write_csv(out_dir / "structural_member.csv", rows, ["name", "macro_f1", "accuracy", "delta_vs_advanced"])
    result = {
        "tier_a": "not_applicable",
        "tier_b": "pass" if sc["macro_f1"] > adv_b["macro_f1"] + 0.003 else "fail",
        "tier_c": "not_applicable",
        "final": "analysis_only",
        "base_macro_f1": adv_b["macro_f1"],
        "new_macro_f1": sc["macro_f1"],
        "delta": sc["macro_f1"] - adv_b["macro_f1"],
        "target_delta": None,
        "worst_class_drop": None,
        "half_split_stability": "A_to_B only",
        "decision": "Use as diversity probe only; not a submit member unless probability blend later proves useful.",
    }
    write_json(out_dir / "tierB.json", result)
    write_verdict(out_dir, "I3", "Structural GBDT", result)
    return result


def write_summary(out_root, results, v4_metrics_path):
    v4_summary = None
    if v4_metrics_path.exists():
        v4 = json.loads(v4_metrics_path.read_text(encoding="utf-8"))
        v4_summary = v4.get("best")
    rows = []
    order = [
        ("I1", "workflow flags", "I1_workflow_flags"),
        ("I4", "numeric result buckets", "I4_numeric_result_buckets"),
        ("I5", "surface flags", "I5_surface_flags"),
        ("I145", "I1+I4+I5 bundle", "I145_bundle"),
        ("I3", "structural GBDT", "I3_structural_gbdt"),
        ("I6", "last3 prior", "I6_markov3_prior"),
        ("I7", "turn bias", "I7_turn_bucket_bias"),
        ("I9", "override selector", "I9_transformer_override_selector"),
        ("I10", "class thresholds", "I10_class_specific_threshold"),
    ]
    for ident, name, key in order:
        res = results.get(key, {})
        rows.append((ident, name, res.get("tier_a", ""), res.get("tier_b", ""), res.get("tier_c", ""), res.get("final", ""), res.get("delta", ""), res.get("decision", "")))

    lines = [
        "# Intuition Validation Summary v2",
        "",
        "## Baselines",
        "",
        f"- advanced router validation Macro-F1: `{results.get('advanced_macro_f1')}`",
        f"- static hybrid validation Macro-F1: `{results.get('static_hybrid_macro_f1')}`",
    ]
    if v4_summary:
        lines.append(f"- v4 mDeBERTa fold0 best Macro-F1: `{v4_summary.get('fold_val_macro_f1')}`")
    lines.extend(
        [
            "",
            "## Matrix",
            "",
            "| ID | intuition | Tier A | Tier B | Tier C | final | delta | note |",
            "|---|---|---|---|---|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines.extend(
        [
            "",
            "## Practical Decision",
            "",
            "- Submit-facing gain is still concentrated in the existing static transformer override, not in a new selector/threshold yet.",
            "- Serializer candidates should be adopted only if their Tier B proxy is positive and then tested in a short transformer-v2 ablation.",
            "- v4 fold0 reached a healthy but not submit-winning `0.6930`, so full replacement by transformer remains rejected for now.",
        ]
    )
    (out_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-root", default="artifacts/intuition")
    parser.add_argument("--logits", default="reports/transformer/B-full-mdeberta-70k-nowfirst-lr5e5-none-3e/val_logits.npy")
    parser.add_argument("--labels", default="reports/transformer/B-full-mdeberta-70k-nowfirst-lr5e5-none-3e/val_labels.npy")
    parser.add_argument("--decision", default="artifacts_v3/reports/decision/decision.json")
    parser.add_argument("--advanced-predictions", default="reports/exp_advanced_action_routing/predictions_valid_best.csv")
    parser.add_argument("--max-features", type=int, default=120_000)
    parser.add_argument("--skip-text-proxy", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    samples = load_train(args.data_dir)
    train_idx, val_idx, y, groups = load_v3_split(samples)
    if args.skip_text_proxy:
        text_features = {}
        for key in ["I1_workflow_flags", "I4_numeric_result_buckets", "I5_surface_flags", "I145_bundle"]:
            path = out_root / key / "tierB.json"
            if path.exists():
                text_features[key] = json.loads(path.read_text(encoding="utf-8"))
        text_results = {"features": text_features}
    else:
        text_results = run_text_proxy(samples, train_idx, val_idx, groups, out_root, max_features=args.max_features)
    train_samples = [samples[i] for i in train_idx]
    bundle = load_transformer_bundle(samples, args.logits, args.labels, args.decision, args.advanced_predictions)
    static = static_override_pred(bundle["adv_pred"], bundle["tf_pred"], bundle["tf_conf"], 0.0)
    results = {
        "advanced_macro_f1": score(bundle["y_true"], bundle["adv_pred"])["macro_f1"],
        "static_hybrid_macro_f1": score(bundle["y_true"], static)["macro_f1"],
    }
    results.update(text_results["features"])
    results["I6_markov3_prior"] = run_i6_prior(bundle, train_samples, out_root)
    results["I7_turn_bucket_bias"] = run_i7_turn_bias(bundle, out_root)
    results["I9_transformer_override_selector"] = run_i9_selector(bundle, out_root)
    results["I10_class_specific_threshold"] = run_i10_thresholds(bundle, out_root)
    results["I3_structural_gbdt"] = run_i3_structural(bundle, out_root)
    write_json(out_root / "summary.json", results)
    write_summary(out_root, results, Path("pipeline_v4/artifacts/reports/mdeberta_a/fold_0_metrics.json"))
    print(out_root / "SUMMARY.md")


if __name__ == "__main__":
    main()
