import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script import advanced_open_files, safe_text  # noqa: E402
from scripts.run_inspect_bottleneck_experiments import (  # noqa: E402
    ALL_CLASSES,
    build_vectorizer,
    fast_router_text,
    last_action_turn,
    load_fold_split,
    load_labels,
    predict_fast_router,
    read_jsonl,
    score_macro,
    session_of,
    train_fast_router,
)


EXECUTE = ["run_bash", "run_tests", "lint_or_typecheck"]
EXECUTE_SET = set(EXECUTE)

LINT_PATTERNS = [
    r"\blint\b",
    r"\blinter\b",
    r"\btype\s*check\b",
    r"\btypecheck\b",
    r"\btsc\b",
    r"\bmypy\b",
    r"\bruff\b",
    r"\beslint\b",
    r"\bpyright\b",
    r"\bflake8\b",
    r"\bblack\s+--check\b",
    r"\bprettier\s+--check\b",
    r"\bactionlint\b",
    r"\bclippy\b",
    r"\bcargo\s+clippy\b",
    r"\bgo\s+vet\b",
    r"\bvet\b",
    r"static\s+(check|analysis)",
    r"type\s+error",
    r"format\s+check",
    r"정적",
    r"타입",
    r"린트",
]

TEST_PATTERNS = [
    r"\btest\b",
    r"\btests\b",
    r"\bspec\b",
    r"\bsuite\b",
    r"\bpytest\b",
    r"\bjest\b",
    r"\bvitest\b",
    r"\bnpm\s+test\b",
    r"\bcargo\s+test\b",
    r"\bgo\s+test\b",
    r"\bunit\b",
    r"\be2e\b",
    r"\bintegration\b",
    r"\bregression\b",
    r"happy\s+path",
    r"coverage",
    r"green",
    r"테스트",
]

BASH_PATTERNS = [
    r"\bbuild\b",
    r"\bcompile\b",
    r"\binstall\b",
    r"\bstart\b",
    r"\bboot\b",
    r"\bserve\b",
    r"dev\s+server",
    r"\bnpm\s+run\b",
    r"\bnpm\s+install\b",
    r"\bpip\s+install\b",
    r"\bcargo\s+build\b",
    r"\bgo\s+build\b",
    r"\bmake\b",
    r"\bdocker\b",
    r"\bgradlew\b",
    r"full\s+build",
    r"run\s+command",
    r"what'?s it complaining",
    r"traceback",
    r"빌드",
    r"실행",
]


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


def write_text(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def has_pattern(text, patterns):
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def execute_rule(sample):
    prompt = safe_text(sample.get("current_prompt"), 1400)
    last = last_action_turn(sample)
    last_result = safe_text(last.get("result_summary"), 500) if last else ""
    last_args = safe_text(last.get("args"), 400) if last else ""
    text = f"{prompt}\n{last_args}\n{last_result}".lower()

    lint = has_pattern(text, LINT_PATTERNS)
    test = has_pattern(text, TEST_PATTERNS)
    bash = has_pattern(text, BASH_PATTERNS)

    # Prioritize explicit static-analysis tools over generic "test/build" words.
    if lint:
        return "lint_or_typecheck", "lint_keyword"
    if test and not bash:
        return "run_tests", "test_keyword"
    if bash and not test:
        return "run_bash", "bash_keyword"
    if test and bash:
        if has_pattern(text, [r"\bnpm\s+test\b", r"\bcargo\s+test\b", r"\bgo\s+test\b", r"\bpytest\b", r"\bjest\b", r"\bvitest\b"]):
            return "run_tests", "explicit_test_command"
        if has_pattern(text, [r"\bnpm\s+run\s+build\b", r"\bcargo\s+build\b", r"\bgo\s+build\b", r"full\s+build"]):
            return "run_bash", "explicit_build_command"
        return None, "ambiguous_test_bash"
    return None, "no_rule"


def last_action_name(sample):
    last = last_action_turn(sample)
    return str(last.get("name")) if last else "NONE"


def result_bucket(sample):
    last = last_action_turn(sample)
    if not last:
        return "none"
    result = safe_text(last.get("result_summary"), 600).lower()
    if any(x in result for x in ["error", "traceback", "exception", "fail", "failed", "exit=1", "exit 1"]):
        return "fail"
    if any(x in result for x in ["pass", "passed", "green", "ok", "success", "exit=0", "exit 0"]):
        return "ok"
    return "other"


def execute_text(sample):
    rule, reason = execute_rule(sample)
    last = last_action_turn(sample)
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    lang_mix = ws.get("language_mix", {}) or {}
    top_lang = "none"
    if lang_mix:
        top_lang = max(lang_mix.items(), key=lambda item: float(item[1]))[0]
    actions = [h.get("name") for h in sample.get("history", []) or [] if h.get("role") == "assistant_action" and h.get("name")]
    return "\n".join(
        [
            "[NOW] " + safe_text(sample.get("current_prompt"), 1200),
            f"[RULE] pred={rule or 'none'} reason={reason}",
            f"[LAST] action={last_action_name(sample)} bucket={result_bucket(sample)} "
            f"args={safe_text(last.get('args'), 400) if last else 'none'} "
            f"result={safe_text(last.get('result_summary'), 500) if last else 'none'}",
            "[SEQ] " + (" > ".join(actions[-6:]) if actions else "none"),
            "[OPEN] " + (" ".join(advanced_open_files(sample)[:8]) or "none"),
            f"[META] turn={meta.get('turn_index')} pref={meta.get('language_pref')} "
            f"ci={ws.get('last_ci_status')} dirty={ws.get('git_dirty')} top_lang={top_lang}",
            "[BASE] " + fast_router_text(sample),
        ]
    )


def train_execute_model(kind, samples, y, features):
    texts = [execute_text(sample) for sample in samples]
    vectorizer = build_vectorizer(features, min_df=2, char_heavy=(kind == "logreg_char_heavy"))
    x = vectorizer.fit_transform(texts)
    if kind == "linear_svc":
        model = LinearSVC(C=1.0, class_weight="balanced", random_state=42, dual="auto", max_iter=2500)
    else:
        model = LogisticRegression(
            C=2.0,
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
    model.fit(x, y)
    return vectorizer, model


def predict_execute(kind, vectorizer, model, samples):
    x = vectorizer.transform([execute_text(sample) for sample in samples])
    if kind.startswith("logreg"):
        probs = model.predict_proba(x)
        idx = probs.argmax(axis=1)
        pred = np.asarray([str(model.classes_[i]) for i in idx], dtype=object)
        conf = probs.max(axis=1)
        return pred, conf
    score = model.decision_function(x)
    idx = score.argmax(axis=1)
    pred = np.asarray([str(model.classes_[i]) for i in idx], dtype=object)
    sorted_score = np.sort(score, axis=1)
    margin = sorted_score[:, -1] - sorted_score[:, -2]
    conf = 1.0 / (1.0 + np.exp(-margin))
    return pred, conf


def classwise_rows(y_true, pred):
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=ALL_CLASSES, zero_division=0)
    rows = []
    for i, cls in enumerate(ALL_CLASSES):
        rows.append({"class": cls, "precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])})
    return rows


def execute_class_f1(y_true, pred):
    return float(f1_score(y_true, pred, labels=EXECUTE, average="macro", zero_division=0))


def apply_rule(base_pred, samples, scope):
    out = np.asarray(base_pred, dtype=object).copy()
    changed = []
    for i, sample in enumerate(samples):
        if not scope[i]:
            continue
        rule, reason = execute_rule(sample)
        if rule is None:
            continue
        if out[i] != rule:
            changed.append((i, out[i], rule, reason))
        out[i] = rule
    return out, changed


def evaluate_override(y_val, base_pred, new_pred, scope, name):
    changed = new_pred != base_pred
    before = base_pred[changed] == y_val[changed]
    after = new_pred[changed] == y_val[changed]
    return {
        "name": name,
        "scope_count": int(scope.sum()),
        "override_count": int(changed.sum()),
        "base_macro_f1": score_macro(y_val, base_pred),
        "new_macro_f1": score_macro(y_val, new_pred),
        "macro_delta": score_macro(y_val, new_pred) - score_macro(y_val, base_pred),
        "base_execute_f1": execute_class_f1(y_val, base_pred),
        "new_execute_f1": execute_class_f1(y_val, new_pred),
        "execute_f1_delta": execute_class_f1(y_val, new_pred) - execute_class_f1(y_val, base_pred),
        "net_correct_delta": int(after.sum() - before.sum()),
        "override_precision": float(after.mean()) if len(after) else 0.0,
        "base_precision_same_rows": float(before.mean()) if len(before) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../open/data")
    parser.add_argument("--fold-file", default="pipeline_v4/folds/fold_assignments.csv")
    parser.add_argument("--out-root", default="reports/execute_router_experiment")
    parser.add_argument("--router-features", type=int, default=50_000)
    parser.add_argument("--execute-features", type=int, default=45_000)
    parser.add_argument("--refresh-router", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    samples = read_jsonl(data_dir / "train.jsonl")
    labels = load_labels(data_dir / "train_labels.csv")
    y = np.asarray([labels[sample["id"]] for sample in samples], dtype=object)
    train_idx, val_idx, split_name = load_fold_split(samples, args.fold_file, fold=0)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    router = train_fast_router(samples, y, train_idx, Path("reports/inspect_bottleneck"), args.router_features, refresh=args.refresh_router)
    base_pred, _, base_infos = predict_fast_router(val_samples, router)
    base_macro = score_macro(y_val, base_pred)
    base_execute = execute_class_f1(y_val, base_pred)

    strict_scope = np.asarray([pred in EXECUTE_SET for pred in base_pred], dtype=bool)
    top2_scope = np.asarray([info["pred"] in EXECUTE_SET or info["top2"] in EXECUTE_SET for info in base_infos], dtype=bool)

    rule_rows = []
    for scope_name, scope in [("strict_base_execute", strict_scope), ("base_or_top2_execute", top2_scope)]:
        rule_pred, changed = apply_rule(base_pred, val_samples, scope)
        rule_rows.append(evaluate_override(y_val, base_pred, rule_pred, scope, f"rule_{scope_name}"))

    train_exec_mask = np.asarray([label in EXECUTE_SET for label in y_train], dtype=bool)
    val_exec_mask = np.asarray([label in EXECUTE_SET for label in y_val], dtype=bool)
    exec_train_samples = [sample for sample, ok in zip(train_samples, train_exec_mask) if ok]
    exec_train_y = y_train[train_exec_mask]

    resolver_rows = []
    changed_rows = []
    best = None
    model_bundles = {}
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    for kind in ["logreg_word_char", "logreg_char_heavy", "linear_svc"]:
        started = time.time()
        vectorizer, model = train_execute_model(kind, exec_train_samples, exec_train_y, args.execute_features)
        pred_exec_all, conf_all = predict_execute(kind, vectorizer, model, val_samples)
        isolated_f1 = execute_class_f1(y_val[val_exec_mask], pred_exec_all[val_exec_mask])
        model_bundles[kind] = (vectorizer, model, pred_exec_all, conf_all)
        for scope_name, scope in [("strict_base_execute", strict_scope), ("base_or_top2_execute", top2_scope)]:
            for threshold in thresholds:
                mask = scope & (conf_all >= threshold)
                new_pred = np.asarray(base_pred, dtype=object).copy()
                new_pred[mask] = pred_exec_all[mask]
                row = evaluate_override(y_val, base_pred, new_pred, mask, f"{kind}_{scope_name}_thr{threshold}")
                row.update(
                    {
                        "model": kind,
                        "scope": scope_name,
                        "threshold": threshold,
                        "isolated_execute_true_f1": isolated_f1,
                        "fit_predict_sec": round(time.time() - started, 3),
                    }
                )
                resolver_rows.append(row)
                if best is None or row["macro_delta"] > best["macro_delta"]:
                    best = row

    if best:
        kind = best["model"]
        _, _, pred_exec_all, conf_all = model_bundles[kind]
        scope = strict_scope if best["scope"] == "strict_base_execute" else top2_scope
        mask = scope & (conf_all >= float(best["threshold"]))
        final_pred = np.asarray(base_pred, dtype=object).copy()
        final_pred[mask] = pred_exec_all[mask]
        for i, ok in enumerate(mask):
            if ok and final_pred[i] != base_pred[i]:
                changed_rows.append(
                    {
                        "id": val_samples[i]["id"],
                        "from_base": base_pred[i],
                        "to_execute_resolver": final_pred[i],
                        "true": y_val[i],
                        "correct_before": int(base_pred[i] == y_val[i]),
                        "correct_after": int(final_pred[i] == y_val[i]),
                        "confidence": float(conf_all[i]),
                        "prompt": safe_text(val_samples[i].get("current_prompt"), 260),
                    }
                )
    else:
        final_pred = base_pred

    class_before = {row["class"]: row for row in classwise_rows(y_val, base_pred)}
    class_after = {row["class"]: row for row in classwise_rows(y_val, final_pred)}
    class_rows = []
    for cls in ALL_CLASSES:
        before = class_before[cls]
        after = class_after[cls]
        class_rows.append(
            {
                "class": cls,
                "f1_before": before["f1"],
                "f1_after": after["f1"],
                "f1_delta": after["f1"] - before["f1"],
                "precision_before": before["precision"],
                "precision_after": after["precision"],
                "recall_before": before["recall"],
                "recall_after": after["recall"],
                "support": before["support"],
            }
        )

    write_csv(out_root / "rule_results.csv", rule_rows, list(rule_rows[0].keys()))
    write_csv(out_root / "resolver_results.csv", resolver_rows, list(resolver_rows[0].keys()))
    write_csv(out_root / "classwise_before_after.csv", class_rows, list(class_rows[0].keys()))
    write_csv(
        out_root / "changed_flow.csv",
        changed_rows,
        ["id", "from_base", "to_execute_resolver", "true", "correct_before", "correct_after", "confidence", "prompt"],
    )
    write_json(
        out_root / "best_config.json",
        {
            "split": split_name,
            "base_macro_f1": base_macro,
            "base_execute_f1": base_execute,
            "best": best,
            "changed_count": len(changed_rows),
        },
    )

    examples = ["# Execute Router Changed Examples", ""]
    for row in changed_rows[:40]:
        examples.append(
            f"- `{row['id']}` {row['from_base']} -> {row['to_execute_resolver']} true={row['true']} "
            f"before={row['correct_before']} after={row['correct_after']} conf={row['confidence']:.3f} prompt={row['prompt']}"
        )
    write_text(out_root / "examples.md", examples)

    verdict = "PASS" if best and best["macro_delta"] >= 0.001 and best["net_correct_delta"] > 0 else "FAIL"
    lines = [
        "# Execute Router Rule + Resolver Experiment",
        "",
        f"- split: `{split_name}`",
        f"- base: `fast_flat` local proxy",
        f"- validation rows: `{len(val_idx)}`",
        f"- execute train rows: `{len(exec_train_samples)}`",
        f"- base Macro-F1: `{base_macro:.6f}`",
        f"- base execute Macro-F1: `{base_execute:.6f}`",
        f"- strict execute scope rows: `{int(strict_scope.sum())}`",
        f"- base/top2 execute scope rows: `{int(top2_scope.sum())}`",
        "",
        "## Rule Results",
        "",
    ]
    for row in rule_rows:
        lines.append(
            f"- `{row['name']}` macro_delta={row['macro_delta']:.6f} "
            f"execute_delta={row['execute_f1_delta']:.6f} net={row['net_correct_delta']} overrides={row['override_count']}"
        )
    lines.extend(["", "## Best Resolver", ""])
    if best:
        lines.append(
            f"- `{best['name']}` macro_delta=`{best['macro_delta']:.6f}`, "
            f"execute_delta=`{best['execute_f1_delta']:.6f}`, net=`{best['net_correct_delta']}`, "
            f"overrides=`{best['override_count']}`, isolated_execute_f1=`{best['isolated_execute_true_f1']:.6f}`"
        )
    else:
        lines.append("- no resolver result")
    lines.extend(["", f"- verdict: `{verdict}`"])
    write_text(out_root / "summary.md", lines)

    with open(ROOT / "research.md", "a", encoding="utf-8") as f:
        f.write(
            "\n## Execute Router Rule + Resolver Experiment\n\n"
            f"- timestamp: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n"
            f"- base Macro-F1: `{base_macro:.6f}`; base execute Macro-F1: `{base_execute:.6f}`.\n"
            f"- best: `{best['name'] if best else 'none'}`; "
            f"macro_delta=`{best['macro_delta'] if best else 0:.6f}`; "
            f"execute_delta=`{best['execute_f1_delta'] if best else 0:.6f}`; "
            f"net=`{best['net_correct_delta'] if best else 0}`.\n"
            f"- verdict: `{verdict}`.\n"
        )
    print(f"wrote {out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
