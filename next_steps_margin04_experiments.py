import argparse
import csv
import json
import re
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from compact_state_experiments import ALL_CLASSES, compact_text
from routing_margin_experiments import ACTION_TO_GROUP, GROUP_TO_ACTIONS


GROUPS = list(GROUP_TO_ACTIONS)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels(path):
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def session_id(sample_id):
    return sample_id.rsplit("-step_", 1)[0] if "-step_" in sample_id else sample_id


def clean(value, max_chars=900):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(text.split())[:max_chars]


def has_any(text, patterns):
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def rule_router(sample):
    text = clean(sample.get("current_prompt"), 1200)
    t = text.lower()

    rules = [
        (
            "communicate",
            "respond_only",
            0.95,
            [
                r"summarize",
                r"summary",
                r"recap",
                r"wrap.?up",
                r"brief",
                r"what we did",
                r"done here",
                r"finish",
            ],
        ),
        (
            "communicate",
            "plan_task",
            0.90,
            [
                r"\bplan\b",
                r"steps?",
                r"break.+down",
                r"where to start",
                r"outline",
                r"roadmap",
                r"approach",
            ],
        ),
        (
            "communicate",
            "web_search",
            0.90,
            [
                r"\bweb\b",
                r"latest",
                r"best.?practice",
                r"recommended",
                r"official.?docs?",
                r"online",
                r"current",
                r"look.?up",
            ],
        ),
        (
            "execute",
            "lint_or_typecheck",
            0.90,
            [
                r"actionlint",
                r"\blint\b",
                r"type.?check",
                r"\btsc\b",
                r"\bmypy\b",
                r"\bruff\b",
                r"static analysis",
            ],
        ),
        (
            "execute",
            "run_tests",
            0.85,
            [
                r"\btests?\b",
                r"\bpytest\b",
                r"cargo test",
                r"npm test",
                r"unit test",
                r"integration test",
                r"\be2e\b",
                r"\bspec\b",
            ],
        ),
        (
            "execute",
            "run_bash",
            0.80,
            [
                r"\bbuild\b",
                r"runserver",
                r"dev server",
                r"npm run dev",
                r"pip install",
                r"pod install",
                r"\bcommand\b",
                r"\bshell\b",
            ],
        ),
        (
            "modify",
            "write_file",
            0.90,
            [
                r"new file",
                r"create .*file",
                r"make .*file",
                r"scaffold",
                r"write .*file",
            ],
        ),
        (
            "modify",
            "apply_patch",
            0.75,
            [
                r"both ",
                r"in one shot",
                r"coupled",
                r"multiple files?",
                r"several files?",
                r"apply patch",
            ],
        ),
        (
            "modify",
            "edit_file",
            0.75,
            [
                r"\bfix\b",
                r"\badd\b",
                r"\bpatch\b",
                r"replace",
                r"change",
                r"rename",
                r"rewrite",
                r"guard",
                r"wire",
                r"handle",
                r"update",
                r"remove",
            ],
        ),
        (
            "inspect",
            "list_directory",
            0.80,
            [
                r"list what",
                r"directory",
                r"folder",
                r"project root",
                r"\btree\b",
                r"\bls\b",
            ],
        ),
        (
            "inspect",
            "glob_pattern",
            0.80,
            [
                r"\*\.",
                r"\*\*/",
                r"glob",
                r"all .*files",
                r"matching files?",
                r"tsx files?",
                r"sql files?",
                r"swift files?",
                r"gradle files?",
            ],
        ),
        (
            "inspect",
            "grep_search",
            0.75,
            [
                r"\bgrep\b",
                r"\bfind\b",
                r"\bsearch\b",
                r"where ",
                r"occurrences",
                r"definition",
                r"import",
                r"reference",
            ],
        ),
        (
            "inspect",
            "read_file",
            0.75,
            [
                r"show me",
                r"pull.*up",
                r"\bread\b",
                r"\bopen\b",
                r"look at",
                r"peek at",
                r"what.?s inside",
            ],
        ),
    ]

    for group, action, conf, patterns in rules:
        if has_any(t, patterns):
            return {"group": group, "action_hint": action, "conf": conf}
    return {"group": "unknown", "action_hint": "unknown", "conf": 0.0}


def bucket_number(value, bins):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for name, upper in bins:
        if value <= upper:
            return name
    return "huge"


def serialize_args(args):
    if not isinstance(args, dict):
        return clean(args, 500)
    parts = []
    for key, value in sorted(args.items()):
        parts.append(f"{key}={clean(value, 250)}")
    return " ".join(parts)


def path_tokens(paths):
    out = []
    for path in paths or []:
        text = clean(path, 300).lower().replace("\\", "/")
        if not text:
            continue
        out.append("OPEN_FILE=" + text)
        if "." in text.rsplit("/", 1)[-1]:
            out.append("OPEN_EXT=" + text.rsplit(".", 1)[-1])
        for part in [p for p in text.split("/") if p][-4:]:
            out.append("OPEN_PART=" + part)
    return " ".join(out)


def serialize_doc_style(sample, window=8):
    current = clean(sample.get("current_prompt"), 1400)
    meta = sample.get("session_meta", {}) or {}
    workspace = meta.get("workspace", {}) or {}
    history = sample.get("history", []) or []
    rule = rule_router(sample)

    user_hist = []
    action_hist = []
    result_hist = []
    action_names = []
    recent = history[-window:]
    for item in recent:
        role = item.get("role", "")
        if role == "user":
            user_hist.append(clean(item.get("content"), 700))
        elif role == "assistant_action":
            name = clean(item.get("name"), 80)
            args = serialize_args(item.get("args", {}))
            result = clean(item.get("result_summary"), 700)
            action_names.append(name)
            action_hist.append(f"{name} {args}")
            result_hist.append(result)

    all_actions = [
        h.get("name", "")
        for h in history
        if h.get("role") == "assistant_action" and h.get("name")
    ]
    last_action = all_actions[-1] if all_actions else "NONE"
    last2 = ">".join(all_actions[-2:]) if len(all_actions) >= 2 else "NONE>" + last_action
    recent_action_tokens = []
    for i, action in enumerate(reversed(all_actions[-5:]), start=1):
        recent_action_tokens.append(f"ACTION_BACK_{i}={action}")

    lang_mix = workspace.get("language_mix", {}) or {}
    lang_tokens = []
    if isinstance(lang_mix, dict):
        for lang, ratio in sorted(lang_mix.items()):
            try:
                share = int(round(float(ratio) * 10))
            except (TypeError, ValueError):
                share = 0
            lang_tokens.append(f"LANG={lang} LANG_SHARE={lang}_{share}")

    rule_tokens = [
        f"RULE_GROUP={rule['group']}",
        f"RULE_HINT={rule['action_hint']}",
    ]
    if rule["conf"] >= 0.9:
        rule_tokens.extend([f"RULE_STRONG={rule['action_hint']}"] * 2)
    elif rule["conf"] >= 0.75:
        rule_tokens.append(f"RULE_MEDIUM={rule['action_hint']}")

    text = f"""
    CURRENT={current}
    CURRENT_AGAIN={current}
    RULE_CONF={rule['conf']}
    RULE_TOKENS={' '.join(rule_tokens)}
    LAST_ACTION={last_action}
    LAST2_ACTION={last2}
    RECENT_ACTIONS={' '.join(recent_action_tokens)}
    HISTORY_USERS={' [U] '.join(user_hist)}
    HISTORY_ACTIONS={' [A] '.join(action_hist)}
    HISTORY_RESULTS={' [R] '.join(result_hist)}
    OPEN_FILES={path_tokens(workspace.get('open_files', []))}
    LAST_CI_STATUS={clean(workspace.get('last_ci_status'), 80)}
    GIT_DIRTY={workspace.get('git_dirty', 'unknown')}
    LANGUAGE_PREF={clean(meta.get('language_pref'), 80)}
    USER_TIER={clean(meta.get('user_tier'), 80)}
    TURN_BIN={bucket_number(meta.get('turn_index'), [('turn_early', 2), ('turn_mid', 7), ('turn_late', 12)])}
    BUDGET_BIN={bucket_number(meta.get('budget_tokens_remaining'), [('budget_low', 10000), ('budget_mid', 50000), ('budget_high', 120000)])}
    ELAPSED_BIN={bucket_number(meta.get('elapsed_session_sec'), [('elapsed_start', 240), ('elapsed_mid', 900), ('elapsed_late', 1800)])}
    LOC_BIN={bucket_number(workspace.get('loc'), [('loc_lt_2k', 2000), ('loc_lt_10k', 10000), ('loc_lt_30k', 30000)])}
    LANGUAGE_MIX={' '.join(lang_tokens)}
    """
    return " ".join(text.split())


def build_texts(samples, feature_name):
    if feature_name == "compact_flags":
        return [compact_text(sample, "flags") for sample in samples]
    if feature_name == "doc_rule8":
        return [serialize_doc_style(sample, window=8) for sample in samples]
    if feature_name == "doc_rule12":
        return [serialize_doc_style(sample, window=12) for sample in samples]
    if feature_name == "compact_plus_doc8":
        return [
            compact_text(sample, "flags") + " DOC_RULE " + serialize_doc_style(sample, window=8)
            for sample in samples
        ]
    if feature_name == "compact_plus_doc12":
        return [
            compact_text(sample, "flags") + " DOC_RULE " + serialize_doc_style(sample, window=12)
            for sample in samples
        ]
    raise ValueError(feature_name)


def build_vectorizer(max_features=220_000, char_hi=5, word_hi=2):
    half = max_features // 2
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, word_hi),
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
                    ngram_range=(3, char_hi),
                    min_df=2,
                    max_features=half,
                    sublinear_tf=True,
                    lowercase=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=ALL_CLASSES, average="macro", zero_division=0)


def eval_pred(name, y_true, y_pred, results_path, extra=None):
    row = {
        "name": name,
        "status": "ok",
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "seconds": 0.0,
    }
    if extra:
        row.update(extra)
    append_jsonl(results_path, row)
    print(f"{name}: macro_f1={row['macro_f1']:.6f} acc={row['accuracy']:.6f}", flush=True)
    return row


def decision_scores(model, x, classes):
    scores = model.decision_function(x)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    out = np.full((x.shape[0], len(classes)), -1e9, dtype=np.float32)
    for i, cls in enumerate(model.classes_):
        out[:, classes.index(str(cls))] = scores[:, i]
    return out


def proba_scores(model, x, classes):
    proba = model.predict_proba(x)
    out = np.full((x.shape[0], len(classes)), -1e9, dtype=np.float32)
    for i, cls in enumerate(model.classes_):
        out[:, classes.index(str(cls))] = np.log(proba[:, i] + 1e-9)
    return out


def top_margin(scores, classes):
    order = np.argsort(scores, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    margin = scores[np.arange(scores.shape[0]), top1] - scores[np.arange(scores.shape[0]), top2]
    pred = np.array([classes[i] for i in top1], dtype=object)
    return pred, margin


def fine_predict(fine_models, group_pred, x):
    preds = np.array(["respond_only"] * x.shape[0], dtype=object)
    for group, model in fine_models.items():
        idx = np.where(group_pred == group)[0]
        if len(idx):
            preds[idx] = model.predict(x[idx])
    return preds


def fine_score_predict(fine_models, group_pred, x, model_kind):
    preds = np.array(["respond_only"] * x.shape[0], dtype=object)
    for group, model in fine_models.items():
        idx = np.where(group_pred == group)[0]
        if not len(idx):
            continue
        actions = GROUP_TO_ACTIONS[group]
        scores = decision_scores(model, x[idx], actions) if model_kind == "svc" else proba_scores(model, x[idx], actions)
        preds[idx] = np.array([actions[i] for i in scores.argmax(axis=1)], dtype=object)
    return preds


def true_groups(y):
    return np.array([ACTION_TO_GROUP[action] for action in y], dtype=object)


def train_fine_models(x_train, y_train, model_kind, c_value):
    fine_models = {}
    for group, actions in GROUP_TO_ACTIONS.items():
        mask = np.isin(y_train, actions)
        if model_kind == "svc":
            model = LinearSVC(C=c_value, class_weight="balanced", random_state=42, dual="auto", max_iter=5000)
        elif model_kind == "logreg":
            model = LogisticRegression(max_iter=700, C=c_value, class_weight="balanced", random_state=42)
        else:
            raise ValueError(model_kind)
        start = time.time()
        model.fit(x_train[mask], y_train[mask])
        print(f"fine_{model_kind} {group} rows={int(mask.sum())} sec={time.time() - start:.1f}", flush=True)
        fine_models[group] = model
    return fine_models


def run_one_config(config, x_train, x_val, y_train, y_val, y_group_train, y_group_val, results_path):
    feature_name, c_value, fine_kind = config
    prefix = f"{feature_name}_svcC{c_value}_{fine_kind}"
    print(f"\n=== {prefix} ===", flush=True)

    start = time.time()
    flat = LinearSVC(C=c_value, class_weight="balanced", random_state=42, dual="auto", max_iter=5000)
    flat.fit(x_train, y_train)
    flat_pred = flat.predict(x_val)
    eval_pred(prefix + "_flat14", y_val, flat_pred, results_path, {"seconds": time.time() - start})

    start = time.time()
    coarse = LinearSVC(C=c_value, class_weight="balanced", random_state=42, dual="auto", max_iter=5000)
    coarse.fit(x_train, y_group_train)
    group_scores = decision_scores(coarse, x_val, GROUPS)
    group_pred, margins = top_margin(group_scores, GROUPS)
    group_acc = float((group_pred == y_group_val).mean())
    eval_pred(
        prefix + "_coarse_diagnostics",
        y_val,
        flat_pred,
        results_path,
        {"seconds": time.time() - start, "group_accuracy": group_acc},
    )
    print(f"{prefix} group_acc={group_acc:.6f}", flush=True)

    fine_models = train_fine_models(x_train, y_train, fine_kind, c_value)
    fine_by_coarse = fine_score_predict(fine_models, group_pred, x_val, fine_kind)
    fine_by_oracle = fine_score_predict(fine_models, y_group_val, x_val, fine_kind)
    eval_pred(prefix + "_fine_by_coarse_all", y_val, fine_by_coarse, results_path, {"group_accuracy": group_acc})
    eval_pred(prefix + "_fine_oracle_group", y_val, fine_by_oracle, results_path)

    flat_group = true_groups(flat_pred)
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        high = margins >= threshold
        if high.any():
            group_acc_covered = float((group_pred[high] == y_group_val[high]).mean())
        else:
            group_acc_covered = 0.0

        hard = flat_pred.copy()
        hard[high] = fine_by_coarse[high]
        eval_pred(
            prefix + f"_hard_t{threshold}",
            y_val,
            hard,
            results_path,
            {"threshold": threshold, "coverage": float(high.mean()), "group_acc_covered": group_acc_covered},
        )

        safety = flat_pred.copy()
        disagree = high & (flat_group != group_pred)
        safety[disagree] = fine_by_coarse[disagree]
        eval_pred(
            prefix + f"_safety_t{threshold}",
            y_val,
            safety,
            results_path,
            {
                "threshold": threshold,
                "coverage": float(high.mean()),
                "changed": int(disagree.sum()),
                "group_acc_covered": group_acc_covered,
            },
        )

    report = classification_report(y_val, fine_by_coarse, labels=ALL_CLASSES, output_dict=True, zero_division=0)
    rare_f1 = {
        cls: report.get(cls, {}).get("f1-score", 0.0)
        for cls in ["write_file", "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only"]
    }
    append_jsonl(
        results_path,
        {
            "name": prefix + "_class_report_focus",
            "status": "ok",
            "macro_f1": macro_f1(y_val, fine_by_coarse),
            "rare_f1": rare_f1,
        },
    )


def update_research(summary, run_dir):
    with open("research.md", "a", encoding="utf-8") as f:
        f.write("\n## Next Steps Margin04 Experiments\n")
        f.write(f"- Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Run dir: `{run_dir}`\n")
        if summary.get("best"):
            best = summary["best"]
            f.write(f"- Best Macro-F1: `{best['macro_f1']:.6f}` via `{best['name']}`\n")
        f.write("- Key idea: doc-style rule hints, LinearSVC flat/coarse/fine routing, GroupShuffleSplit-first validation.\n")
        f.write("\nTop results:\n")
        for row in summary.get("top10", []):
            f.write(f"- `{row['macro_f1']:.6f}` `{row['name']}`\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--run-dir", default="./next_steps_margin04_runs")
    parser.add_argument("--features", default="compact_flags,doc_rule8,doc_rule12")
    parser.add_argument("--c-values", default="0.7,1.5")
    parser.add_argument("--fine-kinds", default="svc")
    parser.add_argument("--split", choices=["group", "stratified"], default="group")
    parser.add_argument("--max-features", type=int, default=220_000)
    parser.add_argument("--char-hi", type=int, default=5)
    parser.add_argument("--word-hi", type=int, default=2)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    (run_dir / "pid.txt").write_text(str(__import__("os").getpid()), encoding="utf-8")
    results_path.write_text("", encoding="utf-8")
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    try:
        samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
        labels = load_labels(Path(args.data_dir) / "train_labels.csv")
        y = np.array([labels[s["id"]] for s in samples], dtype=object)
        y_group = true_groups(y)
        indices = np.arange(len(samples))

        if args.split == "group":
            groups = np.array([session_id(s["id"]) for s in samples], dtype=object)
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(splitter.split(indices, y, groups=groups))
        else:
            train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)

        y_train = y[train_idx]
        y_val = y[val_idx]
        y_group_train = y_group[train_idx]
        y_group_val = y_group[val_idx]
        print(f"split={args.split} train={len(train_idx)} val={len(val_idx)}", flush=True)

        feature_names = [item.strip() for item in args.features.split(",") if item.strip()]
        c_values = [float(item.strip()) for item in args.c_values.split(",") if item.strip()]
        fine_kinds = [item.strip() for item in args.fine_kinds.split(",") if item.strip()]

        for feature_name in feature_names:
            print(f"\n### feature={feature_name}", flush=True)
            start = time.time()
            texts = build_texts(samples, feature_name)
            x_train_text = [texts[i] for i in train_idx]
            x_val_text = [texts[i] for i in val_idx]
            vectorizer = build_vectorizer(args.max_features, args.char_hi, args.word_hi)
            x_train = vectorizer.fit_transform(x_train_text)
            x_val = vectorizer.transform(x_val_text)
            print(f"vectorized {feature_name} shape={x_train.shape} sec={time.time() - start:.1f}", flush=True)

            for c_value in c_values:
                for fine_kind in fine_kinds:
                    try:
                        run_one_config(
                            (feature_name, c_value, fine_kind),
                            x_train,
                            x_val,
                            y_train,
                            y_val,
                            y_group_train,
                            y_group_val,
                            results_path,
                        )
                    except Exception as exc:
                        row = {
                            "name": f"{feature_name}_svcC{c_value}_{fine_kind}",
                            "status": "error",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        append_jsonl(results_path, row)
                        print(f"ERROR {row['name']}: {exc!r}", flush=True)

        rows = []
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("status") == "ok" and "macro_f1" in row and "class_report" not in row.get("name", ""):
                        rows.append(row)
        rows.sort(key=lambda row: row["macro_f1"], reverse=True)
        summary = {
            "best": rows[0] if rows else None,
            "top10": rows[:10],
            "split": args.split,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        update_research(summary, str(run_dir))
        print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        stderr_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise exc


if __name__ == "__main__":
    main()
