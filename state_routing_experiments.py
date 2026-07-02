import argparse
import csv
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


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

GROUPS = {
    "explore": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "modify": ["edit_file", "write_file", "apply_patch"],
    "execute": ["run_bash", "run_tests", "lint_or_typecheck"],
    "dialogue": ["ask_user", "plan_task", "web_search", "respond_only"],
}
CLASS_TO_GROUP = {cls: group for group, classes in GROUPS.items() for cls in classes}


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


def clean_text(value, max_chars=1200):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(text.split())[:max_chars]


def bucket(value, cuts):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for name, upper in cuts:
        if value <= upper:
            return name
    return "huge"


def path_tokens(paths):
    out = []
    for raw in paths or []:
        path = clean_text(raw, 300).lower().replace("\\", "/")
        if not path:
            continue
        out.append("OPEN_FILE=" + path)
        parts = [part for part in path.split("/") if part]
        out.extend("OPEN_PART=" + part for part in parts[-5:])
        if "." in parts[-1]:
            ext = parts[-1].rsplit(".", 1)[-1]
            out.append("OPEN_EXT=" + ext)
            out.append("OPEN_BASENAME=" + parts[-1].rsplit(".", 1)[0])
    return " ".join(out)


def action_features(history):
    actions = [
        item.get("name", "")
        for item in history
        if item.get("role") == "assistant_action" and item.get("name")
    ]
    tokens = []
    if not actions:
        return actions, "HIST_EMPTY=1 LAST_ACT=NONE"
    tokens.append("HIST_EMPTY=0")
    tokens.append("LAST_ACT=" + actions[-1])
    for pos, action in enumerate(reversed(actions[-8:]), start=1):
        tokens.append(f"ACT_BACK_{pos}={action}")
        tokens.append("RECENT_ACT=" + action)
    for a, b in zip(actions[-8:], actions[-7:]):
        tokens.append(f"ACT_BIGRAM={a}>{b}")
    for a, b, c in zip(actions[-8:], actions[-7:], actions[-6:]):
        tokens.append(f"ACT_TRIGRAM={a}>{b}>{c}")
    counts = Counter(actions)
    tokens.extend(f"ACT_COUNT_{name}={count}" for name, count in sorted(counts.items()))
    tokens.append("ACT_SEQ=" + ">".join(actions[-8:]))
    return actions, " ".join(tokens)


def state_text(sample, mode="v1"):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    hist = sample.get("history", []) or []
    prompt = clean_text(sample.get("current_prompt"), 1400)
    actions, action_text = action_features(hist)

    users = [item for item in hist if item.get("role") == "user"]
    tools = [item for item in hist if item.get("role") == "assistant_action"]
    recent_users = users[-4:]
    recent_tools = tools[-5:]

    parts = []
    prompt_repeat = 2 if mode in {"v2", "v3"} else 1
    parts.extend(["CURRENT_PROMPT " + prompt] * prompt_repeat)

    for i, item in enumerate(reversed(recent_users), start=1):
        text = clean_text(item.get("content"), 1000)
        parts.append(f"USER_BACK_{i} " + text)
        if i <= 2 and mode in {"v2", "v3"}:
            parts.append(f"USER_BACK_{i}_AGAIN " + text)

    parts.append(action_text)

    for i, item in enumerate(reversed(recent_tools), start=1):
        name = clean_text(item.get("name"), 80)
        result = clean_text(item.get("result_summary"), 1000)
        args = clean_text(item.get("args"), 1000)
        parts.append(f"TOOL_BACK_{i}={name}")
        parts.append(f"RESULT_BACK_{i} " + result)
        parts.append(f"ARGS_BACK_{i} " + args)
        if i == 1 and mode in {"v2", "v3"}:
            parts.append("LAST_RESULT_AGAIN " + result)
            parts.append("LAST_ARGS_AGAIN " + args)

    turn = meta.get("turn_index", 0)
    elapsed = meta.get("elapsed_session_sec", 0)
    budget = meta.get("budget_tokens_remaining", 0)
    lang_mix = ws.get("language_mix", {}) or {}
    lang_tokens = []
    for lang, ratio in sorted(lang_mix.items()):
        try:
            share = int(round(float(ratio) * 10))
        except (TypeError, ValueError):
            share = 0
        lang_tokens.append(f"CODE_LANG={lang}")
        lang_tokens.append(f"CODE_LANG_SHARE={lang}_{share}")

    parts.extend(
        [
            f"USER_TIER={clean_text(meta.get('user_tier'), 40)}",
            f"LANG_PREF={clean_text(meta.get('language_pref'), 40)}",
            f"TURN={turn}",
            "TURN_BIN=" + bucket(turn, [("early", 2), ("mid", 6), ("late", 10)]),
            "ELAPSED_BIN=" + bucket(elapsed, [("start", 240), ("middle", 900), ("long", 1800)]),
            "BUDGET_BIN=" + bucket(budget, [("low", 10_000), ("mid", 60_000), ("high", 140_000)]),
            f"CI={clean_text(ws.get('last_ci_status'), 40)}",
            f"GIT_DIRTY={ws.get('git_dirty', 'unknown')}",
            "LOC_BIN=" + bucket(ws.get("loc"), [("small", 2_000), ("medium", 10_000), ("large", 30_000)]),
            " ".join(lang_tokens),
            path_tokens(ws.get("open_files", [])),
        ]
    )
    if mode == "v3":
        # Give the local workflow state more weight without flooding with long natural language.
        parts.append(action_text)
        parts.append(path_tokens(ws.get("open_files", [])))
    return " || ".join(part for part in parts if part)


def last_actions(sample):
    actions = [
        item.get("name", "")
        for item in sample.get("history", []) or []
        if item.get("role") == "assistant_action" and item.get("name")
    ]
    last1 = actions[-1] if actions else "NONE"
    last2 = ">".join(actions[-2:]) if len(actions) >= 2 else "NONE>" + last1
    last3 = ">".join(actions[-3:]) if len(actions) >= 3 else "NONE>" + last2
    return last1, last2, last3


def build_vectorizer(max_features=260_000, min_df=2, char_max=5):
    half = max_features // 2
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=half,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, char_max),
        min_df=min_df,
        max_features=half,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )
    return FeatureUnion([("word", word), ("char", char)])


def align_scores(classes, scores):
    out = np.full((scores.shape[0], len(ALL_CLASSES)), -1e9, dtype=np.float32)
    for i, cls in enumerate(classes):
        out[:, ALL_CLASSES.index(cls)] = scores[:, i]
    return out


def model_score_matrix(model, x_val):
    if hasattr(model, "predict_proba"):
        scores = np.log(model.predict_proba(x_val) + 1e-9)
        return align_scores(model.classes_, scores)
    scores = model.decision_function(x_val)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return align_scores(model.classes_, scores)


def normalize_rows(scores):
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores - scores.mean(axis=1, keepdims=True)
    return scores / (scores.std(axis=1, keepdims=True) + 1e-6)


def pred_from_scores(scores):
    return np.array([ALL_CLASSES[i] for i in scores.argmax(axis=1)], dtype=object)


def evaluate_scores(name, scores, y_val, results_path, extra=None):
    pred = pred_from_scores(scores)
    f1 = f1_score(y_val, pred, labels=ALL_CLASSES, average="macro", zero_division=0)
    acc = accuracy_score(y_val, pred)
    row = {"name": name, "status": "ok", "macro_f1": f1, "accuracy": acc, "seconds": 0.0}
    if extra:
        row.update(extra)
    append_jsonl(results_path, row)
    print(f"{name}: macro_f1={f1:.6f} acc={acc:.6f}", flush=True)
    return row


def transition_log_prior(samples_train, y_train, samples_val, key_index, smooth=1.0):
    counts = defaultdict(lambda: Counter())
    global_counts = Counter(y_train)
    global_total = sum(global_counts.values())
    global_logp = np.array(
        [
            math.log((global_counts[cls] + smooth) / (global_total + smooth * len(ALL_CLASSES)))
            for cls in ALL_CLASSES
        ],
        dtype=np.float32,
    )
    for sample, label in zip(samples_train, y_train):
        key = last_actions(sample)[key_index]
        counts[key][label] += 1
    matrix = []
    for sample in samples_val:
        key = last_actions(sample)[key_index]
        if key not in counts:
            matrix.append(global_logp)
            continue
        c = counts[key]
        total = sum(c.values())
        row = [
            math.log((c[cls] + smooth) / (total + smooth * len(ALL_CLASSES)))
            for cls in ALL_CLASSES
        ]
        matrix.append(row)
    return np.array(matrix, dtype=np.float32)


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def train_eval(name, model, x_train, y_train, x_val, y_val, results_path):
    start = time.time()
    print(f"\n=== {name} ===", flush=True)
    try:
        model.fit(x_train, y_train)
        scores = model_score_matrix(model, x_val)
        pred = pred_from_scores(scores)
        f1 = f1_score(y_val, pred, labels=ALL_CLASSES, average="macro", zero_division=0)
        acc = accuracy_score(y_val, pred)
        report = classification_report(
            y_val,
            pred,
            labels=ALL_CLASSES,
            output_dict=True,
            zero_division=0,
        )
        row = {
            "name": name,
            "status": "ok",
            "macro_f1": f1,
            "accuracy": acc,
            "seconds": time.time() - start,
        }
        append_jsonl(results_path, row)
        print(f"{name}: macro_f1={f1:.6f} acc={acc:.6f} sec={row['seconds']:.1f}", flush=True)
        return {"name": name, "model": model, "scores": scores, "row": row, "report": report}
    except Exception as exc:
        row = {
            "name": name,
            "status": "error",
            "seconds": time.time() - start,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        append_jsonl(results_path, row)
        print(f"ERROR {name}: {exc!r}", flush=True)
        return {"name": name, "model": None, "scores": None, "row": row, "report": None}


def train_group_specialists(x_train, y_train, x_val, results_path):
    group_scores = np.full((x_val.shape[0], len(ALL_CLASSES)), -1e9, dtype=np.float32)
    for group, classes in GROUPS.items():
        mask = np.array([label in classes for label in y_train])
        if mask.sum() < len(classes):
            continue
        model = LogisticRegression(max_iter=700, C=2.0, class_weight="balanced", random_state=42)
        start = time.time()
        model.fit(x_train[mask], y_train[mask])
        scores = model_score_matrix(model, x_val)
        for cls in classes:
            idx = ALL_CLASSES.index(cls)
            group_scores[:, idx] = scores[:, idx]
        append_jsonl(
            results_path,
            {
                "name": f"group_specialist_{group}",
                "status": "ok",
                "seconds": time.time() - start,
                "train_rows": int(mask.sum()),
                "macro_f1": 0.0,
                "accuracy": 0.0,
                "auxiliary": True,
            },
        )
        print(f"group specialist {group}: rows={mask.sum()} sec={time.time()-start:.1f}", flush=True)
    return group_scores


def update_research_md(path, summary):
    lines = []
    lines.append("\n## State Routing Experiments\n")
    lines.append(f"- Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if summary.get("best"):
        b = summary["best"]
        lines.append(f"- Best Macro-F1: `{b['macro_f1']:.6f}` via `{b['name']}`\n")
    lines.append("- Key idea: richer state serialization + transition priors + group specialists.\n")
    lines.append("\nTop results:\n")
    for row in summary.get("top10", [])[:10]:
        lines.append(f"- `{row['macro_f1']:.6f}` `{row['name']}`\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--run-dir", default="./state_router_runs")
    parser.add_argument("--max-features", type=int, default=260_000)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
    labels = load_labels(Path(args.data_dir) / "train_labels.csv")
    y = np.array([labels[s["id"]] for s in samples], dtype=object)
    idx = np.arange(len(samples))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    outputs = []
    for mode in ["v1", "v2", "v3"]:
        print(f"\nbuild state text mode={mode}", flush=True)
        texts = [state_text(sample, mode) for sample in samples]
        x_train_text = [texts[i] for i in train_idx]
        x_val_text = [texts[i] for i in val_idx]
        vectorizer = build_vectorizer(max_features=args.max_features, min_df=2, char_max=5)
        start = time.time()
        x_train = vectorizer.fit_transform(x_train_text)
        x_val = vectorizer.transform(x_val_text)
        print(f"vectorized mode={mode} shape={x_train.shape} sec={time.time()-start:.1f}", flush=True)

        lr = train_eval(
            f"state_{mode}_union_logreg_c2",
            LogisticRegression(max_iter=900, class_weight="balanced", C=2.0, random_state=42),
            x_train,
            y_train,
            x_val,
            y_val,
            results_path,
        )
        outputs.append(lr)
        outputs.append(
            train_eval(
                f"state_{mode}_union_lsvc_c0p7",
                LinearSVC(C=0.7, class_weight="balanced", random_state=42, dual="auto", max_iter=2500),
                x_train,
                y_train,
                x_val,
                y_val,
                results_path,
            )
        )
        if mode == "v2":
            outputs.append(
                train_eval(
                    f"state_{mode}_union_sgd_log",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=1e-5,
                        max_iter=80,
                        tol=1e-4,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=args.threads,
                    ),
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    results_path,
                )
            )

        if lr["scores"] is not None:
            base = normalize_rows(lr["scores"])
            prior1 = normalize_rows(transition_log_prior(train_samples, y_train, val_samples, 0, smooth=0.7))
            prior2 = normalize_rows(transition_log_prior(train_samples, y_train, val_samples, 1, smooth=1.5))
            prior3 = normalize_rows(transition_log_prior(train_samples, y_train, val_samples, 2, smooth=3.0))
            for a1 in [0.05, 0.10, 0.15, 0.20]:
                for a2 in [0.00, 0.05, 0.10, 0.15]:
                    scores = base + a1 * prior1 + a2 * prior2
                    evaluate_scores(
                        f"state_{mode}_logreg_transition_a1_{a1}_a2_{a2}",
                        scores,
                        y_val,
                        results_path,
                        {"base": lr["name"], "a1": a1, "a2": a2},
                    )
            for a3 in [0.03, 0.06, 0.10]:
                scores = base + 0.12 * prior1 + 0.08 * prior2 + a3 * prior3
                evaluate_scores(
                    f"state_{mode}_logreg_transition_last123_a3_{a3}",
                    scores,
                    y_val,
                    results_path,
                    {"base": lr["name"], "a1": 0.12, "a2": 0.08, "a3": a3},
                )

        if mode == "v2":
            group_scores = train_group_specialists(x_train, y_train, x_val, results_path)
            if lr["scores"] is not None:
                base = normalize_rows(lr["scores"])
                group_norm = normalize_rows(np.maximum(group_scores, -50))
                for w in [0.05, 0.10, 0.15, 0.25]:
                    scores = base + w * group_norm
                    evaluate_scores(
                        f"state_{mode}_logreg_group_specialist_w{w}",
                        scores,
                        y_val,
                        results_path,
                        {"base": lr["name"], "group_weight": w},
                    )

    ok_rows = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") == "ok" and not row.get("auxiliary") and "macro_f1" in row:
                ok_rows.append(row)
    ok_rows = sorted(ok_rows, key=lambda r: r["macro_f1"], reverse=True)
    summary = {
        "best": ok_rows[0] if ok_rows else None,
        "top10": ok_rows[:10],
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    update_research_md("research.md", summary)
    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
