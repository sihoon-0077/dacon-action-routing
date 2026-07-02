import argparse
import csv
import json
import math
import re
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from script import serialize_sample


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
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_labels(path):
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def clean(value, max_chars=500):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(text.split())[:max_chars]


def actions(sample):
    return [
        h.get("name", "")
        for h in sample.get("history", []) or []
        if h.get("role") == "assistant_action" and h.get("name")
    ]


def last_actions(sample):
    acts = actions(sample)
    last1 = acts[-1] if acts else "NONE"
    last2 = ">".join(acts[-2:]) if len(acts) >= 2 else "NONE>" + last1
    return last1, last2


def flag_tokens(sample):
    hist = sample.get("history", []) or []
    prompt = clean(sample.get("current_prompt"), 700).lower()
    tokens = []

    recent_tools = [h for h in hist if h.get("role") == "assistant_action"][-4:]
    for i, tool in enumerate(reversed(recent_tools), start=1):
        name = clean(tool.get("name"), 80)
        result = clean(tool.get("result_summary"), 900).lower()
        args = clean(tool.get("args"), 900).lower()
        blob = result + " " + args
        tokens.append(f"FLAG_TOOL_BACK_{i}={name}")
        for flag, patterns in {
            "failed": ["fail", "failed", "error", "traceback", "exception", "nonzero"],
            "passed": ["pass", "passed", "success", "ok", "green"],
            "test": ["test", "pytest", "spec", "suite"],
            "lint": ["lint", "type", "mypy", "ruff", "flake", "tsc"],
            "found": ["found", "match", "matches", "occurrence", "result"],
            "not_found": ["not found", "no match", "0 match", "missing"],
            "read": ["read", "opened", "lines", "content"],
            "changed": ["edited", "patched", "modified", "wrote", "created", "updated"],
            "command": ["command", "bash", "shell", "exit"],
        }.items():
            if any(p in blob for p in patterns):
                tokens.append(f"RESULT_{i}_{flag}=1")
        for path in re.findall(r"[\w./\\-]+\.\w{1,8}", blob)[:8]:
            path = path.replace("\\", "/")
            tokens.append("ARG_PATH=" + path)
            tokens.append("ARG_EXT=" + path.rsplit(".", 1)[-1])
        for key in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', args)[:12]:
            tokens.append("ARG_KEY=" + key.lower())

    for flag, patterns in {
        "prompt_run": ["run", "돌려", "실행", "build", "test", "pytest", "테스트"],
        "prompt_lint": ["lint", "typecheck", "mypy", "ruff", "tsc", "타입"],
        "prompt_open": ["open", "read", "show", "열어", "보여", "확인"],
        "prompt_search": ["search", "grep", "find", "where", "찾", "어디"],
        "prompt_list": ["list", "tree", "folder", "directory", "목록"],
        "prompt_glob": ["glob", "*.", "all files", "matching", "패턴"],
        "prompt_fix": ["fix", "change", "edit", "update", "add", "remove", "고쳐", "수정", "추가"],
        "prompt_summary": ["summary", "summarize", "recap", "wrap", "마무리", "요약"],
        "prompt_plan": ["plan", "step", "단계", "쪼개", "계획"],
        "prompt_web": ["web", "google", "latest", "docs", "online", "검색해"],
    }.items():
        if any(p in prompt for p in patterns):
            tokens.append(flag + "=1")
    return " ".join(tokens)


def compact_text(sample, mode):
    base = serialize_sample(sample, "compact")
    if mode == "base":
        return base
    if mode == "flags":
        return base + " FLAGS " + flag_tokens(sample)
    if mode == "flags_x2":
        flags = flag_tokens(sample)
        return base + " FLAGS " + flags + " FLAGS_AGAIN " + flags
    raise ValueError(mode)


def build_vectorizer(max_features, min_df=2):
    half = max_features // 2
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=min_df,
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
                    min_df=min_df,
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


def align_scores(classes, scores):
    out = np.full((scores.shape[0], len(ALL_CLASSES)), -1e9, dtype=np.float32)
    for i, cls in enumerate(classes):
        out[:, ALL_CLASSES.index(str(cls))] = scores[:, i]
    return out


def score_matrix(model, x):
    if hasattr(model, "predict_proba"):
        return align_scores(model.classes_, np.log(model.predict_proba(x) + 1e-9))
    scores = model.decision_function(x)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return align_scores(model.classes_, scores)


def normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    scores -= scores.mean(axis=1, keepdims=True)
    return scores / (scores.std(axis=1, keepdims=True) + 1e-6)


def predict(scores):
    return np.array([ALL_CLASSES[i] for i in scores.argmax(axis=1)], dtype=object)


def evaluate_scores(name, scores, y_val, results_path, extra=None):
    pred = predict(scores)
    f1 = f1_score(y_val, pred, labels=ALL_CLASSES, average="macro", zero_division=0)
    acc = accuracy_score(y_val, pred)
    row = {"name": name, "status": "ok", "macro_f1": f1, "accuracy": acc, "seconds": 0.0}
    if extra:
        row.update(extra)
    append_jsonl(results_path, row)
    print(f"{name}: macro_f1={f1:.6f} acc={acc:.6f}", flush=True)
    return row


def transition_prior(train_samples, y_train, val_samples, key_index, smooth):
    counts = defaultdict(Counter)
    global_counts = Counter(y_train)
    global_total = sum(global_counts.values())
    global_row = np.array(
        [math.log((global_counts[c] + smooth) / (global_total + smooth * len(ALL_CLASSES))) for c in ALL_CLASSES],
        dtype=np.float32,
    )
    for sample, label in zip(train_samples, y_train):
        counts[last_actions(sample)[key_index]][label] += 1
    rows = []
    for sample in val_samples:
        key = last_actions(sample)[key_index]
        if key not in counts:
            rows.append(global_row)
            continue
        c = counts[key]
        total = sum(c.values())
        rows.append([math.log((c[cls] + smooth) / (total + smooth * len(ALL_CLASSES))) for cls in ALL_CLASSES])
    return np.array(rows, dtype=np.float32)


def group_score_matrix(model, x_val):
    group_classes = list(model.classes_)
    scores = np.log(model.predict_proba(x_val) + 1e-9)
    out = np.zeros((x_val.shape[0], len(ALL_CLASSES)), dtype=np.float32)
    for gi, group in enumerate(group_classes):
        for cls in GROUPS[group]:
            out[:, ALL_CLASSES.index(cls)] = scores[:, gi]
    return out


def rule_scores(samples):
    out = np.zeros((len(samples), len(ALL_CLASSES)), dtype=np.float32)
    for i, sample in enumerate(samples):
        prompt = clean(sample.get("current_prompt"), 700).lower()
        last1, _ = last_actions(sample)
        hist = sample.get("history", []) or []
        last_result = ""
        for item in reversed(hist):
            if item.get("role") == "assistant_action":
                last_result = clean(item.get("result_summary"), 700).lower()
                break

        def add(cls, value):
            out[i, ALL_CLASSES.index(cls)] += value

        if any(p in prompt for p in ["run", "돌려", "실행", "build"]):
            add("run_bash", 1.0)
            add("run_tests", 0.7)
        if any(p in prompt for p in ["test", "pytest", "spec", "테스트"]):
            add("run_tests", 1.3)
        if any(p in prompt for p in ["lint", "typecheck", "mypy", "ruff", "tsc"]):
            add("lint_or_typecheck", 1.4)
        if any(p in prompt for p in ["open", "read", "show", "열어", "보여"]):
            add("read_file", 1.2)
        if any(p in prompt for p in ["where", "find", "search", "grep", "찾", "어디"]):
            add("grep_search", 1.1)
        if any(p in prompt for p in ["list", "tree", "directory", "folder", "목록"]):
            add("list_directory", 1.1)
        if any(p in prompt for p in ["glob", "*.", "matching", "pattern"]):
            add("glob_pattern", 1.3)
        if any(p in prompt for p in ["summary", "summarize", "recap", "wrap", "마무리", "요약"]):
            add("respond_only", 1.5)
        if any(p in prompt for p in ["plan", "step", "단계", "쪼개", "계획"]):
            add("plan_task", 1.0)
        if any(p in prompt for p in ["web", "google", "latest", "online"]):
            add("web_search", 1.4)
        if any(p in prompt for p in ["fix", "edit", "change", "update", "고쳐", "수정"]):
            add("edit_file", 1.1)
            add("apply_patch", 0.5)
        if any(p in prompt for p in ["new file", "create file", "새 파일", "만들"]):
            add("write_file", 1.2)

        if last1 in {"edit_file", "apply_patch", "write_file"}:
            add("run_tests", 0.6)
            add("lint_or_typecheck", 0.35)
            add("run_bash", 0.25)
        if last1 in {"grep_search", "glob_pattern", "list_directory"}:
            add("read_file", 0.55)
            add("grep_search", 0.25)
        if last1 == "read_file":
            add("edit_file", 0.45)
            add("grep_search", 0.25)
        if last1 in {"run_tests", "lint_or_typecheck"} and any(p in last_result for p in ["fail", "error", "traceback", "failed"]):
            add("edit_file", 0.75)
            add("apply_patch", 0.45)
    return out


def train_eval(name, model, x_train, y_train, x_val, y_val, results_path):
    start = time.time()
    print(f"\n=== {name} ===", flush=True)
    try:
        model.fit(x_train, y_train)
        scores = score_matrix(model, x_val)
        row = evaluate_scores(name, scores, y_val, results_path, {"seconds": time.time() - start})
        return {"name": name, "model": model, "scores": scores, "row": row}
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
        return {"name": name, "model": None, "scores": None, "row": row}


def update_research_md(summary):
    with open("research.md", "a", encoding="utf-8") as f:
        f.write("\n## Compact State Score Experiments\n")
        f.write(f"- Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if summary.get("best"):
            b = summary["best"]
            f.write(f"- Best Macro-F1: `{b['macro_f1']:.6f}` via `{b['name']}`\n")
        f.write("- Key idea: keep compact text; add small transition/group/rule score adjustments.\n")
        f.write("\nTop results:\n")
        for row in summary.get("top10", [])[:10]:
            f.write(f"- `{row['macro_f1']:.6f}` `{row['name']}`\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--run-dir", default="./compact_state_runs")
    parser.add_argument("--max-features", type=int, default=220_000)
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

    all_outputs = []
    for mode in ["base", "flags", "flags_x2"]:
        print(f"\nMODE {mode}", flush=True)
        texts = [compact_text(sample, mode) for sample in samples]
        x_train_text = [texts[i] for i in train_idx]
        x_val_text = [texts[i] for i in val_idx]
        vectorizer = build_vectorizer(args.max_features, min_df=2)
        start = time.time()
        x_train = vectorizer.fit_transform(x_train_text)
        x_val = vectorizer.transform(x_val_text)
        print(f"vectorized {mode} shape={x_train.shape} sec={time.time()-start:.1f}", flush=True)

        lr = train_eval(
            f"compact_{mode}_logreg_c2",
            LogisticRegression(max_iter=900, C=2.0, class_weight="balanced", random_state=42),
            x_train,
            y_train,
            x_val,
            y_val,
            results_path,
        )
        all_outputs.append(lr)
        lsvc = train_eval(
            f"compact_{mode}_lsvc_c0p5",
            LinearSVC(C=0.5, class_weight="balanced", random_state=42, dual="auto", max_iter=2500),
            x_train,
            y_train,
            x_val,
            y_val,
            results_path,
        )
        all_outputs.append(lsvc)

        if lr["scores"] is None:
            continue
        base_scores = normalize(lr["scores"])
        prior1 = normalize(transition_prior(train_samples, y_train, val_samples, 0, smooth=1.0))
        prior2 = normalize(transition_prior(train_samples, y_train, val_samples, 1, smooth=2.0))
        rules = normalize(rule_scores(val_samples))

        group_labels = np.array([CLASS_TO_GROUP[label] for label in y_train], dtype=object)
        group_model = LogisticRegression(max_iter=500, C=2.0, class_weight="balanced", random_state=42)
        group_model.fit(x_train, group_labels)
        group_scores = normalize(group_score_matrix(group_model, x_val))

        for a1 in [0.02, 0.04, 0.06, 0.08, 0.10]:
            for a2 in [0.00, 0.02, 0.04, 0.06]:
                evaluate_scores(
                    f"compact_{mode}_lr_prior_a1_{a1}_a2_{a2}",
                    base_scores + a1 * prior1 + a2 * prior2,
                    y_val,
                    results_path,
                    {"base": lr["name"], "a1": a1, "a2": a2},
                )
        for gw in [0.03, 0.06, 0.10, 0.15]:
            evaluate_scores(
                f"compact_{mode}_lr_group_gw_{gw}",
                base_scores + gw * group_scores,
                y_val,
                results_path,
                {"base": lr["name"], "group_weight": gw},
            )
        for rw in [0.02, 0.04, 0.06, 0.10]:
            evaluate_scores(
                f"compact_{mode}_lr_rules_rw_{rw}",
                base_scores + rw * rules,
                y_val,
                results_path,
                {"base": lr["name"], "rule_weight": rw},
            )
        for a1 in [0.04, 0.06, 0.08]:
            for gw in [0.04, 0.08]:
                for rw in [0.02, 0.04]:
                    evaluate_scores(
                        f"compact_{mode}_lr_combo_a1_{a1}_gw_{gw}_rw_{rw}",
                        base_scores + a1 * prior1 + 0.03 * prior2 + gw * group_scores + rw * rules,
                        y_val,
                        results_path,
                        {"base": lr["name"], "a1": a1, "a2": 0.03, "group_weight": gw, "rule_weight": rw},
                    )

    ok_rows = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") == "ok" and "macro_f1" in row:
                ok_rows.append(row)
    ok_rows = sorted(ok_rows, key=lambda row: row["macro_f1"], reverse=True)
    summary = {"best": ok_rows[0] if ok_rows else None, "top10": ok_rows[:10], "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    update_research_md(summary)
    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
