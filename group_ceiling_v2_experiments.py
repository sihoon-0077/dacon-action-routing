import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from advanced_action_routing_experiments import group_extra_tokens
from compact_state_experiments import ALL_CLASSES, compact_text
from routing_margin_experiments import ACTION_TO_GROUP, GROUP_TO_ACTIONS


SEED = 42


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


def clean(value, max_chars=1200):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(text.split())[:max_chars]


def current_prompt(sample):
    return clean(sample.get("current_prompt"), 1200)


def assistant_actions(sample):
    return [
        h
        for h in sample.get("history", []) or []
        if h.get("role") == "assistant_action"
    ]


def last_action(sample):
    acts = assistant_actions(sample)
    return acts[-1].get("name", "NONE") if acts else "NONE"


def last_result(sample):
    acts = assistant_actions(sample)
    return clean(acts[-1].get("result_summary"), 1000) if acts else ""


def workspace(sample):
    meta = sample.get("session_meta", {}) or {}
    return meta.get("workspace", {}) or {}


def has_any(text, words):
    text = text.lower()
    return float(any(word in text for word in words))


def bucket(value, cuts):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return len(cuts)
    for i, cut in enumerate(cuts):
        if value <= cut:
            return i
    return len(cuts)


def group_numeric_features(sample, group):
    prompt = current_prompt(sample)
    low = prompt.lower()
    hist = sample.get("history", []) or []
    meta = sample.get("session_meta", {}) or {}
    ws = workspace(sample)
    result = last_result(sample).lower()
    la = last_action(sample)
    open_files = ws.get("open_files", []) or []

    features = [
        len(prompt),
        len(prompt.split()),
        prompt.count("?"),
        float("?" in prompt),
        len(hist),
        meta.get("turn_index", 0) or 0,
        (meta.get("budget_tokens_remaining", 0) or 0) / 100000.0,
        (meta.get("elapsed_session_sec", 0) or 0) / 1000.0,
        float(bool(ws.get("git_dirty"))),
        len(open_files),
        bucket(ws.get("loc"), [2000, 10000, 30000]),
    ]

    ci = ws.get("last_ci_status", "none")
    features.extend(float(ci == value) for value in ["passed", "failed", "none"])
    features.extend(float(la == action) for action in ALL_CLASSES)
    features.extend(
        [
            float("error" in result or "traceback" in result or "exception" in result),
            float("fail" in result or "failed" in result),
            float("pass" in result or "green" in result or "success" in result),
            float("match" in result or "occurrence" in result or "found" in result),
            float("no match" in result or "not found" in result),
            float("file" in result or "listed" in result),
        ]
    )

    if group == "communicate":
        features.extend(
            [
                has_any(low, ["which", "should i", "or ", "어느", "뭐가", "할까", "괜찮", "?"]),
                has_any(low, ["plan", "step", "approach", "strategy", "계획", "단계", "순서"]),
                has_any(low, ["web", "latest", "online", "recommended", "official", "docs", "최신", "공식"]),
                has_any(low, ["summary", "summarize", "recap", "wrap", "brief", "요약", "정리", "마무리"]),
                low.count("?"),
                float(low.strip().endswith("?")),
                has_any(low, ["http", "://", "version", "release", "package", "library"]),
            ]
        )
    elif group == "execute":
        features.extend(
            [
                has_any(low, ["test", "pytest", "npm test", "cargo test", "unit", "integration", "e2e", "spec", "테스트"]),
                has_any(low, ["lint", "typecheck", "type check", "mypy", "ruff", "tsc", "static", "타입"]),
                has_any(low, ["build", "run", "server", "dev server", "command", "bash", "script", "실행", "빌드"]),
                has_any(low, ["install", "pip install", "npm install", "pod install", "설치"]),
                float(ci == "failed"),
                float(ci == "passed"),
                float(la in {"edit_file", "write_file", "apply_patch"}),
            ]
        )
    elif group == "modify":
        features.extend(
            [
                has_any(low, ["fix", "edit", "change", "replace", "refactor", "update", "고쳐", "수정"]),
                has_any(low, ["new file", "create", "write", "scaffold", "새 파일", "생성"]),
                has_any(low, ["patch", "diff", "multiple", "both", "several", "여러", "같이"]),
                float(la in {"edit_file", "apply_patch", "write_file"}),
                len(re.findall(r"[\w./\\-]+\.[A-Za-z0-9]{1,8}", low)),
            ]
        )
    else:
        features.extend(
            [
                has_any(low, ["open", "read", "show", "look", "peek", "열어", "보여", "읽"]),
                has_any(low, ["find", "where", "grep", "search", "occurrence", "찾", "검색", "어디"]),
                has_any(low, ["list", "folder", "directory", "tree", "목록", "폴더", "루트"]),
                has_any(low, ["glob", "*.", "**/", "matching", "pattern", "패턴"]),
                len(re.findall(r"[\w./\\-]+\.[A-Za-z0-9]{1,8}", low)),
            ]
        )

    return features


def build_text(sample, group, variant):
    if variant == "prompt":
        return current_prompt(sample)
    if variant == "compact":
        return compact_text(sample, "flags")
    if variant == "specialized":
        return compact_text(sample, "flags") + " GROUP_EXTRA " + group_extra_tokens(sample, group)
    if variant == "specialized_x2":
        extra = group_extra_tokens(sample, group)
        return compact_text(sample, "flags") + " GROUP_EXTRA " + extra + " GROUP_EXTRA_AGAIN " + extra
    raise ValueError(variant)


def vectorize(train_texts, val_texts, max_features):
    word_features = max_features // 2
    char_features = max_features - word_features
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=word_features,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=char_features,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )
    w_train = word.fit_transform(train_texts)
    w_val = word.transform(val_texts)
    c_train = char.fit_transform(train_texts)
    c_val = char.transform(val_texts)
    return w_train, w_val, hstack([w_train, c_train]).tocsr(), hstack([w_val, c_val]).tocsr()


def evaluate(name, x_train, x_val, y_train, y_val, classes):
    model = LogisticRegression(max_iter=1600, C=3.0, class_weight="balanced", random_state=SEED)
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    row = {
        "name": name,
        "macro_f1": float(f1_score(y_val, pred, labels=np.arange(len(classes)), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_val, pred)),
    }
    per = f1_score(y_val, pred, labels=np.arange(len(classes)), average=None, zero_division=0)
    row["per_class_f1"] = {cls: float(score) for cls, score in zip(classes, per)}
    row["confusion"] = confusion_matrix(y_val, pred, labels=np.arange(len(classes))).tolist()
    print(f"{name}: macro_f1={row['macro_f1']:.6f} acc={row['accuracy']:.6f}", flush=True)
    return row


def run_group(samples, y_all, sessions, group, out_dir, max_features):
    classes = GROUP_TO_ACTIONS[group]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx = np.array([i for i, label in enumerate(y_all) if label in classes], dtype=int)
    group_samples = [samples[i] for i in idx]
    y = np.array([class_to_idx[y_all[i]] for i in idx], dtype=int)
    sess = np.array([sessions[i] for i in idx], dtype=object)
    train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(np.zeros(len(y)), y, groups=sess))

    print(f"\n=== {group} isolate {len(idx)} rows {len(classes)}-way ===", flush=True)
    numeric = np.array([group_numeric_features(sample, group) for sample in group_samples], dtype=np.float32)
    scaler = StandardScaler()
    n_train = csr_matrix(scaler.fit_transform(numeric[train_idx]))
    n_val = csr_matrix(scaler.transform(numeric[val_idx]))

    rows = []
    for variant in ["prompt", "compact", "specialized", "specialized_x2"]:
        start = time.time()
        texts = [build_text(sample, group, variant) for sample in group_samples]
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        w_train, w_val, wc_train, wc_val = vectorize(train_texts, val_texts, max_features)
        print(f"{group}/{variant} vectorized sec={time.time() - start:.1f}", flush=True)
        rows.append(evaluate(f"{group}_{variant}_word", w_train, w_val, y[train_idx], y[val_idx], classes))
        rows.append(evaluate(f"{group}_{variant}_word_char", wc_train, wc_val, y[train_idx], y[val_idx], classes))
        rows.append(
            evaluate(
                f"{group}_{variant}_word_char_num",
                hstack([wc_train, n_train]).tocsr(),
                hstack([wc_val, n_val]).tocsr(),
                y[train_idx],
                y[val_idx],
                classes,
            )
        )

    rows.sort(key=lambda row: row["macro_f1"], reverse=True)
    group_dir = out_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "metrics.json").write_text(json.dumps({"group": group, "classes": classes, "top": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(group_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "macro_f1", "accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"name": row["name"], "macro_f1": row["macro_f1"], "accuracy": row["accuracy"]})
    return rows[0], rows


def estimate_macro_upper(group_best):
    values = {}
    for group, best in group_best.items():
        values.update(best["per_class_f1"])
    return float(sum(values[action] for action in ALL_CLASSES) / len(ALL_CLASSES))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./reports/exp_v2_group_ceiling")
    parser.add_argument("--max-features", type=int, default=80_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")

    samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
    labels = load_labels(Path(args.data_dir) / "train_labels.csv")
    y_all = np.array([labels[s["id"]] for s in samples], dtype=object)
    sessions = np.array([session_id(s["id"]) for s in samples], dtype=object)

    group_best = {}
    all_rows = []
    for group in ["inspect", "modify", "execute", "communicate"]:
        best, rows = run_group(samples, y_all, sessions, group, out_dir, args.max_features)
        group_best[group] = best
        for row in rows:
            row_copy = dict(row)
            row_copy.pop("confusion", None)
            all_rows.append(row_copy)

    upper = estimate_macro_upper(group_best)
    summary = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "group_best": group_best,
        "isolated_macro_upper_estimate": upper,
        "notes": "Group-isolated F1 is not directly comparable to full pipeline F1, but it estimates whether each group has enough signal.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "macro_f1", "accuracy"])
        writer.writeheader()
        for row in sorted(all_rows, key=lambda item: item["macro_f1"], reverse=True):
            writer.writerow({"name": row["name"], "macro_f1": row["macro_f1"], "accuracy": row["accuracy"]})

    with open("research.md", "a", encoding="utf-8") as f:
        f.write("\n## NEXT_EXPERIMENT_v2 Group Ceiling\n")
        f.write(f"- Finished: {summary['finished_at']}\n")
        f.write(f"- Run dir: `{out_dir}`\n")
        f.write(f"- Isolated macro upper estimate: `{upper:.6f}`\n")
        for group, best in group_best.items():
            f.write(f"- `{group}` best `{best['macro_f1']:.6f}` via `{best['name']}`\n")

    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
