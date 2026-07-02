import argparse
import csv
import json
import math
import os
import re
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from compact_state_experiments import ALL_CLASSES, compact_text
from routing_margin_experiments import ACTION_TO_GROUP, GROUP_TO_ACTIONS


SEED = 42
GROUPS = list(GROUP_TO_ACTIONS)
PAIR_PRIORITY = [
    ("read_file", "grep_search"),
    ("grep_search", "glob_pattern"),
    ("list_directory", "glob_pattern"),
    ("edit_file", "apply_patch"),
    ("edit_file", "write_file"),
    ("run_bash", "run_tests"),
    ("run_tests", "lint_or_typecheck"),
    ("ask_user", "plan_task"),
    ("plan_task", "web_search"),
    ("respond_only", "plan_task"),
]


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


def bucket_number(value, bins):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for name, upper in bins:
        if value <= upper:
            return name
    return "huge"


def action_names(sample):
    return [
        h.get("name", "")
        for h in sample.get("history", []) or []
        if h.get("role") == "assistant_action" and h.get("name")
    ]


def last_action(sample):
    acts = action_names(sample)
    return acts[-1] if acts else "NONE"


def last2_action(sample):
    acts = action_names(sample)
    return ">".join(acts[-2:]) if len(acts) >= 2 else "NONE>" + (acts[-1] if acts else "NONE")


def last_result_summary(sample):
    for item in reversed(sample.get("history", []) or []):
        if item.get("role") == "assistant_action":
            return clean(item.get("result_summary"), 1000)
    return ""


def result_type(sample):
    text = last_result_summary(sample).lower()
    if not text:
        return "RESULT_NONE"
    if any(x in text for x in ["traceback", "exception", "error", "conflict"]):
        return "RESULT_ERROR"
    if any(x in text for x in ["failed", "fail", "nonzero", "exit 1", "exit=1"]):
        return "RESULT_FAIL"
    if any(x in text for x in ["passed", "success", "ok", "green", "exit 0", "exit=0"]):
        return "RESULT_PASS"
    if any(x in text for x in ["no match", "0 match", "not found", "missing"]):
        return "RESULT_NO_MATCH"
    if any(x in text for x in ["matches", "occurrences", "found", "results"]):
        return "RESULT_MATCHES"
    if any(x in text for x in ["listed", "files", "directory", "tree"]):
        return "RESULT_LISTED"
    if any(x in text for x in ["read", "opened", "lines", "content"]):
        return "RESULT_READ_OK"
    if any(x in text for x in ["edited", "patched", "modified", "wrote", "created", "updated"]):
        return "RESULT_EDIT_OK"
    return "RESULT_UNKNOWN"


def workspace(sample):
    meta = sample.get("session_meta", {}) or {}
    return meta.get("workspace", {}) or {}


def open_files(sample):
    files = workspace(sample).get("open_files", []) or []
    return [clean(path, 300).replace("\\", "/") for path in files]


def path_ext_tokens(text, prefix):
    tokens = []
    for path in re.findall(r"[\w./\\-]+\.[A-Za-z0-9]{1,8}", text)[:20]:
        norm = path.replace("\\", "/").lower()
        ext = norm.rsplit(".", 1)[-1]
        tokens.append(f"{prefix}_PATH={norm}")
        tokens.append(f"{prefix}_EXT={ext}")
    return tokens


def count_bucket_from_text(text, prefix):
    nums = []
    for match in re.findall(r"\b(\d{1,5})\b", text):
        try:
            nums.append(int(match))
        except ValueError:
            pass
    if not nums:
        return f"{prefix}_COUNT_NONE"
    max_num = max(nums)
    return f"{prefix}_COUNT={bucket_number(max_num, [('zero', 0), ('one', 1), ('few', 5), ('some', 20), ('many', 100)])}"


def simple_rule_hint(sample):
    prompt = clean(sample.get("current_prompt"), 1200).lower()
    rules = [
        ("respond_only", ["요약", "정리", "마무리", "summarize", "summary", "recap", "wrap up", "brief"]),
        ("lint_or_typecheck", ["actionlint", "typecheck", "type check", "lint", "tsc", "mypy", "ruff", "타입"]),
        ("write_file", ["new file", "create file", "make file", "scaffold", "새 파일", "파일 하나"]),
        ("web_search", ["latest", "최신", "best practice", "recommended", "official docs", "공식 문서", "paper"]),
        ("plan_task", ["plan", "계획", "단계", "순서", "break down", "where to start"]),
        ("run_tests", ["pytest", "npm test", "cargo test", "unit test", "integration test", "e2e", "테스트"]),
        ("run_bash", ["build", "dev server", "runserver", "npm run dev", "pip install", "실행"]),
        ("grep_search", ["grep", "search", "find", "where", "occurrence", "찾", "어디", "검색"]),
        ("glob_pattern", ["*.", "**/", "glob", "all files", "matching files"]),
        ("list_directory", ["directory", "folder", "tree", "project root", "목록", "폴더", "루트"]),
        ("read_file", ["open", "read", "show me", "look at", "peek", "열어", "보여", "읽어"]),
        ("edit_file", ["fix", "change", "replace", "update", "remove", "rename", "고쳐", "수정"]),
    ]
    for action, patterns in rules:
        if any(pattern in prompt for pattern in patterns):
            return action
    return "unknown"


def generic_tokens(sample):
    meta = sample.get("session_meta", {}) or {}
    ws = workspace(sample)
    prompt = clean(sample.get("current_prompt"), 1200)
    result = last_result_summary(sample)
    tokens = [
        f"LAST_ACTION={last_action(sample)}",
        f"LAST2_ACTION={last2_action(sample)}",
        f"RESULT_TYPE={result_type(sample)}",
        f"RULE_HINT={simple_rule_hint(sample)}",
        f"RULE_GROUP={ACTION_TO_GROUP.get(simple_rule_hint(sample), 'unknown')}",
        f"CI={clean(ws.get('last_ci_status'), 80)}",
        f"GIT_DIRTY={ws.get('git_dirty', 'unknown')}",
        f"TURN_BIN={bucket_number(meta.get('turn_index'), [('turn_early', 2), ('turn_mid', 7), ('turn_late', 12)])}",
        f"BUDGET_BIN={bucket_number(meta.get('budget_tokens_remaining'), [('budget_low', 10000), ('budget_mid', 50000), ('budget_high', 120000)])}",
        f"PROMPT_LEN={bucket_number(len(prompt), [('short', 30), ('mid', 100), ('long', 250)])}",
        count_bucket_from_text(result, "LAST_RESULT"),
    ]
    for path in open_files(sample)[:10]:
        tokens.append("OPEN_FILE=" + path.lower())
        if "." in path.rsplit("/", 1)[-1]:
            tokens.append("OPEN_EXT=" + path.rsplit(".", 1)[-1].lower())
    tokens.extend(path_ext_tokens(prompt + " " + result, "MENTIONED"))
    return tokens


def group_extra_tokens(sample, group):
    prompt = clean(sample.get("current_prompt"), 1200)
    prompt_l = prompt.lower()
    result_l = last_result_summary(sample).lower()
    blob = prompt_l + " " + result_l
    tokens = generic_tokens(sample)

    def add(name, patterns):
        if has_any(blob, patterns):
            tokens.append(name)

    if group == "inspect":
        add("HAS_EXPLICIT_FILENAME", [r"[\w./\\-]+\.[A-Za-z0-9]{1,8}"])
        add("HAS_GLOB_PATTERN", [r"\*\.", r"\*\*/", r"\bglob\b", r"all .*files", r"matching files"])
        add("HAS_DIRECTORY_WORD", [r"directory", r"folder", r"tree", r"project root", r"\bls\b", r"목록", r"폴더", r"루트"])
        add("HAS_SEARCH_WORD", [r"\bgrep\b", r"\bsearch\b", r"\bfind\b", r"where ", r"occurrence", r"definition", r"reference", r"찾", r"어디", r"검색"])
        add("HAS_IMPORT_WORD", [r"\bimport\b", r"from .* import", r"require\("])
        add("HAS_OPEN_WORD", [r"\bopen\b", r"\bread\b", r"show me", r"look at", r"peek", r"what.?s inside", r"열어", r"보여", r"읽"])
        add("HAS_LIST_WORD", [r"\blist\b", r"\btree\b", r"what files", r"목록"])
        for ext in ["py", "tsx", "ts", "js", "jsx", "json", "yaml", "yml", "md", "sql", "java", "kt", "swift", "go", "rs"]:
            if re.search(rf"\.{ext}\b", blob):
                tokens.append(f"MENTIONED_EXT_{ext}")
    elif group == "modify":
        add("HAS_NEW_FILE_WORD", [r"new file", r"create .*file", r"make .*file", r"scaffold", r"새 파일", r"파일 하나"])
        add("HAS_MULTI_FILE_WORD", [r"both", r"in one shot", r"multiple files", r"several files", r"coupled", r"같이", r"여러"])
        add("HAS_PATCH_WORD", [r"\bpatch\b", r"apply_patch", r"diff"])
        add("HAS_FIX_WORD", [r"\bfix\b", r"bug", r"broken", r"error", r"fail", r"고쳐", r"수정"])
        add("HAS_ADD_WORD", [r"\badd\b", r"append", r"insert", r"추가"])
        add("HAS_REPLACE_WORD", [r"replace", r"rename", r"change", r"rewrite", r"바꿔"])
        tokens.append(f"OPEN_FILES_COUNT={bucket_number(len(open_files(sample)), [('none', 0), ('one', 1), ('few', 4), ('some', 10)])}")
    elif group == "execute":
        add("HAS_TEST_WORD", [r"\btests?\b", r"\bpytest\b", r"cargo test", r"npm test", r"unit test", r"integration test", r"\be2e\b", r"\bspec\b", r"테스트"])
        add("HAS_LINT_WORD", [r"actionlint", r"\blint\b", r"type.?check", r"\btsc\b", r"\bmypy\b", r"\bruff\b", r"static analysis", r"타입"])
        add("HAS_BUILD_WORD", [r"\bbuild\b", r"cargo build", r"npm run build", r"빌드"])
        add("HAS_SERVER_WORD", [r"dev server", r"runserver", r"npm run dev", r"serve", r"server", r"서버"])
        add("HAS_INSTALL_WORD", [r"pip install", r"npm install", r"pod install", r"bundle install", r"설치"])
        tokens.append("LAST_CI_FAILED" if workspace(sample).get("last_ci_status") == "failed" else "LAST_CI_NOT_FAILED")
    elif group == "communicate":
        add("HAS_SUMMARY_WORD", [r"요약", r"정리", r"마무리", r"summarize", r"summary", r"recap", r"wrap.?up", r"brief"])
        add("HAS_PLAN_WORD", [r"\bplan\b", r"계획", r"단계", r"순서", r"break.+down", r"where to start", r"approach"])
        add("HAS_WEB_WORD", [r"latest", r"최신", r"best.?practice", r"recommended", r"official docs?", r"공식 문서", r"paper", r"online"])
        add("HAS_QUESTION_WORD", [r"should i", r"which", r"어느", r"뭐가", r"어떻게 할까", r"괜찮"])
        tokens.append("HISTORY_EMPTY" if not sample.get("history") else "HISTORY_PRESENT")
    return " ".join(tokens)


def group_text(sample, group, variant):
    base = compact_text(sample, "flags")
    if variant == "same_text":
        return base
    if variant == "specialized":
        return base + " GROUP_EXTRA " + group_extra_tokens(sample, group)
    if variant == "specialized_x2":
        extra = group_extra_tokens(sample, group)
        return base + " GROUP_EXTRA " + extra + " GROUP_EXTRA_AGAIN " + extra
    raise ValueError(variant)


def pair_text(sample, pair):
    group = ACTION_TO_GROUP[pair[0]]
    return compact_text(sample, "flags") + " PAIR_EXTRA " + group_extra_tokens(sample, group) + f" PAIR={pair[0]}_VS_{pair[1]}"


def build_vectorizer(max_features=220_000):
    half = max_features // 2
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
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


def aligned_log_proba(model, x, classes):
    proba = model.predict_proba(x)
    out = np.full((x.shape[0], len(classes)), -1e9, dtype=np.float32)
    for i, cls in enumerate(model.classes_):
        out[:, classes.index(str(cls))] = np.log(proba[:, i] + 1e-9)
    return out


def top2_from_scores(scores, classes):
    order = np.argsort(scores, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    pred1 = np.array([classes[i] for i in top1], dtype=object)
    pred2 = np.array([classes[i] for i in top2], dtype=object)
    margin = scores[np.arange(scores.shape[0]), top1] - scores[np.arange(scores.shape[0]), top2]
    return pred1, pred2, margin


def top2_from_proba(model, x, classes):
    proba = model.predict_proba(x)
    aligned = np.zeros((x.shape[0], len(classes)), dtype=np.float32)
    for i, cls in enumerate(model.classes_):
        aligned[:, classes.index(str(cls))] = proba[:, i]
    return top2_from_scores(aligned, classes), aligned


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=ALL_CLASSES, average="macro", zero_division=0)


def eval_predictions(name, y_true, y_pred, out_dir, extra=None):
    row = {
        "name": name,
        "macro_f1": float(macro_f1(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if extra:
        row.update(extra)
    with open(out_dir / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"{name}: macro_f1={row['macro_f1']:.6f} acc={row['accuracy']:.6f}", flush=True)
    return row


def write_class_report(path, y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=ALL_CLASSES, output_dict=True, zero_division=0)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["action", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for action in ALL_CLASSES:
            row = report[action]
            writer.writerow(
                {
                    "action": action,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1-score"],
                    "support": int(row["support"]),
                }
            )


def write_confusions(path, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=ALL_CLASSES)
    rows = []
    for i, true in enumerate(ALL_CLASSES):
        for j, pred in enumerate(ALL_CLASSES):
            if i != j and cm[i, j] > 0:
                rows.append((int(cm[i, j]), true, pred))
    rows.sort(reverse=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["count", "true", "pred"])
        writer.writeheader()
        for count, true, pred in rows:
            writer.writerow({"count": count, "true": true, "pred": pred})


def train_flat_and_coarse(samples, y, y_group, train_idx, val_idx, max_features):
    texts = [compact_text(sample, "flags") for sample in samples]
    x_train_text = [texts[i] for i in train_idx]
    x_val_text = [texts[i] for i in val_idx]
    start = time.time()
    vectorizer = build_vectorizer(max_features)
    x_train = vectorizer.fit_transform(x_train_text)
    x_val = vectorizer.transform(x_val_text)
    print(f"shared vectorized shape={x_train.shape} sec={time.time() - start:.1f}", flush=True)

    start = time.time()
    coarse = LinearSVC(C=2.0, class_weight="balanced", random_state=SEED, dual="auto", max_iter=2500)
    coarse.fit(x_train, y_group[train_idx])
    coarse_scores = coarse.decision_function(x_val)
    if coarse_scores.ndim == 1:
        coarse_scores = np.stack([-coarse_scores, coarse_scores], axis=1)
    coarse_pred, _, coarse_margin = top2_from_scores(coarse_scores, list(coarse.classes_))
    print(f"coarse fit sec={time.time() - start:.1f} group_acc={(coarse_pred == y_group[val_idx]).mean():.6f}", flush=True)

    start = time.time()
    flat = LogisticRegression(max_iter=700, C=2.0, class_weight="balanced", random_state=SEED)
    flat.fit(x_train, y[train_idx])
    print(f"flat14 fit sec={time.time() - start:.1f}", flush=True)
    (_, _, flat_margin), flat_proba = top2_from_proba(flat, x_val, ALL_CLASSES)
    flat_pred = np.array([ALL_CLASSES[i] for i in flat_proba.argmax(axis=1)], dtype=object)
    flat_conf = flat_proba.max(axis=1)

    return {
        "texts": texts,
        "vectorizer": vectorizer,
        "x_train": x_train,
        "x_val": x_val,
        "coarse": coarse,
        "coarse_pred": coarse_pred,
        "coarse_margin": coarse_margin,
        "flat": flat,
        "flat_pred": flat_pred,
        "flat_conf": flat_conf,
        "flat_margin": flat_margin,
    }


def predict_shared_fine(shared, y, train_idx, val_idx):
    fine_models = {}
    x_train = shared["x_train"]
    x_val = shared["x_val"]
    group_pred = shared["coarse_pred"]
    n = len(val_idx)
    scores = np.full((n, len(ALL_CLASSES)), -1e9, dtype=np.float32)
    proba = np.zeros((n, len(ALL_CLASSES)), dtype=np.float32)
    for group, actions in GROUP_TO_ACTIONS.items():
        mask = np.isin(y[train_idx], actions)
        start = time.time()
        model = LogisticRegression(max_iter=700, C=2.0, class_weight="balanced", random_state=SEED)
        model.fit(x_train[mask], y[train_idx][mask])
        fine_models[group] = model
        idx = np.where(group_pred == group)[0]
        if len(idx):
            group_logp = aligned_log_proba(model, x_val[idx], actions)
            group_proba = np.exp(group_logp)
            for j, action in enumerate(actions):
                scores[idx, ALL_CLASSES.index(action)] = group_logp[:, j]
                proba[idx, ALL_CLASSES.index(action)] = group_proba[:, j]
        print(f"shared fine {group} rows={int(mask.sum())} sec={time.time() - start:.1f}", flush=True)
    pred = np.array([ALL_CLASSES[i] for i in scores.argmax(axis=1)], dtype=object)
    order = np.argsort(proba, axis=1)
    top1 = np.array([ALL_CLASSES[i] for i in order[:, -1]], dtype=object)
    top2 = np.array([ALL_CLASSES[i] for i in order[:, -2]], dtype=object)
    fine_margin = proba[np.arange(n), order[:, -1]] - proba[np.arange(n), order[:, -2]]
    return {
        "name": "shared_compact_fine_logreg",
        "pred": pred,
        "scores": scores,
        "proba": proba,
        "top1": top1,
        "top2": top2,
        "margin": fine_margin,
        "models": fine_models,
        "vectorizers": None,
        "variant": "shared",
    }


def train_group_specific(samples, y, train_idx, val_idx, group_pred, variant, max_features):
    models = {}
    vectorizers = {}
    n = len(val_idx)
    scores = np.full((n, len(ALL_CLASSES)), -1e9, dtype=np.float32)
    proba = np.zeros((n, len(ALL_CLASSES)), dtype=np.float32)
    for group, actions in GROUP_TO_ACTIONS.items():
        train_mask = np.array([y[i] in actions for i in train_idx], dtype=bool)
        group_train_idx = train_idx[train_mask]
        start = time.time()
        vec = build_vectorizer(max_features)
        x_train_text = [group_text(samples[i], group, variant) for i in group_train_idx]
        x_train = vec.fit_transform(x_train_text)
        model = LogisticRegression(max_iter=800, C=2.0, class_weight="balanced", random_state=SEED)
        model.fit(x_train, y[group_train_idx])
        models[group] = model
        vectorizers[group] = vec
        val_rows = np.where(group_pred == group)[0]
        if len(val_rows):
            x_val_text = [group_text(samples[val_idx[row]], group, variant) for row in val_rows]
            x_val = vec.transform(x_val_text)
            group_logp = aligned_log_proba(model, x_val, actions)
            group_proba = np.exp(group_logp)
            for j, action in enumerate(actions):
                scores[val_rows, ALL_CLASSES.index(action)] = group_logp[:, j]
                proba[val_rows, ALL_CLASSES.index(action)] = group_proba[:, j]
        print(
            f"group-specific {variant} {group} train={len(group_train_idx)} pred_rows={len(val_rows)} sec={time.time()-start:.1f}",
            flush=True,
        )
    pred = np.array([ALL_CLASSES[i] for i in scores.argmax(axis=1)], dtype=object)
    order = np.argsort(proba, axis=1)
    top1 = np.array([ALL_CLASSES[i] for i in order[:, -1]], dtype=object)
    top2 = np.array([ALL_CLASSES[i] for i in order[:, -2]], dtype=object)
    fine_margin = proba[np.arange(n), order[:, -1]] - proba[np.arange(n), order[:, -2]]
    return {
        "name": f"group_specific_{variant}",
        "pred": pred,
        "scores": scores,
        "proba": proba,
        "top1": top1,
        "top2": top2,
        "margin": fine_margin,
        "models": models,
        "vectorizers": vectorizers,
        "variant": variant,
    }


def transition_key(sample, key_name):
    if key_name == "last_action":
        return last_action(sample)
    if key_name == "last_action_result":
        return last_action(sample) + "|" + result_type(sample)
    if key_name == "last_action_rule":
        return last_action(sample) + "|" + simple_rule_hint(sample)
    if key_name == "last2_action":
        return last2_action(sample)
    raise ValueError(key_name)


def transition_prior_scores(samples, y, train_idx, val_idx, key_name, smooth):
    counts = defaultdict(Counter)
    global_counts = Counter(y[train_idx])
    for i in train_idx:
        counts[transition_key(samples[i], key_name)][y[i]] += 1
    global_total = sum(global_counts.values())
    global_log = np.array(
        [math.log((global_counts.get(action, 0) + smooth) / (global_total + smooth * len(ALL_CLASSES))) for action in ALL_CLASSES],
        dtype=np.float32,
    )
    rows = []
    for i in val_idx:
        key = transition_key(samples[i], key_name)
        if key not in counts:
            rows.append(global_log)
            continue
        row_counts = counts[key]
        total = sum(row_counts.values())
        rows.append(
            [
                math.log((row_counts.get(action, 0) + smooth) / (total + smooth * len(ALL_CLASSES)))
                for action in ALL_CLASSES
            ]
        )
    return np.array(rows, dtype=np.float32)


def apply_transition_prior(base, prior, group_pred, alpha):
    scores = base["scores"].copy()
    for i, group in enumerate(group_pred):
        for action in GROUP_TO_ACTIONS[str(group)]:
            j = ALL_CLASSES.index(action)
            scores[i, j] += alpha * prior[i, j]
    return np.array([ALL_CLASSES[i] for i in scores.argmax(axis=1)], dtype=object), scores


def apply_flat_fallback(base, flat_pred, flat_conf, group_pred, fine_thr, flat_thr):
    pred = base["pred"].copy()
    flat_group = np.array([ACTION_TO_GROUP[action] for action in flat_pred], dtype=object)
    mask = (base["margin"] < fine_thr) & (flat_group == group_pred) & (flat_conf >= flat_thr)
    pred[mask] = flat_pred[mask]
    return pred, int(mask.sum())


def train_pair_resolvers(samples, y, train_idx, max_features):
    resolvers = {}
    for raw_pair in PAIR_PRIORITY:
        pair = tuple(sorted(raw_pair))
        group = ACTION_TO_GROUP[pair[0]]
        if ACTION_TO_GROUP[pair[1]] != group:
            continue
        idx = [i for i in train_idx if y[i] in pair]
        if len(idx) < 50:
            continue
        start = time.time()
        vec = build_vectorizer(max_features)
        x_text = [pair_text(samples[i], pair) for i in idx]
        x = vec.fit_transform(x_text)
        model = LogisticRegression(max_iter=700, C=2.0, class_weight="balanced", random_state=SEED)
        model.fit(x, y[idx])
        resolvers[pair] = {"vectorizer": vec, "model": model, "group": group}
        print(f"pair resolver {pair[0]} vs {pair[1]} rows={len(idx)} sec={time.time()-start:.1f}", flush=True)
    return resolvers


def apply_pair_resolvers(samples, val_idx, base, scores, resolvers, margin_thr):
    pred = np.array([ALL_CLASSES[i] for i in scores.argmax(axis=1)], dtype=object)
    top1, top2, margin = top2_from_scores(np.exp(scores), ALL_CLASSES)
    changed = 0
    for row, (a, b, m) in enumerate(zip(top1, top2, margin)):
        pair = tuple(sorted((str(a), str(b))))
        if pair not in resolvers or m > margin_thr:
            continue
        resolver = resolvers[pair]
        text = pair_text(samples[val_idx[row]], pair)
        x = resolver["vectorizer"].transform([text])
        pair_pred = str(resolver["model"].predict(x)[0])
        if pair_pred != pred[row]:
            changed += 1
            pred[row] = pair_pred
    return pred, changed


def normalize_prompt(text):
    text = text.lower()
    text = re.sub(r"[\w./\\-]+\.[A-Za-z0-9]{1,8}", "<FILE>", text)
    text = re.sub(r"\d+", "<NUM>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:240]


def memory_key(sample, key_name):
    prompt = normalize_prompt(clean(sample.get("current_prompt"), 1000))
    if key_name == "prompt":
        return prompt
    if key_name == "prompt_last":
        return prompt + "|" + last_action(sample)
    if key_name == "rule_last":
        return simple_rule_hint(sample) + "|" + last_action(sample)
    raise ValueError(key_name)


def memory_prior_scores(samples, y, train_idx, val_idx, key_name, min_count, ratio_thr):
    counts = defaultdict(Counter)
    for i in train_idx:
        counts[memory_key(samples[i], key_name)][y[i]] += 1
    rows = np.zeros((len(val_idx), len(ALL_CLASSES)), dtype=np.float32)
    coverage = 0
    for row, i in enumerate(val_idx):
        c = counts.get(memory_key(samples[i], key_name))
        if not c:
            continue
        total = sum(c.values())
        action, n = c.most_common(1)[0]
        if total >= min_count and n / total >= ratio_thr:
            coverage += 1
            for cls in ALL_CLASSES:
                rows[row, ALL_CLASSES.index(cls)] = c.get(cls, 0) / total
    return rows, coverage


def apply_memory(base_scores, memory_scores, group_pred, beta):
    proba = np.exp(base_scores)
    combined = proba.copy()
    active = memory_scores.sum(axis=1) > 0
    combined[active] = (1 - beta) * proba[active] + beta * memory_scores[active]
    for i, group in enumerate(group_pred):
        allowed = set(GROUP_TO_ACTIONS[str(group)])
        for action in ALL_CLASSES:
            if action not in allowed:
                combined[i, ALL_CLASSES.index(action)] = -1e9
    return np.array([ALL_CLASSES[i] for i in combined.argmax(axis=1)], dtype=object), int(active.sum())


def write_predictions(path, samples, val_idx, y_true, y_pred, group_pred, shared, base):
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id",
            "y_true",
            "y_pred",
            "group_true",
            "group_pred",
            "flat_pred",
            "fine_pred",
            "coarse_margin",
            "fine_margin",
            "is_correct",
            "current_prompt",
            "last_action",
            "last_result_type",
            "open_files",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, i in enumerate(val_idx):
            writer.writerow(
                {
                    "id": samples[i].get("id", ""),
                    "y_true": y_true[row],
                    "y_pred": y_pred[row],
                    "group_true": ACTION_TO_GROUP[y_true[row]],
                    "group_pred": group_pred[row],
                    "flat_pred": shared["flat_pred"][row],
                    "fine_pred": base["pred"][row],
                    "coarse_margin": shared["coarse_margin"][row],
                    "fine_margin": base["margin"][row],
                    "is_correct": int(y_true[row] == y_pred[row]),
                    "current_prompt": clean(samples[i].get("current_prompt"), 600),
                    "last_action": last_action(samples[i]),
                    "last_result_type": result_type(samples[i]),
                    "open_files": " ".join(open_files(samples[i])[:8]),
                }
            )


def update_registry(out_dir, rows):
    registry = Path("reports") / "experiment_registry.csv"
    registry.parent.mkdir(exist_ok=True)
    exists = registry.exists()
    with open(registry, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "experiment", "macro_f1", "accuracy", "decision", "notes"],
        )
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "experiment": row["name"],
                    "macro_f1": row["macro_f1"],
                    "accuracy": row["accuracy"],
                    "decision": row.get("decision", ""),
                    "notes": row.get("notes", ""),
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./reports/exp_advanced_action_routing")
    parser.add_argument("--split", choices=["group", "stratified"], default="group")
    parser.add_argument("--max-features-shared", type=int, default=220_000)
    parser.add_argument("--max-features-group", type=int, default=180_000)
    parser.add_argument("--max-features-pair", type=int, default=80_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.jsonl").write_text("", encoding="utf-8")
    (out_dir / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")

    try:
        samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
        labels = load_labels(Path(args.data_dir) / "train_labels.csv")
        y = np.array([labels[s["id"]] for s in samples], dtype=object)
        y_group = np.array([ACTION_TO_GROUP[action] for action in y], dtype=object)
        indices = np.arange(len(samples))
        if args.split == "group":
            groups = np.array([session_id(s["id"]) for s in samples], dtype=object)
            train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(indices, y, groups=groups))
        else:
            train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=SEED)
        print(f"split={args.split} train={len(train_idx)} val={len(val_idx)}", flush=True)

        shared = train_flat_and_coarse(samples, y, y_group, train_idx, val_idx, args.max_features_shared)
        y_val = y[val_idx]
        group_pred = shared["coarse_pred"]
        rows = []

        shared_fine = predict_shared_fine(shared, y, train_idx, val_idx)
        rows.append(eval_predictions("phase0_shared_compact_current", y_val, shared_fine["pred"], out_dir))

        candidates = [shared_fine]
        for variant in ["same_text", "specialized", "specialized_x2"]:
            candidate = train_group_specific(samples, y, train_idx, val_idx, group_pred, variant, args.max_features_group)
            candidates.append(candidate)
            rows.append(eval_predictions(f"phase2_{candidate['name']}", y_val, candidate["pred"], out_dir))

        best_base = max(candidates, key=lambda c: macro_f1(y_val, c["pred"]))
        print(f"best base={best_base['name']} f1={macro_f1(y_val, best_base['pred']):.6f}", flush=True)
        best_pred = best_base["pred"]
        best_scores = best_base["scores"]
        best_name = best_base["name"]
        best_row = {"name": best_name, "macro_f1": macro_f1(y_val, best_pred), "accuracy": accuracy_score(y_val, best_pred)}

        for fine_thr in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
            for flat_thr in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
                pred, changed = apply_flat_fallback(best_base, shared["flat_pred"], shared["flat_conf"], group_pred, fine_thr, flat_thr)
                row = eval_predictions(
                    f"phase3_flat_fallback_ft{fine_thr}_pt{flat_thr}",
                    y_val,
                    pred,
                    out_dir,
                    {"changed": changed, "fine_thr": fine_thr, "flat_thr": flat_thr},
                )
                rows.append(row)
                if row["macro_f1"] > best_row["macro_f1"]:
                    best_row, best_pred, best_scores, best_name = row, pred, best_base["scores"], row["name"]

        prior_best_scores = best_scores
        for key_name in ["last_action", "last_action_result", "last_action_rule", "last2_action"]:
            for smooth in [0.5, 1.0, 2.0, 5.0]:
                prior = transition_prior_scores(samples, y, train_idx, val_idx, key_name, smooth)
                for alpha in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
                    pred, scores = apply_transition_prior(best_base, prior, group_pred, alpha)
                    row = eval_predictions(
                        f"phase5_prior_{key_name}_s{smooth}_a{alpha}",
                        y_val,
                        pred,
                        out_dir,
                        {"key": key_name, "smooth": smooth, "alpha": alpha},
                    )
                    rows.append(row)
                    if row["macro_f1"] > best_row["macro_f1"]:
                        best_row, best_pred, prior_best_scores, best_name = row, pred, scores, row["name"]

        resolvers = train_pair_resolvers(samples, y, train_idx, args.max_features_pair)
        for thr in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
            pred, changed = apply_pair_resolvers(samples, val_idx, best_base, prior_best_scores, resolvers, thr)
            row = eval_predictions(f"phase6_pair_resolver_t{thr}", y_val, pred, out_dir, {"changed": changed, "pair_thr": thr})
            rows.append(row)
            if row["macro_f1"] > best_row["macro_f1"]:
                best_row, best_pred, best_name = row, pred, row["name"]

        for key_name in ["prompt", "prompt_last", "rule_last"]:
            for min_count in [2, 3, 5]:
                for ratio_thr in [0.80, 0.85, 0.90, 0.95]:
                    memory, coverage = memory_prior_scores(samples, y, train_idx, val_idx, key_name, min_count, ratio_thr)
                    if coverage == 0:
                        continue
                    for beta in [0.03, 0.05, 0.08, 0.10, 0.15]:
                        pred, active = apply_memory(prior_best_scores, memory, group_pred, beta)
                        row = eval_predictions(
                            f"phase7_memory_{key_name}_m{min_count}_r{ratio_thr}_b{beta}",
                            y_val,
                            pred,
                            out_dir,
                            {"coverage": active, "key": key_name, "min_count": min_count, "ratio_thr": ratio_thr, "beta": beta},
                        )
                        rows.append(row)
                        if row["macro_f1"] > best_row["macro_f1"]:
                            best_row, best_pred, best_name = row, pred, row["name"]

        write_class_report(out_dir / "class_report_best.csv", y_val, best_pred)
        write_confusions(out_dir / "top_confusion_pairs_best.csv", y_val, best_pred)
        write_predictions(out_dir / "predictions_valid_best.csv", samples, val_idx, y_val, best_pred, group_pred, shared, best_base)

        rows_sorted = sorted(rows, key=lambda row: row["macro_f1"], reverse=True)
        summary = {
            "best": best_row,
            "top20": rows_sorted[:20],
            "split": args.split,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_current": rows[0],
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        update_registry(out_dir, rows_sorted[:20])
        with open("research.md", "a", encoding="utf-8") as f:
            f.write("\n## Advanced Action Routing Experiments\n")
            f.write(f"- Finished: {summary['finished_at']}\n")
            f.write(f"- Run dir: `{out_dir}`\n")
            f.write(f"- Split: `{args.split}`\n")
            f.write(f"- Baseline reproduced Macro-F1: `{rows[0]['macro_f1']:.6f}`\n")
            f.write(f"- Best Macro-F1: `{best_row['macro_f1']:.6f}` via `{best_row['name']}`\n")
            f.write("- Tested: group-specific vectorizers/serializers, fine-margin flat fallback, transition prior, pairwise resolvers, memory lookup.\n")
            f.write("\nTop results:\n")
            for row in rows_sorted[:10]:
                f.write(f"- `{row['macro_f1']:.6f}` `{row['name']}`\n")
        print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        (out_dir / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
