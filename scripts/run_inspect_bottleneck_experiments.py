import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script import (  # noqa: E402
    ADVANCED_ACTION_TO_GROUP,
    ADVANCED_GROUP_TO_ACTIONS,
    ALL_CLASSES,
    advanced_group_text,
    advanced_last2_action,
    advanced_open_files,
    compact_flags_text,
    safe_text,
)


INSPECT = {"read_file", "grep_search", "list_directory", "glob_pattern"}
INSPECT_PAIRS = [
    ("read_file", "grep_search"),
    ("grep_search", "glob_pattern"),
    ("list_directory", "glob_pattern"),
    ("read_file", "list_directory"),
]
BASE_PAIR_RESOLVERS = [
    ("read_file", "grep_search"),
    ("grep_search", "glob_pattern"),
    ("list_directory", "glob_pattern"),
]
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]
FILE_EXTS = {
    "py", "tsx", "ts", "js", "jsx", "json", "yaml", "yml", "go", "rs", "toml",
    "java", "kt", "sql", "tf", "sh", "md", "txt", "vue", "gradle", "lock",
    "html", "css", "scss", "dockerfile", "ini", "cfg",
}
FILE_RE = re.compile(r"(?i)(?:[\w.-]+/)+[\w.-]+\.[a-z0-9]{1,12}\b|[\w.-]+\.[a-z0-9]{1,12}\b")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


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


def append_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def session_of(sample_id):
    return sample_id.rsplit("-step_", 1)[0] if "-step_" in sample_id else sample_id


def load_fold_split(samples, fold_file, fold=0):
    fold_file = Path(fold_file)
    if fold_file.exists():
        fold_of = {}
        with open(fold_file, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                fold_of[row["id"]] = int(row["fold"])
        train_idx = [i for i, sample in enumerate(samples) if fold_of.get(sample["id"]) != fold]
        val_idx = [i for i, sample in enumerate(samples) if fold_of.get(sample["id"]) == fold]
        return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64), "pipeline_v4_fold0"

    groups = np.asarray([session_of(sample["id"]) for sample in samples], dtype=object)
    y_dummy = np.zeros(len(samples), dtype=np.int64)
    train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(np.arange(len(samples)), y_dummy, groups))
    return train_idx, val_idx, "generated_group_split_seed42"


def build_vectorizer(max_features=90_000, min_df=2, char_heavy=False):
    word_features = max_features // 3 if char_heavy else max_features // 2
    char_features = max_features - word_features
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_features=word_features,
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
                    max_features=char_features,
                    sublinear_tf=True,
                    lowercase=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def train_logreg(x, y, c=2.0, max_iter=350):
    return LogisticRegression(
        max_iter=max_iter,
        C=c,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    ).fit(x, y)


def transition_counts(samples, y):
    counts = defaultdict(Counter)
    for sample, label in zip(samples, y):
        counts[advanced_last2_action(sample)][label] += 1
    return {key: dict(value) for key, value in counts.items()}


def transition_prior_matrix(samples, counts, global_counts, smooth=1.0):
    global_total = sum(global_counts.values())
    global_row = np.array(
        [math.log((global_counts.get(cls, 0) + smooth) / (global_total + smooth * len(ALL_CLASSES))) for cls in ALL_CLASSES],
        dtype=np.float32,
    )
    rows = []
    for sample in samples:
        row_counts = counts.get(advanced_last2_action(sample))
        if not row_counts:
            rows.append(global_row)
            continue
        total = sum(row_counts.values())
        rows.append(
            [math.log((row_counts.get(cls, 0) + smooth) / (total + smooth * len(ALL_CLASSES))) for cls in ALL_CLASSES]
        )
    return np.asarray(rows, dtype=np.float32)


def base_pair_text(sample, pair):
    return compact_flags_text(sample) + " PAIR_EXTRA " + advanced_group_text(sample, ADVANCED_ACTION_TO_GROUP[pair[0]]) + f" PAIR={pair[0]}_VS_{pair[1]}"


def train_base_pair_resolvers(samples, y, pair_features):
    resolvers = {}
    y = np.asarray(y, dtype=object)
    for raw_pair in BASE_PAIR_RESOLVERS:
        pair = tuple(sorted(raw_pair))
        idx = np.where(np.isin(y, pair))[0]
        if len(idx) < 100:
            continue
        texts = [base_pair_text(samples[i], pair) for i in idx]
        vectorizer = build_vectorizer(pair_features, min_df=2)
        x = vectorizer.fit_transform(texts)
        model = train_logreg(x, y[idx], c=2.0, max_iter=300)
        resolvers[pair] = {"vectorizer": vectorizer, "model": model}
    return resolvers


def train_fold_router(samples, y, train_idx, out_root, router_features, pair_features, refresh=False):
    cache_path = out_root / "_cache" / f"fold_router_rf{router_features}_pf{pair_features}.joblib"
    if cache_path.exists() and not refresh:
        print(f"load router cache: {cache_path}", flush=True)
        return joblib.load(cache_path)

    train_samples = [samples[i] for i in train_idx]
    y_train = np.asarray([y[i] for i in train_idx], dtype=object)
    y_group = np.asarray([ADVANCED_ACTION_TO_GROUP[label] for label in y_train], dtype=object)

    started = time.time()
    coarse_vectorizer = build_vectorizer(router_features, min_df=2)
    x_coarse = coarse_vectorizer.fit_transform([compact_flags_text(sample) for sample in train_samples])
    coarse_model = LinearSVC(C=2.0, class_weight="balanced", random_state=42, dual="auto", max_iter=2500)
    coarse_model.fit(x_coarse, y_group)
    print(f"coarse trained shape={x_coarse.shape} elapsed={time.time() - started:.1f}s", flush=True)

    group_vectorizers = {}
    group_models = {}
    for group, actions in ADVANCED_GROUP_TO_ACTIONS.items():
        idx = np.where(np.isin(y_train, actions))[0]
        texts = [advanced_group_text(train_samples[i], group) for i in idx]
        vectorizer = build_vectorizer(router_features, min_df=2)
        x = vectorizer.fit_transform(texts)
        model = train_logreg(x, y_train[idx], c=2.0, max_iter=450)
        group_vectorizers[group] = vectorizer
        group_models[group] = model
        print(f"group {group} trained rows={len(idx)} shape={x.shape}", flush=True)

    pair_resolvers = train_base_pair_resolvers(train_samples, y_train, pair_features)
    artifact = {
        "coarse_vectorizer": coarse_vectorizer,
        "coarse_model": coarse_model,
        "group_vectorizers": group_vectorizers,
        "group_models": group_models,
        "transition_last2": transition_counts(train_samples, y_train),
        "global_counts": dict(Counter(y_train)),
        "pair_resolvers": pair_resolvers,
        "config": {"prior_alpha": 0.3, "prior_smooth": 1.0, "pair_threshold": 0.08},
        "router_features": router_features,
        "pair_features": pair_features,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, cache_path, compress=3)
    print(f"saved router cache: {cache_path}", flush=True)
    return artifact


def group_log_proba(model, x, actions):
    proba = model.predict_proba(x)
    out = np.full((x.shape[0], len(actions)), -1e9, dtype=np.float32)
    for i, cls in enumerate(model.classes_):
        if str(cls) in actions:
            out[:, actions.index(str(cls))] = np.log(proba[:, i] + 1e-9)
    return out


def predict_router(samples, artifact, apply_pair_resolvers=True):
    coarse_x = artifact["coarse_vectorizer"].transform([compact_flags_text(sample) for sample in samples])
    group_pred = np.asarray(artifact["coarse_model"].predict(coarse_x), dtype=object)

    scores = np.full((len(samples), len(ALL_CLASSES)), -1e9, dtype=np.float32)
    for group, actions in ADVANCED_GROUP_TO_ACTIONS.items():
        idx = np.where(group_pred == group)[0]
        if not len(idx):
            continue
        texts = [advanced_group_text(samples[i], group) for i in idx]
        x = artifact["group_vectorizers"][group].transform(texts)
        group_scores = group_log_proba(artifact["group_models"][group], x, actions)
        for j, action in enumerate(actions):
            scores[idx, ALL_CLASSES.index(action)] = group_scores[:, j]

    prior = transition_prior_matrix(
        samples,
        artifact["transition_last2"],
        artifact["global_counts"],
        artifact["config"].get("prior_smooth", 1.0),
    )
    alpha = float(artifact["config"].get("prior_alpha", 0.3))
    for i, group in enumerate(group_pred):
        for action in ADVANCED_GROUP_TO_ACTIONS[str(group)]:
            j = ALL_CLASSES.index(action)
            scores[i, j] += alpha * prior[i, j]

    prob_like = np.exp(np.clip(scores, -50, 50))
    order = np.argsort(prob_like, axis=1)
    top1 = np.asarray([ALL_CLASSES[i] for i in order[:, -1]], dtype=object)
    top2 = np.asarray([ALL_CLASSES[i] for i in order[:, -2]], dtype=object)
    margin = prob_like[np.arange(len(samples)), order[:, -1]] - prob_like[np.arange(len(samples)), order[:, -2]]
    pred = top1.copy()

    if apply_pair_resolvers:
        threshold = float(artifact["config"].get("pair_threshold", 0.08))
        for i, (a, b, m) in enumerate(zip(top1, top2, margin)):
            pair = tuple(sorted((str(a), str(b))))
            resolver = artifact["pair_resolvers"].get(pair)
            if resolver is None or m > threshold:
                continue
            text = base_pair_text(samples[i], pair)
            x = resolver["vectorizer"].transform([text])
            pred[i] = str(resolver["model"].predict(x)[0])

    rows = []
    for i, sample in enumerate(samples):
        rows.append(
            {
                "id": sample["id"],
                "pred": str(pred[i]),
                "top1": str(top1[i]),
                "top2": str(top2[i]),
                "margin": float(margin[i]),
                "group": str(group_pred[i]),
            }
        )
    return pred.astype(object), scores, rows


def text_low(sample):
    return safe_text(sample.get("current_prompt"), 1500).lower()


def mentioned_files(sample):
    text = text_low(sample).replace("\\", "/")
    return [m.group(0).lower() for m in FILE_RE.finditer(text)]


def ext_of(path):
    low = str(path).lower().replace("\\", "/")
    if low.endswith("dockerfile"):
        return "dockerfile"
    if "." not in low.rsplit("/", 1)[-1]:
        return ""
    return low.rsplit(".", 1)[-1]


def has_any(text, needles):
    return any(needle in text for needle in needles)


def target_granularity(sample):
    prompt = text_low(sample)
    files = mentioned_files(sample)
    open_basenames = {Path(str(path).replace("\\", "/")).name.lower() for path in advanced_open_files(sample)}
    if any("/" in f and ext_of(f) in FILE_EXTS for f in files):
        return "TARGET_EXACT_PATH"
    if any(ext_of(f) in FILE_EXTS for f in files) or any(base and base in prompt for base in open_basenames):
        return "TARGET_BASENAME"
    if re.search(r"(?i)(\*\.[a-z0-9]+|\*\*/|all .*files|files? .*\.?[a-z0-9]{1,8}|[a-z0-9]{1,8} files)", prompt):
        return "TARGET_EXTENSION_ONLY"
    if has_any(prompt, ["directory", "folder", "project root", " root", "tree", "dir ", "목록", "폴더", "디렉터리", "루트"]):
        return "TARGET_DIRECTORY"
    if re.search(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", sample.get("current_prompt", "")) or has_any(
        prompt, ["symbol", "identifier", "definition", "usage", "import", "called", "정의", "참조", "쓰이는", "어디서"]
    ):
        return "TARGET_SYMBOL"
    return "TARGET_UNKNOWN"


def operation_intent(sample):
    prompt = text_low(sample)
    if has_any(prompt, ["glob", "**/", "*.", "all files", "matching files", "패턴", "파일 전부"]):
        return "OP_ENUM_FILES"
    if has_any(prompt, ["list", "tree", "directory", "folder", "project root", "what's in", "목록", "뭐 있", "폴더", "루트"]):
        return "OP_LIST_DIR"
    if has_any(prompt, ["grep", "search", "find", "where", "occurrence", "usage", "definition", "reference", "import", "찾아", "검색", "어디", "정의", "참조"]):
        return "OP_SEARCH"
    if has_any(prompt, ["open", "read", "show me", "peek", "look at", "pull up", "열어", "읽어", "보여", "봐줘"]):
        return "OP_OPEN"
    return "OP_UNKNOWN"


def last_action_turn(sample):
    for turn in reversed(sample.get("history", []) or []):
        if turn.get("role") == "assistant_action":
            return turn
    return None


def parse_count(text):
    nums = []
    for match in re.findall(r"\b(\d{1,5})\b", text or ""):
        try:
            nums.append(int(match))
        except ValueError:
            pass
    return max(nums) if nums else None


def previous_observation(sample):
    last = last_action_turn(sample)
    if not last:
        return "OBS_NONE"
    action = str(last.get("name", ""))
    result = safe_text(last.get("result_summary"), 1000).lower()
    if any(x in result for x in ["error", "traceback", "exception", "permission denied", "filenotfound"]):
        return "OBS_ERROR"
    count = parse_count(result)
    if action == "grep_search":
        if "no match" in result or "0 match" in result or count == 0:
            return "OBS_GREP_ZERO"
        if count is not None and count <= 3:
            return "OBS_GREP_FEW"
        return "OBS_GREP_MANY"
    if action == "glob_pattern":
        if "0 files" in result or "0 matched" in result or count == 0:
            return "OBS_GLOB_ZERO"
        if count is not None and count <= 3:
            return "OBS_GLOB_FEW"
        return "OBS_GLOB_MANY"
    if action == "list_directory":
        if "empty" in result or "0 items" in result:
            return "OBS_LIST_EMPTY"
        return "OBS_LIST_ENTRIES"
    if action == "read_file":
        return "OBS_READ_OK"
    if action == "web_search":
        return "OBS_WEB"
    if action == "plan_task":
        return "OBS_PLAN"
    if action == "ask_user":
        return "OBS_ASK"
    return "OBS_OTHER"


def location_certainty(sample):
    target = target_granularity(sample)
    prompt = text_low(sample)
    last = last_action_turn(sample)
    if target in {"TARGET_EXACT_PATH", "TARGET_BASENAME"}:
        return "LOC_KNOWN_HIGH"
    if last and str(last.get("name")) == "read_file" and has_any(prompt, ["that file", "there", "same file", "그 파일", "거기", "그거"]):
        return "LOC_KNOWN_HIGH"
    if advanced_open_files(sample):
        return "LOC_KNOWN_MED"
    if last and str(last.get("name")) in {"grep_search", "glob_pattern", "list_directory"}:
        return "LOC_KNOWN_MED"
    return "LOC_UNKNOWN"


def inspect_state(sample):
    last = last_action_turn(sample)
    return {
        "target_granularity": target_granularity(sample),
        "operation_intent": operation_intent(sample),
        "location_certainty": location_certainty(sample),
        "previous_observation": previous_observation(sample),
        "last_action": str(last.get("name")) if last else "NONE",
    }


def state_key(state):
    return "|".join(
        [
            state["target_granularity"],
            state["operation_intent"],
            state["location_certainty"],
            state["previous_observation"],
            state["last_action"],
        ]
    )


def fast_router_text(sample):
    state = inspect_state(sample)
    last = last_action_turn(sample)
    last_args = safe_text(last.get("args"), 260) if last else "none"
    last_result = safe_text(last.get("result_summary"), 400) if last else "none"
    actions = [h.get("name") for h in sample.get("history", []) or [] if h.get("role") == "assistant_action" and h.get("name")]
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    lang_mix = ws.get("language_mix", {}) or {}
    top_lang = "none"
    if lang_mix:
        top_lang = max(lang_mix.items(), key=lambda item: float(item[1]))[0]
    return "\n".join(
        [
            "[NOW] " + safe_text(sample.get("current_prompt"), 1000),
            "[STATE] " + " ".join(f"{k}={v}" for k, v in state.items()),
            "[SEQ] " + " > ".join(actions[-6:]) if actions else "[SEQ] none",
            f"[LAST] args={last_args} result={last_result}",
            "[OPEN] " + " ".join(advanced_open_files(sample)[:8]),
            f"[META] turn={meta.get('turn_index')} tier={meta.get('user_tier')} pref={meta.get('language_pref')} "
            f"ci={ws.get('last_ci_status')} dirty={ws.get('git_dirty')} top_lang={top_lang}",
        ]
    )


def train_fast_router(samples, y, train_idx, out_root, router_features, refresh=False):
    cache_path = out_root / "_cache" / f"fast_flat_router_rf{router_features}.joblib"
    if cache_path.exists() and not refresh:
        print(f"load fast router cache: {cache_path}", flush=True)
        return joblib.load(cache_path)
    train_samples = [samples[i] for i in train_idx]
    y_train = np.asarray([y[i] for i in train_idx], dtype=object)
    vectorizer = build_vectorizer(router_features, min_df=2)
    x = vectorizer.fit_transform([fast_router_text(sample) for sample in train_samples])
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=25,
        tol=1e-3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y_train)
    artifact = {"kind": "fast_flat_router", "vectorizer": vectorizer, "model": model, "router_features": router_features}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, cache_path, compress=3)
    print(f"saved fast router cache: {cache_path} shape={x.shape}", flush=True)
    return artifact


def predict_fast_router(samples, artifact):
    x = artifact["vectorizer"].transform([fast_router_text(sample) for sample in samples])
    proba = artifact["model"].predict_proba(x)
    classes = [str(cls) for cls in artifact["model"].classes_]
    aligned = np.zeros((x.shape[0], len(ALL_CLASSES)), dtype=np.float32)
    for i, cls in enumerate(classes):
        aligned[:, ALL_CLASSES.index(cls)] = proba[:, i]
    order = np.argsort(aligned, axis=1)
    pred = np.asarray([ALL_CLASSES[i] for i in order[:, -1]], dtype=object)
    top2 = np.asarray([ALL_CLASSES[i] for i in order[:, -2]], dtype=object)
    margin = aligned[np.arange(len(samples)), order[:, -1]] - aligned[np.arange(len(samples)), order[:, -2]]
    rows = []
    for i, sample in enumerate(samples):
        rows.append(
            {
                "id": sample["id"],
                "pred": str(pred[i]),
                "top1": str(pred[i]),
                "top2": str(top2[i]),
                "margin": float(margin[i]),
                "group": ADVANCED_ACTION_TO_GROUP.get(str(pred[i]), "unknown"),
            }
        )
    log_scores = np.log(np.clip(aligned, 1e-9, 1.0))
    return pred, log_scores, rows


def run_n2b(samples, y, val_idx, adv_pred, adv_rows, out_dir):
    val_samples = [samples[i] for i in val_idx]
    y_val = np.asarray([y[i] for i in val_idx], dtype=object)
    states = [inspect_state(sample) for sample in val_samples]
    keys = [state_key(state) for state in states]
    inspect_error = np.asarray([(yt in INSPECT or yp in INSPECT) and yt != yp for yt, yp in zip(y_val, adv_pred)], dtype=bool)
    total_errors = int(inspect_error.sum())

    by_state = defaultdict(Counter)
    state_support = Counter(keys)
    state_error = Counter()
    state_confusions = defaultdict(Counter)
    for key, yt, yp, is_error in zip(keys, y_val, adv_pred, inspect_error):
        if yt in INSPECT:
            by_state[key][yt] += 1
        if is_error:
            state_error[key] += 1
            state_confusions[key][f"{yp}->{yt}"] += 1

    rows = []
    for key, counts in by_state.items():
        support = sum(counts.values())
        if support == 0:
            continue
        top_action, top_count = counts.most_common(1)[0]
        state = dict(zip(["target_granularity", "operation_intent", "location_certainty", "previous_observation", "last_action"], key.split("|")))
        rows.append(
            {
                "state_key": key,
                **state,
                "support": support,
                "read_file_count": counts["read_file"],
                "grep_search_count": counts["grep_search"],
                "list_directory_count": counts["list_directory"],
                "glob_pattern_count": counts["glob_pattern"],
                "top_action": top_action,
                "purity": top_count / support,
                "advanced_error_count": state_error[key],
                "advanced_error_coverage": state_error[key] / max(total_errors, 1),
            }
        )
    rows.sort(key=lambda row: (row["purity"], row["support"]), reverse=True)
    high = [row for row in rows if int(row["support"]) >= 30 and float(row["purity"]) >= 0.60]

    dist_rows = []
    for subset_name, mask in [
        ("true_inspect", np.asarray([label in INSPECT for label in y_val])),
        ("deploy_inspect", np.asarray([pred in INSPECT for pred in adv_pred])),
        ("inspect_error", inspect_error),
    ]:
        counter = Counter(keys[i] for i, ok in enumerate(mask) if ok)
        for key, count in counter.most_common():
            state = dict(zip(["target_granularity", "operation_intent", "location_certainty", "previous_observation", "last_action"], key.split("|")))
            dist_rows.append({"subset": subset_name, "state_key": key, **state, "count": count})

    coverage_rows = []
    for key, count in state_error.most_common():
        coverage_rows.append(
            {
                "state_key": key,
                "error_count": count,
                "error_coverage": count / max(total_errors, 1),
                "top_confusions": "; ".join(f"{k}:{v}" for k, v in state_confusions[key].most_common(5)),
            }
        )

    write_csv(out_dir / "state_to_action_purity.csv", rows, list(rows[0].keys()) if rows else ["state_key"])
    write_csv(out_dir / "high_purity_states.csv", high, list(rows[0].keys()) if rows else ["state_key"])
    write_csv(out_dir / "inspect_state_distribution.csv", dist_rows, list(dist_rows[0].keys()) if dist_rows else ["subset"])
    write_csv(out_dir / "state_coverage_by_error_type.csv", coverage_rows, list(coverage_rows[0].keys()) if coverage_rows else ["state_key"])

    example_lines = ["# N2b High-Purity State Examples", ""]
    for row in high[:20]:
        example_lines.append(f"## {row['state_key']}")
        example_lines.append(f"- support: {row['support']}, top_action: `{row['top_action']}`, purity: `{row['purity']:.3f}`")
        shown = 0
        for sample, label, pred, key in zip(val_samples, y_val, adv_pred, keys):
            if key == row["state_key"] and shown < 3:
                example_lines.append(f"- `{sample['id']}` true=`{label}` adv=`{pred}` prompt={safe_text(sample.get('current_prompt'), 220)}")
                shown += 1
        example_lines.append("")
    append_lines(out_dir / "examples_high_purity_states.md", example_lines)

    high_error_coverage = sum(int(row["advanced_error_count"]) for row in high) / max(total_errors, 1)
    soft_count = sum(1 for row in rows if int(row["support"]) >= 30 and float(row["purity"]) >= 0.55)
    verdict = "PASS" if (len(high) >= 10 or high_error_coverage >= 0.30) else "SOFT_PASS" if soft_count >= 20 else "FAIL"
    summary = [
        "# N2b Inspect State-Machine Forensics",
        "",
        f"- validation rows: `{len(val_idx)}`",
        f"- inspect-related advanced errors: `{total_errors}`",
        f"- high-purity states support>=30 purity>=0.60: `{len(high)}`",
        f"- soft states support>=30 purity>=0.55: `{soft_count}`",
        f"- high-purity error coverage: `{high_error_coverage:.4f}`",
        f"- verdict: `{verdict}`",
        "",
        "Top high-purity states:",
    ]
    for row in high[:10]:
        summary.append(f"- `{row['state_key']}` support={row['support']} purity={row['purity']:.3f} top={row['top_action']}")
    append_lines(out_dir / "summary.md", summary)
    return {"verdict": verdict, "high_states": len(high), "high_error_coverage": high_error_coverage, "total_errors": total_errors}


def inspect_serializer(sample, info):
    state = inspect_state(sample)
    last = last_action_turn(sample)
    last_args = safe_text(last.get("args"), 300) if last else "none"
    last_result = safe_text(last.get("result_summary"), 500) if last else "none"
    files = mentioned_files(sample)
    actions = [h.get("name") for h in sample.get("history", []) or [] if h.get("role") == "assistant_action" and h.get("name")]
    ws = sample.get("session_meta", {}).get("workspace", {}) if sample.get("session_meta") else {}
    return "\n".join(
        [
            "[NOW] " + safe_text(sample.get("current_prompt"), 1000),
            "[INSPECT_STATE] " + " ".join(f"{k}={v}" for k, v in state.items()),
            f"[ADV] pred={info['pred']} top2={info['top1']}|{info['top2']} margin={margin_bucket(info['margin'])}",
            f"[LAST] action={state['last_action']} args={last_args} bucket={state['previous_observation']} result={last_result}",
            "[FILES] open=" + " ".join(advanced_open_files(sample)[:8]) + " mentioned=" + " ".join(files[:10]),
            "[SEQ] actions=" + " > ".join(actions[-6:]) if actions else "[SEQ] actions=none",
            f"[META] turn={sample.get('session_meta', {}).get('turn_index')} ci={ws.get('last_ci_status')} dirty={ws.get('git_dirty')} lang={sample.get('session_meta', {}).get('language_pref')}",
        ]
    )


def margin_bucket(value):
    if value <= 0.01:
        return "m00"
    if value <= 0.05:
        return "m05"
    if value <= 0.10:
        return "m10"
    if value <= 0.25:
        return "m25"
    return "m_hi"


def pair_deploy_mask(pair, infos):
    pair_set = set(pair)
    mask = []
    for info in infos:
        top_pair = {info["top1"], info["top2"]}
        mask.append(info["pred"] in pair_set or top_pair == pair_set)
    return np.asarray(mask, dtype=bool)


def predict_pair_model(model_kind, model, vectorizer, texts):
    x = vectorizer.transform(texts)
    if model_kind.startswith("logreg"):
        proba = model.predict_proba(x)
        idx = proba.argmax(axis=1)
        pred = np.asarray([str(model.classes_[i]) for i in idx], dtype=object)
        conf = proba.max(axis=1)
        return pred, conf
    score = model.decision_function(x)
    if score.ndim > 1:
        idx = score.argmax(axis=1)
        pred = np.asarray([str(model.classes_[i]) for i in idx], dtype=object)
        sorted_score = np.sort(score, axis=1)
        raw_conf = sorted_score[:, -1] - sorted_score[:, -2]
    else:
        pred = np.where(score >= 0, str(model.classes_[1]), str(model.classes_[0])).astype(object)
        raw_conf = np.abs(score)
    conf = 1.0 / (1.0 + np.exp(-raw_conf))
    return pred, conf


def train_pair_candidate(kind, texts, labels, pair_features):
    char_heavy = kind == "logreg_char_heavy"
    vectorizer = build_vectorizer(pair_features, min_df=2, char_heavy=char_heavy)
    x = vectorizer.fit_transform(texts)
    if kind == "linear_svc":
        model = LinearSVC(C=1.0, class_weight="balanced", random_state=42, dual="auto", max_iter=2500).fit(x, labels)
    else:
        model = train_logreg(x, labels, c=2.0, max_iter=350)
    return vectorizer, model


def score_macro(y_true, pred):
    return float(f1_score(y_true, pred, labels=ALL_CLASSES, average="macro", zero_division=0))


def run_n2c(samples, y, train_idx, val_idx, train_infos, val_infos, base_pred, out_dir, pair_features):
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    y_train = np.asarray([y[i] for i in train_idx], dtype=object)
    y_val = np.asarray([y[i] for i in val_idx], dtype=object)
    base_pred = np.asarray(base_pred, dtype=object)
    base_macro = score_macro(y_val, base_pred)

    groups = np.asarray([session_of(sample["id"]) for sample in val_samples], dtype=object)
    half_a, half_b = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(np.arange(len(val_samples)), y_val, groups))
    half_sets = {"halfA": set(half_a), "halfB": set(half_b)}

    pair_true_rows = []
    pair_deploy_rows = []
    stability_rows = []
    best_by_pair = {}
    trained = {}
    model_kinds = ["logreg_word_char", "logreg_char_heavy", "linear_svc"]

    for pair in INSPECT_PAIRS:
        pair = tuple(pair)
        train_mask = np.asarray([label in pair for label in y_train], dtype=bool)
        val_true_mask = np.asarray([label in pair for label in y_val], dtype=bool)
        deploy_mask = pair_deploy_mask(pair, val_infos)
        if train_mask.sum() < 100 or val_true_mask.sum() < 20:
            continue
        train_texts = [inspect_serializer(sample, info) for sample, info, ok in zip(train_samples, train_infos, train_mask) if ok]
        train_labels = y_train[train_mask]
        val_texts_all = [inspect_serializer(sample, info) for sample, info in zip(val_samples, val_infos)]

        for kind in model_kinds:
            started = time.time()
            vectorizer, model = train_pair_candidate(kind, train_texts, train_labels, pair_features)
            trained[(pair, kind)] = {"vectorizer": vectorizer, "model": model}
            val_pred_all, val_conf_all = predict_pair_model(kind, model, vectorizer, val_texts_all)
            true_pair_f1 = float(f1_score(y_val[val_true_mask], val_pred_all[val_true_mask], labels=list(pair), average="macro", zero_division=0))
            pair_true_rows.append(
                {
                    "pair": "__".join(pair),
                    "model": kind,
                    "true_pair_count": int(val_true_mask.sum()),
                    "true_pair_f1": true_pair_f1,
                    "fit_predict_sec": round(time.time() - started, 3),
                }
            )

            for threshold in THRESHOLDS:
                override_mask = deploy_mask & (val_conf_all >= threshold)
                new_pred = base_pred.copy()
                new_pred[override_mask] = val_pred_all[override_mask]
                before_correct = base_pred[override_mask] == y_val[override_mask]
                after_correct = new_pred[override_mask] == y_val[override_mask]
                override_count = int(override_mask.sum())
                net_delta = int(after_correct.sum() - before_correct.sum())
                row = {
                    "pair": "__".join(pair),
                    "model": kind,
                    "threshold": threshold,
                    "true_pair_f1": true_pair_f1,
                    "deployment_subset_count": int(deploy_mask.sum()),
                    "override_count": override_count,
                    "advanced_correct_on_override": int(before_correct.sum()),
                    "resolver_correct_on_override": int(after_correct.sum()),
                    "net_correct_delta": net_delta,
                    "override_precision": float(after_correct.mean()) if override_count else 0.0,
                    "advanced_precision_same_rows": float(before_correct.mean()) if override_count else 0.0,
                    "overall_macro_f1_delta": score_macro(y_val, new_pred) - base_macro,
                }
                half_ok = True
                for half_name, half_idx in half_sets.items():
                    half_mask = np.asarray([i in half_idx for i in range(len(y_val))]) & override_mask
                    half_before = base_pred[half_mask] == y_val[half_mask]
                    half_after = new_pred[half_mask] == y_val[half_mask]
                    half_net = int(half_after.sum() - half_before.sum())
                    row[f"{half_name}_net_correct_delta"] = half_net
                    row[f"{half_name}_override_count"] = int(half_mask.sum())
                    half_ok = half_ok and half_net > 0
                    stability_rows.append(
                        {
                            "pair": "__".join(pair),
                            "model": kind,
                            "threshold": threshold,
                            "half": half_name,
                            "override_count": int(half_mask.sum()),
                            "net_correct_delta": half_net,
                        }
                    )
                row["adopt_candidate"] = (
                    net_delta >= 20
                    and row["override_precision"] >= row["advanced_precision_same_rows"] + 0.03
                    and row["overall_macro_f1_delta"] >= 0.001
                    and half_ok
                )
                pair_deploy_rows.append(row)
                if row["adopt_candidate"]:
                    current = best_by_pair.get(pair)
                    if current is None or row["overall_macro_f1_delta"] > current["overall_macro_f1_delta"]:
                        best_by_pair[pair] = row

    write_csv(out_dir / "pair_true_results.csv", pair_true_rows, list(pair_true_rows[0].keys()) if pair_true_rows else ["pair"])
    write_csv(out_dir / "pair_deployment_results.csv", pair_deploy_rows, list(pair_deploy_rows[0].keys()) if pair_deploy_rows else ["pair"])
    write_csv(out_dir / "half_split_stability.csv", stability_rows, list(stability_rows[0].keys()) if stability_rows else ["pair"])

    combined_pred = base_pred.copy()
    adopted_payload = []
    changed_rows = []
    for pair, row in sorted(best_by_pair.items(), key=lambda item: item[1]["overall_macro_f1_delta"], reverse=True):
        kind = row["model"]
        threshold = float(row["threshold"])
        bundle = trained[(pair, kind)]
        texts = [inspect_serializer(sample, info) for sample, info in zip(val_samples, val_infos)]
        pair_pred, pair_conf = predict_pair_model(kind, bundle["model"], bundle["vectorizer"], texts)
        mask = pair_deploy_mask(pair, val_infos) & (pair_conf >= threshold)
        old_pred = combined_pred.copy()
        combined_pred[mask] = pair_pred[mask]
        adopted_payload.append({"pair": list(pair), "model": kind, "threshold": threshold, "override_count": int(mask.sum())})
        for i, ok in enumerate(mask):
            if ok and old_pred[i] != combined_pred[i]:
                changed_rows.append(
                    {
                        "id": val_samples[i]["id"],
                        "pair": "__".join(pair),
                        "from_advanced": old_pred[i],
                        "to_resolver": combined_pred[i],
                        "true": y_val[i],
                        "correct_before": int(old_pred[i] == y_val[i]),
                        "correct_after": int(combined_pred[i] == y_val[i]),
                    }
                )

    combined_macro = score_macro(y_val, combined_pred)
    write_json(
        out_dir / "adopted_pair_resolvers.json",
        {
            "base_macro_f1": base_macro,
            "combined_macro_f1": combined_macro,
            "combined_delta": combined_macro - base_macro,
            "adopted": adopted_payload,
        },
    )
    write_csv(
        out_dir / "changed_flow_pair_resolvers.csv",
        changed_rows,
        ["id", "pair", "from_advanced", "to_resolver", "true", "correct_before", "correct_after"],
    )

    verdict = "PASS" if combined_macro - base_macro >= 0.003 else "FAIL"
    summary = [
        "# N2c Inspect Pairwise Resolvers",
        "",
        f"- base Macro-F1: `{base_macro:.6f}`",
        f"- combined Macro-F1: `{combined_macro:.6f}`",
        f"- combined delta: `{combined_macro - base_macro:.6f}`",
        f"- adopted pair resolvers: `{len(adopted_payload)}`",
        f"- verdict: `{verdict}`",
        "",
        "Best adopt candidates:",
    ]
    for row in sorted([r for r in pair_deploy_rows if r.get("adopt_candidate")], key=lambda r: r["overall_macro_f1_delta"], reverse=True)[:10]:
        summary.append(
            f"- `{row['pair']}` {row['model']} thr={row['threshold']} delta={row['overall_macro_f1_delta']:.6f} "
            f"net={row['net_correct_delta']} overrides={row['override_count']}"
        )
    append_lines(out_dir / "summary.md", summary)
    return {"verdict": verdict, "base_macro": base_macro, "combined_macro": combined_macro, "adopted": adopted_payload}


def find_teacher_artifacts():
    patterns = [
        "pipeline_v4/artifacts/oof/*/oof_probs.npy",
        "artifacts/calibration/*/calibrated_probs.npy",
        "reports/transformer/*/val_logits.npy",
        "reports/transformer/*/val_probs.npy",
    ]
    found = []
    for pattern in patterns:
        found.extend(str(path) for path in ROOT.glob(pattern))
    return sorted(found)


def write_skipped_n2d(out_dir, teacher_artifacts):
    payload = {
        "status": "skipped",
        "reason": "No aligned mDeBERTa/XLM-R teacher logits or probabilities were found in this clone.",
        "searched_artifacts": teacher_artifacts,
    }
    write_json(out_dir / "adopted_student.json", payload)
    for name in ["student_models.csv", "distill_weight_sweep.csv", "advanced_student_blend.csv", "half_split_stability.csv"]:
        write_csv(out_dir / name, [], ["status", "reason"])
    append_lines(
        out_dir / "summary.md",
        [
            "# N2d Inspect Teacher-Distilled Student",
            "",
            "- status: `SKIPPED`",
            "- reason: no aligned teacher logits/probs found in repository.",
            "- next action: train or copy v3/v4 validation logits, then rerun this experiment.",
        ],
    )
    return payload


def candidate_scores(sample, info):
    state = inspect_state(sample)
    score = 0
    inspect_score = 0
    margin = float(info["margin"])
    if margin <= 0.01:
        score += 3
    elif margin <= 0.05:
        score += 2
    elif margin <= 0.10:
        score += 1
    if info["pred"] in INSPECT:
        score += 3
        inspect_score += 3
    if info["top1"] in INSPECT or info["top2"] in INSPECT:
        score += 2
        inspect_score += 2
    if tuple(sorted((info["top1"], info["top2"]))) in [tuple(sorted(p)) for p in INSPECT_PAIRS]:
        score += 3
        inspect_score += 3
    if state["target_granularity"] in {"TARGET_EXACT_PATH", "TARGET_BASENAME"}:
        score += 1
        inspect_score += 1
    if state["target_granularity"] in {"TARGET_EXTENSION_ONLY", "TARGET_DIRECTORY", "TARGET_SYMBOL"}:
        score += 2
        inspect_score += 2
    if state["previous_observation"] in {"OBS_GREP_ZERO", "OBS_GREP_FEW"}:
        score += 2
        inspect_score += 2
    if state["previous_observation"] in {"OBS_GLOB_MANY", "OBS_LIST_ENTRIES"}:
        score += 1
        inspect_score += 1
    if info["pred"] in {"read_file", "grep_search", "list_directory", "glob_pattern", "edit_file", "write_file", "apply_patch", "respond_only"}:
        score += 1
    return score, inspect_score


def select_top(indices, scores, k):
    return [i for i, _ in sorted(((i, scores[i]) for i in indices), key=lambda item: (-item[1], item[0]))[:k]]


def run_n4i(samples, y, val_idx, infos, base_pred, out_dir):
    val_samples = [samples[i] for i in val_idx]
    y_val = np.asarray([y[i] for i in val_idx], dtype=object)
    n_val = len(val_samples)
    general_scores = []
    inspect_scores = []
    for sample, info in zip(val_samples, infos):
        general, inspect = candidate_scores(sample, info)
        general_scores.append(general)
        inspect_scores.append(inspect)
    general_scores = np.asarray(general_scores)
    inspect_scores = np.asarray(inspect_scores)
    base_wrong = np.asarray(base_pred, dtype=object) != y_val
    inspect_true = np.asarray([label in INSPECT for label in y_val])
    inspect_adv = np.asarray([info["pred"] in INSPECT or info["top1"] in INSPECT or info["top2"] in INSPECT for info in infos])

    distribution_rows = []
    quality_rows = []
    sweep_rows = []
    pareto = []
    policies = ["general_topK", "inspect_quota", "inspect_first_fill", "pair_quota"]
    full_train_n = 70000
    for k_full in [12000, 16000, 20000]:
        k = min(n_val, max(1, round(k_full * n_val / full_train_n)))
        for policy in policies:
            selected = set()
            all_idx = list(range(n_val))
            inspect_idx = [i for i, ok in enumerate(inspect_adv) if ok]
            if policy == "general_topK":
                selected.update(select_top(all_idx, general_scores, k))
            elif policy == "inspect_quota":
                quota = k // 2
                selected.update(select_top(inspect_idx, inspect_scores, quota))
                remaining = [i for i in all_idx if i not in selected]
                selected.update(select_top(remaining, general_scores, k - len(selected)))
            elif policy == "inspect_first_fill":
                selected.update(select_top(inspect_idx, inspect_scores, min(k, len(inspect_idx))))
                if len(selected) < k:
                    remaining = [i for i in all_idx if i not in selected]
                    selected.update(select_top(remaining, general_scores, k - len(selected)))
            elif policy == "pair_quota":
                pair_quota = {
                    tuple(sorted(("read_file", "grep_search"))): round(k * 0.30),
                    tuple(sorted(("grep_search", "glob_pattern"))): round(k * 0.25),
                    tuple(sorted(("list_directory", "glob_pattern"))): round(k * 0.20),
                    tuple(sorted(("read_file", "list_directory"))): round(k * 0.10),
                }
                for pair, quota in pair_quota.items():
                    pair_idx = [i for i, info in enumerate(infos) if tuple(sorted((info["top1"], info["top2"]))) == pair or info["pred"] in pair]
                    selected.update(select_top(pair_idx, inspect_scores, max(0, quota)))
                if len(selected) < k:
                    remaining = [i for i in all_idx if i not in selected]
                    selected.update(select_top(remaining, general_scores, k - len(selected)))
            selected = set(list(selected)[:k])
            selected_arr = np.asarray([i in selected for i in range(n_val)])
            config_id = f"{policy}_K{k_full}"
            distribution_rows.append(
                {
                    "config_id": config_id,
                    "K_full": k_full,
                    "K_val_scaled": k,
                    "candidate_policy": policy,
                    "selected_count": int(selected_arr.sum()),
                    "selected_inspect_count": int((selected_arr & inspect_adv).sum()),
                    "selected_noninspect_count": int((selected_arr & ~inspect_adv).sum()),
                    "selected_true_inspect_count": int((selected_arr & inspect_true).sum()),
                    "selected_advanced_wrong_count": int((selected_arr & base_wrong).sum()),
                }
            )
            quality = {
                "config_id": config_id,
                "candidate_policy": policy,
                "K_full": k_full,
                "selected_count": int(selected_arr.sum()),
                "advanced_error_coverage": float((selected_arr & base_wrong).sum() / max(base_wrong.sum(), 1)),
                "inspect_true_coverage": float((selected_arr & inspect_true).sum() / max(inspect_true.sum(), 1)),
                "inspect_advanced_pred_coverage": float((selected_arr & inspect_adv).sum() / max(inspect_adv.sum(), 1)),
                "inspect_error_coverage": float((selected_arr & base_wrong & (inspect_true | inspect_adv)).sum() / max((base_wrong & (inspect_true | inspect_adv)).sum(), 1)),
                "requires_transformer_for_net_win": True,
            }
            quality_rows.append(quality)
            for override in ["inspect_only", "inspect_no_read", "inspect_margin_0.2", "inspect_margin_0.4", "pair_consistent", "hybrid_safe"]:
                sweep_rows.append({**quality, "override_policy": override, "status": "candidate_only_no_transformer_logits"})
            pareto.append(quality)

    pareto = sorted(pareto, key=lambda row: (row["inspect_error_coverage"], row["advanced_error_coverage"], -row["selected_count"]), reverse=True)[:8]
    write_csv(out_dir / "candidate_distribution.csv", distribution_rows, list(distribution_rows[0].keys()))
    write_csv(out_dir / "candidate_quality.csv", quality_rows, list(quality_rows[0].keys()))
    write_csv(out_dir / "n4i_sweep_results.csv", sweep_rows, list(sweep_rows[0].keys()))
    write_csv(out_dir / "pareto_configs.csv", pareto, list(pareto[0].keys()))
    write_csv(
        out_dir / "runtime_matrix.csv",
        [
            {
                "status": "skipped",
                "reason": "No local transformer checkpoint/logits available; runtime must be measured after checkpoint is staged.",
            }
        ],
        ["status", "reason"],
    )
    write_csv(
        out_dir / "override_precision_by_action.csv",
        [{"status": "skipped", "reason": "requires transformer predictions"}],
        ["status", "reason"],
    )
    write_csv(
        out_dir / "changed_flow.csv",
        [{"status": "skipped", "reason": "requires transformer predictions"}],
        ["status", "reason"],
    )
    summary = [
        "# N4i Inspect-Focused Candidate Gating",
        "",
        "- status: `CANDIDATE_ONLY`",
        "- reason: no transformer checkpoint/logits available in this clone.",
        "- produced candidate distribution and validation-only coverage metrics.",
        "",
        "Top candidate configs by inspect-error coverage:",
    ]
    for row in pareto[:5]:
        summary.append(
            f"- `{row['config_id']}` selected={row['selected_count']} inspect_error_cov={row['inspect_error_coverage']:.3f} "
            f"advanced_error_cov={row['advanced_error_coverage']:.3f}"
        )
    append_lines(out_dir / "summary.md", summary)
    return {"status": "candidate_only", "top": pareto[:4]}


def write_public_probe_plan(out_root, n4i_result):
    lines = [
        "# Inspect Bottleneck Public Probe Plan",
        "",
        "No public probe zips were built in this run because no deployable transformer checkpoint is present.",
        "",
        "Candidate families to probe after staging a compatible checkpoint:",
        "",
        "| priority | config | reason |",
        "|---:|---|---|",
    ]
    for i, row in enumerate(n4i_result.get("top", []), 1):
        lines.append(
            f"| {i} | `{row['config_id']}` | inspect_error_cov={row['inspect_error_coverage']:.3f}, "
            f"advanced_error_cov={row['advanced_error_coverage']:.3f} |"
        )
    if not n4i_result.get("top"):
        lines.extend(
            [
                "| 1 | `256_20k_general_current_best` | baseline public probe from plan |",
                "| 2 | `320_16k_inspect_quota` | inspect-heavy coverage |",
                "| 3 | `320_20k_inspect_first_fill` | high inspect recall |",
                "| 4 | `384_12k_pair_quota` | pair ambiguity focus |",
            ]
        )
    append_lines(out_root / "public_probe_plan.md", lines)


def update_research_md(path, n2b, n2c, n2d, n4i):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "",
        "## Inspect Bottleneck Experiments",
        "",
        f"- timestamp: `{stamp}`",
        "- hypothesis: inspect bottleneck should be handled by state decomposition and high-confidence pair correction, not by a standalone 4-class inspect classifier.",
        f"- N2b state-machine: verdict=`{n2b['verdict']}`, high_states=`{n2b['high_states']}`, high_error_coverage=`{n2b['high_error_coverage']:.4f}`.",
        f"- N2c pair resolvers: verdict=`{n2c['verdict']}`, base_macro=`{n2c['base_macro']:.6f}`, combined_macro=`{n2c['combined_macro']:.6f}`, adopted=`{len(n2c['adopted'])}`.",
        f"- N2d distill student: status=`{n2d['status']}`; teacher logits/checkpoint not present in this clone.",
        f"- N4i candidate gating: status=`{n4i['status']}`; candidate-only coverage metrics produced, no transformer runtime/probe built.",
        "- decision: use N2b/N2c outputs as cheap evidence first; stage compatible transformer logits/checkpoint before N2d/N4i override validation.",
        "",
    ]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../open/data")
    parser.add_argument("--fold-file", default="pipeline_v4/folds/fold_assignments.csv")
    parser.add_argument("--out-root", default="reports/inspect_bottleneck")
    parser.add_argument("--router-mode", choices=["fast_flat", "advanced_like"], default="fast_flat")
    parser.add_argument("--router-features", type=int, default=90_000)
    parser.add_argument("--pair-features", type=int, default=60_000)
    parser.add_argument("--refresh-router", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(data_dir / "train.jsonl")
    labels = load_labels(data_dir / "train_labels.csv")
    y = np.asarray([labels[sample["id"]] for sample in samples], dtype=object)
    train_idx, val_idx, split_name = load_fold_split(samples, args.fold_file, fold=0)
    print(f"loaded samples={len(samples)} train={len(train_idx)} val={len(val_idx)} split={split_name}", flush=True)

    if args.router_mode == "advanced_like":
        router = train_fold_router(samples, y, train_idx, out_root, args.router_features, args.pair_features, refresh=args.refresh_router)
        train_pred, _, train_infos = predict_router([samples[i] for i in train_idx], router, apply_pair_resolvers=True)
        val_pred, _, val_infos = predict_router([samples[i] for i in val_idx], router, apply_pair_resolvers=True)
    else:
        router = train_fast_router(samples, y, train_idx, out_root, args.router_features, refresh=args.refresh_router)
        train_pred, _, train_infos = predict_fast_router([samples[i] for i in train_idx], router)
        val_pred, _, val_infos = predict_fast_router([samples[i] for i in val_idx], router)
    y_val = y[val_idx]
    base_macro = score_macro(y_val, val_pred)
    base_acc = float((val_pred == y_val).mean())
    print(f"base fold router macro={base_macro:.6f} acc={base_acc:.6f}", flush=True)

    n2b = run_n2b(samples, y, val_idx, val_pred, val_infos, out_root / "n2b_state_machine")
    n2c = run_n2c(samples, y, train_idx, val_idx, train_infos, val_infos, val_pred, out_root / "n2c_pair_resolvers", args.pair_features)
    teacher_artifacts = find_teacher_artifacts()
    n2d = write_skipped_n2d(out_root / "n2d_distill_student", teacher_artifacts)
    n4i = run_n4i(samples, y, val_idx, val_infos, val_pred, out_root / "n4i_candidate_gating")
    write_public_probe_plan(out_root, n4i)
    update_research_md(ROOT / "research.md", n2b, n2c, n2d, n4i)

    summary = [
        "# Inspect Bottleneck Experiment Summary",
        "",
        f"- split: `{split_name}`",
        f"- router_mode: `{args.router_mode}`",
        f"- train rows: `{len(train_idx)}`",
        f"- validation rows: `{len(val_idx)}`",
        f"- local fold router Macro-F1: `{base_macro:.6f}`",
        f"- local fold router accuracy: `{base_acc:.6f}`",
        "",
        "## Decisions",
        "",
        f"- N2b: `{n2b['verdict']}` high_states={n2b['high_states']} high_error_coverage={n2b['high_error_coverage']:.4f}",
        f"- N2c: `{n2c['verdict']}` base={n2c['base_macro']:.6f} combined={n2c['combined_macro']:.6f} adopted={len(n2c['adopted'])}",
        f"- N2d: `{n2d['status']}` because teacher logits/probs are unavailable.",
        f"- N4i: `{n4i['status']}` candidate-only coverage metrics generated.",
        "",
        "## Output directories",
        "",
        "- `n2b_state_machine/`",
        "- `n2c_pair_resolvers/`",
        "- `n2d_distill_student/`",
        "- `n4i_candidate_gating/`",
        "- `public_probe_plan.md`",
    ]
    append_lines(out_root / "summary.md", summary)
    print(f"wrote summary: {out_root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
