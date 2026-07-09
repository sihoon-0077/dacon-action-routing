import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "inspect_autoresearch"

ACTIONS = [
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
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
INSPECT = ["read_file", "grep_search", "list_directory", "glob_pattern"]
INSPECT_SET = set(INSPECT)

# Directed confusion pairs: (true_action, current_base_prediction).
TARGET_PAIRS = [
    ("grep_search", "read_file"),
    ("read_file", "list_directory"),
    ("read_file", "grep_search"),
    ("grep_search", "list_directory"),
    ("list_directory", "read_file"),
]


FILE_RE = re.compile(
    r"(?i)(?:[\w.-]+[/\\])+[\w.-]+\.[a-z0-9]{1,12}\b|[\w.-]+\.(?:py|js|ts|tsx|jsx|json|md|yaml|yml|txt|csv|sql|ipynb|toml|rs|go|java|cpp|c|h)\b"
)
SPACE_RE = re.compile(r"\s+")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def safe_text(value, limit=900):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return SPACE_RE.sub(" ", value).strip()[:limit]


def bucket_float(x, cuts):
    try:
        x = float(x)
    except Exception:
        return "unk"
    for name, upper in cuts:
        if x <= upper:
            return name
    return "hi"


def norm_prompt(text):
    text = safe_text(text, 1200).lower()
    text = re.sub(r"`[^`]+`", " <quote> ", text)
    text = re.sub(r"['\"][^'\"]+['\"]", " <quote> ", text)
    text = FILE_RE.sub(" <file> ", text)
    text = re.sub(r"\b\d+\b", " <num> ", text)
    return SPACE_RE.sub(" ", text).strip()


def extract_files(text):
    return {m.group(0).replace("\\", "/").lower() for m in FILE_RE.finditer(text or "")}


def last_actions(sample, n=6):
    out = []
    for turn in reversed(sample.get("history") or []):
        if turn.get("role") == "assistant_action":
            out.append(str(turn.get("name") or "none"))
            if len(out) >= n:
                break
    return list(reversed(out))


def last_action(sample):
    acts = last_actions(sample, 1)
    return acts[-1] if acts else "none"


def result_bucket(text):
    low = safe_text(text, 700).lower()
    if not low:
        return "none"
    if any(x in low for x in ["traceback", "exception", "error", "failed", "fail", "permission denied"]):
        return "fail"
    if any(x in low for x in ["no matches", "0 matches", "not found", "zero match"]):
        return "zero_match"
    m = re.search(r"(\d+)\s*(matches?|files?|items?|occurrences?|results?)", low)
    if m:
        n = int(m.group(1))
        if n == 0:
            return "zero_match"
        if n <= 3:
            return "few"
        if n <= 20:
            return "some"
        return "many"
    if any(x in low for x in ["match", "found", "occurrence", "result"]):
        return "matches"
    if any(x in low for x in ["read", "opened", "lines"]):
        return "read_ok"
    if any(x in low for x in ["listed", "entries", "directory", "files"]):
        return "listed"
    return "other"


def last_result(sample):
    for turn in reversed(sample.get("history") or []):
        if turn.get("role") == "assistant_action":
            return result_bucket(turn.get("result_summary"))
    return "none"


def inspect_streak(sample):
    streak = 0
    for act in reversed(last_actions(sample, 12)):
        if act in INSPECT_SET:
            streak += 1
        else:
            break
    if streak == 0:
        return "s0"
    if streak == 1:
        return "s1"
    if streak == 2:
        return "s2"
    if streak <= 4:
        return "s3_4"
    return "s5p"


def open_profile(sample):
    ws = ((sample.get("session_meta") or {}).get("workspace") or {})
    files = ws.get("open_files") or []
    if not files:
        return "none"
    if len(files) >= 3:
        return "many3p"
    exts = {Path(str(p).lower()).suffix.lstrip(".") for p in files if Path(str(p)).suffix}
    if exts and exts <= {"py"}:
        return "py_only"
    if exts and exts <= {"js", "jsx", "ts", "tsx"}:
        return "js_only"
    if len(exts) == 1:
        return "one_" + next(iter(exts))
    return "mixed"


def prompt_file_rel(sample):
    prompt_files = extract_files(sample.get("current_prompt") or "")
    if not prompt_files:
        return "no_file"
    ws = ((sample.get("session_meta") or {}).get("workspace") or {})
    open_names = set()
    for p in ws.get("open_files") or []:
        p = str(p).replace("\\", "/").lower()
        open_names.add(p)
        open_names.add(p.rsplit("/", 1)[-1])
    prompt_names = set(prompt_files)
    prompt_names.update(x.rsplit("/", 1)[-1] for x in prompt_files)
    return "open" if (prompt_names & open_names) else "not_open"


def prompt_intent(sample):
    t = safe_text(sample.get("current_prompt"), 1200).lower()
    if any(x in t for x in ["find", "search", "grep", "where", "어디", "찾", "검색"]):
        return "find"
    if any(x in t for x in ["open", "read", "show", "열", "읽", "봐", "확인"]):
        return "read"
    if any(x in t for x in ["list", "ls", "tree", "목록", "구조", "파일들"]):
        return "list"
    if any(x in t for x in ["glob", "pattern", "패턴"]):
        return "glob"
    return "other"


def recent_history_text(sample):
    chunks = []
    for turn in (sample.get("history") or [])[-8:]:
        role = turn.get("role")
        if role == "user":
            chunks.append("[HU] " + safe_text(turn.get("content"), 450))
        elif role == "assistant_action":
            chunks.append("[HA] " + safe_text(turn.get("name"), 60))
            chunks.append("[HR] " + result_bucket(turn.get("result_summary")))
            chunks.append("[HS] " + safe_text(turn.get("result_summary"), 300))
            chunks.append("[HG] " + safe_text(turn.get("args"), 400))
    return " ".join(chunks)


def rich_text(sample, base_pred, base_scores, teacher_scores):
    meta = sample.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    base_order = np.argsort(base_scores)[::-1]
    teacher_order = np.argsort(teacher_scores)[::-1]
    base_margin = base_scores[base_order[0]] - base_scores[base_order[1]]
    teacher_margin = teacher_scores[teacher_order[0]] - teacher_scores[teacher_order[1]]
    toks = [
        "[NOW] " + safe_text(sample.get("current_prompt"), 1200),
        "[NOW_NORM] " + norm_prompt(sample.get("current_prompt")),
        "[HIST] " + recent_history_text(sample),
        "[ACTIONS] " + ">".join(last_actions(sample, 6)),
        f"[STATE] last={last_action(sample)} result={last_result(sample)} streak={inspect_streak(sample)}",
        f"[OPEN] profile={open_profile(sample)} rel={prompt_file_rel(sample)} intent={prompt_intent(sample)} open_n={len(ws.get('open_files') or [])}",
        f"[META] ci={ws.get('last_ci_status', 'none')} dirty={int(bool(ws.get('git_dirty', False)))} pref={meta.get('language_pref', 'none')} turn={meta.get('turn_index', 'none')}",
        f"[BASE] pred={base_pred} top1={ACTIONS[base_order[0]]} top2={ACTIONS[base_order[1]]} margin={bucket_float(base_margin, [('m0', .05), ('m1', .15), ('m2', .35), ('m3', .8)])}",
        f"[TEACHER] top1={ACTIONS[teacher_order[0]]} top2={ACTIONS[teacher_order[1]]} margin={bucket_float(teacher_margin, [('tm0', .05), ('tm1', .15), ('tm2', .35), ('tm3', .8)])}",
    ]
    for action in INSPECT:
        i = ACTION_TO_ID[action]
        toks.append(f"bp_{action}={bucket_float(base_scores[i], [('b0', -3), ('b1', -1.5), ('b2', -.6), ('b3', -.15)])}")
        toks.append(f"tp_{action}={bucket_float(teacher_scores[i], [('t0', .05), ('t1', .15), ('t2', .35), ('t3', .65)])}")
    return "\n".join(toks)


def coarse_key(sample, base_pred, level):
    parts = [
        f"base={base_pred}",
        f"last={last_action(sample)}",
        f"result={last_result(sample)}",
        f"streak={inspect_streak(sample)}",
    ]
    if level >= 2:
        parts.extend([f"open={open_profile(sample)}", f"rel={prompt_file_rel(sample)}", f"intent={prompt_intent(sample)}"])
    if level >= 3:
        parts.append("last2=" + ">".join(last_actions(sample, 2)))
        ws = ((sample.get("session_meta") or {}).get("workspace") or {})
        parts.append(f"ci={ws.get('last_ci_status', 'none')}")
    return " | ".join(parts)


def macro_f1(y, pred):
    return float(f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0))


def inspect_f1(y, pred):
    return float(f1_score(y, pred, labels=INSPECT, average="macro", zero_division=0))


def pair_counts(y, before, after):
    rows = []
    for true_action, pred_action in TARGET_PAIRS:
        before_mask = (y == true_action) & (before == pred_action)
        after_mask = (y == true_action) & (after == true_action)
        fixed = int((before_mask & (after == true_action)).sum())
        rows.append({
            "true_action": true_action,
            "base_pred": pred_action,
            "base_errors": int(before_mask.sum()),
            "fixed_by_variant": fixed,
        })
    return rows


def fold_values(y, pred, folds):
    values = []
    for fold in sorted(set(folds.tolist())):
        m = folds == fold
        values.append(macro_f1(y[m], pred[m]))
    return values


def load_base():
    adv = np.load(ROOT / "artifacts" / "advanced_oof_strict" / "advanced_oof_probs.npy").astype(np.float32)
    d2 = np.load(ROOT / "reports" / "distill_step2_strict" / "mlp_oof" / "D2-M5" / "oof_probs.npy").astype(np.float32)
    teacher = np.load(ROOT / "artifacts" / "distill_step2_strict" / "teacher_oof" / "teacher_oof_probs.npy").astype(np.float32)
    cfg = read_json(ROOT / "reports" / "distill_step2_strict" / "blends" / "best_config.json")
    probs = 0.5 * adv + 0.5 * d2
    bias = np.array([float(cfg["bias"]["bias_by_class"].get(a, 0.0)) for a in ACTIONS], dtype=np.float32)
    scores = np.log(np.clip(probs, 1e-12, 1.0)) + bias[None, :]
    pred = np.array([ACTIONS[i] for i in scores.argmax(axis=1)], dtype=object)
    return pred, scores, teacher


def build_vectorizer(name):
    if name == "word":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120_000, sublinear_tf=True, lowercase=True)
    if name == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120_000, sublinear_tf=True, lowercase=True)
    if name == "union":
        return FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=90_000, sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=60_000, sublinear_tf=True, lowercase=True)),
        ])
    raise ValueError(name)


def run_pair_flip_config(texts, y, folds, base_pred, config_name, vec_name, c_value):
    scores_by_pair = np.full((len(TARGET_PAIRS), len(y)), np.nan, dtype=np.float32)
    for pair_idx, (true_action, base_action) in enumerate(TARGET_PAIRS):
        for fold in sorted(set(folds.tolist())):
            train = (folds != fold) & (base_pred == base_action) & np.isin(y, INSPECT)
            val = (folds == fold) & (base_pred == base_action)
            if train.sum() < 80 or val.sum() == 0:
                continue
            target = (y[train] == true_action).astype(np.int64)
            if target.sum() < 20 or target.sum() == len(target):
                continue
            vectorizer = build_vectorizer(vec_name)
            x_train = vectorizer.fit_transform(texts[train])
            clf = LogisticRegression(C=c_value, class_weight="balanced", max_iter=500, solver="liblinear", random_state=42)
            clf.fit(x_train, target)
            proba = clf.predict_proba(vectorizer.transform(texts[val]))[:, list(clf.classes_).index(1)]
            scores_by_pair[pair_idx, np.where(val)[0]] = proba.astype(np.float32)

    rows = []
    variants = []
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        out = base_pred.copy()
        changed = 0
        for pair_idx, (true_action, base_action) in enumerate(TARGET_PAIRS):
            mask = (base_pred == base_action) & np.isfinite(scores_by_pair[pair_idx]) & (scores_by_pair[pair_idx] >= threshold)
            changed += int((out[mask] != true_action).sum())
            out[mask] = true_action
        rows.append((f"{config_name}_thr{threshold:.2f}", out, changed))
        variants.append(out)
    return rows


def run_rule_config(samples, y, folds, base_pred, level, min_support, min_purity):
    out = base_pred.copy()
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        val = folds == fold
        table = defaultdict(Counter)
        for i in np.where(train & np.isin(base_pred, INSPECT))[0]:
            table[coarse_key(samples[i], base_pred[i], level)][y[i]] += 1
        val_idx = np.where(val & np.isin(base_pred, INSPECT))[0]
        for i in val_idx:
            counts = table.get(coarse_key(samples[i], base_pred[i], level))
            if not counts:
                continue
            action, count = counts.most_common(1)[0]
            total = sum(counts.values())
            if action in INSPECT_SET and total >= min_support and count / total >= min_purity:
                out[i] = action
    return out


def write_summary(rows, base_macro, base_inspect):
    rows = sorted(rows, key=lambda r: (r["macro_f1"], r["inspect_f1"]), reverse=True)
    write_csv(OUT / "results.csv", rows, list(rows[0].keys()))
    best = rows[0]
    lines = [
        "# Inspect Autoresearch",
        "",
        f"- base Macro-F1: `{base_macro:.6f}`",
        f"- base inspect4 Macro-F1: `{base_inspect:.6f}`",
        f"- best: `{best['name']}`",
        f"- best Macro-F1: `{best['macro_f1']:.6f}`",
        f"- best delta: `{best['delta']:.6f}`",
        f"- best inspect delta: `{best['inspect_delta']:.6f}`",
        f"- changed: `{best['changed']}`",
        "",
        "## Top Variants",
        "",
        "| name | Macro-F1 | delta | inspect4 | inspect_delta | changed | min_fold_delta | fixed_target_errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:20]:
        lines.append(
            f"| `{row['name']}` | `{row['macro_f1']:.6f}` | `{row['delta']:.6f}` | "
            f"`{row['inspect_f1']:.6f}` | `{row['inspect_delta']:.6f}` | `{row['changed']}` | "
            f"`{row['min_fold_delta']:.6f}` | `{row['fixed_target_errors']}` |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "- This is a Karpathy-autoresearch-style cheap loop specialized to the five biggest inspect confusions.",
        "- Adopt only if Macro-F1, inspect4 F1, and fold stability are all positive. A high changed count with negative delta means the flip rule is too aggressive.",
    ])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def use_cached_result():
    if os.environ.get("FORCE_INSPECT_AUTORESEARCH") == "1":
        return False
    results_path = OUT / "results.csv"
    summary_path = OUT / "summary.md"
    script_path = Path(__file__).resolve()
    if not (results_path.exists() and summary_path.exists()):
        return False
    if results_path.stat().st_mtime < script_path.stat().st_mtime:
        return False
    print("# Inspect Autoresearch", flush=True)
    print("", flush=True)
    print("- cached: `true`", flush=True)
    print("- reason: previous inspect pair/rule sweep is newer than the script; set FORCE_INSPECT_AUTORESEARCH=1 to rerun.", flush=True)
    print("", flush=True)
    print(summary_path.read_text(encoding="utf-8").split("## Top Variants")[0], flush=True)
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if use_cached_result():
        return
    samples = read_jsonl(ROOT / "data" / "train.jsonl")
    label_map = load_labels(ROOT / "data" / "train_labels.csv")
    y = np.array([label_map[s["id"]] for s in samples], dtype=object)
    folds = np.load(ROOT / "artifacts" / "distill_step2_strict" / "fold_ids.npy")
    base_pred, base_scores, teacher_probs = load_base()
    base_macro = macro_f1(y, base_pred)
    base_inspect = inspect_f1(y, base_pred)
    texts = np.array(
        [rich_text(s, base_pred[i], base_scores[i], teacher_probs[i]) for i, s in enumerate(samples)],
        dtype=object,
    )

    rows = []

    def add_variant(name, pred, changed):
        m = macro_f1(y, pred)
        ins = inspect_f1(y, pred)
        fvals = fold_values(y, pred, folds)
        pcounts = pair_counts(y, base_pred, pred)
        fixed = sum(r["fixed_by_variant"] for r in pcounts)
        rows.append({
            "name": name,
            "macro_f1": m,
            "delta": m - base_macro,
            "inspect_f1": ins,
            "inspect_delta": ins - base_inspect,
            "changed": int(changed),
            "fixed_target_errors": int(fixed),
            "min_fold_macro_f1": min(fvals),
            "min_fold_delta": min(fvals) - min(fold_values(y, base_pred, folds)),
            "folds": ";".join(f"{v:.6f}" for v in fvals),
        })

    add_variant("base_strict_distill_bias", base_pred, 0)

    pair_configs = [
        ("pair_word_c1", "word", 1.0),
        ("pair_word_c3", "word", 3.0),
        ("pair_union_c1", "union", 1.0),
    ]
    for name, vec_name, c_value in pair_configs:
        print(f"running {name}", flush=True)
        for variant_name, pred, changed in run_pair_flip_config(texts, y, folds, base_pred, name, vec_name, c_value):
            add_variant(variant_name, pred, changed)

    for level in [1, 2, 3]:
        for min_support in [20, 40, 80, 120]:
            for min_purity in [0.55, 0.60, 0.65, 0.70, 0.75]:
                pred = run_rule_config(samples, y, folds, base_pred, level, min_support, min_purity)
                add_variant(f"rule_l{level}_s{min_support}_p{min_purity}", pred, int((pred != base_pred).sum()))

    rows = write_summary(rows, base_macro, base_inspect)
    print((OUT / "summary.md").read_text(encoding="utf-8").split("## Top Variants")[0], flush=True)


if __name__ == "__main__":
    main()
