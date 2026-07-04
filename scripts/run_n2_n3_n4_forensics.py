import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script import (
    ADVANCED_GROUP_TO_ACTIONS,
    ALL_CLASSES,
    advanced_aligned_log_proba,
    advanced_group_text,
    advanced_pair_text,
    advanced_transition_prior_matrix,
    compact_flags_text,
    v4_prefilter_priority,
)


DATA_DIR = ROOT / "data"
REPORT_ROOT = ROOT / "reports"
FOLD_FILE = ROOT / "pipeline_v4" / "folds" / "fold_assignments.csv"
ADVANCED_MODEL = ROOT / "model" / "advanced_router.pkl"
MDEBERTA_OOF = ROOT / "pipeline_v4" / "artifacts" / "oof" / "mdeberta384_v2_384_5e"

INSPECT = ["read_file", "grep_search", "list_directory", "glob_pattern"]
TRIAD = ["ask_user", "plan_task", "web_search"]
COMM = ["ask_user", "plan_task", "web_search", "respond_only"]
OVERRIDE_ACTIONS = set(["read_file", "grep_search", "list_directory", "glob_pattern", "edit_file", "write_file", "apply_patch", "respond_only"])
TAUS = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
K_VALUES = [1000, 2000, 4000, 8000, 12000, 16000, 20000, 25000, 30000]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_labels(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def safe_text(value, max_chars=1200):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(text.split())[:max_chars]


def load_samples():
    labels = read_labels(DATA_DIR / "train_labels.csv")
    samples = read_jsonl(DATA_DIR / "train.jsonl")
    for sample in samples:
        sample["action"] = labels[sample["id"]]
    return samples


def load_fold0_split(samples):
    fold_by_id = {}
    with open(FOLD_FILE, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fold_by_id[row["id"]] = int(row["fold"])
    val_idx = np.array([i for i, s in enumerate(samples) if fold_by_id[s["id"]] == 0], dtype=int)
    train_idx = np.array([i for i, s in enumerate(samples) if fold_by_id[s["id"]] != 0], dtype=int)
    return train_idx, val_idx


def build_vectorizer(max_features=120_000, min_df=2):
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


def action_turns(sample):
    return [t for t in sample.get("history", []) or [] if t.get("role") == "assistant_action"]


def user_turns(sample):
    return [t for t in sample.get("history", []) or [] if t.get("role") == "user"]


def last_action_turn(sample):
    turns = action_turns(sample)
    return turns[-1] if turns else {}


def action_seq(sample, n=6):
    names = [t.get("name", "") for t in action_turns(sample) if t.get("name")]
    return " > ".join(names[-n:]) if names else "none"


def extract_paths(text):
    text = safe_text(text, 5000)
    patterns = [
        r"[\w./\\@~{}$-]+\.(?:py|tsx|ts|js|jsx|json|yaml|yml|go|rs|toml|java|kt|sql|tf|sh|md|txt|vue|gradle|lock)",
        r"\*\*/[^\s,;]+",
        r"\*\.[A-Za-z0-9]+",
    ]
    paths = []
    for pat in patterns:
        paths.extend(re.findall(pat, text, flags=re.IGNORECASE))
    norm = []
    for path in paths[:20]:
        path = path.strip("\"'`()[]{}<>").replace("\\", "/")
        if path and path not in norm:
            norm.append(path)
    exts = []
    basenames = []
    for path in norm:
        base = path.rsplit("/", 1)[-1]
        basenames.append(base)
        if "." in base:
            exts.append(base.rsplit(".", 1)[-1].lower())
    return {
        "paths": norm,
        "basenames": basenames,
        "exts": sorted(set(exts)),
        "has_glob": any("*" in p for p in norm) or bool(re.search(r"\*\.", text)),
        "has_dir_slash": any("/" in p and "." not in p.rsplit("/", 1)[-1] for p in norm),
        "has_specific_file": any("." in p.rsplit("/", 1)[-1] and "*" not in p for p in norm),
    }


def count_bucket(text):
    nums = []
    for m in re.findall(r"\b\d{1,5}\b", safe_text(text, 3000)):
        try:
            nums.append(int(m))
        except ValueError:
            pass
    if not nums:
        return "none"
    mx = max(nums)
    if mx == 0:
        return "zero"
    if mx == 1:
        return "one"
    if mx <= 3:
        return "few_2_3"
    if mx <= 10:
        return "mid_4_10"
    return "many_11_plus"


def result_bucket(text):
    low = safe_text(text, 3000).lower()
    if not low:
        return "RESULT_NONE"
    if any(x in low for x in ["traceback", "exception", "error", "failed", "fail", "not found", "no such"]):
        if any(x in low for x in ["no match", "0 match", "not found"]):
            return "RESULT_ZERO_MATCH"
        return "RESULT_ERROR"
    if any(x in low for x in ["0 match", "no match", "no results"]):
        return "RESULT_ZERO_MATCH"
    if any(x in low for x in ["read", "opened", "lines", "content"]):
        return "RESULT_READ_OK"
    if any(x in low for x in ["listed", "directory", "entries", "tree"]):
        return "RESULT_LISTED_DIR"
    if any(x in low for x in ["matches", "occurrences", "found", "results", "files"]):
        bucket = count_bucket(low)
        if bucket in {"zero"}:
            return "RESULT_ZERO_MATCH"
        if bucket in {"one", "few_2_3"}:
            return "RESULT_FEW_MATCH"
        return "RESULT_MANY_MATCH"
    return "RESULT_UNKNOWN"


def prompt_has(patterns, sample):
    prompt = (sample.get("current_prompt") or "").lower()
    return any(re.search(p, prompt, flags=re.IGNORECASE) for p in patterns)


def inspect_trigger(sample):
    return prompt_has(
        [
            r"\bopen\b",
            r"\bread\b",
            r"\bshow\b",
            r"\bgrep\b",
            r"\bsearch\b",
            r"\bfind\b",
            r"\bwhere\b",
            r"\blist\b",
            r"\btree\b",
            r"\bfolder\b",
            r"\bdirectory\b",
            r"\bglob\b",
            r"\*\.",
            r"\*\*/",
            r"파일",
            r"열어",
            r"보여",
            r"찾",
            r"목록",
            r"폴더",
        ],
        sample,
    )


def comm_trigger(sample):
    return prompt_has(
        [
            r"latest",
            r"best practice",
            r"recommended",
            r"official",
            r"\bplan\b",
            r"break down",
            r"where to start",
            r"should i",
            r"which",
            r"summar",
            r"recap",
            r"wrap",
            r"최신",
            r"권장",
            r"공식",
            r"계획",
            r"단계",
            r"요약",
            r"정리",
            r"마무리",
            r"어떻게",
            r"뭐가",
        ],
        sample,
    )


def respond_only_protect(sample):
    return prompt_has(
        [r"summar", r"summary", r"recap", r"wrap", r"brief", r"done here", r"요약", r"정리", r"마무리"],
        sample,
    )


def inspect_text(sample):
    prompt = safe_text(sample.get("current_prompt"), 1400)
    last = last_action_turn(sample)
    last_args = last.get("args") or {}
    last_result = safe_text(last.get("result_summary"), 1200)
    ws = (sample.get("session_meta", {}) or {}).get("workspace", {}) or {}
    open_files = [safe_text(p, 260).replace("\\", "/") for p in (ws.get("open_files", []) or [])[:12]]
    blob = " ".join([prompt, safe_text(last_args, 1200), last_result, " ".join(open_files)])
    paths = extract_paths(blob)
    flags = [
        f"last_action={safe_text(last.get('name'), 80) or 'none'}",
        f"result_bucket={result_bucket(last_result)}",
        f"count_bucket={count_bucket(last_result)}",
        f"file_known={int(paths['has_specific_file'] or bool(open_files))}",
        f"path_known={int(bool(paths['paths']))}",
        f"glob_known={int(paths['has_glob'])}",
        f"dir_known={int(bool(open_files) or paths['has_dir_slash'])}",
        f"seq={action_seq(sample)}",
    ]
    for path in paths["paths"][:10]:
        flags.append(f"mentioned_path={path.lower()}")
    for ext in paths["exts"][:10]:
        flags.append(f"mentioned_ext={ext}")
    for path in open_files:
        flags.append(f"open_file={path.lower()}")
        if "." in path.rsplit("/", 1)[-1]:
            flags.append(f"open_ext={path.rsplit('.', 1)[-1].lower()}")
    recent_users = " ".join(safe_text(t.get("content"), 500) for t in user_turns(sample)[-3:])
    return "\n".join(
        [
            "[NOW] " + prompt,
            "[LAST] action={} args={} bucket={} result={}".format(
                safe_text(last.get("name"), 80) or "none",
                safe_text(last_args, 600),
                result_bucket(last_result),
                last_result,
            ),
            "[KNOWN] " + " ".join(flags),
            "[FILES] " + " ".join(open_files),
            "[HIST] " + recent_users,
        ]
    )


def comm_text(sample):
    prompt = safe_text(sample.get("current_prompt"), 1400)
    last = last_action_turn(sample)
    last_result = safe_text(last.get("result_summary"), 1000)
    ws = (sample.get("session_meta", {}) or {}).get("workspace", {}) or {}
    open_files = ws.get("open_files", []) or []
    flags = [
        f"last_action={safe_text(last.get('name'), 80) or 'none'}",
        f"last_bucket={result_bucket(last_result)}",
        f"seq={action_seq(sample)}",
        f"has_recent_read={int(any(t.get('name') == 'read_file' for t in action_turns(sample)[-5:]))}",
        f"has_recent_grep={int(any(t.get('name') == 'grep_search' for t in action_turns(sample)[-5:]))}",
        f"has_recent_web={int(any(t.get('name') == 'web_search' for t in action_turns(sample)[-5:]))}",
        f"has_recent_ask={int(any(t.get('name') == 'ask_user' for t in action_turns(sample)[-5:]))}",
        f"has_concrete_file={int(bool(open_files) or extract_paths(prompt)['has_specific_file'])}",
        f"question_form={int(prompt.strip().endswith('?'))}",
        f"respond_protect={int(respond_only_protect(sample))}",
        f"comm_trigger={int(comm_trigger(sample))}",
    ]
    recent = []
    for turn in (sample.get("history", []) or [])[-8:]:
        if turn.get("role") == "user":
            recent.append("hist_user=" + safe_text(turn.get("content"), 500))
        elif turn.get("role") == "assistant_action":
            recent.append(
                "hist_action={} hist_bucket={} hist_result={}".format(
                    safe_text(turn.get("name"), 80),
                    result_bucket(turn.get("result_summary", "")),
                    safe_text(turn.get("result_summary"), 300),
                )
            )
    return "\n".join(["[NOW] " + prompt, "[STATE] " + " ".join(flags), "[LAST] " + last_result, "[HIST] " + " ".join(recent)])


def predict_advanced_details(samples, artifact):
    coarse_texts = [compact_flags_text(sample) for sample in samples]
    coarse_x = artifact["coarse_vectorizer"].transform(coarse_texts)
    group_pred = artifact["coarse_model"].predict(coarse_x)
    scores = np.full((len(samples), len(ALL_CLASSES)), -1e9, dtype=np.float32)
    for group, actions in ADVANCED_GROUP_TO_ACTIONS.items():
        idx = np.where(group_pred == group)[0]
        if not len(idx):
            continue
        texts = [advanced_group_text(samples[i], group) for i in idx]
        x = artifact["group_vectorizers"][group].transform(texts)
        group_scores = advanced_aligned_log_proba(artifact["group_models"][group], x, actions)
        for j, action in enumerate(actions):
            scores[idx, ALL_CLASSES.index(action)] = group_scores[:, j]
    prior = advanced_transition_prior_matrix(
        samples,
        artifact["transition_last2"],
        artifact["global_counts"],
        artifact["config"].get("prior_smooth", 1.0),
    )
    alpha = artifact["config"].get("prior_alpha", 0.3)
    for i, group in enumerate(group_pred):
        for action in ADVANCED_GROUP_TO_ACTIONS[str(group)]:
            j = ALL_CLASSES.index(action)
            scores[i, j] += alpha * prior[i, j]
    prob_like = np.exp(np.clip(scores, -50, 50))
    order = np.argsort(prob_like, axis=1)
    top1 = np.array([ALL_CLASSES[i] for i in order[:, -1]], dtype=object)
    top2 = np.array([ALL_CLASSES[i] for i in order[:, -2]], dtype=object)
    margin = prob_like[np.arange(len(samples)), order[:, -1]] - prob_like[np.arange(len(samples)), order[:, -2]]
    preds = top1.copy()
    pair_thr = artifact["config"].get("pair_threshold", 0.08)
    for i, (a, b, m) in enumerate(zip(top1, top2, margin)):
        pair = tuple(sorted((str(a), str(b))))
        resolver = artifact["pair_resolvers"].get(pair)
        if resolver is None or m > pair_thr:
            continue
        text = advanced_pair_text(samples[i], pair)
        x = resolver["vectorizer"].transform([text])
        preds[i] = str(resolver["model"].predict(x)[0])
    return {
        "pred": np.array([str(x) for x in preds], dtype=object),
        "top1": top1,
        "top2": top2,
        "margin": margin,
        "group_pred": np.array([str(x) for x in group_pred], dtype=object),
        "scores": scores,
    }


def proba_margin(proba):
    order = np.argsort(proba, axis=1)
    top = proba[np.arange(proba.shape[0]), order[:, -1]]
    second = proba[np.arange(proba.shape[0]), order[:, -2]]
    return top - second


def fit_text_model(texts, labels, max_features=120_000):
    vectorizer = build_vectorizer(max_features=max_features, min_df=2)
    x = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=900, C=2.0, class_weight="balanced", random_state=42, n_jobs=None)
    model.fit(x, labels)
    return vectorizer, model


def predict_text_model(vectorizer, model, texts, classes):
    x = vectorizer.transform(texts)
    proba_raw = model.predict_proba(x)
    proba = np.zeros((len(texts), len(classes)), dtype=np.float32)
    for i, cls in enumerate(model.classes_):
        proba[:, classes.index(str(cls))] = proba_raw[:, i]
    pred = np.array([classes[i] for i in proba.argmax(axis=1)], dtype=object)
    margin = proba_margin(proba)
    return pred, proba, margin


def write_class_report(path, y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    rows = []
    for label in labels:
        row = report.get(label, {})
        rows.append(
            {
                "class": label,
                "precision": row.get("precision", 0.0),
                "recall": row.get("recall", 0.0),
                "f1": row.get("f1-score", 0.0),
                "support": row.get("support", 0),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_confusion(path, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(path)


def changed_matrix(y_true, before, after):
    rows = []
    for src in ALL_CLASSES:
        for dst in ALL_CLASSES:
            mask = (before == src) & (after == dst) & (before != after)
            if not mask.any():
                continue
            rows.append(
                {
                    "from_action": src,
                    "to_action": dst,
                    "count": int(mask.sum()),
                    "correct_before": int((before[mask] == y_true[mask]).sum()),
                    "correct_after": int((after[mask] == y_true[mask]).sum()),
                    "net_gain": int((after[mask] == y_true[mask]).sum() - (before[mask] == y_true[mask]).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["net_gain", "count"], ascending=[False, False]) if rows else pd.DataFrame()


def macro(y, pred):
    return f1_score(y, pred, labels=ALL_CLASSES, average="macro", zero_division=0)


def macro_subset(y, pred, labels):
    return f1_score(y, pred, labels=labels, average="macro", zero_division=0)


def run_n2(samples, train_idx, val_idx, y_val, adv_pred):
    out = REPORT_ROOT / "n2_inspect_specialist"
    out.mkdir(parents=True, exist_ok=True)
    y_all = np.array([s["action"] for s in samples], dtype=object)
    train_inspect = np.array([i for i in train_idx if y_all[i] in INSPECT], dtype=int)
    val_inspect = np.array([i for i in val_idx if y_all[i] in INSPECT], dtype=int)
    vectorizer, model = fit_text_model([inspect_text(samples[i]) for i in train_inspect], y_all[train_inspect], 100_000)
    val_texts_all = [inspect_text(samples[i]) for i in val_idx]
    spec_pred_all, spec_proba_all, spec_margin_all = predict_text_model(vectorizer, model, val_texts_all, INSPECT)
    val_pos = {idx: p for p, idx in enumerate(val_idx)}
    inspect_positions = np.array([val_pos[i] for i in val_inspect], dtype=int)
    iso_pred = spec_pred_all[inspect_positions]
    iso_true = y_all[val_inspect]
    isolated_f1 = macro_subset(iso_true, iso_pred, INSPECT)
    adv_iso = adv_pred[inspect_positions]
    adv_iso_f1 = macro_subset(iso_true, adv_iso, INSPECT)
    write_class_report(out / "isolated_f1.csv", iso_true, iso_pred, INSPECT)
    write_confusion(out / "inspect_confusion.csv", iso_true, iso_pred, INSPECT)

    sweep = []
    best = None
    for tau in TAUS:
        candidate = np.array([p in INSPECT or inspect_trigger(samples[idx]) for p, idx in enumerate(val_idx)], dtype=bool)
        use = candidate & (spec_margin_all >= tau)
        final = adv_pred.copy()
        final[use] = spec_pred_all[use]
        row = {
            "tau": tau,
            "overall_macro_f1": macro(y_val, final),
            "delta_vs_advanced_full": macro(y_val, final) - macro(y_val, adv_pred),
            "inspect_macro_f1_true_subset": macro_subset(y_val[np.isin(y_val, INSPECT)], final[np.isin(y_val, INSPECT)], INSPECT),
            "changed_count": int((final != adv_pred).sum()),
            "candidate_count": int(candidate.sum()),
            "used_count": int(use.sum()),
        }
        sweep.append(row)
        if best is None or row["overall_macro_f1"] > best["overall_macro_f1"]:
            best = row | {"pred": final}
    pd.DataFrame(sweep).to_csv(out / "deployable_sweep.csv", index=False)
    changed_matrix(y_val, adv_pred, best["pred"]).to_csv(out / "changed_matrix.csv", index=False)
    joblib.dump({"vectorizer": vectorizer, "model": model, "classes": INSPECT}, out / "inspect_specialist.pkl", compress=3)
    summary = [
        "# N2 Inspect Specialist v2",
        "",
        f"- train inspect rows: `{len(train_inspect)}`",
        f"- val inspect rows: `{len(val_inspect)}`",
        f"- isolated inspect Macro-F1: `{isolated_f1:.6f}`",
        f"- advanced_full isolated inspect Macro-F1 참고값: `{adv_iso_f1:.6f}`",
        f"- advanced_full overall Macro-F1 참고값: `{macro(y_val, adv_pred):.6f}`",
        f"- best deployable tau: `{best['tau']}`",
        f"- best deployable Macro-F1: `{best['overall_macro_f1']:.6f}`",
        f"- delta vs advanced_full: `{best['delta_vs_advanced_full']:.6f}`",
        f"- changed_count: `{best['changed_count']}`",
        "",
        "Note: `advanced_full` was trained on all train rows, so deployable delta is a direction/proxy, not a clean OOF score.",
    ]
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {"isolated_f1": isolated_f1, "advanced_iso_f1": adv_iso_f1, "best": best}


def run_n3(samples, train_idx, val_idx, y_val, adv_pred):
    out = REPORT_ROOT / "n3_comm_triad"
    out.mkdir(parents=True, exist_ok=True)
    y_all = np.array([s["action"] for s in samples], dtype=object)
    train_triad = np.array([i for i in train_idx if y_all[i] in TRIAD], dtype=int)
    val_triad = np.array([i for i in val_idx if y_all[i] in TRIAD], dtype=int)
    train_comm = np.array([i for i in train_idx if y_all[i] in COMM], dtype=int)
    val_comm = np.array([i for i in val_idx if y_all[i] in COMM], dtype=int)

    tri_vec, tri_model = fit_text_model([comm_text(samples[i]) for i in train_triad], y_all[train_triad], 80_000)
    comm_vec, comm_model = fit_text_model([comm_text(samples[i]) for i in train_comm], y_all[train_comm], 100_000)
    val_texts_all = [comm_text(samples[i]) for i in val_idx]
    tri_pred_all, _, tri_margin_all = predict_text_model(tri_vec, tri_model, val_texts_all, TRIAD)
    comm_pred_all, _, comm_margin_all = predict_text_model(comm_vec, comm_model, val_texts_all, COMM)
    val_pos = {idx: p for p, idx in enumerate(val_idx)}
    tri_positions = np.array([val_pos[i] for i in val_triad], dtype=int)
    comm_positions = np.array([val_pos[i] for i in val_comm], dtype=int)

    tri_true = y_all[val_triad]
    comm_true = y_all[val_comm]
    tri_iso = macro_subset(tri_true, tri_pred_all[tri_positions], TRIAD)
    comm_iso = macro_subset(comm_true, comm_pred_all[comm_positions], COMM)
    write_class_report(out / "triad_isolated.csv", tri_true, tri_pred_all[tri_positions], TRIAD)
    write_class_report(out / "comm4_isolated.csv", comm_true, comm_pred_all[comm_positions], COMM)
    write_confusion(out / "triad_confusion.csv", tri_true, tri_pred_all[tri_positions], TRIAD)

    base_macro = macro(y_val, adv_pred)
    sweep = []
    best = None
    for tau in TAUS:
        candidate = np.array([p in COMM or comm_trigger(samples[idx]) for p, idx in enumerate(val_idx)], dtype=bool)
        protect = np.array([respond_only_protect(samples[idx]) and adv_pred[p] == "respond_only" for p, idx in enumerate(val_idx)], dtype=bool)
        use = candidate & (~protect) & (comm_margin_all >= tau)
        final = adv_pred.copy()
        final[use] = comm_pred_all[use]
        row = {
            "tau": tau,
            "overall_macro_f1": macro(y_val, final),
            "delta_vs_advanced_full": macro(y_val, final) - base_macro,
            "comm4_macro_f1_true_subset": macro_subset(y_val[np.isin(y_val, COMM)], final[np.isin(y_val, COMM)], COMM),
            "respond_only_f1": f1_score(y_val, final, labels=["respond_only"], average="macro", zero_division=0),
            "changed_count": int((final != adv_pred).sum()),
            "candidate_count": int(candidate.sum()),
            "used_count": int(use.sum()),
            "protected_count": int(protect.sum()),
        }
        sweep.append(row)
        if best is None or row["overall_macro_f1"] > best["overall_macro_f1"]:
            best = row | {"pred": final}
    pd.DataFrame(sweep).to_csv(out / "deployable_sweep.csv", index=False)
    changed_matrix(y_val, adv_pred, best["pred"]).to_csv(out / "changed_matrix.csv", index=False)

    false_rows = []
    final_best = best["pred"]
    for p, idx in enumerate(val_idx):
        if adv_pred[p] != final_best[p] and final_best[p] != y_val[p]:
            false_rows.append(
                {
                    "id": samples[idx]["id"],
                    "true": y_val[p],
                    "before": adv_pred[p],
                    "after": final_best[p],
                    "prompt": safe_text(samples[idx].get("current_prompt"), 300),
                }
            )
        if len(false_rows) >= 40:
            break
    lines = ["# N3 false override examples", ""]
    for row in false_rows:
        lines.append(f"- `{row['id']}` true=`{row['true']}` before=`{row['before']}` after=`{row['after']}` prompt={row['prompt']}")
    (out / "false_override_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    joblib.dump({"triad": (tri_vec, tri_model), "comm4": (comm_vec, comm_model), "classes": {"triad": TRIAD, "comm": COMM}}, out / "comm_specialists.pkl", compress=3)
    summary = [
        "# N3 Communication Triad Specialist",
        "",
        f"- train triad rows: `{len(train_triad)}`",
        f"- train comm rows: `{len(train_comm)}`",
        f"- triad isolated Macro-F1: `{tri_iso:.6f}`",
        f"- comm4 isolated Macro-F1: `{comm_iso:.6f}`",
        f"- advanced_full overall Macro-F1 참고값: `{base_macro:.6f}`",
        f"- best deployable tau: `{best['tau']}`",
        f"- best deployable Macro-F1: `{best['overall_macro_f1']:.6f}`",
        f"- delta vs advanced_full: `{best['delta_vs_advanced_full']:.6f}`",
        f"- changed_count: `{best['changed_count']}`",
        f"- respond_only F1 at best: `{best['respond_only_f1']:.6f}`",
        "",
        "Note: deployable delta uses full-data advanced router as base and is a proxy.",
    ]
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {"tri_iso": tri_iso, "comm_iso": comm_iso, "best": best}


def load_transformer_fold0(val_idx, samples):
    ids_path = MDEBERTA_OOF / "fold_0_ids.txt"
    probs_path = MDEBERTA_OOF / "fold_0_probs.npy"
    if not ids_path.exists() or not probs_path.exists():
        return None
    ids = ids_path.read_text(encoding="utf-8").splitlines()
    probs = np.load(probs_path)
    row_of_id = {sid: i for i, sid in enumerate(ids)}
    order = [row_of_id[samples[i]["id"]] for i in val_idx]
    return probs[order]


def run_n4(samples, val_idx, y_val, adv):
    out = REPORT_ROOT / "n4_candidate_gating"
    out.mkdir(parents=True, exist_ok=True)
    probs = load_transformer_fold0(val_idx, samples)
    if probs is None:
        raise FileNotFoundError(f"missing transformer fold0 probs in {MDEBERTA_OOF}")
    tf_pred = np.array([ALL_CLASSES[i] for i in probs.argmax(axis=1)], dtype=object)
    tf_conf = probs.max(axis=1)
    tf_margin = proba_margin(probs)
    adv_pred = adv["pred"]
    priority = np.array([v4_prefilter_priority(samples[idx], adv_pred[p]) for p, idx in enumerate(val_idx)], dtype=np.float32)
    rank = np.argsort(-priority)

    base_macro = macro(y_val, adv_pred)
    tf_direct_macro = macro(y_val, tf_pred)
    all_hybrid = adv_pred.copy()
    mask_all = np.array([(p in OVERRIDE_ACTIONS) for p in tf_pred], dtype=bool)
    all_hybrid[mask_all] = tf_pred[mask_all]
    hybrid_all_macro = macro(y_val, all_hybrid)

    rows = []
    for k in K_VALUES:
        kk = min(k, len(val_idx))
        selected = np.zeros(len(val_idx), dtype=bool)
        selected[rank[:kk]] = True
        override = selected & np.array([(p in OVERRIDE_ACTIONS) for p in tf_pred], dtype=bool)
        final = adv_pred.copy()
        final[override] = tf_pred[override]
        changed = final != adv_pred
        profit = (tf_pred == y_val).astype(int) - (adv_pred == y_val).astype(int)
        rows.append(
            {
                "k": kk,
                "overall_macro_f1": macro(y_val, final),
                "delta_vs_advanced_full": macro(y_val, final) - base_macro,
                "selected_count": int(selected.sum()),
                "override_count": int(override.sum()),
                "changed_count": int(changed.sum()),
                "positive_profit_count": int(((profit > 0) & selected).sum()),
                "negative_profit_count": int(((profit < 0) & selected).sum()),
                "net_profit": int(profit[selected].sum()),
                "candidate_positive_rate": float(((profit > 0) & selected).sum() / max(selected.sum(), 1)),
                "candidate_negative_rate": float(((profit < 0) & selected).sum() / max(selected.sum(), 1)),
                "estimated_runtime_min": round(6.05 * kk / 20000.0, 2),
            }
        )
    rank_curve = pd.DataFrame(rows)
    rank_curve.to_csv(out / "rank_curve.csv", index=False)
    best_k = int(rank_curve.sort_values("overall_macro_f1", ascending=False).iloc[0]["k"])
    selected_best = np.zeros(len(val_idx), dtype=bool)
    selected_best[rank[:best_k]] = True
    override_best = selected_best & np.array([(p in OVERRIDE_ACTIONS) for p in tf_pred], dtype=bool)
    final_best = adv_pred.copy()
    final_best[override_best] = tf_pred[override_best]
    changed_matrix(y_val, adv_pred, final_best).to_csv(out / "changed_matrix.csv", index=False)

    pd.DataFrame(
        [
            {
                "advanced_pred": action,
                "total_rows": int((adv_pred == action).sum()),
                "selected_rows_k20000": int(((adv_pred == action) & selected_best).sum()),
                "selected_ratio_kbest": float(((adv_pred == action) & selected_best).sum() / max((adv_pred == action).sum(), 1)),
            }
            for action in ALL_CLASSES
        ]
    ).to_csv(out / "coverage_by_advanced_pred.csv", index=False)
    pd.DataFrame(
        [
            {
                "true_label": action,
                "total_rows": int((y_val == action).sum()),
                "selected_rows_kbest": int(((y_val == action) & selected_best).sum()),
                "selected_ratio_kbest": float(((y_val == action) & selected_best).sum() / max((y_val == action).sum(), 1)),
            }
            for action in ALL_CLASSES
        ]
    ).to_csv(out / "coverage_by_true_label.csv", index=False)

    profit_rows = []
    profit = (tf_pred == y_val).astype(int) - (adv_pred == y_val).astype(int)
    for action in ALL_CLASSES:
        mask = selected_best & (tf_pred == action)
        if not mask.any():
            continue
        profit_rows.append(
            {
                "transformer_pred": action,
                "override_count": int(mask.sum()),
                "transformer_precision": float((tf_pred[mask] == y_val[mask]).mean()),
                "advanced_precision_same_rows": float((adv_pred[mask] == y_val[mask]).mean()),
                "net_delta": int(profit[mask].sum()),
                "avg_transformer_conf": float(tf_conf[mask].mean()),
                "avg_transformer_margin": float(tf_margin[mask].mean()),
            }
        )
    pd.DataFrame(profit_rows).sort_values("net_delta", ascending=False).to_csv(out / "override_precision_by_action.csv", index=False)

    pd.DataFrame(
        {
            "id": [samples[i]["id"] for i in val_idx],
            "true": y_val,
            "advanced_pred": adv_pred,
            "transformer_pred": tf_pred,
            "advanced_margin": adv["margin"],
            "transformer_conf": tf_conf,
            "transformer_margin": tf_margin,
            "priority": priority,
            "selected_kbest": selected_best.astype(int),
            "profit": profit,
        }
    ).to_csv(out / "profitability_by_candidate.csv", index=False)

    pd.DataFrame(
        [
            {"metric": "advanced_full_macro_f1", "value": base_macro},
            {"metric": "transformer_direct_macro_f1", "value": tf_direct_macro},
            {"metric": "hybrid_all_override_actions_macro_f1", "value": hybrid_all_macro},
            {"metric": "best_rank_curve_k", "value": best_k},
            {"metric": "best_rank_curve_macro_f1", "value": float(rank_curve["overall_macro_f1"].max())},
        ]
    ).to_csv(out / "summary_metrics.csv", index=False)
    best_row = rank_curve.sort_values("overall_macro_f1", ascending=False).iloc[0].to_dict()
    summary = [
        "# N4 Candidate Gating Forensic",
        "",
        f"- validation rows: `{len(val_idx)}`",
        f"- advanced_full Macro-F1 참고값: `{base_macro:.6f}`",
        f"- transformer direct Macro-F1: `{tf_direct_macro:.6f}`",
        f"- hybrid all override-actions Macro-F1: `{hybrid_all_macro:.6f}`",
        f"- best rank-curve K: `{best_k}`",
        f"- best rank-curve Macro-F1: `{best_row['overall_macro_f1']:.6f}`",
        f"- delta vs advanced_full: `{best_row['delta_vs_advanced_full']:.6f}`",
        f"- changed_count at best K: `{int(best_row['changed_count'])}`",
        f"- estimated runtime at best K: `{best_row['estimated_runtime_min']} min`",
        "",
        "Note: candidate ranking uses the current `v4_prefilter_priority` and full-data advanced router as base.",
    ]
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {"base_macro": base_macro, "tf_direct": tf_direct_macro, "hybrid_all": hybrid_all_macro, "best": best_row}


def append_research(n2, n3, n4):
    text = f"""

## N2/N3/N4 Cheap-Proxy Forensic Result

### Setup
- validation: `fold0` from `pipeline_v4/folds/fold_assignments.csv`
- no new transformer training
- base reference: current full-data `advanced_router.pkl` predictions on fold0. This is marked as `advanced_full` because it is not an OOF base.
- transformer reference for N4: `mdeberta384_v2_384_5e` fold0 probabilities.

### N2 Inspect Specialist

| Metric | Value |
|---|---:|
| isolated inspect Macro-F1 | `{n2['isolated_f1']:.6f}` |
| advanced_full isolated inspect Macro-F1 참고값 | `{n2['advanced_iso_f1']:.6f}` |
| best deployable tau | `{n2['best']['tau']}` |
| best deployable Macro-F1 | `{n2['best']['overall_macro_f1']:.6f}` |
| delta vs advanced_full | `{n2['best']['delta_vs_advanced_full']:.6f}` |
| changed_count | `{n2['best']['changed_count']}` |

Decision:
- {'soft-pass candidate' if n2['best']['delta_vs_advanced_full'] > 0.0015 else 'reject for submit'}.
- Full details: `reports/n2_inspect_specialist/summary.md`.

### N3 Communication Triad

| Metric | Value |
|---|---:|
| triad isolated Macro-F1 | `{n3['tri_iso']:.6f}` |
| comm4 isolated Macro-F1 | `{n3['comm_iso']:.6f}` |
| best deployable tau | `{n3['best']['tau']}` |
| best deployable Macro-F1 | `{n3['best']['overall_macro_f1']:.6f}` |
| delta vs advanced_full | `{n3['best']['delta_vs_advanced_full']:.6f}` |
| changed_count | `{n3['best']['changed_count']}` |

Decision:
- {'soft-pass candidate' if n3['best']['delta_vs_advanced_full'] > 0.0015 else 'reject for submit'}.
- Full details: `reports/n3_comm_triad/summary.md`.

### N4 Candidate Gating

| Metric | Value |
|---|---:|
| advanced_full Macro-F1 참고값 | `{n4['base_macro']:.6f}` |
| transformer direct Macro-F1 | `{n4['tf_direct']:.6f}` |
| hybrid all override-actions Macro-F1 | `{n4['hybrid_all']:.6f}` |
| best rank-curve K | `{int(n4['best']['k'])}` |
| best rank-curve Macro-F1 | `{n4['best']['overall_macro_f1']:.6f}` |
| delta vs advanced_full | `{n4['best']['delta_vs_advanced_full']:.6f}` |
| estimated runtime at best K | `{n4['best']['estimated_runtime_min']} min` |

Decision:
- Candidate coverage is worth probing further if public runtime remains under 10 minutes.
- Full details: `reports/n4_candidate_gating/summary.md`.
"""
    path = REPORT_ROOT / "n2_n3_n4_research_append.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def main():
    samples = load_samples()
    train_idx, val_idx = load_fold0_split(samples)
    y_val = np.array([samples[i]["action"] for i in val_idx], dtype=object)
    artifact = joblib.load(ADVANCED_MODEL)
    adv = predict_advanced_details([samples[i] for i in val_idx], artifact)
    n2 = run_n2(samples, train_idx, val_idx, y_val, adv["pred"])
    n3 = run_n3(samples, train_idx, val_idx, y_val, adv["pred"])
    n4 = run_n4(samples, val_idx, y_val, adv)
    append_research(n2, n3, n4)
    print(json.dumps({"n2": n2, "n3": n3, "n4": n4}, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
