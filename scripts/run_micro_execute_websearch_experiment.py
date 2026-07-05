import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script import ALL_CLASSES, advanced_open_files, safe_text  # noqa: E402
from scripts.run_inspect_bottleneck_experiments import (  # noqa: E402
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
EXECUTE_PAIR = {"run_bash", "run_tests"}
COMMUNICATE = {"ask_user", "plan_task", "web_search", "respond_only"}
WEB_RISK_CLASSES = ["plan_task", "ask_user", "grep_search"]


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


def write_md(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def has_re(text, patterns):
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def last_action_name(sample):
    last = last_action_turn(sample)
    return str(last.get("name")) if last else "NONE"


def last_blob(sample):
    last = last_action_turn(sample)
    if not last:
        return ""
    return "\n".join(
        [
            safe_text(last.get("name"), 80),
            safe_text(last.get("args"), 800),
            safe_text(last.get("result_summary"), 1200),
        ]
    )


def sample_blob(sample):
    return "\n".join(
        [
            safe_text(sample.get("current_prompt"), 1600),
            last_blob(sample),
            " ".join(advanced_open_files(sample)[:10]),
        ]
    ).lower()


def result_bucket(sample):
    text = last_blob(sample).lower()
    if not text:
        return "none"
    if any(x in text for x in ["traceback", "exception", "error", "failed", "fail", "exit 1", "exit=1", "nonzero"]):
        return "fail_or_error"
    if any(x in text for x in ["passed", "pass", "success", "green", "ok", "exit 0", "exit=0"]):
        return "pass_or_ok"
    return "other"


def bool_int(value):
    return int(bool(value))


def open_file_exts(sample):
    exts = []
    for path in advanced_open_files(sample):
        low = path.lower().replace("\\", "/")
        name = low.rsplit("/", 1)[-1]
        if "." in name:
            exts.append(name.rsplit(".", 1)[-1])
        elif name in {"dockerfile", "makefile"}:
            exts.append(name)
    return exts


def execute_flags(sample):
    text = sample_blob(sample)
    prompt = safe_text(sample.get("current_prompt"), 1600).lower()
    last_action = last_action_name(sample)
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    open_paths = " ".join(advanced_open_files(sample)).lower().replace("\\", "/")

    explicit_test = has_re(
        text,
        [
            r"\bpytest\b",
            r"\bnpm\s+(run\s+)?test\b",
            r"\byarn\s+test\b",
            r"\bpnpm\s+test\b",
            r"\bcargo\s+test\b",
            r"\bgo\s+test\b",
            r"\bjest\b",
            r"\bvitest\b",
            r"\brspec\b",
            r"\btox\b",
        ],
    )
    test_file = has_re(
        text + "\n" + open_paths,
        [
            r"(^|[/\\])tests?([/\\]|$)",
            r"(^|[/\\])e2e([/\\]|$)",
            r"\btest_[\w.-]+\.py\b",
            r"\b[\w.-]+_test\.(py|go|rs)\b",
            r"\b[\w.-]+\.(test|spec)\.(js|jsx|ts|tsx)\b",
        ],
    )
    test_word = explicit_test or test_file or has_re(
        text,
        [
            r"\btests?\b",
            r"\bspecs?\b",
            r"\bsuite\b",
            r"\bunit\b",
            r"\bintegration\b",
            r"\be2e\b",
            r"\bregression\b",
            r"\bcoverage\b",
            r"\bgreen\b",
        ],
    )

    lint = has_re(
        text,
        [
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
            r"\bgo\s+vet\b",
            r"\bstatic\s+(analysis|check)\b",
            r"\bformat\s+check\b",
            r"\btype\s+errors?\b",
        ],
    )
    build = has_re(text, [r"\bbuild\b", r"\bcompile\b", r"\bnpm\s+run\s+build\b", r"\bcargo\s+build\b", r"\bgo\s+build\b"])
    server = has_re(text, [r"\bdev\s+server\b", r"\brunserver\b", r"\bserve\b", r"\bserver\b", r"\bnpm\s+run\s+dev\b"])
    install = has_re(text, [r"\binstall\b", r"\bnpm\s+install\b", r"\bpip\s+install\b", r"\bpoetry\s+install\b", r"\bbundle\s+install\b"])
    kubectl = has_re(text, [r"\bkubectl\b", r"\brollout\s+status\b", r"\bdocker\b"])
    explicit_bash = build or server or install or kubectl or has_re(
        text,
        [
            r"\bmake\b",
            r"\bsh\s+[\w./-]+",
            r"\bbash\s+[\w./-]+",
            r"\bpython\s+[\w./-]+\.py\b",
            r"\bnode\s+[\w./-]+\.js\b",
            r"\bnpm\s+run\s+(?!test\b)[\w:-]+",
        ],
    )

    last_edit = last_action in {"edit_file", "apply_patch", "write_file"}
    bucket = result_bucket(sample)
    ci = safe_text(ws.get("last_ci_status"), 80).lower()
    config_exts = {"json", "toml", "yaml", "yml", "ini", "cfg", "lock", "dockerfile", "makefile"}
    exts = set(open_file_exts(sample))
    return {
        "EX_HAS_TEST_WORD": test_word,
        "EX_HAS_EXPLICIT_TEST_COMMAND": explicit_test,
        "EX_HAS_TEST_FILE_PATH": test_file,
        "EX_HAS_SPEC_WORD": has_re(text, [r"\bspecs?\b", r"\bsuite\b"]),
        "EX_HAS_GREEN_PASS_WORD": has_re(text, [r"\bgreen\b", r"\bpass(ed)?\b"]),
        "EX_HAS_BUILD_WORD": build,
        "EX_HAS_SERVER_WORD": server,
        "EX_HAS_INSTALL_WORD": install,
        "EX_HAS_EXPLICIT_BASH_COMMAND": explicit_bash,
        "EX_HAS_RUNSERVER": has_re(text, [r"\brunserver\b", r"\bnpm\s+run\s+dev\b"]),
        "EX_HAS_ROLLOUT_OR_KUBECTL": kubectl,
        "EX_HAS_LINT_WORD": lint,
        "EX_HAS_TYPECHECK_WORD": has_re(text, [r"\btype\s*check\b", r"\btypecheck\b", r"\btsc\b", r"\bmypy\b", r"\bpyright\b"]),
        "EX_HAS_STATIC_ANALYSIS_WORD": has_re(text, [r"\bstatic\s+(analysis|check)\b", r"\blint\b", r"\bclippy\b", r"\bgo\s+vet\b"]),
        "EX_LINT_PROTECT": lint,
        "LAST_ACTION_EDIT_OR_PATCH_OR_WRITE": last_edit,
        "LAST_ACTION_RUN_TESTS": last_action == "run_tests",
        "LAST_ACTION_RUN_BASH": last_action == "run_bash",
        "LAST_ACTION_LINT": last_action == "lint_or_typecheck",
        "LAST_RESULT_PASS": bucket == "pass_or_ok",
        "LAST_RESULT_FAIL_OR_ERROR": bucket == "fail_or_error",
        "CI_FAILED": "fail" in ci or "error" in ci,
        "CI_PASSED": "pass" in ci or "success" in ci or "green" in ci,
        "OPEN_HAS_TEST_FILE": test_file,
        "OPEN_HAS_CONFIG_FILE": bool(exts & config_exts),
    }


def execute_rule_candidate(sample, rule_name):
    flags = execute_flags(sample)
    lint = flags["EX_LINT_PROTECT"]
    bashish = flags["EX_HAS_BUILD_WORD"] or flags["EX_HAS_SERVER_WORD"] or flags["EX_HAS_INSTALL_WORD"] or flags["EX_HAS_ROLLOUT_OR_KUBECTL"]
    if rule_name == "R_test_explicit":
        if not lint and (flags["EX_HAS_EXPLICIT_TEST_COMMAND"] or flags["EX_HAS_TEST_FILE_PATH"]):
            return "run_tests"
    elif rule_name == "R_test_contextual":
        if flags["EX_HAS_TEST_WORD"] and flags["LAST_ACTION_EDIT_OR_PATCH_OR_WRITE"] and not lint and not bashish:
            return "run_tests"
    elif rule_name == "R_bash_explicit":
        if not lint and flags["EX_HAS_EXPLICIT_BASH_COMMAND"] and not flags["EX_HAS_EXPLICIT_TEST_COMMAND"]:
            return "run_bash"
    elif rule_name == "R_bash_contextual":
        if bashish and not flags["EX_HAS_TEST_FILE_PATH"] and not lint and not flags["EX_HAS_EXPLICIT_TEST_COMMAND"]:
            return "run_bash"
    elif rule_name == "R_execute_priority":
        if lint:
            return None
        if flags["EX_HAS_EXPLICIT_TEST_COMMAND"] or flags["EX_HAS_TEST_FILE_PATH"]:
            return "run_tests"
        if flags["EX_HAS_EXPLICIT_BASH_COMMAND"]:
            return "run_bash"
        if flags["EX_HAS_TEST_WORD"] and flags["LAST_ACTION_EDIT_OR_PATCH_OR_WRITE"] and not bashish:
            return "run_tests"
        if bashish:
            return "run_bash"
    return None


def flag_summary(flags):
    return " ".join(key for key, value in flags.items() if value)


def classwise_f1(y_true, pred, labels=ALL_CLASSES):
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    return {
        label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, label in enumerate(labels)
    }


def stable_half(sample_id):
    sid = session_of(sample_id)
    digest = hashlib.md5(sid.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2


def half_delta(y_true, base_pred, new_pred, half_mask):
    if int(half_mask.sum()) == 0:
        return 0.0
    return score_macro(y_true[half_mask], new_pred[half_mask]) - score_macro(y_true[half_mask], base_pred[half_mask])


def false_positive_summary(y_true, pred, fire_mask):
    fp_labels = y_true[fire_mask & (pred != y_true)]
    return json.dumps(Counter(fp_labels).most_common(6), ensure_ascii=False)


def evaluate_candidate_override(name, y_true, base_pred, candidates, scope, half_a, half_b, target_classes, delta_classes):
    fire_mask = scope & np.asarray([candidate is not None for candidate in candidates], dtype=bool)
    new_pred = np.asarray(base_pred, dtype=object).copy()
    for i, candidate in enumerate(candidates):
        if fire_mask[i]:
            new_pred[i] = candidate

    changed = fire_mask & (new_pred != base_pred)
    before_fire = base_pred[fire_mask] == y_true[fire_mask]
    after_fire = new_pred[fire_mask] == y_true[fire_mask]
    before_changed = base_pred[changed] == y_true[changed]
    after_changed = new_pred[changed] == y_true[changed]
    base_macro = score_macro(y_true, base_pred)
    new_macro = score_macro(y_true, new_pred)
    base_cw = classwise_f1(y_true, base_pred)
    new_cw = classwise_f1(y_true, new_pred)
    true_target = np.isin(y_true, list(target_classes))
    corrected_target = fire_mask & (new_pred == y_true) & true_target

    row = {
        "name": name,
        "support": int(fire_mask.sum()),
        "override_count": int(changed.sum()),
        "precision": float(after_fire.mean()) if len(after_fire) else 0.0,
        "advanced_precision_same_rows": float(before_fire.mean()) if len(before_fire) else 0.0,
        "override_precision": float(after_changed.mean()) if len(after_changed) else 0.0,
        "base_precision_changed_rows": float(before_changed.mean()) if len(before_changed) else 0.0,
        "recall_on_target_class": float(corrected_target.sum() / max(int(true_target.sum()), 1)),
        "net_gain_rows": int(after_fire.sum() - before_fire.sum()) if len(after_fire) else 0,
        "net_gain_changed_rows": int(after_changed.sum() - before_changed.sum()) if len(after_changed) else 0,
        "base_macro_f1": base_macro,
        "new_macro_f1": new_macro,
        "macro_f1_delta": new_macro - base_macro,
        "halfA_delta": half_delta(y_true, base_pred, new_pred, half_a),
        "halfB_delta": half_delta(y_true, base_pred, new_pred, half_b),
        "false_positive_top_classes": false_positive_summary(y_true, new_pred, fire_mask),
    }
    for cls in delta_classes:
        row[f"{cls}_f1_delta"] = new_cw[cls]["f1"] - base_cw[cls]["f1"]
    return row, new_pred, fire_mask


def execute_serializer(sample):
    flags = execute_flags(sample)
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    actions = [h.get("name") for h in sample.get("history", []) or [] if h.get("role") == "assistant_action" and h.get("name")]
    return "\n".join(
        [
            "[NOW] " + safe_text(sample.get("current_prompt"), 1200),
            "[EXEC_FLAG] " + " ".join(f"{k}={bool_int(v)}" for k, v in sorted(flags.items())),
            f"[LAST] action={last_action_name(sample)} result_bucket={result_bucket(sample)} details={last_blob(sample)[:900]}",
            "[SEQ] " + (" > ".join(actions[-6:]) if actions else "none"),
            "[FILES] open=" + (" ".join(advanced_open_files(sample)[:10]) or "none") + " ext=" + " ".join(open_file_exts(sample)),
            f"[META] ci={safe_text(ws.get('last_ci_status'), 80)} turn={meta.get('turn_index')} dirty={ws.get('git_dirty')}",
            "[BASE] " + fast_router_text(sample),
        ]
    )


def train_pair_resolver(train_samples, y_train, max_features):
    pair_idx = np.where(np.isin(y_train, list(EXECUTE_PAIR)))[0]
    texts = [execute_serializer(train_samples[i]) for i in pair_idx]
    vectorizer = build_vectorizer(max_features=max_features, min_df=2, char_heavy=False)
    x = vectorizer.fit_transform(texts)
    model = LogisticRegression(
        C=2.0,
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(x, y_train[pair_idx])
    return vectorizer, model, len(pair_idx)


def predict_pair_resolver(vectorizer, model, samples):
    x = vectorizer.transform([execute_serializer(sample) for sample in samples])
    probs = model.predict_proba(x)
    idx = probs.argmax(axis=1)
    pred = np.asarray([str(model.classes_[i]) for i in idx], dtype=object)
    conf = probs.max(axis=1)
    return pred, conf


def web_flags(sample):
    text = sample_blob(sample)
    prompt = safe_text(sample.get("current_prompt"), 1600).lower()
    path_file = has_re(
        text,
        [
            r"\b(file|path|directory|folder|repo|codebase)\b",
            r"[\w./\\-]+\.(py|tsx|ts|js|jsx|go|rs|java|kt|sql|json|yaml|yml|toml|md|txt|lock)\b",
            r"\b(package\.json|requirements\.txt|go\.mod|cargo\.toml|dockerfile)\b",
        ],
    )
    internal = has_re(
        text,
        [
            r"\b(import|definition|defined|where\s+is|grep|usage|usages|occurrences?|symbol|call\s+sites?)\b",
            r"\bin\s+(this|the)\s+(repo|repository|codebase|project)\b",
            r"\bfrom\s+.*\s+import\b",
        ],
    )
    plan = has_re(prompt, [r"\b(plan|steps?|break\s+down|roadmap|sequence|approach|strategy)\b"])
    ask_uncertainty = has_re(prompt, [r"\b(not\s+sure|should\s+i|which\s+one|choose|clarify|unclear|do\s+you\s+want)\b"])
    respond_summary = has_re(prompt, [r"\b(summarize|summary|recap|wrap\s+up|brief|explain|tell\s+me)\b"])
    explicit_web = has_re(
        prompt,
        [
            r"\bweb\s+search\b",
            r"\bsearch\s+(the\s+)?web\b",
            r"\blook\s+up\s+(online|on\s+the\s+web)\b",
            r"\bonline\b",
            r"\binternet\b",
            r"\bgoogle\b",
            r"\bbrowse\b",
        ],
    )
    latest_docs = has_re(
        prompt,
        [
            r"\blatest\b",
            r"\bcurrent\b",
            r"\bup[- ]to[- ]date\b",
            r"\brecent\b",
            r"\btoday\b",
            r"\bofficial\s+(docs|documentation)\b",
            r"\bdocs?\b",
            r"\bdocumentation\b",
            r"\brelease\s+notes?\b",
            r"\bchangelog\b",
            r"\bversion\s+compat",
            r"\bdeprecated\b",
        ],
    )
    best_practice = has_re(
        prompt,
        [
            r"\brecommended\b",
            r"\brecommendation\b",
            r"\bbest\s+practices?\b",
            r"\bsane\s+default\b",
            r"\bindustry\s+standard\b",
            r"\bwhich\s+(library|package|api|model)\b",
        ],
    )
    paper = has_re(
        prompt,
        [
            r"\boriginal\s+paper\b",
            r"\bpaper\s+formulation\b",
            r"\barxiv\b",
            r"\bstandard\s+formulation\b",
            r"\bsinusoidal\b",
            r"\bresearch\s+paper\b",
        ],
    )
    return {
        "W_POS_EXPLICIT_WEB": explicit_web,
        "W_POS_LATEST_DOCS": latest_docs,
        "W_POS_RECOMMENDED_BESTPRACTICE": best_practice,
        "W_POS_PAPER_FORMULATION": paper,
        "W_NEG_INTERNAL_CODE_SEARCH": internal,
        "W_NEG_PATH_FILE_SYMBOL": path_file,
        "W_NEG_PLAN_TASK": plan,
        "W_NEG_ASK_UNCERTAINTY": ask_uncertainty,
        "W_NEG_RESPOND_SUMMARY": respond_summary,
        "W_ANY_NEGATIVE": internal or path_file or plan or ask_uncertainty or respond_summary,
    }


def web_rule_candidate(sample, rule_name):
    flags = web_flags(sample)
    if rule_name == "W1_explicit_web":
        if flags["W_POS_EXPLICIT_WEB"] and not flags["W_ANY_NEGATIVE"]:
            return "web_search"
    elif rule_name == "W2_latest_docs":
        if flags["W_POS_LATEST_DOCS"] and not flags["W_NEG_INTERNAL_CODE_SEARCH"] and not flags["W_NEG_PATH_FILE_SYMBOL"]:
            return "web_search"
    elif rule_name == "W3_recommended_bestpractice":
        if flags["W_POS_RECOMMENDED_BESTPRACTICE"] and not flags["W_NEG_PLAN_TASK"] and not flags["W_NEG_INTERNAL_CODE_SEARCH"]:
            return "web_search"
    elif rule_name == "W4_paper_formulation":
        if flags["W_POS_PAPER_FORMULATION"] and not flags["W_NEG_INTERNAL_CODE_SEARCH"]:
            return "web_search"
    elif rule_name == "W5_external_knowledge_combo":
        if (
            flags["W_POS_LATEST_DOCS"]
            or flags["W_POS_RECOMMENDED_BESTPRACTICE"]
            or flags["W_POS_PAPER_FORMULATION"]
        ) and not flags["W_NEG_INTERNAL_CODE_SEARCH"] and not flags["W_NEG_PATH_FILE_SYMBOL"] and not flags["W_NEG_RESPOND_SUMMARY"]:
            return "web_search"
    return None


def scope_masks(base_pred, infos, samples):
    top2_pair = np.asarray([info["top2"] for info in infos], dtype=object)
    pred_pair = np.isin(base_pred, list(EXECUTE_PAIR))
    top2_has_pair = pred_pair | np.isin(top2_pair, list(EXECUTE_PAIR))
    exact_pair = np.asarray(
        [set([str(info["top1"]), str(info["top2"])]) == EXECUTE_PAIR for info in infos],
        dtype=bool,
    )
    no_lint = np.asarray([not execute_flags(sample)["EX_LINT_PROTECT"] for sample in samples], dtype=bool)
    web_top2 = np.asarray([info["top2"] == "web_search" or info["top1"] == "web_search" for info in infos], dtype=bool)
    return {
        "all": np.ones(len(base_pred), dtype=bool),
        "pred_pair_no_lint": pred_pair & no_lint,
        "top2_pair_no_lint": top2_has_pair & no_lint,
        "exact_pair_no_lint": exact_pair & no_lint,
        "pred_execute_no_lint": np.isin(base_pred, EXECUTE) & no_lint,
        "pred_communicate": np.isin(base_pred, list(COMMUNICATE)),
        "pred_or_top2_web": (base_pred == "web_search") | web_top2,
        "communicate_or_top2_web": np.isin(base_pred, list(COMMUNICATE)) | web_top2,
    }


def adoption_execute_rule(row):
    return (
        row["precision"] >= 0.85
        and row["support"] >= 50
        and row["net_gain_rows"] >= 20
        and row["macro_f1_delta"] >= 0.001
        and row["lint_or_typecheck_f1_delta"] >= -0.003
        and row["halfA_delta"] > 0
        and row["halfB_delta"] > 0
    )


def adoption_execute_resolver(row):
    return (
        row["net_gain_rows"] >= 20
        and row["precision"] >= row["advanced_precision_same_rows"] + 0.03
        and row["macro_f1_delta"] >= 0.0015
        and row["lint_or_typecheck_f1_delta"] >= -0.003
        and row["halfA_delta"] > 0
        and row["halfB_delta"] > 0
    )


def adoption_web_hard(row):
    return (
        row["precision"] >= 0.80
        and row["support"] >= 30
        and row["net_gain_rows"] >= 10
        and row["macro_f1_delta"] >= 0.001
        and row["plan_task_f1_delta"] >= -0.003
        and row["ask_user_f1_delta"] >= -0.003
        and row["grep_search_f1_delta"] >= -0.003
        and row["halfA_delta"] > 0
        and row["halfB_delta"] > 0
    )


def medium_candidate_boost(row):
    return row["precision"] >= 0.60 and row["support"] >= 50


def top_row(rows, predicate=None):
    candidates = [row for row in rows if predicate(row)] if predicate else list(rows)
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["macro_f1_delta"], row["net_gain_rows"], row["precision"]))


def apply_row_config(base_pred, val_samples, row, rule_func):
    out = np.asarray(base_pred, dtype=object).copy()
    if row is None:
        return out
    rule_name = row["rule_name"]
    scope_name = row["application_subset"]
    scopes = row["_scopes"]
    scope = scopes[scope_name]
    for i, sample in enumerate(val_samples):
        if not scope[i]:
            continue
        candidate = rule_func(sample, rule_name)
        if candidate is not None:
            out[i] = candidate
    return out


def run_experiment(args):
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    execute_dir = out_root / "execute_pair"
    web_dir = out_root / "web_search"
    combined_dir = out_root / "combined"
    samples = read_jsonl(data_dir / "train.jsonl")
    labels = load_labels(data_dir / "train_labels.csv")
    y = np.asarray([labels[sample["id"]] for sample in samples], dtype=object)
    train_idx, val_idx, split_name = load_fold_split(samples, args.fold_file, fold=0)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    router = train_fast_router(
        samples,
        y,
        train_idx,
        Path(args.router_cache_root),
        args.router_features,
        refresh=args.refresh_router,
    )
    base_pred, _, infos = predict_fast_router(val_samples, router)
    base_macro = score_macro(y_val, base_pred)
    base_acc = float((base_pred == y_val).mean())
    halves = np.asarray([stable_half(sample["id"]) for sample in val_samples], dtype=np.int64)
    half_a = halves == 0
    half_b = halves == 1
    scopes = scope_masks(base_pred, infos, val_samples)
    print(f"loaded train={len(train_idx)} val={len(val_idx)} base_macro={base_macro:.6f}", flush=True)

    class_rows = []
    base_cw = classwise_f1(y_val, base_pred)
    for cls in EXECUTE:
        class_rows.append({"class": cls, **base_cw[cls]})
    write_csv(execute_dir / "execute_classwise_f1.csv", class_rows, ["class", "precision", "recall", "f1", "support"])

    confusion_pairs = [
        ("run_tests", "run_bash"),
        ("run_bash", "run_tests"),
        ("lint_or_typecheck", "run_tests"),
        ("lint_or_typecheck", "run_bash"),
        ("run_tests", "lint_or_typecheck"),
        ("run_bash", "lint_or_typecheck"),
    ]
    confusion_rows = [
        {"true_label": true, "pred_label": pred, "count": int(((y_val == true) & (base_pred == pred)).sum())}
        for true, pred in confusion_pairs
    ]
    write_csv(execute_dir / "run_tests_vs_run_bash_confusion.csv", confusion_rows, ["true_label", "pred_label", "count"])

    error_rows = []
    seen_pair_count = Counter()
    for sample, truth, pred in zip(val_samples, y_val, base_pred):
        pair = (truth, pred)
        if pair not in confusion_pairs or seen_pair_count[pair] >= 50:
            continue
        seen_pair_count[pair] += 1
        flags = execute_flags(sample)
        error_rows.append(
            {
                "id": sample["id"],
                "current_prompt": safe_text(sample.get("current_prompt"), 500),
                "last_action": last_action_name(sample),
                "last_result_summary": safe_text(last_blob(sample), 500),
                "open_files": " ".join(advanced_open_files(sample)[:8]),
                "base_pred": pred,
                "true_label": truth,
                "flag_summary": flag_summary(flags),
            }
        )
    write_csv(
        execute_dir / "execute_error_cases.csv",
        error_rows,
        ["id", "current_prompt", "last_action", "last_result_summary", "open_files", "base_pred", "true_label", "flag_summary"],
    )

    flag_rows = []
    all_exec_flags = [execute_flags(sample) for sample in val_samples]
    for flag in sorted(all_exec_flags[0]):
        mask = np.asarray([flags[flag] for flags in all_exec_flags], dtype=bool)
        counts = Counter(y_val[mask])
        flag_rows.append(
            {
                "flag": flag,
                "count": int(mask.sum()),
                "label_counts": json.dumps(counts.most_common(), ensure_ascii=False),
                "precision_run_tests": float((y_val[mask] == "run_tests").mean()) if int(mask.sum()) else 0.0,
                "precision_run_bash": float((y_val[mask] == "run_bash").mean()) if int(mask.sum()) else 0.0,
                "precision_lint_or_typecheck": float((y_val[mask] == "lint_or_typecheck").mean()) if int(mask.sum()) else 0.0,
                "base_accuracy_when_flag": float((base_pred[mask] == y_val[mask]).mean()) if int(mask.sum()) else 0.0,
            }
        )
    write_csv(
        execute_dir / "execute_flag_distribution.csv",
        flag_rows,
        [
            "flag",
            "count",
            "label_counts",
            "precision_run_tests",
            "precision_run_bash",
            "precision_lint_or_typecheck",
            "base_accuracy_when_flag",
        ],
    )

    execute_rule_rows = []
    execute_rules = ["R_test_explicit", "R_test_contextual", "R_bash_explicit", "R_bash_contextual", "R_execute_priority"]
    execute_scopes = ["all", "pred_pair_no_lint", "top2_pair_no_lint", "exact_pair_no_lint", "pred_execute_no_lint"]
    for rule_name in execute_rules:
        candidates = [execute_rule_candidate(sample, rule_name) for sample in val_samples]
        for scope_name in execute_scopes:
            row, _, _ = evaluate_candidate_override(
                f"{rule_name}__{scope_name}",
                y_val,
                base_pred,
                candidates,
                scopes[scope_name],
                half_a,
                half_b,
                EXECUTE_PAIR,
                EXECUTE,
            )
            row["rule_name"] = rule_name
            row["application_subset"] = scope_name
            row["adopt_hard_override"] = adoption_execute_rule(row)
            row["candidate_boost_only"] = (not row["adopt_hard_override"]) and row["precision"] >= 0.65 and row["support"] >= 50
            execute_rule_rows.append(row)
    execute_rule_fields = [
        "name",
        "rule_name",
        "application_subset",
        "support",
        "override_count",
        "precision",
        "advanced_precision_same_rows",
        "override_precision",
        "base_precision_changed_rows",
        "recall_on_target_class",
        "net_gain_rows",
        "net_gain_changed_rows",
        "base_macro_f1",
        "new_macro_f1",
        "macro_f1_delta",
        "run_bash_f1_delta",
        "run_tests_f1_delta",
        "lint_or_typecheck_f1_delta",
        "halfA_delta",
        "halfB_delta",
        "false_positive_top_classes",
        "adopt_hard_override",
        "candidate_boost_only",
    ]
    write_csv(execute_dir / "execute_rule_results.csv", execute_rule_rows, execute_rule_fields)

    vectorizer, pair_model, pair_train_rows = train_pair_resolver(train_samples, y_train, args.pair_features)
    pair_pred, pair_conf = predict_pair_resolver(vectorizer, pair_model, val_samples)
    resolver_rows = []
    for scope_name in ["pred_pair_no_lint", "top2_pair_no_lint", "exact_pair_no_lint"]:
        for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
            candidates = [str(pair_pred[i]) if pair_conf[i] >= threshold else None for i in range(len(val_samples))]
            row, _, _ = evaluate_candidate_override(
                f"pair_logreg__{scope_name}__thr{threshold:.2f}",
                y_val,
                base_pred,
                candidates,
                scopes[scope_name],
                half_a,
                half_b,
                EXECUTE_PAIR,
                EXECUTE,
            )
            true_pair_mask = np.isin(y_val, list(EXECUTE_PAIR))
            row.update(
                {
                    "model": "tfidf_word_char_logreg",
                    "application_subset": scope_name,
                    "threshold": threshold,
                    "pair_train_rows": pair_train_rows,
                    "true_pair_isolated_f1": float(
                        f1_score(y_val[true_pair_mask], pair_pred[true_pair_mask], labels=sorted(EXECUTE_PAIR), average="macro", zero_division=0)
                    ),
                    "adopt_resolver": adoption_execute_resolver(row),
                    "candidate_boost_only": (not adoption_execute_resolver(row)) and row["precision"] >= 0.65 and row["support"] >= 50,
                }
            )
            resolver_rows.append(row)
    resolver_fields = [
        "name",
        "model",
        "application_subset",
        "threshold",
        "pair_train_rows",
        "support",
        "override_count",
        "precision",
        "advanced_precision_same_rows",
        "override_precision",
        "base_precision_changed_rows",
        "recall_on_target_class",
        "net_gain_rows",
        "net_gain_changed_rows",
        "base_macro_f1",
        "new_macro_f1",
        "macro_f1_delta",
        "run_bash_f1_delta",
        "run_tests_f1_delta",
        "lint_or_typecheck_f1_delta",
        "true_pair_isolated_f1",
        "halfA_delta",
        "halfB_delta",
        "false_positive_top_classes",
        "adopt_resolver",
        "candidate_boost_only",
    ]
    write_csv(execute_dir / "execute_pair_resolver.csv", resolver_rows, resolver_fields)

    web_flag_rows = []
    all_web_flags = [web_flags(sample) for sample in val_samples]
    for flag in sorted(all_web_flags[0]):
        mask = np.asarray([flags[flag] for flags in all_web_flags], dtype=bool)
        counts = Counter(y_val[mask])
        web_flag_rows.append(
            {
                "flag": flag,
                "support": int(mask.sum()),
                "precision_web_search": float((y_val[mask] == "web_search").mean()) if int(mask.sum()) else 0.0,
                "base_accuracy_when_flag": float((base_pred[mask] == y_val[mask]).mean()) if int(mask.sum()) else 0.0,
                "label_counts": json.dumps(counts.most_common(), ensure_ascii=False),
            }
        )
    write_csv(web_dir / "web_rule_audit.csv", web_flag_rows, ["flag", "support", "precision_web_search", "base_accuracy_when_flag", "label_counts"])

    web_rule_rows = []
    web_rules = [
        "W1_explicit_web",
        "W2_latest_docs",
        "W3_recommended_bestpractice",
        "W4_paper_formulation",
        "W5_external_knowledge_combo",
    ]
    web_scopes = ["all", "pred_communicate", "pred_or_top2_web", "communicate_or_top2_web"]
    for rule_name in web_rules:
        candidates = [web_rule_candidate(sample, rule_name) for sample in val_samples]
        for scope_name in web_scopes:
            row, _, _ = evaluate_candidate_override(
                f"{rule_name}__{scope_name}",
                y_val,
                base_pred,
                candidates,
                scopes[scope_name],
                half_a,
                half_b,
                {"web_search"},
                ["web_search", *WEB_RISK_CLASSES],
            )
            row["rule_name"] = rule_name
            row["application_subset"] = scope_name
            row["adopt_hard_override"] = adoption_web_hard(row)
            row["candidate_boost_only"] = (not row["adopt_hard_override"]) and medium_candidate_boost(row)
            web_rule_rows.append(row)
    web_fields = [
        "name",
        "rule_name",
        "application_subset",
        "support",
        "override_count",
        "precision",
        "advanced_precision_same_rows",
        "override_precision",
        "base_precision_changed_rows",
        "recall_on_target_class",
        "net_gain_rows",
        "net_gain_changed_rows",
        "base_macro_f1",
        "new_macro_f1",
        "macro_f1_delta",
        "web_search_f1_delta",
        "plan_task_f1_delta",
        "ask_user_f1_delta",
        "grep_search_f1_delta",
        "halfA_delta",
        "halfB_delta",
        "false_positive_top_classes",
        "adopt_hard_override",
        "candidate_boost_only",
    ]
    write_csv(web_dir / "web_rule_results.csv", web_rule_rows, web_fields)

    web_fp_rows = []
    best_web_any = top_row(web_rule_rows)
    if best_web_any:
        candidates = [web_rule_candidate(sample, best_web_any["rule_name"]) for sample in val_samples]
        scope = scopes[best_web_any["application_subset"]]
        for sample, truth, pred, candidate, ok in zip(val_samples, y_val, base_pred, candidates, scope):
            if not ok or candidate is None or truth == "web_search":
                continue
            web_fp_rows.append(
                {
                    "id": sample["id"],
                    "rule_name": best_web_any["rule_name"],
                    "application_subset": best_web_any["application_subset"],
                    "true_label": truth,
                    "base_pred": pred,
                    "current_prompt": safe_text(sample.get("current_prompt"), 600),
                    "last_action": last_action_name(sample),
                    "web_flag_summary": flag_summary(web_flags(sample)),
                }
            )
            if len(web_fp_rows) >= 120:
                break
    write_csv(
        web_dir / "web_false_positive_cases.csv",
        web_fp_rows,
        ["id", "rule_name", "application_subset", "true_label", "base_pred", "current_prompt", "last_action", "web_flag_summary"],
    )

    boost_rows = []
    for row in web_rule_rows:
        if row["candidate_boost_only"] or row["adopt_hard_override"]:
            boost_rows.append(
                {
                    "rule_name": row["rule_name"],
                    "application_subset": row["application_subset"],
                    "support": row["support"],
                    "precision": row["precision"],
                    "web_search_recall_on_target": row["recall_on_target_class"],
                    "recommended_mode": "hard_override" if row["adopt_hard_override"] else "candidate_boost_only",
                    "candidate_score_boost_1": 1,
                    "candidate_score_boost_2": 2,
                    "candidate_score_boost_3": 3,
                }
            )
    write_csv(
        web_dir / "web_candidate_boost_results.csv",
        boost_rows,
        [
            "rule_name",
            "application_subset",
            "support",
            "precision",
            "web_search_recall_on_target",
            "recommended_mode",
            "candidate_score_boost_1",
            "candidate_score_boost_2",
            "candidate_score_boost_3",
        ],
    )

    best_exec_hard = top_row(execute_rule_rows, adoption_execute_rule)
    best_exec_resolver = top_row(resolver_rows, adoption_execute_resolver)
    best_web_hard = top_row(web_rule_rows, adoption_web_hard)

    combined_rows = []
    combined_preds = {"C0_advanced_only": np.asarray(base_pred, dtype=object).copy()}
    combined_preds["C1_advanced_plus_execute_hard"] = apply_row_config(base_pred, val_samples, {**best_exec_hard, "_scopes": scopes} if best_exec_hard else None, execute_rule_candidate)
    combined_preds["C2_advanced_plus_web_hard"] = apply_row_config(base_pred, val_samples, {**best_web_hard, "_scopes": scopes} if best_web_hard else None, web_rule_candidate)
    c3 = combined_preds["C1_advanced_plus_execute_hard"].copy()
    if best_web_hard:
        c3 = apply_row_config(c3, val_samples, {**best_web_hard, "_scopes": scopes}, web_rule_candidate)
    combined_preds["C3_advanced_plus_execute_web_hard"] = c3

    for name, pred in combined_preds.items():
        changed = pred != base_pred
        combined_rows.append(
            {
                "config": name,
                "status": "evaluated",
                "macro_f1": score_macro(y_val, pred),
                "macro_f1_delta": score_macro(y_val, pred) - base_macro,
                "accuracy": float((pred == y_val).mean()),
                "changed_count": int(changed.sum()),
                "halfA_delta": half_delta(y_val, base_pred, pred, half_a),
                "halfB_delta": half_delta(y_val, base_pred, pred, half_b),
                "runtime_estimate": "cpu_micro_rules_only",
            }
        )
    for name in [
        "C4_N4_current",
        "C5_N4_plus_execute_boost",
        "C6_N4_plus_web_boost",
        "C7_N4_plus_execute_web_boost",
        "C8_execute_web_hard_plus_N4",
    ]:
        combined_rows.append(
            {
                "config": name,
                "status": "candidate_only_no_transformer_logits",
                "macro_f1": "",
                "macro_f1_delta": "",
                "accuracy": "",
                "changed_count": "",
                "halfA_delta": "",
                "halfB_delta": "",
                "runtime_estimate": "requires staged N4 transformer logits/checkpoint",
            }
        )
    write_csv(
        combined_dir / "combined_micro_rule_results.csv",
        combined_rows,
        ["config", "status", "macro_f1", "macro_f1_delta", "accuracy", "changed_count", "halfA_delta", "halfB_delta", "runtime_estimate"],
    )

    best_combined_pred = max(
        [pred for name, pred in combined_preds.items()],
        key=lambda pred: score_macro(y_val, pred),
    )
    after_cw = classwise_f1(y_val, best_combined_pred)
    delta_rows = []
    for cls in ALL_CLASSES:
        delta_rows.append(
            {
                "class": cls,
                "base_f1": base_cw[cls]["f1"],
                "combined_f1": after_cw[cls]["f1"],
                "f1_delta": after_cw[cls]["f1"] - base_cw[cls]["f1"],
                "support": base_cw[cls]["support"],
            }
        )
    write_csv(combined_dir / "classwise_delta.csv", delta_rows, ["class", "base_f1", "combined_f1", "f1_delta", "support"])

    stability_rows = []
    for row in execute_rule_rows + resolver_rows + web_rule_rows:
        stability_rows.append(
            {
                "name": row["name"],
                "halfA_delta": row["halfA_delta"],
                "halfB_delta": row["halfB_delta"],
                "both_positive": row["halfA_delta"] > 0 and row["halfB_delta"] > 0,
                "macro_f1_delta": row["macro_f1_delta"],
            }
        )
    write_csv(combined_dir / "half_split_stability.csv", stability_rows, ["name", "halfA_delta", "halfB_delta", "both_positive", "macro_f1_delta"])

    candidate_boost_flags = {
        "base": "fast_flat local proxy",
        "split": split_name,
        "execute_candidate_boost_only": [
            {key: row[key] for key in ["name", "rule_name", "application_subset", "support", "precision", "macro_f1_delta"]}
            for row in execute_rule_rows
            if row["candidate_boost_only"]
        ],
        "execute_resolver_candidate_boost_only": [
            {key: row[key] for key in ["name", "application_subset", "threshold", "support", "precision", "macro_f1_delta"]}
            for row in resolver_rows
            if row["candidate_boost_only"]
        ],
        "web_candidate_boost_only": [
            {key: row[key] for key in ["name", "rule_name", "application_subset", "support", "precision", "macro_f1_delta"]}
            for row in web_rule_rows
            if row["candidate_boost_only"]
        ],
        "adopted_hard_rules": {
            "execute": best_exec_hard["name"] if best_exec_hard else None,
            "web": best_web_hard["name"] if best_web_hard else None,
        },
        "adopted_execute_resolver": best_exec_resolver["name"] if best_exec_resolver else None,
    }
    write_json(combined_dir / "candidate_boost_flags.json", candidate_boost_flags)

    def short_row(row):
        if not row:
            return "none"
        return f"{row['name']} delta={row['macro_f1_delta']:.6f} precision={row['precision']:.3f} support={row['support']} net={row['net_gain_rows']}"

    execute_summary = [
        "# Execute Pair Micro Rules",
        "",
        f"- split: `{split_name}`",
        f"- base Macro-F1: `{base_macro:.6f}`",
        f"- base accuracy: `{base_acc:.6f}`",
        f"- pair resolver train rows: `{pair_train_rows}`",
        "",
        "## Best Rows",
        "",
        f"- best hard-rule adopt candidate: `{short_row(best_exec_hard)}`",
        f"- best pair resolver adopt candidate: `{short_row(best_exec_resolver)}`",
        "",
        "## Confusions",
        "",
    ]
    for row in confusion_rows:
        execute_summary.append(f"- true `{row['true_label']}` -> pred `{row['pred_label']}`: `{row['count']}`")
    write_md(execute_dir / "summary.md", execute_summary)

    web_summary = [
        "# Web Search Micro Rules",
        "",
        f"- split: `{split_name}`",
        f"- base web_search F1: `{base_cw['web_search']['f1']:.6f}`",
        f"- best hard-rule adopt candidate: `{short_row(best_web_hard)}`",
        f"- boost-only candidate count: `{len(candidate_boost_flags['web_candidate_boost_only'])}`",
        "",
        "Top raw web rules:",
    ]
    for row in sorted(web_rule_rows, key=lambda item: item["macro_f1_delta"], reverse=True)[:8]:
        web_summary.append(
            f"- `{row['name']}` delta={row['macro_f1_delta']:.6f} precision={row['precision']:.3f} support={row['support']} net={row['net_gain_rows']}"
        )
    write_md(web_dir / "summary.md", web_summary)

    combined_summary = [
        "# Combined Micro Rule Evaluation",
        "",
        f"- base Macro-F1: `{base_macro:.6f}`",
        "",
        "Evaluated configs:",
    ]
    for row in combined_rows:
        combined_summary.append(f"- `{row['config']}` status={row['status']} delta=`{row['macro_f1_delta']}` changed=`{row['changed_count']}`")
    write_md(combined_dir / "summary.md", combined_summary)

    decision = [
        "# Micro Rule Final Decision",
        "",
        f"- timestamp: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- validation split: `{split_name}`",
        f"- base used here: `fast_flat local proxy`, Macro-F1 `{base_macro:.6f}`",
        "",
        "## Decisions",
        "",
        f"- execute hard rule: `{'ADOPT' if best_exec_hard else 'REJECT'}` ({short_row(best_exec_hard)})",
        f"- execute pair resolver: `{'ADOPT' if best_exec_resolver else 'REJECT'}` ({short_row(best_exec_resolver)})",
        f"- web hard override: `{'ADOPT' if best_web_hard else 'REJECT'}` ({short_row(best_web_hard)})",
        f"- web candidate boost only: `{len(candidate_boost_flags['web_candidate_boost_only'])}` rule/scope candidates exported",
        f"- execute candidate boost only: `{len(candidate_boost_flags['execute_candidate_boost_only']) + len(candidate_boost_flags['execute_resolver_candidate_boost_only'])}` candidates exported",
        f"- build submit zip: `NO`",
        "",
        "## Rationale",
        "",
        "- Hard overrides are accepted only when precision, net gain, Macro-F1, protected-class drop, and half-split stability all pass.",
        "- N4 candidate-boost variants are exported as flags only because no aligned transformer logits/checkpoint are staged in this clone.",
        "- Re-run this script on actual advanced_router or hybrid validation predictions before turning any proxy-local finding into a submit zip.",
    ]
    write_md(out_root / "MICRO_RULE_FINAL_DECISION.md", decision)

    with open(ROOT / "research.md", "a", encoding="utf-8") as f:
        f.write(
            "\n## Micro Execute/WebSearch Rule Experiment\n\n"
            f"- timestamp: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n"
            f"- base: `fast_flat local proxy`; split=`{split_name}`; Macro-F1=`{base_macro:.6f}`.\n"
            f"- execute hard rule: `{'ADOPT' if best_exec_hard else 'REJECT'}`; best=`{short_row(best_exec_hard)}`.\n"
            f"- execute pair resolver: `{'ADOPT' if best_exec_resolver else 'REJECT'}`; best=`{short_row(best_exec_resolver)}`.\n"
            f"- web hard override: `{'ADOPT' if best_web_hard else 'REJECT'}`; best=`{short_row(best_web_hard)}`.\n"
            f"- web boost-only candidates: `{len(candidate_boost_flags['web_candidate_boost_only'])}`; submit zip not built.\n"
        )
    print(f"wrote {out_root / 'MICRO_RULE_FINAL_DECISION.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../open/data")
    parser.add_argument("--fold-file", default="pipeline_v4/folds/fold_assignments.csv")
    parser.add_argument("--out-root", default="reports/micro_rules")
    parser.add_argument("--router-cache-root", default="reports/inspect_bottleneck")
    parser.add_argument("--router-features", type=int, default=50_000)
    parser.add_argument("--pair-features", type=int, default=45_000)
    parser.add_argument("--refresh-router", action="store_true")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
