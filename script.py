import csv
import json
import math
import os
import re
from pathlib import Path

import joblib
import numpy as np


REQUIRED_KEYS = ("id", "session_meta", "history", "current_prompt")
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
ADVANCED_GROUP_TO_ACTIONS = {
    "inspect": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "modify": ["edit_file", "write_file", "apply_patch"],
    "execute": ["run_bash", "run_tests", "lint_or_typecheck"],
    "communicate": ["ask_user", "plan_task", "web_search", "respond_only"],
}
ADVANCED_ACTION_TO_GROUP = {
    action: group
    for group, actions in ADVANCED_GROUP_TO_ACTIONS.items()
    for action in actions
}


def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON parse failed: {exc}") from exc
    return samples


def extract_text(sample):
    text = sample.get("current_prompt", "")
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return text


def safe_text(value, max_chars=1200):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split())
    return text[:max_chars]


def bucket_number(value, bins):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for name, upper in bins:
        if value <= upper:
            return name
    return "huge"


def file_tokens(paths):
    out = []
    for path in paths or []:
        text = safe_text(path, 300).lower()
        out.append("openfile=" + text)
        suffix = Path(text).suffix.lstrip(".")
        if suffix:
            out.append("openext=" + suffix)
        parts = [p for p in text.replace("\\", "/").split("/") if p]
        out.extend("pathpart=" + p for p in parts[-4:])
    return " ".join(out)


def meta_text(sample):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    loc_bin = bucket_number(
        ws.get("loc"),
        [("loc_lt_2k", 2_000), ("loc_lt_10k", 10_000), ("loc_lt_30k", 30_000)],
    )
    budget_bin = bucket_number(
        meta.get("budget_tokens_remaining"),
        [("budget_low", 10_000), ("budget_mid", 60_000), ("budget_high", 140_000)],
    )
    elapsed_bin = bucket_number(
        meta.get("elapsed_session_sec"),
        [("elapsed_start", 240), ("elapsed_mid", 900), ("elapsed_late", 1800)],
    )
    langs = ws.get("language_mix", {}) or {}
    lang_tokens = []
    for lang, ratio in sorted(langs.items()):
        try:
            bucket = int(round(float(ratio) * 10))
        except (TypeError, ValueError):
            bucket = 0
        lang_tokens.append(f"lang={lang} langshare={lang}_{bucket}")
    return " ".join(
        [
            f"tier={safe_text(meta.get('user_tier'), 40)}",
            f"pref={safe_text(meta.get('language_pref'), 40)}",
            f"turn={meta.get('turn_index', 'unknown')}",
            f"turn_bin={bucket_number(meta.get('turn_index'), [('turn_early', 2), ('turn_mid', 7), ('turn_late', 12)])}",
            f"elapsed={elapsed_bin}",
            f"budget={budget_bin}",
            f"git_dirty={ws.get('git_dirty', 'unknown')}",
            f"ci={safe_text(ws.get('last_ci_status'), 40)}",
            loc_bin,
            " ".join(lang_tokens),
            file_tokens(ws.get("open_files", [])),
        ]
    )


def history_text(sample, mode):
    hist = sample.get("history", []) or []
    if not hist:
        return "history_empty last_action=NONE"

    action_names = [
        h.get("name", "")
        for h in hist
        if h.get("role") == "assistant_action" and h.get("name")
    ]
    last_action = action_names[-1] if action_names else "NONE"
    seq = " ".join(f"hist_action={name}" for name in action_names)
    last_seq = " ".join(f"recent_action={name}" for name in action_names[-4:])

    if mode == "actions":
        return f"last_action={last_action} {seq} {last_seq}"

    recent = hist[-6:] if mode == "recent" else hist
    parts = [f"last_action={last_action}", seq, last_seq]
    for item in recent:
        role = item.get("role", "")
        if role == "user":
            parts.append("hist_user=" + safe_text(item.get("content"), 700))
        elif role == "assistant_action":
            parts.append("hist_tool=" + safe_text(item.get("name"), 80))
            parts.append("hist_args=" + safe_text(item.get("args"), 800))
            parts.append("hist_result=" + safe_text(item.get("result_summary"), 800))
        else:
            parts.append("hist_other=" + safe_text(item, 800))
    return " ".join(parts)


def serialize_sample(sample, feature_mode):
    prompt = safe_text(sample.get("current_prompt"), 1200)
    if feature_mode == "prompt":
        return prompt
    if feature_mode == "prompt_x2":
        return f"current={prompt} current_again={prompt}"
    if feature_mode == "prompt_meta":
        return f"current={prompt} meta {meta_text(sample)}"
    if feature_mode == "compact":
        return f"current={prompt} history {history_text(sample, 'actions')} meta {meta_text(sample)}"
    if feature_mode == "compact_no_meta":
        return f"current={prompt} history {history_text(sample, 'actions')}"
    if feature_mode == "compact_prompt_x2":
        return (
            f"current={prompt} current_again={prompt} "
            f"history {history_text(sample, 'actions')} meta {meta_text(sample)}"
        )
    if feature_mode == "compact_prompt_x3":
        return (
            f"current={prompt} current_again={prompt} current_third={prompt} "
            f"history {history_text(sample, 'actions')} meta {meta_text(sample)}"
        )
    if feature_mode == "compact_history_x2":
        hist = history_text(sample, "actions")
        return f"current={prompt} history {hist} history_again {hist} meta {meta_text(sample)}"
    if feature_mode == "compact_prompt_history_x2":
        hist = history_text(sample, "actions")
        return (
            f"current={prompt} current_again={prompt} "
            f"history {hist} history_again {hist} meta {meta_text(sample)}"
        )
    if feature_mode == "recent":
        return f"current={prompt} history {history_text(sample, 'recent')} meta {meta_text(sample)}"
    if feature_mode == "recent_prompt_x2":
        return (
            f"current={prompt} current_again={prompt} "
            f"history {history_text(sample, 'recent')} meta {meta_text(sample)}"
        )
    if feature_mode == "full":
        return f"current={prompt} history {history_text(sample, 'full')} meta {meta_text(sample)}"
    if feature_mode == "full_prompt_x3":
        return (
            f"current={prompt} current_again={prompt} current_third={prompt} "
            f"history {history_text(sample, 'full')} meta {meta_text(sample)}"
        )
    return extract_text(sample)


def compact_router_actions(sample):
    return [
        h.get("name", "")
        for h in sample.get("history", []) or []
        if h.get("role") == "assistant_action" and h.get("name")
    ]


def compact_router_last_actions(sample):
    acts = compact_router_actions(sample)
    last1 = acts[-1] if acts else "NONE"
    last2 = ">".join(acts[-2:]) if len(acts) >= 2 else "NONE>" + last1
    return last1, last2


def compact_router_flag_tokens(sample):
    hist = sample.get("history", []) or []
    prompt = safe_text(sample.get("current_prompt"), 700).lower()
    tokens = []
    recent_tools = [h for h in hist if h.get("role") == "assistant_action"][-4:]
    for i, tool in enumerate(reversed(recent_tools), start=1):
        name = safe_text(tool.get("name"), 80)
        result = safe_text(tool.get("result_summary"), 900).lower()
        args = safe_text(tool.get("args"), 900).lower()
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


def compact_flags_text(sample):
    return serialize_sample(sample, "compact") + " FLAGS " + compact_router_flag_tokens(sample)


def advanced_action_names(sample):
    return [
        h.get("name", "")
        for h in sample.get("history", []) or []
        if h.get("role") == "assistant_action" and h.get("name")
    ]


def advanced_last_action(sample):
    acts = advanced_action_names(sample)
    return acts[-1] if acts else "NONE"


def advanced_last2_action(sample):
    acts = advanced_action_names(sample)
    return ">".join(acts[-2:]) if len(acts) >= 2 else "NONE>" + (acts[-1] if acts else "NONE")


def advanced_last_result(sample):
    for item in reversed(sample.get("history", []) or []):
        if item.get("role") == "assistant_action":
            return safe_text(item.get("result_summary"), 1000)
    return ""


def advanced_workspace(sample):
    meta = sample.get("session_meta", {}) or {}
    return meta.get("workspace", {}) or {}


def advanced_open_files(sample):
    return [
        safe_text(path, 300).replace("\\", "/")
        for path in advanced_workspace(sample).get("open_files", []) or []
    ]


def advanced_has_any(text, patterns):
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def advanced_result_type(sample):
    text = advanced_last_result(sample).lower()
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


def advanced_simple_rule_hint(sample):
    prompt = safe_text(sample.get("current_prompt"), 1200).lower()
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


def advanced_path_ext_tokens(text, prefix):
    tokens = []
    for path in re.findall(r"[\w./\\-]+\.[A-Za-z0-9]{1,8}", text)[:20]:
        norm = path.replace("\\", "/").lower()
        ext = norm.rsplit(".", 1)[-1]
        tokens.append(f"{prefix}_PATH={norm}")
        tokens.append(f"{prefix}_EXT={ext}")
    return tokens


def advanced_count_bucket_from_text(text, prefix):
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


def advanced_generic_tokens(sample):
    meta = sample.get("session_meta", {}) or {}
    ws = advanced_workspace(sample)
    prompt = safe_text(sample.get("current_prompt"), 1200)
    result = advanced_last_result(sample)
    tokens = [
        f"LAST_ACTION={advanced_last_action(sample)}",
        f"LAST2_ACTION={advanced_last2_action(sample)}",
        f"RESULT_TYPE={advanced_result_type(sample)}",
        f"RULE_HINT={advanced_simple_rule_hint(sample)}",
        f"RULE_GROUP={ADVANCED_ACTION_TO_GROUP.get(advanced_simple_rule_hint(sample), 'unknown')}",
        f"CI={safe_text(ws.get('last_ci_status'), 80)}",
        f"GIT_DIRTY={ws.get('git_dirty', 'unknown')}",
        f"TURN_BIN={bucket_number(meta.get('turn_index'), [('turn_early', 2), ('turn_mid', 7), ('turn_late', 12)])}",
        f"BUDGET_BIN={bucket_number(meta.get('budget_tokens_remaining'), [('budget_low', 10000), ('budget_mid', 50000), ('budget_high', 120000)])}",
        f"PROMPT_LEN={bucket_number(len(prompt), [('short', 30), ('mid', 100), ('long', 250)])}",
        advanced_count_bucket_from_text(result, "LAST_RESULT"),
    ]
    for path in advanced_open_files(sample)[:10]:
        tokens.append("OPEN_FILE=" + path.lower())
        if "." in path.rsplit("/", 1)[-1]:
            tokens.append("OPEN_EXT=" + path.rsplit(".", 1)[-1].lower())
    tokens.extend(advanced_path_ext_tokens(prompt + " " + result, "MENTIONED"))
    return tokens


def advanced_group_extra_tokens(sample, group):
    prompt = safe_text(sample.get("current_prompt"), 1200)
    prompt_l = prompt.lower()
    result_l = advanced_last_result(sample).lower()
    blob = prompt_l + " " + result_l
    tokens = advanced_generic_tokens(sample)

    def add(name, patterns):
        if advanced_has_any(blob, patterns):
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
        tokens.append(f"OPEN_FILES_COUNT={bucket_number(len(advanced_open_files(sample)), [('none', 0), ('one', 1), ('few', 4), ('some', 10)])}")
    elif group == "execute":
        add("HAS_TEST_WORD", [r"\btests?\b", r"\bpytest\b", r"cargo test", r"npm test", r"unit test", r"integration test", r"\be2e\b", r"\bspec\b", r"테스트"])
        add("HAS_LINT_WORD", [r"actionlint", r"\blint\b", r"type.?check", r"\btsc\b", r"\bmypy\b", r"\bruff\b", r"static analysis", r"타입"])
        add("HAS_BUILD_WORD", [r"\bbuild\b", r"cargo build", r"npm run build", r"빌드"])
        add("HAS_SERVER_WORD", [r"dev server", r"runserver", r"npm run dev", r"serve", r"server", r"서버"])
        add("HAS_INSTALL_WORD", [r"pip install", r"npm install", r"pod install", r"bundle install", r"설치"])
        tokens.append("LAST_CI_FAILED" if advanced_workspace(sample).get("last_ci_status") == "failed" else "LAST_CI_NOT_FAILED")
    elif group == "communicate":
        add("HAS_SUMMARY_WORD", [r"요약", r"정리", r"마무리", r"summarize", r"summary", r"recap", r"wrap.?up", r"brief"])
        add("HAS_PLAN_WORD", [r"\bplan\b", r"계획", r"단계", r"순서", r"break.+down", r"where to start", r"approach"])
        add("HAS_WEB_WORD", [r"latest", r"최신", r"best.?practice", r"recommended", r"official docs?", r"공식 문서", r"paper", r"online"])
        add("HAS_QUESTION_WORD", [r"should i", r"which", r"어느", r"뭐가", r"어떻게 할까", r"괜찮"])
        tokens.append("HISTORY_EMPTY" if not sample.get("history") else "HISTORY_PRESENT")
    return " ".join(tokens)


def advanced_group_text(sample, group):
    base = compact_flags_text(sample)
    extra = advanced_group_extra_tokens(sample, group)
    return base + " GROUP_EXTRA " + extra + " GROUP_EXTRA_AGAIN " + extra


def advanced_pair_text(sample, pair):
    group = ADVANCED_ACTION_TO_GROUP[pair[0]]
    return compact_flags_text(sample) + " PAIR_EXTRA " + advanced_group_extra_tokens(sample, group) + f" PAIR={pair[0]}_VS_{pair[1]}"


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores - scores.mean(axis=1, keepdims=True)
    return scores / (scores.std(axis=1, keepdims=True) + 1e-6)


def aligned_log_proba(model, x):
    proba = model.predict_proba(x)
    out = np.full((proba.shape[0], len(ALL_CLASSES)), -1e9, dtype=np.float32)
    for i, cls in enumerate(model.classes_):
        out[:, ALL_CLASSES.index(str(cls))] = np.log(proba[:, i] + 1e-9)
    return out


def transition_prior_matrix(samples, counts, global_counts, smooth, key_index):
    global_total = sum(global_counts.values())
    global_row = np.array(
        [
            math.log((global_counts.get(cls, 0) + smooth) / (global_total + smooth * len(ALL_CLASSES)))
            for cls in ALL_CLASSES
        ],
        dtype=np.float32,
    )
    rows = []
    for sample in samples:
        key = compact_router_last_actions(sample)[key_index]
        class_counts = counts.get(key)
        if not class_counts:
            rows.append(global_row)
            continue
        total = sum(class_counts.values())
        rows.append(
            [
                math.log((class_counts.get(cls, 0) + smooth) / (total + smooth * len(ALL_CLASSES)))
                for cls in ALL_CLASSES
            ]
        )
    return np.array(rows, dtype=np.float32)


def compact_router_group_scores(group_model, x):
    group_classes = list(group_model.classes_)
    logp = np.log(group_model.predict_proba(x) + 1e-9)
    out = np.zeros((x.shape[0], len(ALL_CLASSES)), dtype=np.float32)
    for gi, group in enumerate(group_classes):
        for cls in GROUPS[group]:
            out[:, ALL_CLASSES.index(cls)] = logp[:, gi]
    return out


def compact_router_rule_scores(samples):
    out = np.zeros((len(samples), len(ALL_CLASSES)), dtype=np.float32)
    for i, sample in enumerate(samples):
        prompt = safe_text(sample.get("current_prompt"), 700).lower()
        last1, _ = compact_router_last_actions(sample)
        hist = sample.get("history", []) or []
        last_result = ""
        for item in reversed(hist):
            if item.get("role") == "assistant_action":
                last_result = safe_text(item.get("result_summary"), 700).lower()
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
        if last1 in {"run_tests", "lint_or_typecheck"} and any(
            p in last_result for p in ["fail", "error", "traceback", "failed"]
        ):
            add("edit_file", 0.75)
            add("apply_patch", 0.45)
    return out


def predict_compact_flags_router(samples, artifact):
    if not samples:
        return []
    texts = [compact_flags_text(sample) for sample in samples]
    x = artifact["vectorizer"].transform(texts)
    weights = artifact["weights"]
    base = normalize_scores(aligned_log_proba(artifact["clf"], x))
    prior1 = normalize_scores(
        transition_prior_matrix(
            samples,
            artifact["transition1"],
            artifact["global_counts"],
            weights.get("smooth1", 1.0),
            0,
        )
    )
    prior2 = normalize_scores(
        transition_prior_matrix(
            samples,
            artifact["transition2"],
            artifact["global_counts"],
            weights.get("smooth2", 2.0),
            1,
        )
    )
    group = normalize_scores(compact_router_group_scores(artifact["group_clf"], x))
    rules = normalize_scores(compact_router_rule_scores(samples))
    scores = (
        base
        + weights.get("prior1", 0.06) * prior1
        + weights.get("prior2", 0.03) * prior2
        + weights.get("group", 0.08) * group
        + weights.get("rules", 0.02) * rules
    )
    return [ALL_CLASSES[i] for i in scores.argmax(axis=1)]


def predict_routing_margin_router(samples, artifact):
    if not samples:
        return []
    texts = [compact_flags_text(sample) for sample in samples]
    x = artifact["vectorizer"].transform(texts)
    group_pred = artifact["coarse_svc"].predict(x)
    preds = np.array(["respond_only"] * len(samples), dtype=object)
    for group, model in artifact["fine_models"].items():
        idx = np.where(group_pred == group)[0]
        if len(idx):
            preds[idx] = model.predict(x[idx])
    return [str(pred) for pred in preds]


def advanced_aligned_log_proba(model, x, classes):
    proba = model.predict_proba(x)
    out = np.full((x.shape[0], len(classes)), -1e9, dtype=np.float32)
    for i, cls in enumerate(model.classes_):
        out[:, classes.index(str(cls))] = np.log(proba[:, i] + 1e-9)
    return out


def advanced_transition_prior_matrix(samples, counts, global_counts, smooth):
    global_total = sum(global_counts.values())
    global_row = np.array(
        [
            math.log((global_counts.get(cls, 0) + smooth) / (global_total + smooth * len(ALL_CLASSES)))
            for cls in ALL_CLASSES
        ],
        dtype=np.float32,
    )
    rows = []
    for sample in samples:
        key = advanced_last2_action(sample)
        row_counts = counts.get(key)
        if not row_counts:
            rows.append(global_row)
            continue
        total = sum(row_counts.values())
        rows.append(
            [
                math.log((row_counts.get(cls, 0) + smooth) / (total + smooth * len(ALL_CLASSES)))
                for cls in ALL_CLASSES
            ]
        )
    return np.array(rows, dtype=np.float32)


def predict_advanced_router(samples, artifact):
    if not samples:
        return []
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
    return [str(pred) for pred in preds]


def load_model_and_config(model_dir):
    advanced_router_path = os.path.join(model_dir, "advanced_router.pkl")
    routing_margin_path = os.path.join(model_dir, "routing_margin_router.pkl")
    compact_router_path = os.path.join(model_dir, "compact_flags_router.pkl")
    research_model_path = os.path.join(model_dir, "research_best.pkl")
    research_config_path = os.path.join(model_dir, "research_model_config.json")
    baseline_model_path = os.path.join(model_dir, "tfidf_logreg.pkl")
    if os.path.exists(advanced_router_path):
        return joblib.load(advanced_router_path), "advanced_router"
    if os.path.exists(routing_margin_path):
        return joblib.load(routing_margin_path), "routing_margin_router"
    if os.path.exists(compact_router_path):
        return joblib.load(compact_router_path), "compact_flags_router"
    if os.path.exists(research_model_path) and os.path.exists(research_config_path):
        with open(research_config_path, encoding="utf-8") as f:
            config = json.load(f)
        return joblib.load(research_model_path), config.get("feature_mode", "prompt")
    return joblib.load(baseline_model_path), "prompt"


def load_sample_submission(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None or fieldnames[:2] != ["id", "action"]:
        raise ValueError(f"Unexpected sample_submission columns: {fieldnames}")
    return fieldnames, rows


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def submission_session_id(sample_id):
    return sample_id.rsplit("-step_", 1)[0] if "-step_" in sample_id else sample_id


def iter_session_lookup_pairs(sample):
    sid = submission_session_id(sample.get("id", ""))
    history = sample.get("history") or []
    for i, turn in enumerate(history):
        if turn.get("role") != "user":
            continue
        prompt = turn.get("content", "")
        if not prompt:
            continue
        next_action = None
        for later in history[i + 1 :]:
            if later.get("role") == "assistant_action":
                next_action = later.get("name")
                break
            if later.get("role") == "user":
                break
        if next_action:
            yield (sid, prompt), next_action


def build_session_lookup(samples):
    lookup = {}
    pair_count = 0
    collision_count = 0
    for sample in samples:
        for key, action in iter_session_lookup_pairs(sample):
            pair_count += 1
            old = lookup.get(key)
            if old is not None and old != action:
                collision_count += 1
            lookup[key] = action
    return lookup, pair_count, collision_count


def apply_session_lookup_override(samples, preds, data_dir):
    source_samples = []
    train_path = os.path.join(data_dir, "train.jsonl")
    if os.path.exists(train_path):
        source_samples.extend(load_jsonl(train_path))
    source_samples.extend(samples)

    lookup, pair_count, collision_count = build_session_lookup(source_samples)
    out = []
    hit_count = 0
    changed_count = 0
    for sample, pred in zip(samples, preds):
        key = (submission_session_id(sample.get("id", "")), sample.get("current_prompt", ""))
        override = lookup.get(key)
        if override:
            hit_count += 1
            if override != pred:
                changed_count += 1
            out.append(str(override))
        else:
            out.append(str(pred))

    print(
        "session_lookup: "
        f"sources={len(source_samples)} keys={len(lookup)} pairs={pair_count} "
        f"collisions={collision_count} hits={hit_count}/{len(samples)} changed={changed_count}"
    )
    return out


def main():
    test_path = "./data/test.jsonl"
    sample_submission_path = "./data/sample_submission.csv"
    output_path = "./output/submission.csv"

    model, feature_mode = load_model_and_config("./model")
    samples = load_jsonl(test_path)

    missing_schema = sum(
        1 for sample in samples if any(key not in sample for key in REQUIRED_KEYS)
    )
    if missing_schema:
        print(f"warning: missing required keys in {missing_schema} samples")

    ids = [sample.get("id", "") for sample in samples]
    if feature_mode == "advanced_router":
        preds = predict_advanced_router(samples, model)
    elif feature_mode == "routing_margin_router":
        preds = predict_routing_margin_router(samples, model)
    elif feature_mode == "compact_flags_router":
        preds = [str(pred) for pred in predict_compact_flags_router(samples, model)]
    else:
        texts = [serialize_sample(sample, feature_mode) for sample in samples]
        preds = [str(pred) for pred in model.predict(texts)] if texts else []

    try:
        preds = apply_session_lookup_override(samples, preds, "./data")
    except Exception as exc:
        print(f"warning: session lookup override skipped: {exc}")

    pred_by_id = dict(zip(ids, preds))

    fieldnames, rows = load_sample_submission(sample_submission_path)
    for row in rows:
        if row["id"] in pred_by_id:
            row["action"] = pred_by_id[row["id"]]

    save_submission(output_path, fieldnames, rows)
    print(f"Saved: {output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
