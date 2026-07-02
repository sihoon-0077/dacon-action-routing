import re

from src.state_features import get_last_actions, iter_history_pairs, result_bucket_v3, turn_bucket


MODIFY_ACTIONS = {"edit_file", "write_file", "apply_patch"}
INSPECT_ACTIONS = {"read_file", "grep_search", "list_directory", "glob_pattern"}
VERIFY_ACTIONS = {"run_tests", "lint_or_typecheck"}
EXECUTE_ACTIONS = {"run_bash", "run_tests", "lint_or_typecheck"}
COMMUNICATE_ACTIONS = {"ask_user", "plan_task", "web_search", "respond_only"}


def _bucket_count(value):
    if value is None:
        return "none"
    try:
        value = int(value)
    except Exception:
        return "none"
    if value == 0:
        return "zero"
    if value == 1:
        return "one"
    if value <= 3:
        return "few_2_3"
    if value <= 10:
        return "mid_4_10"
    return "many_11_plus"


def _bucket_lines(value):
    if value is None:
        return "none"
    try:
        value = int(value)
    except Exception:
        return "none"
    if value < 100:
        return "short_lt100"
    if value < 400:
        return "mid_100_400"
    return "long_400_plus"


def extract_result_bucket(result_summary):
    return result_bucket_v3(result_summary)


def _verify_state(result_summary):
    text = (result_summary or "").lower()
    if not text:
        return "unknown"
    if any(x in text for x in ["traceback", "exception", "permission denied", "conflict", "error"]):
        return "error"
    if any(x in text for x in ["fail", "failed", "red", "exit=1", "exit 1"]):
        return "fail"
    if any(x in text for x in ["lint clean", "typecheck clean", "no issues"]):
        return "clean"
    if any(x in text for x in ["pass", "passed", "green", "exit=0", "exit 0", "ok"]):
        return "pass"
    return "unknown"


def _last_assistant_actions(sample):
    return [
        turn
        for turn in (sample.get("history", []) or [])
        if turn.get("role") == "assistant_action" and turn.get("name")
    ]


def extract_workflow_flags(sample):
    actions = _last_assistant_actions(sample)
    verify_state = "never"
    verify_action = "none"
    last_verify_idx = None
    last_modify = "none"
    last_modify_idx = None

    for idx, turn in enumerate(actions):
        name = turn.get("name")
        if name in MODIFY_ACTIONS:
            last_modify = name
            last_modify_idx = idx
        if name in VERIFY_ACTIONS:
            verify_action = name
            verify_state = _verify_state(turn.get("result_summary", ""))
            last_verify_idx = idx

    edits_after_verify = 0
    if last_verify_idx is not None:
        edits_after_verify = sum(1 for turn in actions[last_verify_idx + 1 :] if turn.get("name") in MODIFY_ACTIONS)

    inspects_since_modify = 0
    after_modify_groups = []
    if last_modify_idx is not None:
        for turn in actions[last_modify_idx + 1 :]:
            name = turn.get("name")
            if name in INSPECT_ACTIONS:
                inspects_since_modify += 1
                after_modify_groups.append("inspect")
            elif name in VERIFY_ACTIONS:
                after_modify_groups.append("verify")
            elif name in COMMUNICATE_ACTIONS:
                after_modify_groups.append("communicate")
            elif name in MODIFY_ACTIONS:
                after_modify_groups.append("modify")
            elif name:
                after_modify_groups.append("execute")

    if not after_modify_groups:
        after_modify = "none"
    else:
        unique = set(after_modify_groups)
        after_modify = next(iter(unique)) if len(unique) == 1 else "mixed"

    return {
        "WF_VERIFY_STATE": verify_state,
        "WF_VERIFY_ACTION": verify_action,
        "WF_EDITS_AFTER_VERIFY": "2plus" if edits_after_verify >= 2 else str(edits_after_verify),
        "WF_INSPECTS_SINCE_MODIFY": "3plus" if inspects_since_modify >= 3 else str(inspects_since_modify),
        "WF_LAST_MODIFY": last_modify,
        "WF_AFTER_MODIFY": after_modify,
    }


def _first_int(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            if match.lastindex:
                return int(match.group(1))
            return 0
    return None


def parse_numeric_result(result_summary):
    text = result_summary or ""
    low = text.lower()
    match_count = _first_int(
        [
            r"(\d+)\s*matches?",
            r"found\s+(\d+)\s+occurrences?",
            r"(\d+)\s+occurrences?",
            r"no matches",
            r"0 matches",
        ],
        low,
    )
    file_count = _first_int(
        [
            r"(\d+)\s*files?\s*matched",
            r"(\d+)\s*files?",
            r"(\d+)\s+entries",
            r"listed .*?:\s*(\d+)\s+items",
            r"empty directory",
        ],
        low,
    )
    line_count = _first_int(
        [
            r"read .*?\((\d+)l\)",
            r"(\d+)\s*lines",
            r"(\d+)l",
        ],
        low,
    )
    if any(x in low for x in ["grep", "matches", "occurrences"]):
        kind = "grep_result"
    elif any(x in low for x in ["glob", "files matched"]):
        kind = "glob_result"
    elif any(x in low for x in ["listed", "entries", "directory"]):
        kind = "list_result"
    elif any(x in low for x in ["read", "opened", "lines"]):
        kind = "read_result"
    elif any(x in low for x in ["test", "pass", "fail", "green", "red"]):
        kind = "test_result"
    elif any(x in low for x in ["patched", "modified", "wrote", "created", "edited"]):
        kind = "edit_result"
    elif "web" in low or "search" in low:
        kind = "web_result"
    elif "plan" in low:
        kind = "plan_result"
    elif "ask" in low or "question" in low:
        kind = "ask_result"
    else:
        kind = "unknown"
    return {
        "match": _bucket_count(match_count),
        "files": _bucket_count(file_count),
        "lines": _bucket_lines(line_count),
        "kind": kind,
    }


def extract_numeric_result_buckets(sample):
    actions = _last_assistant_actions(sample)
    last = actions[-1] if actions else {}
    last_num = parse_numeric_result(last.get("result_summary", ""))
    selected = {
        "grep": None,
        "glob": None,
        "read": None,
    }
    for turn in reversed(actions):
        name = turn.get("name")
        if name == "grep_search" and selected["grep"] is None:
            selected["grep"] = parse_numeric_result(turn.get("result_summary", ""))
        elif name == "glob_pattern" and selected["glob"] is None:
            selected["glob"] = parse_numeric_result(turn.get("result_summary", ""))
        elif name == "read_file" and selected["read"] is None:
            selected["read"] = parse_numeric_result(turn.get("result_summary", ""))
    return {
        "NM_MATCH": last_num["match"],
        "NM_FILES": last_num["files"],
        "NM_LINES": last_num["lines"],
        "NM_KIND": last_num["kind"],
        "LAST_GREP_MATCH": (selected["grep"] or {}).get("match", "none"),
        "LAST_GLOB_FILES": (selected["glob"] or {}).get("files", "none"),
        "LAST_READ_LINES": (selected["read"] or {}).get("lines", "none"),
    }


def extract_surface_flags(sample):
    prompt = sample.get("current_prompt", "") or ""
    low = prompt.lower()
    stripped = prompt.strip()
    sentences = [part for part in re.split(r"[.!?\n]+", stripped) if part.strip()]
    return {
        "PF_END_Q": int(stripped.endswith("?") or stripped.endswith("？")),
        "PF_END_EXCL": int(stripped.endswith("!") or stripped.endswith("！")),
        "PF_HAS_ELLIPSIS": int("..." in stripped or "…" in stripped),
        "PF_HAS_KO_IMPERATIVE": int(bool(re.search(r"(해줘|해봐|하자|보자|줘|돌려|열어|찾아|고쳐|추가|바꿔)\s*[.?!]*$", prompt))),
        "PF_HAS_KO_POLITE_YO": int(bool(re.search(r"(주세요|부탁해|가능하면요|할까요|인가요)\s*[.?!]*$", prompt))),
        "PF_HAS_QUESTION_WORD": int(bool(re.search(r"\b(what|where|how|why|which|should|can you|could you)\b", low))),
        "PF_HAS_PLEASE_THANKS": int(any(x in low for x in ["please", "thanks", "thank you", "부탁", "고마"])),
        "PF_HAS_URGENT": int(any(x in low for x in ["urgent", "asap", "빨리", "급해", "당장"])),
        "PF_HAS_NO_RUSH": int(any(x in low for x in ["no rush", "천천히", "나중에"])),
        "PF_HAS_LAUGH": int(any(x in low for x in ["ㅋㅋ", "ㅎㅎ", "lol", "haha"])),
        "PF_N_SENT": "3plus" if len(sentences) >= 3 else str(len(sentences)),
        "PF_LEN_BUCKET": "short" if len(prompt) < 40 else ("mid" if len(prompt) < 140 else "long"),
        "PF_HAS_SUMMARY_PHRASE": int(bool(re.search(r"(요약|정리|마무리|recap|summarize|wrap.?up|brief)", low))),
        "PF_HAS_PLAN_PHRASE": int(bool(re.search(r"(계획|단계|쪼개|plan|break.?down|where to start)", low))),
        "PF_HAS_WEB_PHRASE": int(bool(re.search(r"(최신|권장|best.?practice|recommended|official docs?|문서)", low))),
    }


def extract_prompt_flags(prompt):
    return extract_surface_flags({"current_prompt": prompt})


def intuition_tokens(sample, feature):
    if feature == "i1":
        data = extract_workflow_flags(sample)
    elif feature == "i4":
        data = extract_numeric_result_buckets(sample)
    elif feature == "i5":
        data = extract_surface_flags(sample)
    elif feature == "i145":
        data = {}
        data.update(extract_workflow_flags(sample))
        data.update(extract_numeric_result_buckets(sample))
        data.update(extract_surface_flags(sample))
    else:
        raise ValueError(feature)
    return " ".join(f"{key}={value}" for key, value in sorted(data.items()))


def selector_feature_dict(sample, advanced_pred, transformer_pred, transformer_probs):
    probs = list(transformer_probs)
    top = sorted(range(len(probs)), key=lambda idx: probs[idx], reverse=True)
    margin = probs[top[0]] - probs[top[1]] if len(top) > 1 else probs[top[0]]
    entropy = -sum(float(p) * __import__("math").log(max(float(p), 1e-12)) for p in probs)
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    last_actions = get_last_actions(sample, 3)
    last_result = "none"
    pairs = iter_history_pairs(sample)
    if pairs and pairs[-1][1]:
        last_result = result_bucket_v3(pairs[-1][1].get("result_summary", ""))
    data = {
        "advanced_pred": advanced_pred,
        "transformer_pred": transformer_pred,
        "same_pred": int(advanced_pred == transformer_pred),
        "same_group": int(_group_of(advanced_pred) == _group_of(transformer_pred)),
        "advanced_group": _group_of(advanced_pred),
        "transformer_group": _group_of(transformer_pred),
        "tf_conf_bucket": _num_bucket(probs[top[0]], [0.35, 0.5, 0.65, 0.8]),
        "tf_margin_bucket": _num_bucket(margin, [0.05, 0.15, 0.3, 0.5]),
        "tf_entropy_bucket": _num_bucket(entropy, [1.0, 1.6, 2.1, 2.5]),
        "last1": last_actions[-1] if len(last_actions) >= 1 else "none",
        "last2": "|".join(last_actions[-2:]) if len(last_actions) >= 2 else "none",
        "last3": "|".join(last_actions[-3:]) if len(last_actions) >= 3 else "none",
        "last_result": last_result,
        "turn_bucket": turn_bucket(meta.get("turn_index")),
        "language_pref": meta.get("language_pref", "unknown"),
        "ci_status": ws.get("last_ci_status", "none"),
        "open_files_bucket": _num_bucket(len(ws.get("open_files", []) or []), [0, 1, 3, 6]),
        "prompt_len_bucket": _num_bucket(len(sample.get("current_prompt", "") or ""), [30, 80, 160, 320]),
        "history_len_bucket": _num_bucket(len(sample.get("history", []) or []), [0, 2, 6, 10]),
    }
    data.update({f"sf_{k}": v for k, v in extract_surface_flags(sample).items()})
    data.update({f"nm_{k}": v for k, v in extract_numeric_result_buckets(sample).items()})
    return data


def _group_of(action):
    if action in INSPECT_ACTIONS:
        return "inspect"
    if action in MODIFY_ACTIONS:
        return "modify"
    if action in EXECUTE_ACTIONS:
        return "execute"
    return "communicate"


def _num_bucket(value, cuts):
    for idx, cut in enumerate(cuts):
        if value <= cut:
            return f"b{idx}"
    return f"b{len(cuts)}"
