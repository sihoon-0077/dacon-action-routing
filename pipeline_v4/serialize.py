import json
import re

FILE_RE = re.compile(r"[\w./-]+\.[a-z]{1,4}\b", re.I)


def clean(value, max_chars=None):
    if value is None:
        text = "none"
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split())
    if not text:
        text = "none"
    return text if max_chars is None else text[:max_chars]


def bucket(value, bins, labels):
    try:
        value = int(value)
    except Exception:
        return "none"
    for upper, label in zip(bins, labels):
        if value < upper:
            return label
    return labels[-1]


def budget_bucket(value):
    return bucket(value, [5000, 20000, 80000], ["b0", "b1", "b2", "b3"])


def loc_bucket(value):
    return bucket(value, [5000, 15000, 40000], ["l0", "l1", "l2", "l3"])


def file_mentions(text):
    return {m.group(0).replace("\\", "/").lower() for m in FILE_RE.finditer(text or "")}


def prompt_flags(sample):
    prompt_files = file_mentions(sample.get("current_prompt", ""))
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    open_files = set()
    for path in ws.get("open_files", []) or []:
        norm = str(path).replace("\\", "/").lower()
        open_files.add(norm)
        open_files.add(norm.rsplit("/", 1)[-1])
    seen_texts = []
    for turn in sample.get("history", []) or []:
        if turn.get("role") == "assistant_action":
            seen_texts.append(clean(turn.get("args")))
    seen = " ".join(seen_texts).replace("\\", "/").lower()
    return {
        "pf": int(bool(prompt_files)),
        "pf_open": int(any(f in open_files or f.rsplit("/", 1)[-1] in open_files for f in prompt_files)),
        "pf_seen": int(any(f in seen or f.rsplit("/", 1)[-1] in seen for f in prompt_files)),
    }


def summarize_args(args):
    if not isinstance(args, dict):
        return "none"
    parts = []
    for key in ["path", "pattern", "target", "scope"]:
        value = args.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{key}={clean(value, 160)}")
    return " ".join(parts) if parts else "none"


def result_bucket_detail(text):
    low = (text or "").lower()
    if not low:
        return "none"
    if any(x in low for x in ["traceback", "exception", "permission denied", "conflict", "error"]):
        return "fail"
    if any(x in low for x in ["fail", "failed", "red", "exit=1", "exit 1"]):
        return "fail"
    if any(x in low for x in ["no matches", "0 matches", "not found", "zero match"]):
        return "zero_match"
    for pattern in [r"(\d+)\s*matches?", r"found\s+(\d+)\s+occurrences?", r"(\d+)\s+occurrences?"]:
        match = re.search(pattern, low)
        if match:
            count = int(match.group(1))
            if count == 0:
                return "zero_match"
            if count <= 3:
                return "few_match"
            if count <= 10:
                return "many_match"
            return "flood"
    if ("read" in low and "line" in low) or "opened" in low:
        return "read_ok"
    if any(x in low for x in ["pass", "passed", "green", "exit=0", "exit 0", "ok", "success"]):
        return "ok"
    return "ok"


def workflow_state(sample):
    test_state = "never"
    last_test_idx = None
    last_modify_idx = None
    actions = []
    for turn in sample.get("history", []) or []:
        if turn.get("role") != "assistant_action":
            continue
        idx = len(actions)
        name = turn.get("name") or "none"
        actions.append(turn)
        if name in {"run_tests", "lint_or_typecheck"}:
            test_state = result_bucket_detail(turn.get("result_summary", ""))
            if test_state == "ok":
                test_state = "pass"
            last_test_idx = idx
        if name in {"edit_file", "write_file", "apply_patch"}:
            last_modify_idx = idx
    edits_after_test = 0
    if last_test_idx is not None:
        edits_after_test = sum(
            1 for turn in actions[last_test_idx + 1 :]
            if turn.get("name") in {"edit_file", "write_file", "apply_patch"}
        )
    insp_since_mod = 0
    if last_modify_idx is not None:
        insp_since_mod = sum(
            1 for turn in actions[last_modify_idx + 1 :]
            if turn.get("name") in {"read_file", "grep_search", "list_directory", "glob_pattern"}
        )
    return {
        "test": test_state if test_state in {"never", "pass", "fail"} else "fail",
        "edits_after_test": "2+" if edits_after_test >= 2 else str(edits_after_test),
        "insp_since_mod": "3+" if insp_since_mod >= 3 else str(insp_since_mod),
    }


def surface_flag_endq(sample):
    prompt = (sample.get("current_prompt") or "").strip()
    return int(prompt.endswith("?") or prompt.endswith("？"))


def last_action_turn(sample):
    for turn in reversed(sample.get("history", []) or []):
        if turn.get("role") == "assistant_action":
            return turn
    return None


def action_sequence(sample, n=6):
    names = [
        turn.get("name")
        for turn in sample.get("history", []) or []
        if turn.get("role") == "assistant_action" and turn.get("name")
    ]
    return " > ".join(names[-n:]) if names else "none"


def history_pairs(sample, max_pairs=6):
    pairs = []
    last_user = None
    for turn in sample.get("history", []) or []:
        if turn.get("role") == "user":
            last_user = clean(turn.get("content"), 900)
        elif turn.get("role") == "assistant_action":
            pairs.append((last_user or "none", turn))
            last_user = None
    return pairs[-max_pairs:]


def serialize_blocks(sample, max_pairs=6):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    flags = prompt_flags(sample)
    lang_mix = ws.get("language_mix", {}) or {}
    mix_items = sorted(lang_mix.items(), key=lambda item: -float(item[1]))[:2]
    fixed = [
        "[NOW] " + clean(sample.get("current_prompt"), 1200),
        "[META] "
        f"tier={clean(meta.get('user_tier'), 40)} "
        f"lang={clean(meta.get('language_pref'), 40)} "
        f"ci={clean(ws.get('last_ci_status'), 40)} "
        f"dirty={ws.get('git_dirty', 'none')} "
        f"turn={clean(meta.get('turn_index'), 40)} "
        f"budget={budget_bucket(meta.get('budget_tokens_remaining'))} "
        f"loc={loc_bucket(ws.get('loc'))}",
        "[OPEN] " + (" ".join(clean(p, 180).replace("\\", "/") for p in (ws.get("open_files", []) or [])) or "none"),
        "[MIX] " + (" ".join(f"{clean(k, 30)}:{float(v):.2f}" for k, v in mix_items) or "none"),
        f"[FLAG] pf={flags['pf']} pf_open={flags['pf_open']} pf_seen={flags['pf_seen']}",
    ]
    pairs = history_pairs(sample, max_pairs=max_pairs)
    history = []
    n = len(pairs)
    for idx, (user_text, action) in enumerate(pairs):
        label = f"H{n - idx}"
        name = clean(action.get("name"), 80)
        args = summarize_args(action.get("args"))
        result = clean(action.get("result_summary"))
        history.append(f"[{label}] U: {user_text}\n[{label}] A: {name}({args}) -> {result}")
    return fixed, history


def serialize_blocks_v2(sample, max_pairs=5):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    flags = prompt_flags(sample)
    wf = workflow_state(sample)
    last = last_action_turn(sample)
    if last:
        last_action = clean(last.get("name"), 80)
        last_args = summarize_args(last.get("args"))
        last_bucket = result_bucket_detail(last.get("result_summary", ""))
        last_result = clean(last.get("result_summary"), 700)
    else:
        last_action, last_args, last_bucket, last_result = "none", "none", "none", "none"
    fixed = [
        "[NOW] " + clean(sample.get("current_prompt"), 1200),
        "[LAST] "
        f"action={last_action} "
        f"args={last_args} "
        f"result_bucket={last_bucket} "
        f"result={last_result}",
        "[STATE] "
        f"test={wf['test']} "
        f"edits_after_test={wf['edits_after_test']} "
        f"insp_since_mod={wf['insp_since_mod']}",
        "[SEQ] actions=" + action_sequence(sample, 6),
        f"[FLAG] pf={flags['pf']} pf_open={flags['pf_open']} pf_seen={flags['pf_seen']} endq={surface_flag_endq(sample)}",
        "[META] "
        f"tier={clean(meta.get('user_tier'), 40)} "
        f"lang={clean(meta.get('language_pref'), 40)} "
        f"ci={clean(ws.get('last_ci_status'), 40)} "
        f"dirty={ws.get('git_dirty', 'none')} "
        f"turn={clean(meta.get('turn_index'), 40)} "
        f"budget={budget_bucket(meta.get('budget_tokens_remaining'))} "
        f"loc={loc_bucket(ws.get('loc'))}",
        "[OPEN] " + (" ".join(clean(p, 180).replace("\\", "/") for p in (ws.get("open_files", []) or [])) or "none"),
    ]
    pairs = history_pairs(sample, max_pairs=max_pairs)
    history = []
    n = len(pairs)
    for idx, (user_text, action) in enumerate(pairs):
        label = f"H{n - idx}"
        name = clean(action.get("name"), 80)
        args = summarize_args(action.get("args"))
        bucket_name = result_bucket_detail(action.get("result_summary", ""))
        result = clean(action.get("result_summary"), 500)
        history.append(f"[{label}] user={user_text}\n[{label}] action={name} args={args} bucket={bucket_name} result={result}")
    return fixed, history


def serialize(sample: dict, variant="v1") -> str:
    fixed, history = serialize_blocks_v2(sample) if variant == "v2" else serialize_blocks(sample)
    return "\n".join(fixed + history)


def serialize_for_tokenizer(sample, tokenizer, max_len, variant="v1"):
    fixed, history = serialize_blocks_v2(sample) if variant == "v2" else serialize_blocks(sample)
    while True:
        text = "\n".join(fixed + history)
        ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        if len(ids) <= max_len or not history:
            return text
        history.pop(0)
