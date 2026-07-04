import json
import re

FILE_RE = re.compile(r"[\w@~./-]+\.[a-z][a-z0-9]{0,9}\b", re.I)
EXT_RE = re.compile(r"\.([a-z][a-z0-9]{0,9})\b", re.I)

INSPECT_ACTIONS = {"read_file", "grep_search", "list_directory", "glob_pattern"}
MODIFY_ACTIONS = {"edit_file", "write_file", "apply_patch"}
EXECUTE_ACTIONS = {"run_bash", "run_tests", "lint_or_typecheck"}


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


def path_basename(path):
    text = str(path or "").replace("\\", "/")
    return text.rsplit("/", 1)[-1] if text else "none"


def path_ext(path):
    match = EXT_RE.search(str(path or ""))
    return match.group(1).lower() if match else "none"


def path_features(sample):
    prompt = sample.get("current_prompt", "") or ""
    prompt_paths = [m.group(0).replace("\\", "/") for m in FILE_RE.finditer(prompt)]
    prompt_exts = sorted({path_ext(path) for path in prompt_paths if path_ext(path) != "none"})[:8]
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    open_files = [str(path).replace("\\", "/") for path in (ws.get("open_files", []) or [])]
    open_exts = sorted({path_ext(path) for path in open_files if path_ext(path) != "none"})[:8]
    return {
        "prompt_file": int(bool(prompt_paths)),
        "prompt_glob": int("*." in prompt or "**/" in prompt or "glob" in prompt.lower()),
        "prompt_slash": int("/" in prompt or "\\" in prompt),
        "prompt_paths": prompt_paths[:5],
        "prompt_basenames": [path_basename(path) for path in prompt_paths[:5]],
        "prompt_exts": prompt_exts,
        "open_files": open_files[:8],
        "open_basenames": [path_basename(path) for path in open_files[:8]],
        "open_exts": open_exts,
        "open_count": len(open_files),
    }


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


def prompt_surface_flags(sample):
    text = sample.get("current_prompt", "") or ""
    low = text.lower()
    stripped = text.strip()
    return {
        "endq": int(stripped.endswith("?") or stripped.endswith("？")),
        "exclaim": int(stripped.endswith("!")),
        "ellipsis": int("..." in text or "…" in text),
        "has_file_word": int(any(x in low for x in ["file", "path", "read", "open", "grep", "glob"])),
        "has_run_word": int(any(x in low for x in ["run", "test", "lint", "typecheck", "execute", "bash"])),
        "has_edit_word": int(any(x in low for x in ["edit", "patch", "write", "modify", "fix", "create"])),
        "has_finish_word": int(any(x in low for x in ["summary", "summarize", "done", "finish", "final"])),
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


def count_bucket_from_result(text):
    low = (text or "").lower()
    if not low:
        return "none"
    if "no matches" in low or "not found" in low:
        return "0"
    patterns = [
        r"\b(\d+)\s+(?:items|entries|files|directories)\b",
        r"\bfound\s+(\d+)\s+(?:files|matches|occurrences|results)\b",
        r"\b(\d+)\s+(?:matches|occurrences|results)\b",
        r"\bmatched\s+(\d+)\s+files\b",
    ]
    counts = []
    for pattern in patterns:
        for match in re.finditer(pattern, low):
            counts.append(int(match.group(1)))
    if not counts:
        return "unknown"
    n = max(counts)
    if n == 0:
        return "0"
    if n <= 3:
        return "1-3"
    if n <= 15:
        return "4-15"
    return "16+"


def result_features(text):
    raw = text or ""
    low = raw.lower()
    flags = []
    if any(x in low for x in ["pass", "green", "exit=0", "exit 0", "lint clean", "ok"]):
        flags.append("RESULT_PASS")
    if any(x in low for x in ["fail", "error", "traceback", "exception", "exit=1", "exit 1"]):
        flags.append("RESULT_FAIL_OR_ERROR")
    if "permission denied" in low:
        flags.append("RESULT_PERMISSION_DENIED")
    if any(x in low for x in ["edit conflict", "context not unique", "target string not found"]):
        flags.append("RESULT_EDIT_CONFLICT")
    if "no matches" in low or re.search(r"\b0\s+matches\b", low):
        flags.append("RESULT_ZERO_MATCH")
    if "matches in" in low or "occurrences" in low or re.search(r"found\s+\d+", low):
        flags.append("RESULT_TEXT_MATCHES")
    if "files matched" in low:
        flags.append("RESULT_FILES_MATCHED")
    if "empty directory" in low:
        flags.append("RESULT_EMPTY_DIRECTORY")
    if any(x in low for x in ["listed", "entries", "items"]):
        flags.append("RESULT_LIST_OK")
    if any(x in low for x in ["read", "lines", "defines:", "classes/functions"]):
        flags.append("RESULT_READ_OK")
    if any(x in low for x in ["new file", "wrote"]):
        flags.append("RESULT_WRITE_OK")
    if any(x in low for x in ["patched", "applied", "modified"]):
        flags.append("RESULT_EDIT_OK")
    return flags or ["RESULT_NONE"]


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


def workflow_state_v22(sample):
    actions = []
    last_test_idx = None
    last_lint_idx = None
    test_state = "never"
    lint_state = "never"
    for turn in sample.get("history", []) or []:
        if turn.get("role") != "assistant_action":
            continue
        idx = len(actions)
        name = turn.get("name") or "none"
        actions.append(turn)
        if name == "run_tests":
            test_state = "fail" if result_bucket_detail(turn.get("result_summary", "")) == "fail" else "pass"
            last_test_idx = idx
        elif name == "lint_or_typecheck":
            lint_state = "fail" if result_bucket_detail(turn.get("result_summary", "")) == "fail" else "pass"
            last_lint_idx = idx

    def edits_after(idx):
        if idx is None:
            return "0"
        n = sum(1 for turn in actions[idx + 1 :] if turn.get("name") in MODIFY_ACTIONS)
        return "2+" if n >= 2 else str(n)

    return {
        "test": test_state,
        "lint": lint_state,
        "edits_after_test": edits_after(last_test_idx),
        "edits_after_lint": edits_after(last_lint_idx),
    }


def inspect_streak_bucket(sample):
    n = 0
    for turn in reversed(sample.get("history", []) or []):
        if turn.get("role") != "assistant_action":
            continue
        if turn.get("name") in INSPECT_ACTIONS:
            n += 1
            continue
        break
    return "4+" if n >= 4 else str(n)


def open_count_bucket(sample):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    n = len(ws.get("open_files", []) or [])
    return "2+" if n >= 2 else str(n)


def prompt_len_bucket(sample):
    n = len((sample.get("current_prompt") or "").strip())
    if n < 40:
        return "s"
    if n < 100:
        return "m"
    return "l"


def normalize_mod_ext(ext):
    ext = (ext or "none").lower()
    if ext in {"py", "ts", "tsx", "js"}:
        return ext
    if ext == "none":
        return "none"
    return "other"


def last_modified_ext(sample):
    for turn in reversed(sample.get("history", []) or []):
        if turn.get("role") != "assistant_action" or turn.get("name") not in MODIFY_ACTIONS:
            continue
        args = turn.get("args")
        values = []
        if isinstance(args, dict):
            for key in ["path", "file", "filename", "target"]:
                value = args.get(key)
                if isinstance(value, str):
                    values.append(value)
                elif isinstance(value, list):
                    values.extend(str(item) for item in value if isinstance(item, str))
            if not values:
                values.append(clean(args))
        for value in values:
            match = FILE_RE.search(value or "")
            if match:
                return normalize_mod_ext(path_ext(match.group(0)))
            ext = path_ext(value)
            if ext != "none":
                return normalize_mod_ext(ext)
    return "none"


def last_list_glob_count_bucket(sample):
    for turn in reversed(sample.get("history", []) or []):
        if turn.get("role") != "assistant_action":
            continue
        name = turn.get("name") or "none"
        if name in {"list_directory", "glob_pattern"}:
            return f"{name}:{count_bucket_from_result(turn.get('result_summary', ''))}"
        return "none"
    return "none"


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


def serialize_blocks_v2_2(sample, max_pairs=5):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    flags = prompt_flags(sample)
    surface = prompt_surface_flags(sample)
    wf = workflow_state_v22(sample)
    last = last_action_turn(sample)
    if last:
        last_action = clean(last.get("name"), 80)
        last_args = summarize_args(last.get("args"))
        last_bucket = result_bucket_detail(last.get("result_summary", ""))
        last_count_bucket = (
            count_bucket_from_result(last.get("result_summary", ""))
            if last.get("name") in {"list_directory", "glob_pattern"}
            else "none"
        )
        last_result = clean(last.get("result_summary"), 700)
    else:
        last_action = last_args = last_bucket = last_count_bucket = last_result = "none"

    fixed = [
        "[NOW] " + clean(sample.get("current_prompt"), 1200),
        "[LAST] "
        f"action={last_action} "
        f"args={last_args} "
        f"result_bucket={last_bucket} "
        f"count_bucket={last_count_bucket} "
        f"result={last_result}",
        "[STATE] "
        f"test={wf['test']} "
        f"lint={wf['lint']} "
        f"edits_after_test={wf['edits_after_test']} "
        f"edits_after_lint={wf['edits_after_lint']} "
        f"insp_streak={inspect_streak_bucket(sample)} "
        f"last_mod_ext={last_modified_ext(sample)} "
        f"open_cnt={open_count_bucket(sample)} "
        f"last_listglob={last_list_glob_count_bucket(sample)}",
        "[SEQ] actions=" + action_sequence(sample, 8),
        "[FLAG] "
        f"pf={flags['pf']} pf_open={flags['pf_open']} pf_seen={flags['pf_seen']} "
        + " ".join(f"{key}={value}" for key, value in surface.items()),
        "[META] "
        f"tier={clean(meta.get('user_tier'), 40)} "
        f"lang={clean(meta.get('language_pref'), 40)} "
        f"ci={clean(ws.get('last_ci_status'), 40)} "
        f"dirty={ws.get('git_dirty', 'none')} "
        f"turn={clean(meta.get('turn_index'), 40)} "
        f"budget={budget_bucket(meta.get('budget_tokens_remaining'))} "
        f"loc={loc_bucket(ws.get('loc'))} "
        f"len_bucket={prompt_len_bucket(sample)}",
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
        count_bucket = (
            count_bucket_from_result(action.get("result_summary", ""))
            if action.get("name") in {"list_directory", "glob_pattern"}
            else "none"
        )
        result = clean(action.get("result_summary"), 500)
        history.append(
            f"[{label}] user={user_text}\n"
            f"[{label}] action={name} args={args} bucket={bucket_name} count={count_bucket} result={result}"
        )
    return fixed, history


def serialize_blocks_xlmr_state_v1(sample, max_pairs=5):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    flags = prompt_flags(sample)
    surface = prompt_surface_flags(sample)
    paths = path_features(sample)
    wf = workflow_state(sample)
    lang_mix = ws.get("language_mix", {}) or {}
    mix_items = sorted(lang_mix.items(), key=lambda item: -float(item[1]))[:4]
    last = last_action_turn(sample)
    if last:
        last_action = clean(last.get("name"), 80)
        last_args = summarize_args(last.get("args"))
        last_bucket = result_bucket_detail(last.get("result_summary", ""))
        last_flags = ",".join(result_features(last.get("result_summary", "")))
        last_result = clean(last.get("result_summary"), 700)
    else:
        last_action = last_args = last_bucket = last_result = "none"
        last_flags = "RESULT_NONE"

    fixed = [
        "[NOW] " + clean(sample.get("current_prompt"), 1200),
        "[LAST] "
        f"action={last_action} "
        f"args={last_args} "
        f"bucket={last_bucket} "
        f"flags={last_flags} "
        f"result={last_result}",
        "[STATE] "
        f"verify={wf['test']} "
        f"edits_after_verify={wf['edits_after_test']} "
        f"inspects_since_modify={wf['insp_since_mod']} "
        f"last_modify={last_action if last_action in {'edit_file', 'write_file', 'apply_patch'} else 'none'}",
        "[SEQ] actions=" + action_sequence(sample, 8),
        "[FILES] "
        f"prompt_file={paths['prompt_file']} "
        f"prompt_glob={paths['prompt_glob']} "
        f"prompt_slash={paths['prompt_slash']} "
        f"prompt_ext={','.join(paths['prompt_exts']) or 'none'} "
        f"mentioned={','.join(clean(p, 80) for p in paths['prompt_basenames']) or 'none'} "
        f"open_count={paths['open_count']} "
        f"open_ext={','.join(paths['open_exts']) or 'none'} "
        f"open={','.join(clean(p, 80) for p in paths['open_basenames']) or 'none'}",
        "[FLAG] "
        f"pf={flags['pf']} pf_open={flags['pf_open']} pf_seen={flags['pf_seen']} "
        + " ".join(f"{key}={value}" for key, value in surface.items()),
        "[META] "
        f"tier={clean(meta.get('user_tier'), 40)} "
        f"lang={clean(meta.get('language_pref'), 40)} "
        f"ci={clean(ws.get('last_ci_status'), 40)} "
        f"dirty={ws.get('git_dirty', 'none')} "
        f"turn={clean(meta.get('turn_index'), 40)} "
        f"budget={budget_bucket(meta.get('budget_tokens_remaining'))} "
        f"loc={loc_bucket(ws.get('loc'))}",
        "[MIX] " + (" ".join(f"{clean(k, 30)}:{float(v):.2f}" for k, v in mix_items) or "none"),
    ]

    pairs = history_pairs(sample, max_pairs=max_pairs)
    history = []
    n = len(pairs)
    for idx, (user_text, action) in enumerate(pairs):
        label = f"H{n - idx}"
        name = clean(action.get("name"), 80)
        args = summarize_args(action.get("args"))
        bucket_name = result_bucket_detail(action.get("result_summary", ""))
        result_flag = ",".join(result_features(action.get("result_summary", "")))
        result = clean(action.get("result_summary"), 500)
        history.append(
            f"[{label}] user={user_text}\n"
            f"[{label}] action={name} args={args} bucket={bucket_name} flags={result_flag} result={result}"
        )
    return fixed, history


def serialize(sample: dict, variant="v1") -> str:
    if variant == "xlmr_state_v1":
        fixed, history = serialize_blocks_xlmr_state_v1(sample)
    elif variant in {"v2_2", "v2.2"}:
        fixed, history = serialize_blocks_v2_2(sample)
    elif variant == "v2":
        fixed, history = serialize_blocks_v2(sample)
    else:
        fixed, history = serialize_blocks(sample)
    return "\n".join(fixed + history)


def serialize_for_tokenizer(sample, tokenizer, max_len, variant="v1"):
    if variant == "xlmr_state_v1":
        fixed, history = serialize_blocks_xlmr_state_v1(sample)
    elif variant in {"v2_2", "v2.2"}:
        fixed, history = serialize_blocks_v2_2(sample)
    elif variant == "v2":
        fixed, history = serialize_blocks_v2(sample)
    else:
        fixed, history = serialize_blocks(sample)
    while True:
        text = "\n".join(fixed + history)
        ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        if len(ids) <= max_len or not history:
            return text
        history.pop(0)
