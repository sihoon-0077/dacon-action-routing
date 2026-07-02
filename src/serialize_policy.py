from src.state_features import (
    budget_bucket,
    budget_bucket_v3,
    clean,
    iter_history_pairs,
    loc_bucket_v3,
    prompt_file_flags,
    turn_bucket,
)


def summarize_args(args, max_chars=220):
    if not isinstance(args, dict) or not args:
        return ""
    useful = ["path", "file", "filename", "target", "pattern", "query", "glob", "command", "cmd", "cwd"]
    parts = []
    for key in useful:
        if key in args and args[key] not in (None, "", []):
            parts.append(f"{key}={clean(args[key], 100)}")
    if not parts:
        for key, value in list(args.items())[:4]:
            parts.append(f"{key}={clean(value, 80)}")
    return " ".join(parts)[:max_chars]


def serialize_policy_state(sample, max_pairs=6, layout="now_first"):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    open_files = ws.get("open_files", []) or []
    language_mix = ws.get("language_mix", {}) or {}
    mix_items = sorted(language_mix.items(), key=lambda x: -float(x[1]))[:3]
    now_chunk = "[NOW] " + clean(sample.get("current_prompt"), 900)
    meta_chunks = [
        "[META] "
        f"tier={clean(meta.get('user_tier'), 40)} "
        f"lang={clean(meta.get('language_pref'), 40)} "
        f"ci={clean(ws.get('last_ci_status'), 40)} "
        f"dirty={ws.get('git_dirty', 'unknown')} "
        f"turn={turn_bucket(meta.get('turn_index'))} "
        f"budget={budget_bucket(meta.get('budget_tokens_remaining', 0))}",
        "[OPEN] " + (" ".join(clean(path, 120).replace("\\", "/") for path in open_files[:8]) or "none"),
        "[MIX] " + (" ".join(f"{clean(k, 20)}:{float(v):.2f}" for k, v in mix_items) or "none"),
    ]
    pairs = []
    for user_text, action in iter_history_pairs(sample)[-max_pairs:]:
        if not action:
            continue
        pairs.append(
            (
                clean(user_text, 420),
                clean(action.get("name"), 80),
                summarize_args(action.get("args"), 220),
                clean(action.get("result_summary"), 420),
            )
        )
    if layout == "now_first":
        chunks = [now_chunk] + meta_chunks
        ordered_pairs = list(reversed(pairs))
    elif layout == "legacy":
        chunks = meta_chunks[:]
        ordered_pairs = pairs
    else:
        raise ValueError(layout)
    for idx, (user_text, action_name, args, result) in enumerate(ordered_pairs, 1):
        chunks.append(f"[H{idx}] U: {user_text} >> A: {action_name} {args} => {result}")
    if layout == "legacy":
        chunks.append(now_chunk)
    return "\n".join(chunks)


def serialize_policy_v3(sample, max_pairs=6):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    open_files = ws.get("open_files", []) or []
    language_mix = ws.get("language_mix", {}) or {}
    mix_items = sorted(language_mix.items(), key=lambda x: -float(x[1]))[:2]
    flags = prompt_file_flags(sample)

    chunks = [
        "[META] "
        f"tier={clean(meta.get('user_tier'), 40) or 'none'} "
        f"lang={clean(meta.get('language_pref'), 40) or 'none'} "
        f"ci={clean(ws.get('last_ci_status'), 40) or 'none'} "
        f"dirty={ws.get('git_dirty', 'none')} "
        f"turn={clean(meta.get('turn_index'), 20) or 'none'} "
        f"budget={budget_bucket_v3(meta.get('budget_tokens_remaining'))} "
        f"loc={loc_bucket_v3(ws.get('loc'))}",
        "[OPEN] " + (" ".join(clean(path, 160).replace("\\", "/") for path in open_files[:10]) or "none"),
        "[MIX] " + (" ".join(f"{clean(k, 20)}:{float(v):.2f}" for k, v in mix_items) or "none"),
        f"[FLAG] pf={flags['pf']} pf_open={flags['pf_open']} pf_seen={flags['pf_seen']}",
    ]

    pairs = iter_history_pairs(sample)[-max_pairs:]
    for idx, (user_text, action) in enumerate(pairs, 1):
        chunks.append(f"[H{idx}] U: {clean(user_text, 520) or 'none'}")
        if action:
            args = summarize_args(action.get("args"), 260)
            action_name = clean(action.get("name"), 80) or "none"
            chunks.append(
                f"[H{idx}] A: {action_name}({args or 'none'}) -> "
                f"{clean(action.get('result_summary'), 520) or 'none'}"
            )
    chunks.append("[NOW] " + (clean(sample.get("current_prompt"), 900) or "none"))
    return "\n".join(chunks)
