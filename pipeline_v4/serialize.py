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


def serialize(sample: dict) -> str:
    fixed, history = serialize_blocks(sample)
    return "\n".join(fixed + history)


def serialize_for_tokenizer(sample, tokenizer, max_len):
    fixed, history = serialize_blocks(sample)
    while True:
        text = "\n".join(fixed + history)
        ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        if len(ids) <= max_len or not history:
            return text
        history.pop(0)
