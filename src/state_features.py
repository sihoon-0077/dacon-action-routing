import json
import re


PATH_RE = re.compile(r"(?:[\w.-]+/)+[\w.-]+")
FILE_RE = re.compile(
    r"\b[\w.-]+\.(py|tsx|ts|js|jsx|go|rs|java|kt|vue|json|yaml|yml|toml|tf|sql|md|txt|sh|gradle|xml)\b",
    re.I,
)
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
ID_STYLE_RE = re.compile(r"\b(?:[A-Za-z]+(?:_[A-Za-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]*)\b")
SPACE_RE = re.compile(r"\s+")


def clean(value, max_chars=900):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(text.split())[:max_chars]


def iter_history_pairs(sample):
    hist = sample.get("history", []) or []
    pairs = []
    i = 0
    while i < len(hist):
        if hist[i].get("role") == "user":
            user_text = hist[i].get("content", "")
            action = None
            j = i + 1
            while j < len(hist):
                if hist[j].get("role") == "assistant_action":
                    action = hist[j]
                    break
                if hist[j].get("role") == "user":
                    break
                j += 1
            pairs.append((user_text, action))
            i = j + 1
        else:
            i += 1
    return pairs


def get_last_actions(sample, n=2):
    names = []
    for item in sample.get("history", []) or []:
        if item.get("role") == "assistant_action" and item.get("name"):
            names.append(item["name"])
    return names[-n:]


def get_last_action(sample):
    actions = get_last_actions(sample, 1)
    return actions[-1] if actions else "NONE"


def bucket_result_summary(text):
    t = (text or "").lower()
    buckets = []
    if not t:
        return "RESULT_EMPTY"
    if any(x in t for x in ["error", "fail", "traceback", "permission denied", "exception", "conflict"]):
        buckets.append("RESULT_ERROR")
    if any(x in t for x in ["pass", "green", "lint clean", "exit=0", "exit 0", "success", "ok"]):
        buckets.append("RESULT_OK")
    if "no matches" in t or "0 matches" in t or "not found" in t:
        buckets.append("RESULT_NO_MATCH")
    if any(x in t for x in ["found", "matches", "occurrences", "results"]):
        buckets.append("RESULT_MATCHES")
    if ("read" in t and "lines" in t) or "read " in t:
        buckets.append("RESULT_READ")
    if any(x in t for x in ["patched", "modified", "applied", "wrote", "new file", "edited", "created"]):
        buckets.append("RESULT_EDITED")
    if any(x in t for x in ["listed", "entries", "empty directory", "files", "directory"]):
        buckets.append("RESULT_LISTED")
    m = re.search(r"(\d+)\s+(?:matches|occurrences|results)", t)
    if m:
        n = int(m.group(1))
        if n == 0:
            buckets.append("MATCH_COUNT_0")
        elif n <= 3:
            buckets.append("MATCH_COUNT_LOW")
        elif n <= 20:
            buckets.append("MATCH_COUNT_MID")
        else:
            buckets.append("MATCH_COUNT_HIGH")
    return " ".join(sorted(set(buckets))) if buckets else "RESULT_OTHER"


def normalize_prompt_basic(text):
    text = (text or "").strip().lower()
    return SPACE_RE.sub(" ", text)


def normalize_prompt_template(text):
    text = normalize_prompt_basic(text)
    text = re.sub(r"`[^`]+`", "<CODE>", text)
    text = PATH_RE.sub("<PATH>", text)
    text = FILE_RE.sub("<FILE>", text)
    text = NUM_RE.sub("<NUM>", text)
    return text


def normalize_prompt_v3(text):
    text = (text or "").strip()
    text = re.sub(r"`[^`]+`", "<CODE>", text)
    text = PATH_RE.sub("<F>", text)
    text = FILE_RE.sub("<F>", text)
    text = ID_STYLE_RE.sub("<ID>", text)
    text = NUM_RE.sub("<N>", text)
    text = text.lower()
    return SPACE_RE.sub(" ", text)


def budget_bucket(x):
    try:
        x = int(x)
    except Exception:
        return "BUDGET_UNKNOWN"
    if x < 10_000:
        return "BUDGET_VERY_LOW"
    if x < 50_000:
        return "BUDGET_LOW"
    if x < 120_000:
        return "BUDGET_MID"
    return "BUDGET_HIGH"


def budget_bucket_v3(x):
    try:
        x = int(x)
    except Exception:
        return "b_unknown"
    if x < 5_000:
        return "b0"
    if x < 20_000:
        return "b1"
    if x < 80_000:
        return "b2"
    return "b3"


def loc_bucket_v3(x):
    try:
        x = int(x)
    except Exception:
        return "l_unknown"
    if x < 5_000:
        return "l0"
    if x < 15_000:
        return "l1"
    if x < 40_000:
        return "l2"
    return "l3"


def turn_bucket(turn):
    try:
        turn = int(turn)
    except Exception:
        return "TURN_UNKNOWN"
    if turn <= 1:
        return "TURN_1"
    if turn <= 3:
        return "TURN_2_3"
    if turn <= 6:
        return "TURN_4_6"
    if turn <= 10:
        return "TURN_7_10"
    return "TURN_11_PLUS"


def last_result_bucket(sample):
    pairs = iter_history_pairs(sample)
    if not pairs or not pairs[-1][1]:
        return "RESULT_NONE"
    return bucket_result_summary(pairs[-1][1].get("result_summary", ""))


def result_bucket_v3(text):
    t = (text or "").lower()
    if not t:
        return "none"
    if any(x in t for x in ["error", "fail", "traceback", "exception", "conflict", "permission denied"]):
        return "fail"
    if any(x in t for x in ["no matches", "0 matches", "not found", "zero match"]):
        return "zero_match"
    if any(x in t for x in ["matches", "occurrences", "found", "results"]):
        return "matches"
    if ("read" in t and "line" in t) or "opened" in t:
        return "read_ok"
    return "ok"


def last_result_bucket_v3(sample):
    pairs = iter_history_pairs(sample)
    if not pairs or not pairs[-1][1]:
        return "none"
    return result_bucket_v3(pairs[-1][1].get("result_summary", ""))


def extract_file_mentions(text):
    text = text or ""
    mentions = set()
    for match in PATH_RE.findall(text):
        mentions.add(match.replace("\\", "/").lower())
        mentions.add(match.replace("\\", "/").rsplit("/", 1)[-1].lower())
    for match in FILE_RE.finditer(text):
        mentions.add(match.group(0).lower())
    return mentions


def history_file_mentions(sample):
    mentions = set()
    for turn in sample.get("history", []) or []:
        if turn.get("role") == "assistant_action":
            args = turn.get("args")
            if isinstance(args, dict):
                for value in args.values():
                    if isinstance(value, str):
                        mentions.update(extract_file_mentions(value))
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                mentions.update(extract_file_mentions(item))
        elif turn.get("role") == "user":
            mentions.update(extract_file_mentions(turn.get("content", "")))
    return mentions


def prompt_file_flags(sample):
    prompt_mentions = extract_file_mentions(sample.get("current_prompt", ""))
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    open_mentions = set()
    for path in ws.get("open_files", []) or []:
        norm = str(path).replace("\\", "/").lower()
        open_mentions.add(norm)
        open_mentions.add(norm.rsplit("/", 1)[-1])
    seen_mentions = history_file_mentions(sample)
    return {
        "pf": int(bool(prompt_mentions)),
        "pf_open": int(bool(prompt_mentions & open_mentions)),
        "pf_seen": int(bool(prompt_mentions & seen_mentions)),
    }


def build_signature_v3(sample, level):
    cur_tpl = normalize_prompt_v3(sample.get("current_prompt", ""))
    last_actions = get_last_actions(sample, 2)
    last1 = last_actions[-1] if len(last_actions) >= 1 else "NONE"
    last2 = "|".join(last_actions[-2:]) if last_actions else "NONE"
    result = last_result_bucket_v3(sample)
    flags = prompt_file_flags(sample)
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}

    if level == "S1":
        parts = [cur_tpl]
    elif level == "S2":
        parts = [cur_tpl, f"last={last1}"]
    elif level == "S3":
        parts = [cur_tpl, f"last={last1}", f"result={result}"]
    elif level == "S4":
        parts = [cur_tpl, f"last={last1}", f"result={result}", f"last2={last2}"]
    elif level == "S5":
        parts = [
            cur_tpl,
            f"last={last1}",
            f"result={result}",
            f"last2={last2}",
            f"pf_open={flags['pf_open']}",
            f"pf_seen={flags['pf_seen']}",
            f"ci={ws.get('last_ci_status', 'none')}",
        ]
    else:
        raise ValueError(level)
    return " || ".join(parts)


SIGNATURE_LEVELS_V3 = ["S1", "S2", "S3", "S4", "S5"]


def build_signature(sample, level):
    cur_raw = normalize_prompt_basic(sample.get("current_prompt", ""))
    cur_tpl = normalize_prompt_template(sample.get("current_prompt", ""))
    last_actions = get_last_actions(sample, 3)
    last1 = last_actions[-1] if len(last_actions) >= 1 else "NONE"
    last2 = "|".join(last_actions[-2:]) if last_actions else "NONE"
    last3 = "|".join(last_actions[-3:]) if last_actions else "NONE"
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    result = last_result_bucket(sample)
    open_files = ws.get("open_files", []) or []
    open_exts = sorted(set(path.rsplit(".", 1)[-1].lower() for path in open_files if "." in path))
    lang_mix = ws.get("language_mix", {}) or {}
    lang_keys = sorted(lang_mix.keys())[:5]

    if level == "S0_raw_prompt":
        parts = [cur_raw]
    elif level == "S1_template_prompt":
        parts = [cur_tpl]
    elif level == "S2_tpl_last1":
        parts = [cur_tpl, f"last1={last1}"]
    elif level == "S3_tpl_last2":
        parts = [cur_tpl, f"last2={last2}"]
    elif level == "S4_tpl_last2_result":
        parts = [cur_tpl, f"last2={last2}", f"result={result}"]
    elif level == "S5_tpl_last3_result_meta":
        parts = [
            cur_tpl,
            f"last3={last3}",
            f"result={result}",
            f"ci={ws.get('last_ci_status', 'none')}",
            f"dirty={int(bool(ws.get('git_dirty', False)))}",
            f"turn={turn_bucket(meta.get('turn_index'))}",
            f"budget={budget_bucket(meta.get('budget_tokens_remaining', 0))}",
        ]
    elif level == "S6_tpl_last3_result_open_lang":
        parts = [
            cur_tpl,
            f"last3={last3}",
            f"result={result}",
            f"open_exts={','.join(open_exts)}",
            f"lang={','.join(lang_keys)}",
            f"ci={ws.get('last_ci_status', 'none')}",
            f"turn={turn_bucket(meta.get('turn_index'))}",
        ]
    else:
        raise ValueError(level)
    return " || ".join(parts)


SIGNATURE_LEVELS = [
    "S0_raw_prompt",
    "S1_template_prompt",
    "S2_tpl_last1",
    "S3_tpl_last2",
    "S4_tpl_last2_result",
    "S5_tpl_last3_result_meta",
    "S6_tpl_last3_result_open_lang",
]
