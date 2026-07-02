import json
import re


PATH_RE = re.compile(r"(?:[\w.-]+/)+[\w.-]+")
FILE_RE = re.compile(
    r"\b[\w.-]+\.(py|tsx|ts|js|jsx|go|rs|java|kt|vue|json|yaml|yml|toml|tf|sql|md|txt|sh|gradle|xml)\b",
    re.I,
)
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
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
