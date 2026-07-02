from collections import Counter, defaultdict

from src.io_utils import get_session_id


def iter_user_next_action(sample, stop_at_next_user=True):
    history = sample.get("history", []) or []
    for i, turn in enumerate(history):
        if turn.get("role") != "user":
            continue
        prompt = turn.get("content", "")
        if not prompt:
            continue
        for later in history[i + 1 :]:
            if later.get("role") == "assistant_action" and later.get("name"):
                yield prompt, later["name"]
                break
            if stop_at_next_user and later.get("role") == "user":
                break


def build_replay_lookup(samples, scoped=True, policy="last"):
    counts = defaultdict(Counter)
    for sample in samples:
        sid = get_session_id(sample.get("id", ""))
        for prompt, action in iter_user_next_action(sample):
            key = (sid, prompt) if scoped else prompt
            counts[key][action] += 1
    lookup = {}
    conflicts = 0
    for key, counter in counts.items():
        if len(counter) > 1:
            conflicts += 1
        if policy == "unique" and len(counter) != 1:
            continue
        lookup[key] = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return lookup, counts, {"keys": len(counts), "lookup_keys": len(lookup), "conflicts": conflicts}


def replay_key(sample, scoped=True):
    sid = get_session_id(sample.get("id", ""))
    prompt = sample.get("current_prompt", "")
    return (sid, prompt) if scoped else prompt
