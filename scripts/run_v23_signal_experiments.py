import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "v23_signal_experiments"

from pipeline_v4.serialize import serialize

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
    "inspect4": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "modify3": ["edit_file", "write_file", "apply_patch"],
    "execute3": ["run_bash", "run_tests", "lint_or_typecheck"],
    "communicate4": ["ask_user", "plan_task", "web_search", "respond_only"],
}

FILE_RE = re.compile(r"[\w@~./\\-]+\.[a-z][a-z0-9]{0,9}\b", re.I)
EXT_RE = re.compile(r"\.([a-z][a-z0-9]{0,9})\b", re.I)
JS_EXTS = {"js", "jsx", "ts", "tsx", "mjs", "cjs"}
PY_EXTS = {"py", "pyi", "ipynb"}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row["action"])
    return np.array(rows, dtype=object)


def path_ext(path):
    match = EXT_RE.search(str(path or ""))
    return match.group(1).lower() if match else "none"


def basename(path):
    text = str(path or "").replace("\\", "/").lower()
    return text.rsplit("/", 1)[-1] if text else "none"


def file_mentions(text):
    return [m.group(0).replace("\\", "/").lower() for m in FILE_RE.finditer(text or "")]


def bucket_number(value, bins, labels):
    try:
        value = int(value)
    except Exception:
        return "none"
    for upper, label in zip(bins, labels):
        if value < upper:
            return label
    return labels[-1]


def open_profile(sample):
    ws = ((sample.get("session_meta") or {}).get("workspace") or {})
    open_files = [str(p).replace("\\", "/").lower() for p in (ws.get("open_files") or [])]
    if not open_files:
        return "none"
    if len(open_files) >= 3:
        return "many3+"
    exts = {path_ext(p) for p in open_files if path_ext(p) != "none"}
    if exts and exts <= PY_EXTS:
        return "py_only"
    if exts and exts <= JS_EXTS:
        return "js_only"
    if len(exts) == 1:
        return f"{next(iter(exts))}_only"
    return "mixed"


def prompt_file_rel(sample):
    prompt_files = file_mentions(sample.get("current_prompt", ""))
    if not prompt_files:
        return "no_file"
    ws = ((sample.get("session_meta") or {}).get("workspace") or {})
    open_files = set()
    for path in ws.get("open_files") or []:
        norm = str(path).replace("\\", "/").lower()
        open_files.add(norm)
        open_files.add(basename(norm))
    for path in prompt_files:
        if path in open_files or basename(path) in open_files:
            return "open"
    return "not_open"


def recursive_find_key(value, wanted):
    out = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) == wanted and item:
                out.append(str(item))
            out.extend(recursive_find_key(item, wanted))
    elif isinstance(value, list):
        for item in value:
            out.extend(recursive_find_key(item, wanted))
    return out


def target_symbols(sample):
    symbols = []
    for turn in sample.get("history") or []:
        if turn.get("role") != "assistant_action":
            continue
        symbols.extend(recursive_find_key(turn.get("args"), "target_symbol"))
    cleaned = []
    for symbol in symbols:
        symbol = re.sub(r"\s+", "_", symbol.strip())[:80]
        if symbol:
            cleaned.append(symbol)
    return cleaned[-5:]


def last_args_target_symbols(sample):
    for turn in reversed(sample.get("history") or []):
        if turn.get("role") == "assistant_action":
            return recursive_find_key(turn.get("args"), "target_symbol")[-3:]
    return []


def v23_meta_tokens(sample):
    meta = sample.get("session_meta") or {}
    elapsed = bucket_number(meta.get("elapsed_session_sec"), [60, 300, 1200], ["e0", "e1", "e2", "e3"])
    budget = bucket_number(meta.get("budget_tokens_remaining"), [5000, 20000, 80000], ["b0", "b1", "b2", "b3"])
    return f"[META23] elapsed={elapsed} budget={budget} budget_low5k={int(budget == 'b0')}"


def v23_open_tokens(sample):
    rel = prompt_file_rel(sample)
    profile = open_profile(sample)
    ws = ((sample.get("session_meta") or {}).get("workspace") or {})
    open_exts = sorted({path_ext(p) for p in (ws.get("open_files") or []) if path_ext(p) != "none"})[:8]
    return f"[OPEN23] open_profile={profile} prompt_file_rel={rel} open_exts={','.join(open_exts) or 'none'}"


def v23_target_symbol_tokens(sample):
    recent = target_symbols(sample)
    last = last_args_target_symbols(sample)
    return (
        f"[ARGS23] target_symbol_present={int(bool(recent))} "
        f"recent_target_symbol={' '.join(recent) if recent else 'none'} "
        f"last_target_symbol={' '.join(last) if last else 'none'}"
    )


def v23_text(sample, parts):
    chunks = []
    if "meta" in parts:
        chunks.append(v23_meta_tokens(sample))
    if "open" in parts:
        chunks.append(v23_open_tokens(sample))
    if "target_symbol" in parts:
        chunks.append(v23_target_symbol_tokens(sample))
    return "\n".join(chunks)


def action_lift_rows(samples, y, feature_name, values):
    global_counts = Counter(y.tolist())
    n = len(y)
    rows = []
    by_value = defaultdict(Counter)
    for value, label in zip(values, y):
        by_value[value][label] += 1
    for value, counts in sorted(by_value.items(), key=lambda item: (-sum(item[1].values()), str(item[0]))):
        support = sum(counts.values())
        for action in ALL_CLASSES:
            p_action = global_counts[action] / n
            p_cond = counts[action] / support if support else 0.0
            rows.append(
                {
                    "feature": feature_name,
                    "value": value,
                    "action": action,
                    "support": support,
                    "count": counts[action],
                    "p_action_given_value": p_cond,
                    "global_p_action": p_action,
                    "lift": p_cond / p_action if p_action else 0.0,
                }
            )
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(y, pred):
    row = {"macro_f1": f1_score(y, pred, labels=ALL_CLASSES, average="macro", zero_division=0)}
    for group, classes in GROUPS.items():
        row[f"{group}_f1"] = f1_score(y, pred, labels=classes, average="macro", zero_division=0)
    return row


def train_predict_variant(samples, y, folds, name, extra_parts):
    texts = []
    for sample in samples:
        text = serialize(sample, "v2_2")
        extra = v23_text(sample, extra_parts)
        if extra:
            text = text + "\n" + extra
        texts.append(text)
    texts = np.array(texts, dtype=object)
    pred = np.empty(len(y), dtype=object)
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        val = folds == fold
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=140_000,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
        )
        x_train = vectorizer.fit_transform(texts[train])
        x_val = vectorizer.transform(texts[val])
        clf = SGDClassifier(
            loss="modified_huber",
            penalty="l2",
            alpha=1e-5,
            max_iter=18,
            tol=1e-3,
            class_weight="balanced",
            random_state=42 + int(fold),
        )
        clf.fit(x_train, y[train])
        pred[val] = clf.predict(x_val)
        print(f"{name}: fold={fold} done train={int(train.sum())} val={int(val.sum())}", flush=True)
    return pred


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(ROOT / "data" / "train.jsonl")
    y = load_labels(ROOT / "data" / "train_labels.csv")
    folds = np.load(ROOT / "artifacts" / "distill_step2_strict" / "fold_ids.npy")

    feature_values = {
        "open_profile": [open_profile(sample) for sample in samples],
        "prompt_file_rel": [prompt_file_rel(sample) for sample in samples],
        "target_symbol_present": ["yes" if target_symbols(sample) else "no" for sample in samples],
        "elapsed_bucket": [
            bucket_number((sample.get("session_meta") or {}).get("elapsed_session_sec"), [60, 300, 1200], ["e0", "e1", "e2", "e3"])
            for sample in samples
        ],
        "budget_bucket_v23": [
            bucket_number((sample.get("session_meta") or {}).get("budget_tokens_remaining"), [5000, 20000, 80000], ["b0", "b1", "b2", "b3"])
            for sample in samples
        ],
    }

    lift_rows = []
    for name, values in feature_values.items():
        lift_rows.extend(action_lift_rows(samples, y, name, values))
    write_csv(
        OUT / "signal_lift.csv",
        lift_rows,
        ["feature", "value", "action", "support", "count", "p_action_given_value", "global_p_action", "lift"],
    )

    variants = [
        ("base_v2_2", []),
        ("v23_open", ["open"]),
        ("v23_meta", ["meta"]),
        ("v23_target_symbol", ["target_symbol"]),
        ("v23_all", ["open", "meta", "target_symbol"]),
    ]
    result_rows = []
    preds = {}
    for name, parts in variants:
        pred = train_predict_variant(samples, y, folds, name, parts)
        preds[name] = pred
        row = {"variant": name, **evaluate(y, pred)}
        result_rows.append(row)
    base = result_rows[0]
    for row in result_rows:
        row["delta_macro_f1"] = row["macro_f1"] - base["macro_f1"]
        row["delta_inspect4_f1"] = row["inspect4_f1"] - base["inspect4_f1"]
        row["delta_execute3_f1"] = row["execute3_f1"] - base["execute3_f1"]
        row["delta_communicate4_f1"] = row["communicate4_f1"] - base["communicate4_f1"]

    write_csv(
        OUT / "proxy_model_results.csv",
        result_rows,
        [
            "variant",
            "macro_f1",
            "inspect4_f1",
            "modify3_f1",
            "execute3_f1",
            "communicate4_f1",
            "delta_macro_f1",
            "delta_inspect4_f1",
            "delta_execute3_f1",
            "delta_communicate4_f1",
        ],
    )

    top_lifts = []
    for row in lift_rows:
        if int(row["support"]) >= 250 and float(row["lift"]) >= 1.5:
            top_lifts.append(row)
    top_lifts = sorted(top_lifts, key=lambda r: (-float(r["lift"]), -int(r["support"])))[:40]

    lines = [
        "# V2.3 Signal Experiments",
        "",
        "## Proxy Model Results",
        "",
        "| variant | Macro-F1 | delta | inspect4 | execute3 | communicate4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            f"| `{row['variant']}` | `{row['macro_f1']:.6f}` | `{row['delta_macro_f1']:.6f}` | "
            f"`{row['inspect4_f1']:.6f}` | `{row['execute3_f1']:.6f}` | `{row['communicate4_f1']:.6f}` |"
        )
    lines.extend(["", "## Top Lift Signals", "", "| feature | value | action | support | lift | p(action|value) |", "|---|---|---|---:|---:|---:|"])
    for row in top_lifts[:20]:
        lines.append(
            f"| `{row['feature']}` | `{row['value']}` | `{row['action']}` | `{row['support']}` | "
            f"`{float(row['lift']):.3f}` | `{float(row['p_action_given_value']):.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Treat lift signals as feature candidates only; adoption requires proxy or distill validation.",
            "- `v23_all` must beat `base_v2_2` before spending GPU on a v2.3 transformer serializer.",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]), flush=True)


if __name__ == "__main__":
    main()
