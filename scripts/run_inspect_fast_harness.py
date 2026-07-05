import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "inspect_fast_harness"

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
INSPECT = ["read_file", "grep_search", "list_directory", "glob_pattern"]
INSPECT_SET = set(INSPECT)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    with open(path, encoding="utf-8", newline="") as f:
        return np.array([row["action"] for row in csv.DictReader(f)], dtype=object)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_text(value, limit=500):
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def norm_template(text):
    text = safe_text(text, 1000).lower()
    text = re.sub(r"`[^`]+`", " <quote> ", text)
    text = re.sub(r"['\"][^'\"]+['\"]", " <quote> ", text)
    text = re.sub(r"[a-zA-Z]:[/\\][^\s]+", " <path> ", text)
    text = re.sub(r"[/\\]?[\w.-]+(?:[/\\][\w .-]+)+", " <path> ", text)
    text = re.sub(r"\b[\w.-]+\.(py|js|ts|tsx|jsx|json|md|yaml|yml|txt|csv|sql|ipynb|toml|rs|go|java|cpp|c|h)\b", " <file> ", text)
    text = re.sub(r"\b\d+\b", " <num> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def last_actions(sample, n=4):
    out = []
    for turn in reversed(sample.get("history") or []):
        if turn.get("role") == "assistant_action":
            out.append(str(turn.get("name") or "none"))
            if len(out) >= n:
                break
    return list(reversed(out))


def recent_user_text(sample, n=3):
    users = []
    for turn in reversed(sample.get("history") or []):
        if turn.get("role") == "user":
            users.append(safe_text(turn.get("content"), 500))
            if len(users) >= n:
                break
    return " ".join(reversed(users))


def result_bucket(text):
    low = safe_text(text, 500).lower()
    if not low:
        return "none"
    if any(x in low for x in ["traceback", "exception", "error", "failed", "fail", "permission denied"]):
        return "fail"
    if any(x in low for x in ["no matches", "0 matches", "not found", "zero match"]):
        return "zero_match"
    match = re.search(r"(\d+)\s*(matches?|files?|items?|occurrences?)", low)
    if match:
        n = int(match.group(1))
        if n == 0:
            return "zero_match"
        if n <= 3:
            return "few"
        if n <= 20:
            return "some"
        return "many"
    if any(x in low for x in ["success", "passed", "ok", "done"]):
        return "success"
    return "other"


def inspect_text(sample):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    open_files = ws.get("open_files", []) or []
    chunks = [
        "[NOW] " + safe_text(sample.get("current_prompt"), 1200),
        "[NOW_NORM] " + norm_template(sample.get("current_prompt")),
        "[RECENT_USER] " + recent_user_text(sample),
        "[ACTIONS] " + " ".join(last_actions(sample, 6)),
        "[OPEN] " + " ".join(str(x).replace("\\", "/") for x in open_files[:12]),
        "[META] "
        + " ".join(
            [
                f"lang={meta.get('language_pref', 'none')}",
                f"ci={ws.get('last_ci_status', 'none')}",
                f"dirty={int(bool(ws.get('git_dirty', False)))}",
                f"open_n={len(open_files)}",
            ]
        ),
    ]
    for idx, turn in enumerate((sample.get("history") or [])[-8:]):
        role = turn.get("role")
        if role == "assistant_action":
            args = turn.get("args") or {}
            arg_text = json.dumps(args, ensure_ascii=False, sort_keys=True)[:500]
            chunks.append(
                f"[H{idx}] ACT={turn.get('name', 'none')} RESULT={result_bucket(turn.get('result_summary'))} "
                f"ARGS={arg_text} SUMMARY={safe_text(turn.get('result_summary'), 400)}"
            )
        elif role == "user":
            chunks.append(f"[H{idx}] USER={safe_text(turn.get('content'), 500)}")
    return "\n".join(chunks)


def prompt_template(sample):
    actions = ">".join(last_actions(sample, 3)) or "none"
    last_result = "none"
    for turn in reversed(sample.get("history") or []):
        if turn.get("role") == "assistant_action":
            last_result = result_bucket(turn.get("result_summary"))
            break
    return f"now={norm_template(sample.get('current_prompt'))} || actions={actions} || result={last_result}"


def base_predictions():
    adv = np.load(ROOT / "artifacts" / "advanced_oof_strict" / "advanced_oof_probs.npy").astype(np.float32)
    d2m5 = np.load(ROOT / "reports" / "distill_step2_strict" / "mlp_oof" / "D2-M5" / "oof_probs.npy").astype(np.float32)
    cfg = read_json(ROOT / "reports" / "distill_step2_strict" / "blends" / "best_config.json")
    probs = 0.5 * adv + 0.5 * d2m5
    bias_by_class = cfg["bias"]["bias_by_class"]
    bias = np.array([float(bias_by_class.get(c, 0.0)) for c in ALL_CLASSES], dtype=np.float32)
    scores = np.log(np.clip(probs, 1e-12, 1.0)) + bias[None, :]
    pred = np.array([ALL_CLASSES[i] for i in scores.argmax(axis=1)], dtype=object)
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    conf = softmax(scores, axis=1).max(axis=1)
    return pred, margin, conf


def metrics(y, pred):
    return {
        "macro_f1": f1_score(y, pred, labels=ALL_CLASSES, average="macro", zero_division=0),
        "inspect_f1": f1_score(y, pred, labels=INSPECT, average="macro", zero_division=0),
    }


def classwise(y, pred):
    return {c: f1_score(y, pred, labels=[c], average="macro", zero_division=0) for c in INSPECT}


def fold_stats(y, pred, folds):
    rows = []
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        rows.append(metrics(y[mask], pred[mask])["macro_f1"])
    return min(rows), float(np.mean(rows)), rows


def template_predict(train_templates, train_y, val_templates, min_support, min_purity):
    table = defaultdict(Counter)
    for tpl, label in zip(train_templates, train_y):
        if label in INSPECT_SET:
            table[tpl][label] += 1
    preds = []
    purities = []
    supports = []
    for tpl in val_templates:
        counts = table.get(tpl)
        if not counts:
            preds.append(None)
            purities.append(0.0)
            supports.append(0)
            continue
        action, count = counts.most_common(1)[0]
        total = sum(counts.values())
        preds.append(action if total >= min_support and count / total >= min_purity else None)
        purities.append(count / total)
        supports.append(total)
    return np.array(preds, dtype=object), np.array(purities), np.array(supports)


def evaluate_template(samples, y, folds, base_pred, thresholds):
    templates = np.array([prompt_template(s) for s in samples], dtype=object)
    variants = []
    for min_support, min_purity in thresholds:
        out = base_pred.copy()
        changed = 0
        for fold in sorted(set(folds.tolist())):
            train = folds != fold
            val = folds == fold
            tpl_pred, purity, support = template_predict(
                templates[train],
                y[train],
                templates[val],
                min_support=min_support,
                min_purity=min_purity,
            )
            val_idx = np.where(val)[0]
            for local_i, action in enumerate(tpl_pred):
                i = val_idx[local_i]
                if action is None or base_pred[i] not in INSPECT_SET:
                    continue
                if out[i] != action:
                    changed += 1
                    out[i] = action
        variants.append((f"template_s{min_support}_p{min_purity}", out, changed))
    return variants


def fit_predict_specialist(samples, y, folds, base_pred, model_name):
    texts = np.array([inspect_text(s) for s in samples], dtype=object)
    out_scores = np.full((len(samples), len(INSPECT)), -1e9, dtype=np.float32)
    for fold in sorted(set(folds.tolist())):
        train = (folds != fold) & np.isin(y, INSPECT)
        val = folds == fold
        if model_name == "svc_word_char":
            vectorizer = FeatureUnion(
                [
                    (
                        "word",
                        TfidfVectorizer(
                            analyzer="word",
                            ngram_range=(1, 2),
                            min_df=2,
                            max_features=180_000,
                            sublinear_tf=True,
                            lowercase=True,
                            strip_accents="unicode",
                        ),
                    ),
                    (
                        "char",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(3, 5),
                            min_df=3,
                            max_features=80_000,
                            sublinear_tf=True,
                            lowercase=True,
                        ),
                    ),
                ]
            )
            clf = LinearSVC(C=0.8, class_weight="balanced", max_iter=3000, dual="auto", tol=1e-4)
        elif model_name == "logreg_word":
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_features=180_000,
                sublinear_tf=True,
                lowercase=True,
                strip_accents="unicode",
            )
            clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=700, solver="lbfgs")
        else:
            raise ValueError(model_name)
        x_train = vectorizer.fit_transform(texts[train])
        clf.fit(x_train, y[train])
        score = clf.decision_function(vectorizer.transform(texts[val]))
        if score.ndim == 1:
            score = np.vstack([-score, score]).T
        class_to_col = {c: i for i, c in enumerate(clf.classes_)}
        val_idx = np.where(val)[0]
        for j, cls in enumerate(INSPECT):
            if cls in class_to_col:
                out_scores[val_idx, j] = score[:, class_to_col[cls]]
    top = out_scores.argmax(axis=1)
    sorted_scores = np.sort(out_scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    pred = np.array([INSPECT[i] for i in top], dtype=object)
    variants = []
    for margin_thr in [0.0, 0.2, 0.5, 1.0, 1.5]:
        out = base_pred.copy()
        mask = np.isin(base_pred, INSPECT) & (margins >= margin_thr)
        out[mask] = pred[mask]
        variants.append((f"{model_name}_m{margin_thr}", out, int(mask.sum())))
    return variants


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(ROOT / "data" / "train.jsonl")
    y = load_labels(ROOT / "data" / "train_labels.csv")
    folds = np.load(ROOT / "artifacts" / "distill_step2_strict" / "fold_ids.npy")
    base_pred, base_margin, base_conf = base_predictions()
    base_m = metrics(y, base_pred)
    rows = []

    def add_variant(name, pred, changed):
        m = metrics(y, pred)
        cw = classwise(y, pred)
        min_fold, mean_fold, fold_values = fold_stats(y, pred, folds)
        rows.append(
            {
                "name": name,
                "macro_f1": m["macro_f1"],
                "delta": m["macro_f1"] - base_m["macro_f1"],
                "inspect_f1": m["inspect_f1"],
                "inspect_delta": m["inspect_f1"] - base_m["inspect_f1"],
                "changed": changed,
                "min_fold_macro_f1": min_fold,
                "mean_fold_macro_f1": mean_fold,
                "read_file_f1": cw["read_file"],
                "grep_search_f1": cw["grep_search"],
                "list_directory_f1": cw["list_directory"],
                "glob_pattern_f1": cw["glob_pattern"],
                "folds": ";".join(f"{x:.6f}" for x in fold_values),
            }
        )

    add_variant("base_strict_distill_bias", base_pred, 0)
    for name in ["svc_word_char", "logreg_word"]:
        for variant_name, pred, changed in fit_predict_specialist(samples, y, folds, base_pred, name):
            add_variant(variant_name, pred, changed)
    for variant_name, pred, changed in evaluate_template(
        samples,
        y,
        folds,
        base_pred,
        thresholds=[(2, 0.60), (3, 0.60), (3, 0.70), (5, 0.70), (5, 0.80), (10, 0.75)],
    ):
        add_variant(variant_name, pred, changed)

    rows = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)
    fields = list(rows[0].keys())
    write_csv(OUT / "results.csv", rows, fields)

    best = rows[0]
    lines = [
        "# Inspect Fast Harness",
        "",
        f"- base Macro-F1: `{base_m['macro_f1']:.6f}`",
        f"- base inspect4 Macro-F1: `{base_m['inspect_f1']:.6f}`",
        f"- best: `{best['name']}`",
        f"- best Macro-F1: `{best['macro_f1']:.6f}`",
        f"- delta: `{best['delta']:.6f}`",
        f"- inspect delta: `{best['inspect_delta']:.6f}`",
        f"- changed: `{best['changed']}`",
        "",
        "## Top Variants",
        "",
        "| name | Macro-F1 | delta | inspect4 | inspect_delta | changed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:12]:
        lines.append(
            f"| `{row['name']}` | `{row['macro_f1']:.6f}` | `{row['delta']:.6f}` | "
            f"`{row['inspect_f1']:.6f}` | `{row['inspect_delta']:.6f}` | `{row['changed']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- Adopt only if strict OOF delta is positive, inspect4 delta is positive, and no fold is materially worse.",
            "- If all variants are negative, do not build a public zip for this inspect specialist family.",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
