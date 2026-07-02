import argparse
import csv
import json
import os
import shutil
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier


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


@dataclass(frozen=True)
class Experiment:
    name: str
    feature_mode: str
    vectorizer: str
    classifier: str
    ngram_max: int = 2
    char_max: int = 5
    max_features: int = 100_000
    min_df: int = 2
    c: float = 1.0
    alpha: float = 0.0001
    max_iter: int = 500


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels(path):
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def safe_text(value, max_chars=1200):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split())
    return text[:max_chars]


def bucket_number(value, bins):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for name, upper in bins:
        if value <= upper:
            return name
    return "huge"


def file_tokens(paths):
    out = []
    for path in paths or []:
        text = safe_text(path, 300).lower()
        out.append("openfile=" + text)
        suffix = Path(text).suffix.lstrip(".")
        if suffix:
            out.append("openext=" + suffix)
        parts = [p for p in text.replace("\\", "/").split("/") if p]
        out.extend("pathpart=" + p for p in parts[-4:])
    return " ".join(out)


def meta_text(sample):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    loc_bin = bucket_number(
        ws.get("loc"),
        [("loc_lt_2k", 2_000), ("loc_lt_10k", 10_000), ("loc_lt_30k", 30_000)],
    )
    budget_bin = bucket_number(
        meta.get("budget_tokens_remaining"),
        [("budget_low", 10_000), ("budget_mid", 60_000), ("budget_high", 140_000)],
    )
    elapsed_bin = bucket_number(
        meta.get("elapsed_session_sec"),
        [("elapsed_start", 240), ("elapsed_mid", 900), ("elapsed_late", 1800)],
    )
    langs = ws.get("language_mix", {}) or {}
    lang_tokens = []
    for lang, ratio in sorted(langs.items()):
        try:
            bucket = int(round(float(ratio) * 10))
        except (TypeError, ValueError):
            bucket = 0
        lang_tokens.append(f"lang={lang} langshare={lang}_{bucket}")
    return " ".join(
        [
            f"tier={safe_text(meta.get('user_tier'), 40)}",
            f"pref={safe_text(meta.get('language_pref'), 40)}",
            f"turn={meta.get('turn_index', 'unknown')}",
            f"turn_bin={bucket_number(meta.get('turn_index'), [('turn_early', 2), ('turn_mid', 7), ('turn_late', 12)])}",
            f"elapsed={elapsed_bin}",
            f"budget={budget_bin}",
            f"git_dirty={ws.get('git_dirty', 'unknown')}",
            f"ci={safe_text(ws.get('last_ci_status'), 40)}",
            loc_bin,
            " ".join(lang_tokens),
            file_tokens(ws.get("open_files", [])),
        ]
    )


def history_text(sample, mode):
    hist = sample.get("history", []) or []
    if not hist:
        return "history_empty last_action=NONE"

    action_names = [
        h.get("name", "")
        for h in hist
        if h.get("role") == "assistant_action" and h.get("name")
    ]
    last_action = action_names[-1] if action_names else "NONE"
    seq = " ".join(f"hist_action={name}" for name in action_names)
    last_seq = " ".join(f"recent_action={name}" for name in action_names[-4:])

    if mode == "actions":
        return f"last_action={last_action} {seq} {last_seq}"

    recent = hist[-6:] if mode == "recent" else hist
    parts = [f"last_action={last_action}", seq, last_seq]
    for item in recent:
        role = item.get("role", "")
        if role == "user":
            parts.append("hist_user=" + safe_text(item.get("content"), 700))
        elif role == "assistant_action":
            parts.append("hist_tool=" + safe_text(item.get("name"), 80))
            parts.append("hist_args=" + safe_text(item.get("args"), 800))
            parts.append("hist_result=" + safe_text(item.get("result_summary"), 800))
        else:
            parts.append("hist_other=" + safe_text(item, 800))
    return " ".join(parts)


def serialize_sample(sample, feature_mode):
    prompt = safe_text(sample.get("current_prompt"), 1200)
    if feature_mode == "prompt":
        return prompt
    if feature_mode == "prompt_x2":
        return f"current={prompt} current_again={prompt}"
    if feature_mode == "prompt_meta":
        return f"current={prompt} meta {meta_text(sample)}"
    if feature_mode == "compact":
        return f"current={prompt} history {history_text(sample, 'actions')} meta {meta_text(sample)}"
    if feature_mode == "compact_no_meta":
        return f"current={prompt} history {history_text(sample, 'actions')}"
    if feature_mode == "compact_prompt_x2":
        return (
            f"current={prompt} current_again={prompt} "
            f"history {history_text(sample, 'actions')} meta {meta_text(sample)}"
        )
    if feature_mode == "compact_prompt_x3":
        return (
            f"current={prompt} current_again={prompt} current_third={prompt} "
            f"history {history_text(sample, 'actions')} meta {meta_text(sample)}"
        )
    if feature_mode == "compact_history_x2":
        hist = history_text(sample, "actions")
        return f"current={prompt} history {hist} history_again {hist} meta {meta_text(sample)}"
    if feature_mode == "compact_prompt_history_x2":
        hist = history_text(sample, "actions")
        return (
            f"current={prompt} current_again={prompt} "
            f"history {hist} history_again {hist} meta {meta_text(sample)}"
        )
    if feature_mode == "recent":
        return f"current={prompt} history {history_text(sample, 'recent')} meta {meta_text(sample)}"
    if feature_mode == "recent_prompt_x2":
        return (
            f"current={prompt} current_again={prompt} "
            f"history {history_text(sample, 'recent')} meta {meta_text(sample)}"
        )
    if feature_mode == "full":
        return f"current={prompt} history {history_text(sample, 'full')} meta {meta_text(sample)}"
    if feature_mode == "full_prompt_x3":
        return (
            f"current={prompt} current_again={prompt} current_third={prompt} "
            f"history {history_text(sample, 'full')} meta {meta_text(sample)}"
        )
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def build_vectorizer(exp):
    word = TfidfVectorizer(
        ngram_range=(1, exp.ngram_max),
        min_df=exp.min_df,
        max_features=exp.max_features,
        sublinear_tf=True,
        lowercase=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, exp.char_max),
        min_df=exp.min_df,
        max_features=exp.max_features,
        sublinear_tf=True,
        lowercase=True,
    )
    if exp.vectorizer == "word":
        return word
    if exp.vectorizer == "char":
        return char
    if exp.vectorizer == "union":
        half = max(20_000, exp.max_features // 2)
        word.set_params(max_features=half)
        char.set_params(max_features=half)
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(f"Unknown vectorizer: {exp.vectorizer}")


def build_classifier(exp):
    if exp.classifier == "logreg":
        return LogisticRegression(
            max_iter=exp.max_iter,
            class_weight="balanced",
            C=exp.c,
            random_state=42,
        )
    if exp.classifier == "linearsvc":
        return LinearSVC(
            C=exp.c,
            class_weight="balanced",
            random_state=42,
            dual="auto",
            max_iter=max(1000, exp.max_iter),
        )
    if exp.classifier == "sgd_hinge":
        return SGDClassifier(
            loss="hinge",
            alpha=exp.alpha,
            max_iter=exp.max_iter,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    if exp.classifier == "sgd_log":
        return SGDClassifier(
            loss="log_loss",
            alpha=exp.alpha,
            max_iter=exp.max_iter,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    if exp.classifier == "cnb":
        return ComplementNB(alpha=exp.alpha)
    raise ValueError(f"Unknown classifier: {exp.classifier}")


def build_pipeline(exp):
    return Pipeline([("tfidf", build_vectorizer(exp)), ("clf", build_classifier(exp))])


def planned_experiments():
    base = [
        Experiment("baseline_prompt_word_lr_c2", "prompt", "word", "logreg", c=2.0, max_features=80_000),
        Experiment("prompt_word_lsvc_c05", "prompt", "word", "linearsvc", c=0.5, max_features=100_000),
        Experiment("prompt_word_lsvc_c1", "prompt", "word", "linearsvc", c=1.0, max_features=120_000),
        Experiment("prompt_word13_lsvc_c1", "prompt", "word", "linearsvc", ngram_max=3, c=1.0, max_features=140_000),
        Experiment("prompt_char_lsvc_c1", "prompt", "char", "linearsvc", c=1.0, max_features=120_000),
        Experiment("prompt_union_lsvc_c1", "prompt", "union", "linearsvc", c=1.0, max_features=160_000),
        Experiment("prompt_meta_word_lsvc_c1", "prompt_meta", "word", "linearsvc", c=1.0, max_features=130_000),
        Experiment("compact_word_lsvc_c1", "compact", "word", "linearsvc", c=1.0, max_features=150_000),
        Experiment("compact_union_lsvc_c1", "compact", "union", "linearsvc", c=1.0, max_features=180_000),
        Experiment("recent_word_lsvc_c1", "recent", "word", "linearsvc", c=1.0, max_features=180_000),
        Experiment("recent_union_lsvc_c1", "recent", "union", "linearsvc", c=1.0, max_features=220_000),
        Experiment("recent_prompt_x2_union_lsvc_c1", "recent_prompt_x2", "union", "linearsvc", c=1.0, max_features=220_000),
        Experiment("full_word_lsvc_c1", "full", "word", "linearsvc", c=1.0, max_features=220_000),
        Experiment("full_prompt_x3_word_lsvc_c1", "full_prompt_x3", "word", "linearsvc", c=1.0, max_features=220_000),
        Experiment("full_prompt_x3_union_lsvc_c1", "full_prompt_x3", "union", "linearsvc", c=1.0, max_features=260_000),
        Experiment("recent_word_lr_c2", "recent", "word", "logreg", c=2.0, max_features=180_000),
        Experiment("recent_union_lr_c2", "recent", "union", "logreg", c=2.0, max_features=220_000),
        Experiment("full_word_lr_c2", "full", "word", "logreg", c=2.0, max_features=220_000),
        Experiment("compact_word_sgd_hinge_a1e4", "compact", "word", "sgd_hinge", alpha=0.0001, max_iter=30, max_features=180_000),
        Experiment("recent_union_sgd_log_a1e5", "recent", "union", "sgd_log", alpha=0.00001, max_iter=30, max_features=220_000),
        Experiment("prompt_cnb_a01", "prompt", "word", "cnb", alpha=0.1, max_features=120_000),
        Experiment("recent_cnb_a01", "recent", "word", "cnb", alpha=0.1, max_features=180_000),
    ]
    extra = []
    for mode in ["compact", "recent", "full_prompt_x3"]:
        for c in [0.25, 0.75, 1.5, 3.0]:
            extra.append(
                Experiment(
                    f"{mode}_union_lsvc_c{str(c).replace('.', 'p')}",
                    mode,
                    "union",
                    "linearsvc",
                    c=c,
                    max_features=220_000,
                )
            )
    for mode in ["recent", "full_prompt_x3"]:
        for max_features in [120_000, 300_000]:
            extra.append(
                Experiment(
                    f"{mode}_word13_lsvc_mf{max_features}",
                    mode,
                    "word",
                    "linearsvc",
                    ngram_max=3,
                    c=1.0,
                    max_features=max_features,
                )
            )
    compact_modes = [
        "compact",
        "compact_no_meta",
        "compact_prompt_x2",
        "compact_prompt_x3",
        "compact_history_x2",
        "compact_prompt_history_x2",
    ]
    for mode in compact_modes:
        for c in [0.15, 0.35, 0.5, 0.6, 0.9, 1.2]:
            extra.append(
                Experiment(
                    f"{mode}_union_lsvc_c{str(c).replace('.', 'p')}_mf180k",
                    mode,
                    "union",
                    "linearsvc",
                    c=c,
                    max_features=180_000,
                )
            )
    for mode in compact_modes:
        for max_features in [100_000, 140_000, 260_000, 340_000]:
            extra.append(
                Experiment(
                    f"{mode}_union_lsvc_c1_mf{max_features}",
                    mode,
                    "union",
                    "linearsvc",
                    c=1.0,
                    max_features=max_features,
                )
            )
    for mode in ["compact", "compact_prompt_x2", "compact_prompt_history_x2"]:
        for char_max in [4, 6]:
            for c in [0.5, 1.0]:
                extra.append(
                    Experiment(
                        f"{mode}_union_lsvc_c{str(c).replace('.', 'p')}_char{char_max}",
                        mode,
                        "union",
                        "linearsvc",
                        c=c,
                        char_max=char_max,
                        max_features=220_000,
                    )
                )
    for mode in ["compact", "compact_prompt_x2", "compact_prompt_history_x2"]:
        for min_df in [1, 3, 5]:
            extra.append(
                Experiment(
                    f"{mode}_union_lsvc_c1_mindf{min_df}",
                    mode,
                    "union",
                    "linearsvc",
                    c=1.0,
                    min_df=min_df,
                    max_features=220_000,
                )
            )
    for mode in ["compact", "compact_prompt_x2"]:
        for c in [0.5, 1.0, 2.0]:
            extra.append(
                Experiment(
                    f"{mode}_word13_lsvc_c{str(c).replace('.', 'p')}",
                    mode,
                    "word",
                    "linearsvc",
                    ngram_max=3,
                    c=c,
                    max_features=260_000,
                )
            )
    for mode in ["compact", "compact_prompt_x2"]:
        for c in [0.5, 1.0, 2.0]:
            extra.append(
                Experiment(
                    f"{mode}_union_lr_c{str(c).replace('.', 'p')}",
                    mode,
                    "union",
                    "logreg",
                    c=c,
                    max_features=220_000,
                    max_iter=800,
                )
            )
    return base + extra


def read_completed(results_path):
    completed = {}
    if not results_path.exists():
        return completed
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") in {"ok", "error"} and row.get("name"):
                completed[row["name"]] = row
    return completed


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def save_best_artifacts(pipe, exp, score, report, run_dir, model_dir):
    run_best_dir = run_dir / "best"
    run_best_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "feature_mode": exp.feature_mode,
        "experiment": asdict(exp),
        "validation_macro_f1": score,
        "classes": ALL_CLASSES,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    joblib.dump(pipe, run_best_dir / "research_best.pkl", compress=3)
    with open(run_best_dir / "research_model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with open(run_best_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    shutil.copy2(run_best_dir / "research_best.pkl", model_dir / "research_best.pkl")
    shutil.copy2(run_best_dir / "research_model_config.json", model_dir / "research_model_config.json")


def make_text_cache(samples, modes):
    cache = {}
    for mode in modes:
        start = time.time()
        cache[mode] = [serialize_sample(sample, mode) for sample in samples]
        print(f"cached mode={mode} rows={len(samples)} sec={time.time() - start:.2f}", flush=True)
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-dir", default="./model")
    parser.add_argument("--run-dir", default="./research_runs")
    parser.add_argument("--time-budget-sec", type=int, default=18_000)
    parser.add_argument("--max-experiments", type=int, default=10_000)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    start_all = time.time()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "runner_status.json"

    samples = load_jsonl(data_dir / "train.jsonl")
    labels = load_labels(data_dir / "train_labels.csv")
    y = [labels[s["id"]] for s in samples]

    split_indices = list(range(len(samples)))
    train_idx, val_idx = train_test_split(
        split_indices,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    y_train = [y[i] for i in train_idx]
    y_val = [y[i] for i in val_idx]

    experiments = planned_experiments()
    if args.smoke:
        experiments = experiments[:2]

    modes = sorted({exp.feature_mode for exp in experiments})
    text_cache = make_text_cache(samples, modes)

    completed = {} if args.no_resume else read_completed(results_path)
    best_score = -1.0
    best_name = None
    for row in completed.values():
        if row.get("status") == "ok" and row.get("macro_f1", -1) > best_score:
            best_score = row["macro_f1"]
            best_name = row["name"]

    label_distribution = dict(Counter(y))
    with open(run_dir / "data_profile.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "samples": len(samples),
                "classes": ALL_CLASSES,
                "label_distribution": label_distribution,
                "history_lengths": dict(Counter(len(s.get("history", [])) for s in samples)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    tried = 0
    for exp in experiments:
        if tried >= args.max_experiments:
            break
        elapsed_all = time.time() - start_all
        if elapsed_all >= args.time_budget_sec:
            break
        if exp.name in completed:
            print(f"skip completed {exp.name}", flush=True)
            continue

        tried += 1
        exp_start = time.time()
        row = {"name": exp.name, "status": "started", "experiment": asdict(exp)}
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)
        print(f"\n=== experiment {tried}: {exp.name} ===", flush=True)

        try:
            texts = text_cache[exp.feature_mode]
            x_train = [texts[i] for i in train_idx]
            x_val = [texts[i] for i in val_idx]
            pipe = build_pipeline(exp)
            pipe.fit(x_train, y_train)
            pred = pipe.predict(x_val)
            score = f1_score(y_val, pred, labels=ALL_CLASSES, average="macro", zero_division=0)
            acc = accuracy_score(y_val, pred)
            report = classification_report(
                y_val,
                pred,
                labels=ALL_CLASSES,
                output_dict=True,
                zero_division=0,
            )
            row = {
                "name": exp.name,
                "status": "ok",
                "macro_f1": score,
                "accuracy": acc,
                "seconds": time.time() - exp_start,
                "experiment": asdict(exp),
                "best_before": best_score,
            }
            append_jsonl(results_path, row)
            print(f"macro_f1={score:.6f} acc={acc:.6f} sec={row['seconds']:.1f}", flush=True)

            if score > best_score:
                print(f"new best: {score:.6f} > {best_score:.6f}; refit full train", flush=True)
                best_score = score
                best_name = exp.name
                full_pipe = build_pipeline(exp)
                full_pipe.fit(texts, y)
                save_best_artifacts(full_pipe, exp, score, report, run_dir, model_dir)
                print("saved best artifacts", flush=True)

        except Exception as exc:
            row = {
                "name": exp.name,
                "status": "error",
                "seconds": time.time() - exp_start,
                "experiment": asdict(exp),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(results_path, row)
            print(f"ERROR {exp.name}: {exc!r}", flush=True)

        summary = {
            "best_name": best_name,
            "best_macro_f1": best_score,
            "elapsed_seconds": time.time() - start_all,
            "completed_count": len(read_completed(results_path)),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"done best={best_name} macro_f1={best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
