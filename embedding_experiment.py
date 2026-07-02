import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from script import serialize_sample


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

ACT2I = {name: i for i, name in enumerate(ALL_CLASSES)}


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


def safe_text(value, max_chars=900):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(text.split())[:max_chars]


def build_embed_text(sample):
    hist = sample.get("history", []) or []
    action_names = [
        item.get("name", "")
        for item in hist
        if item.get("role") == "assistant_action" and item.get("name")
    ]
    parts = []
    if action_names:
        parts.append("Recent action sequence: " + " -> ".join(action_names[-6:]))
        parts.append("Last action: " + action_names[-1])
    else:
        parts.append("Recent action sequence: none")
    for item in hist[-6:]:
        if item.get("role") == "user":
            parts.append("User said: " + safe_text(item.get("content"), 500))
        elif item.get("role") == "assistant_action":
            parts.append(
                "Agent did "
                + safe_text(item.get("name"), 80)
                + ". Result: "
                + safe_text(item.get("result_summary"), 500)
            )
    parts.append("Now user says: " + safe_text(sample.get("current_prompt"), 800))
    return "\n".join(parts)


def build_numeric(sample):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    hist = sample.get("history", []) or []
    prompt = sample.get("current_prompt", "") or ""
    low = prompt.lower()
    actions = [
        item.get("name", "")
        for item in hist
        if item.get("role") == "assistant_action" and item.get("name")
    ]
    last = actions[-1] if actions else "NONE"

    def has_any(words):
        return 1.0 if any(word in low for word in words) else 0.0

    values = [
        len(prompt),
        prompt.count("?"),
        1.0 if "?" in prompt else 0.0,
        len(prompt.split()),
        has_any(["show", "read", "open", "cat ", "contents", "look at"]),
        has_any(["find", "search", "grep", "where", "which file", "usage", "reference"]),
        has_any(["list", "ls ", "directory", "folder", "what's in", "whats in", "tree"]),
        has_any(["*.", "glob", "all files", "matching", "pattern", "extension"]),
        has_any(["test", "pytest", "run the", "happy path", "spec"]),
        has_any(["lint", "typecheck", "mypy", "type check", "flake", "ruff"]),
        has_any(["search the web", "google", "look up online", "documentation", "latest"]),
        1.0 if "/" in prompt or "\\" in prompt else 0.0,
        float(prompt.count(".")),
        float(meta.get("budget_tokens_remaining", 0)) / 100_000.0,
        float(meta.get("turn_index", 0)),
        float(meta.get("elapsed_session_sec", 0)) / 600.0,
        float(ws.get("loc", 0)) / 10_000.0,
        1.0 if ws.get("git_dirty") else 0.0,
        float(len(ws.get("open_files", []) or [])),
        float(len(hist)),
    ]

    ci = ws.get("last_ci_status", "none")
    values.extend(1.0 if ci == key else 0.0 for key in ["passed", "failed", "none"])

    last_onehot = [0.0] * (len(ALL_CLASSES) + 1)
    last_onehot[ACT2I.get(last, len(ALL_CLASSES))] = 1.0
    values.extend(last_onehot)

    counts = [0.0] * len(ALL_CLASSES)
    for action in actions:
        if action in ACT2I:
            counts[ACT2I[action]] += 1.0
    values.extend(counts)

    return values


def build_union_vectorizer(max_features=100_000, min_df=2, char_max=5):
    half = max_features // 2
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=half,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, char_max),
        min_df=min_df,
        max_features=half,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )
    return FeatureUnion([("word", word), ("char", char)])


def encode_embeddings(texts, model_name, cache_path, batch_size):
    cache_path = Path(cache_path)
    if cache_path.exists():
        emb = np.load(cache_path)
        print(f"loaded embeddings {cache_path} shape={emb.shape}", flush=True)
        return emb

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"load embedding model={model_name} device={device}", flush=True)
    model = SentenceTransformer(model_name, device=device)
    start = time.time()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    print(f"encoded embeddings shape={emb.shape} sec={time.time() - start:.1f}", flush=True)
    return emb


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def score_model(name, model, x_train, y_train, x_val, y_val, results_path):
    start = time.time()
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    f1 = f1_score(y_val, pred, labels=ALL_CLASSES, average="macro", zero_division=0)
    acc = accuracy_score(y_val, pred)
    report = classification_report(
        y_val,
        pred,
        labels=ALL_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    row = {
        "name": name,
        "macro_f1": f1,
        "accuracy": acc,
        "seconds": time.time() - start,
        "status": "ok",
    }
    append_jsonl(results_path, row)
    print(f"{name}: macro_f1={f1:.6f} acc={acc:.6f} sec={row['seconds']:.1f}", flush=True)
    return model, row, report


def align_scores(classes, scores):
    aligned = np.full((scores.shape[0], len(ALL_CLASSES)), -1e9, dtype=np.float32)
    for src_idx, cls in enumerate(classes):
        aligned[:, ALL_CLASSES.index(cls)] = scores[:, src_idx]
    return aligned


def save_best_combo(run_dir, model_dir, artifacts, config, score, report):
    run_dir = Path(run_dir)
    model_dir = Path(model_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifacts": artifacts,
        "config": config,
        "validation_macro_f1": score,
        "classes": ALL_CLASSES,
    }
    joblib.dump(payload, run_dir / "embedding_best.pkl", compress=3)
    with open(run_dir / "embedding_best_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "validation_macro_f1": score,
                "classes": ALL_CLASSES,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(run_dir / "embedding_best_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    shutil.copy2(run_dir / "embedding_best.pkl", model_dir / "embedding_best.pkl")
    shutil.copy2(run_dir / "embedding_best_config.json", model_dir / "embedding_best_config.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--run-dir", default="./embedding_runs")
    parser.add_argument("--model-dir", default="./model")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--cache-name", default="minilm_train.npy")
    parser.add_argument("--baseline-threshold", type=float, default=0.6272188794254586)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    labels = load_labels(Path(args.data_dir) / "train_labels.csv")
    samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
    y = np.array([labels[sample["id"]] for sample in samples], dtype=object)
    idx = np.arange(len(samples))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    y_train = y[train_idx]
    y_val = y[val_idx]

    print(f"samples={len(samples)} labels={Counter(y)}", flush=True)
    compact_texts = [serialize_sample(sample, "compact") for sample in samples]
    embed_texts = [build_embed_text(sample) for sample in samples]
    num = np.array([build_numeric(sample) for sample in samples], dtype=np.float32)

    emb = encode_embeddings(
        embed_texts,
        args.embedding_model,
        run_dir / args.cache_name,
        args.batch_size,
    )

    best_score = args.baseline_threshold
    best_name = "existing_best_linear"

    emb_train = emb[train_idx]
    emb_val = emb[val_idx]
    num_train_raw = num[train_idx]
    num_val_raw = num[val_idx]

    scaler = StandardScaler()
    num_train = scaler.fit_transform(num_train_raw).astype(np.float32)
    num_val = scaler.transform(num_val_raw).astype(np.float32)

    emb_svc, emb_svc_row, emb_svc_report = score_model(
        "emb_only_lsvc_c1",
        LinearSVC(C=1.0, class_weight="balanced", random_state=42, dual="auto", max_iter=2000),
        emb_train,
        y_train,
        emb_val,
        y_val,
        results_path,
    )
    emb_lr, _, _ = score_model(
        "emb_only_logreg_c5",
        LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0, random_state=42),
        emb_train,
        y_train,
        emb_val,
        y_val,
        results_path,
    )

    emb_num_train = np.hstack([emb_train, num_train]).astype(np.float32)
    emb_num_val = np.hstack([emb_val, num_val]).astype(np.float32)
    score_model(
        "emb_num_lsvc_c1",
        LinearSVC(C=1.0, class_weight="balanced", random_state=42, dual="auto", max_iter=2000),
        emb_num_train,
        y_train,
        emb_num_val,
        y_val,
        results_path,
    )

    print("fit compact union vectorizer", flush=True)
    vectorizer = build_union_vectorizer(max_features=100_000, min_df=2, char_max=5)
    x_text_train = [compact_texts[i] for i in train_idx]
    x_text_val = [compact_texts[i] for i in val_idx]
    tfidf_train = vectorizer.fit_transform(x_text_train)
    tfidf_val = vectorizer.transform(x_text_val)
    print(f"tfidf train shape={tfidf_train.shape}", flush=True)

    tfidf_svc, tfidf_row, tfidf_report = score_model(
        "tfidf_compact_union_lsvc_c1_mf100k",
        LinearSVC(C=1.0, class_weight="balanced", random_state=42, dual="auto", max_iter=2000),
        tfidf_train,
        y_train,
        tfidf_val,
        y_val,
        results_path,
    )

    if tfidf_row["macro_f1"] > best_score:
        best_score = tfidf_row["macro_f1"]
        best_name = tfidf_row["name"]

    combo_reports = {}
    for emb_scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
        x_train = hstack(
            [
                tfidf_train,
                csr_matrix(emb_train * emb_scale, dtype=np.float32),
            ],
            format="csr",
            dtype=np.float32,
        )
        x_val = hstack(
            [
                tfidf_val,
                csr_matrix(emb_val * emb_scale, dtype=np.float32),
            ],
            format="csr",
            dtype=np.float32,
        )
        for c in [0.5, 1.0]:
            name = f"tfidf_emb_lsvc_c{str(c).replace('.', 'p')}_scale{str(emb_scale).replace('.', 'p')}"
            model, row, report = score_model(
                name,
                LinearSVC(C=c, class_weight="balanced", random_state=42, dual="auto", max_iter=2000),
                x_train,
                y_train,
                x_val,
                y_val,
                results_path,
            )
            combo_reports[name] = report
            if row["macro_f1"] > best_score:
                best_score = row["macro_f1"]
                best_name = name

        x_train_num = hstack(
            [
                tfidf_train,
                csr_matrix(emb_train * emb_scale, dtype=np.float32),
                csr_matrix(num_train, dtype=np.float32),
            ],
            format="csr",
            dtype=np.float32,
        )
        x_val_num = hstack(
            [
                tfidf_val,
                csr_matrix(emb_val * emb_scale, dtype=np.float32),
                csr_matrix(num_val, dtype=np.float32),
            ],
            format="csr",
            dtype=np.float32,
        )
        name = f"tfidf_emb_num_lsvc_c1_scale{str(emb_scale).replace('.', 'p')}"
        model, row, report = score_model(
            name,
            LinearSVC(C=1.0, class_weight="balanced", random_state=42, dual="auto", max_iter=2000),
            x_train_num,
            y_train,
            x_val_num,
            y_val,
            results_path,
        )
        combo_reports[name] = report
        if row["macro_f1"] > best_score:
            best_score = row["macro_f1"]
            best_name = name

    tfidf_scores = align_scores(tfidf_svc.classes_, tfidf_svc.decision_function(tfidf_val))
    emb_scores = align_scores(emb_svc.classes_, emb_svc.decision_function(emb_val))
    tfidf_scores = tfidf_scores / (tfidf_scores.std(axis=1, keepdims=True) + 1e-6)
    emb_scores = emb_scores / (emb_scores.std(axis=1, keepdims=True) + 1e-6)
    ensemble_report = None
    for emb_weight in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        scores = (1.0 - emb_weight) * tfidf_scores + emb_weight * emb_scores
        pred = np.array([ALL_CLASSES[i] for i in scores.argmax(axis=1)], dtype=object)
        f1 = f1_score(y_val, pred, labels=ALL_CLASSES, average="macro", zero_division=0)
        acc = accuracy_score(y_val, pred)
        name = f"score_ensemble_tfidf_emb_w{str(emb_weight).replace('.', 'p')}"
        row = {
            "name": name,
            "macro_f1": f1,
            "accuracy": acc,
            "seconds": 0.0,
            "status": "ok",
        }
        append_jsonl(results_path, row)
        print(f"{name}: macro_f1={f1:.6f} acc={acc:.6f}", flush=True)
        if f1 > best_score:
            best_score = f1
            best_name = name
            ensemble_report = classification_report(
                y_val,
                pred,
                labels=ALL_CLASSES,
                output_dict=True,
                zero_division=0,
            )

    summary = {
        "best_name": best_name,
        "best_macro_f1": best_score,
        "baseline_threshold": args.baseline_threshold,
        "embedding_model": args.embedding_model,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("summary", json.dumps(summary, ensure_ascii=False), flush=True)

    if best_score > args.baseline_threshold:
        print("embedding experiment beat current baseline; refit packaging is needed", flush=True)
    else:
        print("embedding experiment did not beat current baseline; keep linear model", flush=True)


if __name__ == "__main__":
    main()
