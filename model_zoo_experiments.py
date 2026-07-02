import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from embedding_experiment import ALL_CLASSES, build_numeric
from script import serialize_sample


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


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_union_vectorizer(max_features, min_df=2, char_max=5):
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


def to_labels(pred, label_encoder=None):
    arr = np.asarray(pred)
    if arr.dtype.kind in {"i", "u"} and label_encoder is not None:
        return label_encoder.inverse_transform(arr)
    return arr.astype(object)


def row_normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores - scores.mean(axis=1, keepdims=True)
    return scores / (scores.std(axis=1, keepdims=True) + 1e-6)


def align_scores(classes, scores, label_encoder=None):
    aligned = np.full((scores.shape[0], len(ALL_CLASSES)), -1e9, dtype=np.float32)
    for src_idx, cls in enumerate(classes):
        if isinstance(cls, (int, np.integer)) and label_encoder is not None:
            cls = label_encoder.inverse_transform([int(cls)])[0]
        aligned[:, ALL_CLASSES.index(str(cls))] = scores[:, src_idx]
    return aligned


def model_scores(model, x_val, label_encoder=None):
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_val)
        classes = getattr(model, "classes_", ALL_CLASSES)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        return align_scores(classes, scores, label_encoder)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_val)
        classes = getattr(model, "classes_", ALL_CLASSES)
        return align_scores(classes, np.log(np.asarray(proba) + 1e-9), label_encoder)
    pred = to_labels(model.predict(x_val), label_encoder)
    scores = np.zeros((len(pred), len(ALL_CLASSES)), dtype=np.float32)
    for i, label in enumerate(pred):
        scores[i, ALL_CLASSES.index(str(label))] = 1.0
    return scores


def evaluate(name, model, x_train, y_train, x_val, y_val, results_path, label_encoder=None):
    start = time.time()
    row = {"name": name, "status": "started", "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"\n=== {name} ===", flush=True)
    try:
        model.fit(x_train, y_train)
        pred = to_labels(model.predict(x_val), label_encoder)
        y_true = to_labels(y_val, label_encoder)
        macro_f1 = f1_score(y_true, pred, labels=ALL_CLASSES, average="macro", zero_division=0)
        acc = accuracy_score(y_true, pred)
        report = classification_report(
            y_true,
            pred,
            labels=ALL_CLASSES,
            output_dict=True,
            zero_division=0,
        )
        scores = model_scores(model, x_val, label_encoder)
        row.update(
            {
                "status": "ok",
                "macro_f1": macro_f1,
                "accuracy": acc,
                "seconds": time.time() - start,
            }
        )
        append_jsonl(results_path, row)
        print(f"{name}: macro_f1={macro_f1:.6f} acc={acc:.6f} sec={row['seconds']:.1f}", flush=True)
        return {"name": name, "model": model, "row": row, "report": report, "scores": scores}
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "seconds": time.time() - start,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        append_jsonl(results_path, row)
        print(f"ERROR {name}: {exc!r}", flush=True)
        return {"name": name, "model": None, "row": row, "report": None, "scores": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--run-dir", default="./model_zoo_runs")
    parser.add_argument("--embedding-cache", default="./embedding_runs/minilm_train.npy")
    parser.add_argument("--tfidf-max-features", type=int, default=220_000)
    parser.add_argument("--svd-components", type=int, default=256)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
    labels = load_labels(Path(args.data_dir) / "train_labels.csv")
    y_str = np.array([labels[s["id"]] for s in samples], dtype=object)

    label_encoder = LabelEncoder()
    label_encoder.fit(ALL_CLASSES)
    y_int = label_encoder.transform(y_str)

    idx = np.arange(len(samples))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y_str, random_state=42)
    y_train_str, y_val_str = y_str[train_idx], y_str[val_idx]
    y_train_int, y_val_int = y_int[train_idx], y_int[val_idx]

    print(f"samples={len(samples)} train={len(train_idx)} val={len(val_idx)}", flush=True)
    print("build compact texts", flush=True)
    texts = [serialize_sample(s, "compact") for s in samples]
    x_text_train = [texts[i] for i in train_idx]
    x_text_val = [texts[i] for i in val_idx]

    print("fit tfidf union", flush=True)
    vectorizer = build_union_vectorizer(args.tfidf_max_features, min_df=2, char_max=5)
    x_tfidf_train = vectorizer.fit_transform(x_text_train)
    x_tfidf_val = vectorizer.transform(x_text_val)
    print(f"tfidf train shape={x_tfidf_train.shape}", flush=True)

    print("build dense features: SVD + MiniLM embedding + numeric", flush=True)
    svd = TruncatedSVD(n_components=args.svd_components, random_state=42, n_iter=7)
    x_svd_train = svd.fit_transform(x_tfidf_train).astype(np.float32)
    x_svd_val = svd.transform(x_tfidf_val).astype(np.float32)
    print(f"svd explained_variance={svd.explained_variance_ratio_.sum():.4f}", flush=True)

    emb = np.load(args.embedding_cache).astype(np.float32)
    num = np.array([build_numeric(s) for s in samples], dtype=np.float32)
    num_scaler = StandardScaler()
    num_train = num_scaler.fit_transform(num[train_idx]).astype(np.float32)
    num_val = num_scaler.transform(num[val_idx]).astype(np.float32)

    dense_train_raw = np.hstack([x_svd_train, emb[train_idx] * 0.5, num_train]).astype(np.float32)
    dense_val_raw = np.hstack([x_svd_val, emb[val_idx] * 0.5, num_val]).astype(np.float32)
    dense_scaler = StandardScaler()
    dense_train = dense_scaler.fit_transform(dense_train_raw).astype(np.float32)
    dense_val = dense_scaler.transform(dense_val_raw).astype(np.float32)
    print(f"dense shape={dense_train.shape}", flush=True)

    outputs = []

    outputs.append(
        evaluate(
            "01_linear_svc_tfidf_c0p5",
            LinearSVC(C=0.5, class_weight="balanced", random_state=42, dual="auto", max_iter=2500),
            x_tfidf_train,
            y_train_str,
            x_tfidf_val,
            y_val_str,
            results_path,
        )
    )
    outputs.append(
        evaluate(
            "02_logreg_tfidf_c2",
            LogisticRegression(max_iter=900, class_weight="balanced", C=2.0, random_state=42, n_jobs=args.threads),
            x_tfidf_train,
            y_train_str,
            x_tfidf_val,
            y_val_str,
            results_path,
        )
    )
    outputs.append(
        evaluate(
            "03_sgd_hinge_tfidf_alpha1e5",
            SGDClassifier(
                loss="hinge",
                alpha=1e-5,
                max_iter=80,
                tol=1e-4,
                class_weight="balanced",
                n_jobs=args.threads,
                random_state=42,
            ),
            x_tfidf_train,
            y_train_str,
            x_tfidf_val,
            y_val_str,
            results_path,
        )
    )
    outputs.append(
        evaluate(
            "04_complement_nb_tfidf_alpha0p1",
            ComplementNB(alpha=0.1),
            x_tfidf_train,
            y_train_str,
            x_tfidf_val,
            y_val_str,
            results_path,
        )
    )

    try:
        from lightgbm import LGBMClassifier

        outputs.append(
            evaluate(
                "05_lightgbm_svd_emb_num",
                LGBMClassifier(
                    objective="multiclass",
                    num_class=len(ALL_CLASSES),
                    n_estimators=700,
                    learning_rate=0.035,
                    num_leaves=63,
                    min_child_samples=25,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    reg_lambda=2.0,
                    class_weight="balanced",
                    n_jobs=args.threads,
                    random_state=42,
                    verbosity=-1,
                ),
                dense_train_raw,
                y_train_int,
                dense_val_raw,
                y_val_int,
                results_path,
                label_encoder,
            )
        )
    except Exception as exc:
        append_jsonl(results_path, {"name": "05_lightgbm_svd_emb_num", "status": "error", "error": repr(exc)})

    outputs.append(
        evaluate(
            "06_random_forest_svd_emb_num",
            RandomForestClassifier(
                n_estimators=450,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=args.threads,
                random_state=42,
            ),
            dense_train_raw,
            y_train_int,
            dense_val_raw,
            y_val_int,
            results_path,
            label_encoder,
        )
    )
    outputs.append(
        evaluate(
            "07_extra_trees_svd_emb_num",
            ExtraTreesClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                n_jobs=args.threads,
                random_state=42,
            ),
            dense_train_raw,
            y_train_int,
            dense_val_raw,
            y_val_int,
            results_path,
            label_encoder,
        )
    )
    outputs.append(
        evaluate(
            "08_hist_gradient_boosting_svd_emb_num",
            HistGradientBoostingClassifier(
                max_iter=350,
                learning_rate=0.045,
                max_leaf_nodes=63,
                l2_regularization=0.1,
                early_stopping=True,
                class_weight="balanced",
                random_state=42,
            ),
            dense_train,
            y_train_int,
            dense_val,
            y_val_int,
            results_path,
            label_encoder,
        )
    )
    outputs.append(
        evaluate(
            "09_nearest_centroid_svd_emb_num",
            NearestCentroid(metric="euclidean"),
            dense_train,
            y_train_int,
            dense_val,
            y_val_int,
            results_path,
            label_encoder,
        )
    )

    print("\n=== 10_score_voting_ensemble ===", flush=True)
    score_outputs = [o for o in outputs if o.get("scores") is not None]
    ensemble_rows = []
    if score_outputs:
        # Fixed priors first, then a tiny deterministic local search around useful sparse models.
        score_map = {o["name"]: row_normalize_scores(o["scores"]) for o in score_outputs}
        candidate_weights = []
        names = list(score_map)
        candidate_weights.append({name: 1.0 / len(names) for name in names})
        candidate_weights.append(
            {
                "01_linear_svc_tfidf_c0p5": 0.30,
                "02_logreg_tfidf_c2": 0.35,
                "03_sgd_hinge_tfidf_alpha1e5": 0.10,
                "05_lightgbm_svd_emb_num": 0.10,
                "07_extra_trees_svd_emb_num": 0.05,
                "08_hist_gradient_boosting_svd_emb_num": 0.05,
                "04_complement_nb_tfidf_alpha0p1": 0.05,
            }
        )
        for sparse_w in [0.65, 0.75, 0.85]:
            candidate_weights.append(
                {
                    "01_linear_svc_tfidf_c0p5": sparse_w * 0.45,
                    "02_logreg_tfidf_c2": sparse_w * 0.45,
                    "03_sgd_hinge_tfidf_alpha1e5": sparse_w * 0.10,
                    "05_lightgbm_svd_emb_num": (1 - sparse_w) * 0.45,
                    "07_extra_trees_svd_emb_num": (1 - sparse_w) * 0.30,
                    "08_hist_gradient_boosting_svd_emb_num": (1 - sparse_w) * 0.25,
                }
            )

        best = None
        for i, weights in enumerate(candidate_weights):
            total = np.zeros((len(val_idx), len(ALL_CLASSES)), dtype=np.float32)
            used = {}
            weight_sum = 0.0
            for name, weight in weights.items():
                if name in score_map and weight > 0:
                    total += score_map[name] * weight
                    used[name] = weight
                    weight_sum += weight
            if weight_sum <= 0:
                continue
            total /= weight_sum
            pred = np.array([ALL_CLASSES[j] for j in total.argmax(axis=1)], dtype=object)
            macro_f1 = f1_score(y_val_str, pred, labels=ALL_CLASSES, average="macro", zero_division=0)
            acc = accuracy_score(y_val_str, pred)
            row = {
                "name": f"10_score_voting_ensemble_{i}",
                "status": "ok",
                "macro_f1": macro_f1,
                "accuracy": acc,
                "seconds": 0.0,
                "weights": used,
            }
            append_jsonl(results_path, row)
            ensemble_rows.append(row)
            print(f"{row['name']}: macro_f1={macro_f1:.6f} acc={acc:.6f} weights={used}", flush=True)
            if best is None or macro_f1 > best["macro_f1"]:
                best = row

    ok_rows = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") == "ok" and "macro_f1" in row:
                ok_rows.append(row)
    ok_rows = sorted(ok_rows, key=lambda r: r["macro_f1"], reverse=True)
    summary = {
        "best": ok_rows[0] if ok_rows else None,
        "top10": ok_rows[:10],
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": {
            "sparse_features": f"compact TF-IDF word+char max_features={args.tfidf_max_features}",
            "dense_features": f"SVD{args.svd_components} + MiniLM embeddings + numeric/session/action features",
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "svd": svd,
            "num_scaler": num_scaler,
            "dense_scaler": dense_scaler,
            "label_encoder": label_encoder,
            "summary": summary,
        },
        run_dir / "preprocessors.pkl",
        compress=3,
    )
    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
