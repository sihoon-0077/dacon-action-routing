import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.constants import ALL_CLASSES, LABEL2ID, session_of
from pipeline_v4.common.data_io import load_fold_rows, load_train_samples
from pipeline_v4.serialize import (
    action_sequence,
    clean,
    count_bucket_from_result,
    inspect_streak_bucket,
    last_action_turn,
    last_list_glob_count_bucket,
    last_modified_ext,
    open_count_bucket,
    prompt_flags,
    prompt_len_bucket,
    prompt_surface_flags,
    result_bucket_detail,
    serialize,
    summarize_args,
    workflow_state_v22,
)
from script import (
    ADVANCED_ACTION_TO_GROUP,
    ADVANCED_GROUP_TO_ACTIONS,
    advanced_aligned_log_proba,
    advanced_group_text,
    advanced_pair_text,
    advanced_transition_prior_matrix,
    compact_flags_text,
)


SEED = 42
ID2LABEL = {i: c for c, i in LABEL2ID.items()}
GROUP_NAMES = ["inspect", "modify", "execute", "communicate"]
STRONGER_OVERRIDE = {
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "respond_only",
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class Status:
    def __init__(self, path, total_units):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.total_units = float(total_units)
        self.done_units = 0.0
        self.stage = "init"
        self.detail = ""
        self.extra = {}

    def update(self, stage, detail="", done_units=None, add_units=0.0, **extra):
        if done_units is not None:
            self.done_units = float(done_units)
        self.done_units += float(add_units)
        self.stage = stage
        self.detail = detail
        self.extra.update(extra)
        elapsed = time.time() - self.started
        frac = min(max(self.done_units / max(self.total_units, 1.0), 0.0), 0.999)
        eta = elapsed * (1.0 - frac) / max(frac, 1e-6) if frac > 0 else None
        lines = [
            "# Distill Step2 Status",
            "",
            f"- updated_at: `{now()}`",
            f"- stage: `{self.stage}`",
            f"- detail: `{self.detail}`",
            f"- progress_units: `{self.done_units:.2f}/{self.total_units:.2f}`",
            f"- progress_pct: `{frac * 100:.1f}`",
            f"- elapsed_hours: `{elapsed / 3600:.2f}`",
            f"- eta_hours: `{eta / 3600:.2f}`" if eta is not None else "- eta_hours: `unknown`",
            "",
            "## Extra",
        ]
        for key, value in sorted(self.extra.items()):
            lines.append(f"- {key}: `{value}`")
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[{now()}] {stage}: {detail}", flush=True)


def macro_f1(y_true, pred):
    return float(f1_score(y_true, pred, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0))


def metrics_from_probs(y_true, probs):
    pred = probs.argmax(axis=1)
    return {
        "macro_f1": macro_f1(y_true, pred),
        "accuracy": float(accuracy_score(y_true, pred)),
        "nll": float(log_loss(y_true, probs, labels=list(range(len(ALL_CLASSES))))),
    }


def classwise_rows(y_true, pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, pred, labels=np.arange(len(ALL_CLASSES)), zero_division=0
    )
    return [
        {
            "class": ALL_CLASSES[i],
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in range(len(ALL_CLASSES))
    ]


def load_fold_ids(path):
    rows = load_fold_rows(path)
    return {row["id"]: int(row["fold"]) for row in rows}


def build_vectorizer(max_features):
    word_features = int(max_features * 0.625)
    char_features = max_features - word_features
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=word_features,
                    sublinear_tf=True,
                    lowercase=False,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=char_features,
                    sublinear_tf=True,
                    lowercase=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def assemble_teacher(samples, labels, oof_dir, out_dir):
    oof_dir = Path(oof_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    n = len(samples)
    id_to_pos = {sample["id"]: i for i, sample in enumerate(samples)}
    logits = np.zeros((n, len(ALL_CLASSES)), dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    for fold in range(5):
        ids_path = oof_dir / f"fold_{fold}_ids.txt"
        logits_path = oof_dir / f"fold_{fold}_logits.npy"
        if not ids_path.exists() or not logits_path.exists():
            raise FileNotFoundError(f"missing teacher fold {fold}: {ids_path} / {logits_path}")
        fold_ids = ids_path.read_text(encoding="utf-8").splitlines()
        fold_logits = np.load(logits_path).astype(np.float32)
        if fold_logits.shape != (len(fold_ids), len(ALL_CLASSES)):
            raise ValueError(f"teacher fold {fold} shape mismatch: {fold_logits.shape}")
        for row_id, row_logits in zip(fold_ids, fold_logits):
            if row_id not in id_to_pos:
                continue
            pos = id_to_pos[row_id]
            logits[pos] = row_logits
            seen[pos] = True
    if not seen.all():
        raise ValueError(f"teacher OOF missing rows: {int((~seen).sum())}")
    if not np.isfinite(logits).all():
        raise ValueError("teacher logits contain NaN/inf")
    probs = softmax(logits, axis=1).astype(np.float32)
    pred = probs.argmax(axis=1)
    y = np.asarray([LABEL2ID[label] for label in labels], dtype=np.int64)
    metrics = metrics_from_probs(y, probs)
    np.save(out_dir / "teacher_oof_logits.npy", logits)
    np.save(out_dir / "teacher_oof_probs.npy", probs)
    np.save(out_dir / "teacher_oof_pred.npy", pred)
    write_json(
        out_dir / "teacher_oof_meta.json",
        {
            "run": oof_dir.name,
            "n": n,
            "class_order": ALL_CLASSES,
            **metrics,
        },
    )
    return logits, probs, pred, metrics


def dense_dict(sample):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    flags = prompt_flags(sample)
    surface = prompt_surface_flags(sample)
    wf = workflow_state_v22(sample)
    last = last_action_turn(sample)
    last_name = last.get("name") if last else "none"
    last_result = result_bucket_detail(last.get("result_summary", "")) if last else "none"
    data = {
        "tier": str(meta.get("user_tier") or "none"),
        "lang_pref": str(meta.get("language_pref") or "none"),
        "ci": str(ws.get("last_ci_status") or "none"),
        "dirty": int(bool(ws.get("git_dirty", False))),
        "turn_bucket": bucket_num(meta.get("turn_index"), [1, 3, 8, 12], ["t1", "t2_3", "t4_8", "t9_12", "t13p"]),
        "budget_bucket": bucket_num(
            meta.get("budget_tokens_remaining"), [5000, 20000, 80000], ["b0", "b1", "b2", "b3"]
        ),
        "loc_bucket": bucket_num(ws.get("loc"), [5000, 15000, 40000], ["l0", "l1", "l2", "l3"]),
        "last_action": last_name,
        "last2": action_sequence(sample, 2),
        "last_result": last_result,
        "test": wf["test"],
        "lint": wf["lint"],
        "edits_after_test": wf["edits_after_test"],
        "edits_after_lint": wf["edits_after_lint"],
        "insp_streak": inspect_streak_bucket(sample),
        "last_mod_ext": last_modified_ext(sample),
        "open_cnt": open_count_bucket(sample),
        "last_listglob": last_list_glob_count_bucket(sample),
        "len_bucket": prompt_len_bucket(sample),
        "prompt_chars": len(sample.get("current_prompt") or ""),
        "history_len": len(sample.get("history", []) or []),
        "open_n": len(ws.get("open_files", []) or []),
    }
    data.update({f"pf_{k}": int(v) for k, v in flags.items()})
    data.update({f"surface_{k}": int(v) for k, v in surface.items()})
    for path in ws.get("open_files", []) or []:
        ext = str(path).rsplit(".", 1)[-1].lower() if "." in str(path) else "none"
        data[f"open_ext={ext}"] = 1
    for k, v in (ws.get("language_mix", {}) or {}).items():
        data[f"mix_{k}"] = float(v)
    return data


def bucket_num(value, bins, labels):
    try:
        value = int(value)
    except Exception:
        return "unknown"
    for upper, label in zip(bins, labels):
        if value <= upper:
            return label
    return labels[-1]


def predict_advanced_with_scores(samples, artifact, batch_size=2048):
    all_scores = []
    all_preds = []
    all_group_pred = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        coarse_texts = [compact_flags_text(sample) for sample in batch]
        coarse_x = artifact["coarse_vectorizer"].transform(coarse_texts)
        group_pred = artifact["coarse_model"].predict(coarse_x)
        scores = np.full((len(batch), len(ALL_CLASSES)), -1e9, dtype=np.float32)
        for group, actions in ADVANCED_GROUP_TO_ACTIONS.items():
            idx = np.where(group_pred == group)[0]
            if not len(idx):
                continue
            texts = [advanced_group_text(batch[i], group) for i in idx]
            x = artifact["group_vectorizers"][group].transform(texts)
            group_scores = advanced_aligned_log_proba(artifact["group_models"][group], x, actions)
            for j, action in enumerate(actions):
                scores[idx, ALL_CLASSES.index(action)] = group_scores[:, j]
        prior = advanced_transition_prior_matrix(
            batch,
            artifact["transition_last2"],
            artifact["global_counts"],
            artifact["config"].get("prior_smooth", 1.0),
        )
        alpha = float(artifact["config"].get("prior_alpha", 0.3))
        for i, group in enumerate(group_pred):
            for action in ADVANCED_GROUP_TO_ACTIONS[str(group)]:
                j = ALL_CLASSES.index(action)
                scores[i, j] += alpha * prior[i, j]
        prob_like = np.exp(np.clip(scores, -50, 50))
        order = np.argsort(prob_like, axis=1)
        top1 = np.array([ALL_CLASSES[i] for i in order[:, -1]], dtype=object)
        top2 = np.array([ALL_CLASSES[i] for i in order[:, -2]], dtype=object)
        margin = prob_like[np.arange(len(batch)), order[:, -1]] - prob_like[np.arange(len(batch)), order[:, -2]]
        preds = top1.copy()
        pair_thr = float(artifact["config"].get("pair_threshold", 0.08))
        for i, (a, b, m) in enumerate(zip(top1, top2, margin)):
            pair = tuple(sorted((str(a), str(b))))
            resolver = artifact["pair_resolvers"].get(pair)
            if resolver is None or m > pair_thr:
                continue
            text = advanced_pair_text(batch[i], pair)
            x = resolver["vectorizer"].transform([text])
            preds[i] = str(resolver["model"].predict(x)[0])
            scores[i, ALL_CLASSES.index(str(preds[i]))] += 0.2
        all_scores.append(scores)
        all_preds.extend([str(x) for x in preds])
        all_group_pred.extend([str(x) for x in group_pred])
    scores = np.vstack(all_scores)
    probs = softmax(scores, axis=1).astype(np.float32)
    pred_ids = np.asarray([LABEL2ID[p] for p in all_preds], dtype=np.int64)
    group_ids = np.asarray([GROUP_NAMES.index(g) for g in all_group_pred], dtype=np.int64)
    return scores, probs, pred_ids, group_ids


def advanced_feature_matrix(adv_probs, adv_pred, adv_group):
    n = len(adv_pred)
    pred_oh = np.zeros((n, len(ALL_CLASSES)), dtype=np.float32)
    pred_oh[np.arange(n), adv_pred] = 1.0
    group_oh = np.zeros((n, len(GROUP_NAMES)), dtype=np.float32)
    group_oh[np.arange(n), adv_group] = 1.0
    sorted_probs = np.sort(adv_probs, axis=1)
    margin = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1).astype(np.float32)
    conf = sorted_probs[:, -1].reshape(-1, 1).astype(np.float32)
    return np.hstack([adv_probs.astype(np.float32), pred_oh, group_oh, margin, conf]).astype(np.float32)


def build_features(samples, serializer, max_features, svd_dim, artifact_dir, status):
    artifact_dir = Path(artifact_dir)
    ensure_dir(artifact_dir)
    status.update("features", "serializing texts", add_units=0.15)
    texts = [serialize(sample, serializer) for sample in samples]
    with open(artifact_dir / f"texts_{serializer}.jsonl", "w", encoding="utf-8") as f:
        for sample, text in zip(samples, texts):
            f.write(json.dumps({"id": sample["id"], "text": text}, ensure_ascii=False) + "\n")
    status.update("features", "fitting TF-IDF", add_units=0.25)
    vectorizer = build_vectorizer(max_features)
    x_sparse = vectorizer.fit_transform(texts)
    status.update("features", f"TF-IDF shape={x_sparse.shape}, fitting SVD{svd_dim}", add_units=0.5)
    svd = TruncatedSVD(n_components=svd_dim, random_state=SEED, n_iter=7)
    x_svd = svd.fit_transform(x_sparse).astype(np.float32)
    explained = float(svd.explained_variance_ratio_.sum())
    status.update("features", f"SVD done explained={explained:.4f}, dense dicts", add_units=0.2)
    dense_dicts = [dense_dict(sample) for sample in samples]
    dense_encoder = DictVectorizer(sparse=False)
    x_dense = dense_encoder.fit_transform(dense_dicts).astype(np.float32)
    joblib.dump(vectorizer, artifact_dir / "vectorizer.pkl", compress=3)
    svd.components_ = svd.components_.astype(np.float32)
    joblib.dump(svd, artifact_dir / "svd.pkl", compress=3)
    joblib.dump(dense_encoder, artifact_dir / "dense_encoder.pkl", compress=3)
    np.save(artifact_dir / "X_svd.npy", x_svd)
    np.save(artifact_dir / "X_dense.npy", x_dense)
    write_json(
        artifact_dir / "feature_meta.json",
        {
            "serializer": serializer,
            "max_features": max_features,
            "svd_dim": svd_dim,
            "tfidf_shape": list(x_sparse.shape),
            "svd_explained_variance_ratio_sum": explained,
            "dense_dim": int(x_dense.shape[1]),
            "shortcut": "TF-IDF/SVD fitted on full train text for quick distill gate",
        },
    )
    return x_svd, x_dense, vectorizer, svd, dense_encoder


def fit_temperature(logits, y):
    def obj(temp):
        probs = softmax(logits / max(temp, 1e-6), axis=1)
        return log_loss(y, probs, labels=list(range(len(ALL_CLASSES))))

    before = obj(1.0)
    res = minimize_scalar(obj, bounds=(0.5, 5.0), method="bounded")
    temp = float(res.x)
    after = obj(temp)
    return temp, before, after


def run_sgd_oof(name, x, y, fold_ids, out_dir, status, target=None, sample_weight=None):
    target = y if target is None else target
    n = len(y)
    logits = np.zeros((n, len(ALL_CLASSES)), dtype=np.float32)
    pred = np.zeros(n, dtype=np.int64)
    rows = []
    for fold in range(5):
        status.update("fast_students", f"{name} fold{fold}", add_units=0.1)
        train_idx = np.where(fold_ids != fold)[0]
        val_idx = np.where(fold_ids == fold)[0]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_val = scaler.transform(x[val_idx])
        clf = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            penalty="l2",
            max_iter=1500,
            tol=1e-4,
            random_state=SEED + fold,
            class_weight="balanced",
            n_jobs=-1,
        )
        sw = sample_weight[train_idx] if sample_weight is not None else None
        clf.fit(x_train, target[train_idx], sample_weight=sw)
        proba = np.nan_to_num(clf.predict_proba(x_val), nan=0.0, posinf=0.0, neginf=0.0)
        aligned = np.full((len(val_idx), len(ALL_CLASSES)), 1e-9, dtype=np.float32)
        for j, cls in enumerate(clf.classes_):
            aligned[:, int(cls)] = proba[:, j]
        row_sum = aligned.sum(axis=1, keepdims=True)
        bad = row_sum[:, 0] <= 0
        if np.any(bad):
            aligned[bad] = 1.0 / len(ALL_CLASSES)
            row_sum = aligned.sum(axis=1, keepdims=True)
        aligned = aligned / row_sum
        logits[val_idx] = np.log(np.clip(aligned, 1e-9, 1.0))
        pred[val_idx] = aligned.argmax(axis=1)
        rows.append({"fold": fold, **metrics_from_probs(y[val_idx], aligned)})
    probs = softmax(logits, axis=1).astype(np.float32)
    metrics = metrics_from_probs(y, probs)
    out_dir = Path(out_dir) / name
    ensure_dir(out_dir)
    np.save(out_dir / "oof_logits.npy", logits)
    np.save(out_dir / "oof_probs.npy", probs)
    write_json(out_dir / "metrics.json", {"name": name, "oof": metrics, "folds": rows})
    write_csv(out_dir / "classwise.csv", classwise_rows(y, pred), ["class", "precision", "recall", "f1", "support"])
    return {"name": name, "probs": probs, "logits": logits, "metrics": metrics}


class DistillMLP(nn.Module):
    def __init__(self, input_dim, hidden=(512, 256), dropout=(0.15, 0.10)):
        super().__init__()
        layers = []
        prev = input_dim
        for width, drop in zip(hidden, dropout):
            layers.extend([nn.Linear(prev, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(drop)])
            prev = width
        layers.append(nn.Linear(prev, len(ALL_CLASSES)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp_one_fold(x_train, y_train, t_train, x_val, y_val, cfg, device, status_detail):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    model = DistillMLP(x_train.shape[1], cfg["hidden"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.astype(np.int64)),
        torch.from_numpy(t_train.astype(np.float32)),
    )
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    x_val_t = torch.from_numpy(x_val).to(device)
    best = {"f1": -1.0, "epoch": 0, "state": None, "logits": None}
    bad = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for xb, yb, tb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            tb = tb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            hard = ce(logits, yb)
            logp = torch.log_softmax(logits / cfg["temperature"], dim=1)
            soft = kl(logp, tb) * cfg["temperature"] * cfg["temperature"]
            loss = cfg["lambda_hard"] * hard + cfg["lambda_soft"] * soft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu())
        model.eval()
        with torch.inference_mode():
            logits = model(x_val_t).detach().float().cpu().numpy()
        probs = softmax(logits, axis=1)
        f1 = macro_f1(y_val, probs.argmax(axis=1))
        if f1 > best["f1"] + 1e-6:
            best = {
                "f1": float(f1),
                "epoch": epoch,
                "loss": total / max(len(loader), 1),
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "logits": logits.astype(np.float32),
            }
            bad = 0
        else:
            bad += 1
        if epoch % 5 == 0:
            print(f"{status_detail} epoch={epoch} val_f1={f1:.6f} best={best['f1']:.6f}", flush=True)
        if bad >= cfg["patience"]:
            break
    return best, scaler


def run_mlp_oof(configs, feature_sets, y, fold_ids, teacher_probs, out_dir, status):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for cfg in configs:
        x = feature_sets[cfg["feature_set"]][:, : cfg.get("input_dim", feature_sets[cfg["feature_set"]].shape[1])]
        n = len(y)
        logits_all = np.zeros((n, len(ALL_CLASSES)), dtype=np.float32)
        fold_rows = []
        for fold in range(5):
            status.update("mlp_oof", f"{cfg['name']} fold{fold}", add_units=0.22, current_mlp=cfg["name"])
            train_idx = np.where(fold_ids != fold)[0]
            val_idx = np.where(fold_ids == fold)[0]
            best, _ = train_mlp_one_fold(
                x[train_idx],
                y[train_idx],
                teacher_probs[train_idx],
                x[val_idx],
                y[val_idx],
                cfg,
                device,
                f"{cfg['name']} fold{fold}",
            )
            logits_all[val_idx] = best["logits"]
            fold_rows.append({"fold": fold, "best_epoch": best["epoch"], "macro_f1": best["f1"], "loss": best.get("loss")})
        probs = softmax(logits_all, axis=1).astype(np.float32)
        metrics = metrics_from_probs(y, probs)
        cfg_dir = Path(out_dir) / cfg["name"]
        ensure_dir(cfg_dir)
        np.save(cfg_dir / "oof_logits.npy", logits_all)
        np.save(cfg_dir / "oof_probs.npy", probs)
        write_json(cfg_dir / "metrics.json", {"config": serializable_cfg(cfg), "oof": metrics, "folds": fold_rows})
        write_csv(
            cfg_dir / "classwise.csv",
            classwise_rows(y, probs.argmax(axis=1)),
            ["class", "precision", "recall", "f1", "support"],
        )
        results.append({"config": cfg, "probs": probs, "logits": logits_all, "metrics": metrics})
    return results


def serializable_cfg(cfg):
    out = dict(cfg)
    out["hidden"] = list(out["hidden"])
    out["dropout"] = list(out["dropout"])
    return out


def blend_sweep(y, student_results, adv_probs, out_dir, status):
    rows = []
    best = None
    for result in student_results:
        name = result["config"]["name"] if "config" in result else result["name"]
        s_probs = result["probs"]
        for w in [0.3, 0.5, 0.7, 0.9, 1.0]:
            probs = w * s_probs + (1.0 - w) * adv_probs
            m = metrics_from_probs(y, probs)
            row = {"student": name, "w_student": w, **m}
            rows.append(row)
            if best is None or row["macro_f1"] > best["macro_f1"]:
                best = {**row, "probs": probs, "student_probs": s_probs, "student_name": name}
    write_csv(
        Path(out_dir) / "blend_results.csv",
        [{k: v for k, v in row.items() if k != "probs"} for row in rows],
        ["student", "w_student", "macro_f1", "accuracy", "nll"],
    )
    status.update("blend_bias", f"best blend {best['student']} w={best['w_student']} f1={best['macro_f1']:.6f}", add_units=0.3)
    return best, rows


def fit_bias_half_split(y, probs, ids, out_dir):
    groups = np.asarray([session_of(x) for x in ids], dtype=object)
    idx = np.arange(len(y))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    a_idx, b_idx = next(splitter.split(idx, y, groups=groups))
    logp = np.log(np.clip(probs, 1e-12, 1.0))

    def optimize(train_idx):
        bias = np.zeros(len(ALL_CLASSES), dtype=np.float32)
        best = macro_f1(y[train_idx], (logp[train_idx] + bias[None, :]).argmax(axis=1))
        grid = np.arange(-0.5, 0.5001, 0.05)
        for _ in range(2):
            changed = False
            for cls in range(len(ALL_CLASSES)):
                local = bias[cls]
                local_best = best
                for value in grid:
                    cand = bias.copy()
                    cand[cls] = value
                    f1 = macro_f1(y[train_idx], (logp[train_idx] + cand[None, :]).argmax(axis=1))
                    f1 -= 0.01 * float(np.square(cand).sum())
                    if f1 > local_best:
                        local_best = f1
                        local = value
                if local != bias[cls]:
                    changed = True
                bias[cls] = local
                best = local_best
            if not changed:
                break
        return bias

    bias_a = optimize(a_idx)
    bias_b = optimize(b_idx)

    def eval_bias(eval_idx, bias):
        base = macro_f1(y[eval_idx], logp[eval_idx].argmax(axis=1))
        pred = (logp[eval_idx] + bias[None, :]).argmax(axis=1)
        new = macro_f1(y[eval_idx], pred)
        return base, new, new - base

    base_b, new_b, d_ab = eval_bias(b_idx, bias_a)
    base_a, new_a, d_ba = eval_bias(a_idx, bias_b)
    avg = float((d_ab + d_ba) / 2)
    adopted = bool(d_ab >= 0 and d_ba >= 0 and avg >= 0.001)
    full_bias = optimize(idx) if adopted else np.zeros(len(ALL_CLASSES), dtype=np.float32)
    biased_probs = softmax(logp + full_bias[None, :], axis=1)
    payload = {
        "A_to_B": {"base": base_b, "new": new_b, "delta": d_ab},
        "B_to_A": {"base": base_a, "new": new_a, "delta": d_ba},
        "avg_delta": avg,
        "adopted": adopted,
        "bias_by_class": {cls: float(full_bias[i]) for i, cls in enumerate(ALL_CLASSES)},
        "full_metrics": metrics_from_probs(y, biased_probs),
    }
    write_json(Path(out_dir) / "bias_results.json", payload)
    return payload, biased_probs


def train_full_student(x, y, teacher_probs, cfg, model_dir, status):
    status.update("full_train", f"training final {cfg['name']}", add_units=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32)
    model = DistillMLP(x_scaled.shape[1], cfg["hidden"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    ds = TensorDataset(
        torch.from_numpy(x_scaled),
        torch.from_numpy(y.astype(np.int64)),
        torch.from_numpy(teacher_probs.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    epochs = max(int(round(np.mean(cfg.get("best_epochs", [12])))), 8)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            hard = ce(logits, yb)
            soft = kl(torch.log_softmax(logits / cfg["temperature"], dim=1), tb) * cfg["temperature"] * cfg["temperature"]
            loss = cfg["lambda_hard"] * hard + cfg["lambda_soft"] * soft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu())
        print(f"full {cfg['name']} epoch={epoch}/{epochs} loss={total / max(len(loader), 1):.5f}", flush=True)
    model_dir = Path(model_dir)
    ensure_dir(model_dir)
    torch.save({"state_dict": model.state_dict(), "input_dim": x.shape[1], "config": serializable_cfg(cfg)}, model_dir / "student.pt")
    joblib.dump(scaler, model_dir / "scaler.pkl", compress=3)
    return model_dir


def make_mlp_configs(epochs):
    base = {
        "feature_set": "x_adv_768",
        "lambda_hard": 0.65,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 1024,
        "epochs": epochs,
        "patience": 5,
    }
    return [
        {**base, "name": "D2-M1", "lambda_soft": 0.20, "temperature": 2.0, "hidden": (512, 256), "dropout": (0.15, 0.10)},
        {**base, "name": "D2-M2", "lambda_soft": 0.35, "temperature": 2.0, "hidden": (512, 256), "dropout": (0.15, 0.10)},
        {**base, "name": "D2-M3", "lambda_soft": 0.50, "temperature": 2.0, "hidden": (512, 256), "dropout": (0.15, 0.10)},
        {**base, "name": "D2-M4", "lambda_soft": 0.35, "temperature": 3.0, "hidden": (512, 256), "dropout": (0.15, 0.10)},
        {
            **base,
            "name": "D2-M5",
            "feature_set": "x_adv_512",
            "lambda_soft": 0.35,
            "temperature": 2.0,
            "hidden": (256, 128),
            "dropout": (0.10, 0.10),
        },
        {**base, "name": "D2-M6", "lambda_soft": 0.35, "temperature": 2.0, "hidden": (768, 384), "dropout": (0.20, 0.10)},
    ]


def write_summary(path, teacher_metrics, adv_metrics, fast_results, mlp_results, best_blend, bias_payload, adopted, submit_zip):
    lines = [
        "# Distill Step2 Summary",
        "",
        f"- finished_at: `{now()}`",
        f"- teacher OOF Macro-F1: `{teacher_metrics['macro_f1']:.6f}`",
        f"- advanced full-fit feature baseline Macro-F1: `{adv_metrics['macro_f1']:.6f}`",
        f"- best blend Macro-F1: `{best_blend['macro_f1']:.6f}`",
        f"- bias adopted: `{bias_payload['adopted']}`",
        f"- final adopted: `{adopted}`",
        f"- submit zip: `{submit_zip or 'not_built'}`",
        "",
        "## Fast Students",
        "",
        "| Name | Macro-F1 | Accuracy | NLL |",
        "|---|---:|---:|---:|",
    ]
    for res in fast_results:
        m = res["metrics"]
        lines.append(f"| `{res['name']}` | `{m['macro_f1']:.6f}` | `{m['accuracy']:.6f}` | `{m['nll']:.6f}` |")
    lines.extend(["", "## MLP OOF", "", "| Name | Macro-F1 | Accuracy | NLL |", "|---|---:|---:|---:|"])
    for res in mlp_results:
        m = res["metrics"]
        lines.append(f"| `{res['config']['name']}` | `{m['macro_f1']:.6f}` | `{m['accuracy']:.6f}` | `{m['nll']:.6f}` |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Teacher probabilities were used only as training targets, not inference features.",
            "- TF-IDF/SVD was fit on full train text for the quick full battery; record this as unsupervised feature-cache shortcut.",
            "- Advanced router probabilities were recomputed from the existing full-fit artifact because they are available at inference time.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--fold-file", default="pipeline_v4/folds/fold_assignments.csv")
    parser.add_argument("--teacher-oof", default="pipeline_v4/artifacts/oof/mdeberta384_v2_384_5e")
    parser.add_argument("--advanced-model", default="model/advanced_router.pkl")
    parser.add_argument("--artifact-dir", default="artifacts/distill_step2")
    parser.add_argument("--report-dir", default="reports/distill_step2")
    parser.add_argument("--model-dir", default="model/distill_student")
    parser.add_argument("--serializer", default="v2_2")
    parser.add_argument("--max-features", type=int, default=160_000)
    parser.add_argument("--svd-dim", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--smoke-rows", type=int, default=0)
    parser.add_argument("--smoke-configs", default="")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ensure_dir(args.artifact_dir)
    ensure_dir(args.report_dir)
    status = Status(Path(args.report_dir) / "status.md", total_units=10.0)
    (Path(args.artifact_dir) / "runner_pid.txt").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    status.update("start", "loading train data", done_units=0.1)

    samples = load_train_samples(args.data_dir)
    if args.smoke_rows:
        samples = samples[: args.smoke_rows]
    labels = [sample["action"] for sample in samples]
    y = np.asarray([LABEL2ID[label] for label in labels], dtype=np.int64)
    ids = [sample["id"] for sample in samples]
    fold_map = load_fold_ids(args.fold_file)
    fold_ids = np.asarray([fold_map[sample["id"]] for sample in samples], dtype=np.int64)
    write_json(Path(args.artifact_dir) / "class_order.json", ALL_CLASSES)
    np.save(Path(args.artifact_dir) / "y.npy", y)
    np.save(Path(args.artifact_dir) / "fold_ids.npy", fold_ids)

    teacher_logits, teacher_probs, teacher_pred, teacher_metrics = assemble_teacher(
        samples, labels, args.teacher_oof, Path(args.artifact_dir) / "teacher_oof"
    )
    status.update("teacher_audit", f"teacher f1={teacher_metrics['macro_f1']:.6f}", done_units=1.0)
    write_csv(
        Path(args.report_dir) / "d2_0_asset_audit" / "teacher_classwise_f1.csv",
        classwise_rows(y, teacher_pred),
        ["class", "precision", "recall", "f1", "support"],
    )
    write_json(Path(args.report_dir) / "d2_0_asset_audit" / "teacher_metrics.json", teacher_metrics)
    if teacher_metrics["macro_f1"] < 0.710 and not args.smoke_rows:
        raise RuntimeError(f"teacher gate failed: {teacher_metrics['macro_f1']:.6f}")

    x_svd, x_dense, vectorizer, svd, dense_encoder = build_features(
        samples, args.serializer, args.max_features, args.svd_dim, Path(args.artifact_dir) / "features", status
    )
    status.update("advanced", "computing advanced features", done_units=2.2)
    adv_artifact = joblib.load(args.advanced_model)
    adv_scores, adv_probs, adv_pred, adv_group = predict_advanced_with_scores(samples, adv_artifact)
    adv_features = advanced_feature_matrix(adv_probs, adv_pred, adv_group)
    np.save(Path(args.artifact_dir) / "features" / "advanced_probs.npy", adv_probs)
    np.save(Path(args.artifact_dir) / "features" / "advanced_features.npy", adv_features)
    adv_metrics = metrics_from_probs(y, adv_probs)
    write_json(Path(args.report_dir) / "advanced_metrics.json", adv_metrics)

    x_noadv = np.hstack([x_svd, x_dense]).astype(np.float32)
    x_adv = np.hstack([x_svd, x_dense, adv_features]).astype(np.float32)
    feature_sets = {
        "x_noadv_768": x_noadv,
        "x_adv_768": x_adv,
        "x_adv_512": np.hstack([x_svd[:, : min(512, x_svd.shape[1])], x_dense, adv_features]).astype(np.float32),
    }
    status.update("fast_students", "running SGD gates", done_units=3.0, advanced_f1=f"{adv_metrics['macro_f1']:.6f}")
    fast_dir = Path(args.report_dir) / "fast_students"
    fast_results = []
    fast_results.append(run_sgd_oof("D2-G1_hard_noadv", x_noadv, y, fold_ids, fast_dir, status))
    fast_results.append(run_sgd_oof("D2-G2_hard_adv", x_adv, y, fold_ids, fast_dir, status))
    teacher_top1 = teacher_probs.argmax(axis=1).astype(np.int64)
    teacher_conf = teacher_probs.max(axis=1).astype(np.float32)
    for thr, beta in [(0.55, 0.4), (0.65, 0.4), (0.75, 0.6)]:
        target = y.copy()
        mask = teacher_conf >= thr
        target[mask] = teacher_top1[mask]
        sw = np.ones(len(y), dtype=np.float32)
        sw[mask] += beta * teacher_conf[mask]
        fast_results.append(run_sgd_oof(f"D2-G3_pseudo_t{thr}_b{beta}", x_adv, y, fold_ids, fast_dir, status, target, sw))
    hybrid_target = y.copy()
    hybrid = adv_pred.copy()
    mask = np.asarray([ALL_CLASSES[i] in STRONGER_OVERRIDE for i in teacher_top1])
    hybrid[mask] = teacher_top1[mask]
    fast_results.append(run_sgd_oof("D2-G4_hybrid_imitation", x_adv, y, fold_ids, fast_dir, status, hybrid))

    status.update("mlp_oof", "running MLP distillation configs", done_units=4.2)
    configs = make_mlp_configs(args.epochs)
    if args.smoke_configs:
        keep = set(args.smoke_configs.split(","))
        configs = [cfg for cfg in configs if cfg["name"] in keep]
    mlp_results = run_mlp_oof(configs, feature_sets, y, fold_ids, teacher_probs, Path(args.report_dir) / "mlp_oof", status)

    status.update("blend_bias", "sweeping blends", done_units=8.2)
    student_candidates = fast_results + mlp_results
    best_blend, blend_rows = blend_sweep(y, student_candidates, adv_probs, Path(args.report_dir) / "blends", status)
    bias_payload, biased_probs = fit_bias_half_split(y, best_blend["probs"], ids, Path(args.report_dir) / "blends")
    final_probs = biased_probs if bias_payload["adopted"] else best_blend["probs"]
    final_metrics = metrics_from_probs(y, final_probs)
    best_payload = {
        "best_student": best_blend["student_name"],
        "w_student": best_blend["w_student"],
        "blend_metrics": {k: float(v) for k, v in best_blend.items() if k in {"macro_f1", "accuracy", "nll"}},
        "bias": bias_payload,
        "final_metrics": final_metrics,
        "advanced_reference_f1": 0.7113236414043568,
        "adopt_gate": "final_macro_f1 >= 0.716323641",
    }
    write_json(Path(args.report_dir) / "blends" / "best_config.json", best_payload)

    # The current full battery recomputes advanced-router probabilities from a
    # full-fit artifact. That is valid as an inference feature, but not a strict
    # OOF validation signal. Keep the package gate conservative until a strict
    # advanced OOF feature cache exists.
    adopted = False
    submit_zip = None
    if adopted:
        selected = None
        for res in mlp_results:
            if res["config"]["name"] == best_blend["student_name"]:
                selected = res["config"]
                break
        if selected is None:
            selected = make_mlp_configs(args.epochs)[1]
            selected["name"] = "D2-final-from-fast-fallback"
        selected["best_epochs"] = [12]
        train_full_student(feature_sets[selected["feature_set"]], y, teacher_probs, selected, args.model_dir, status)
        joblib.dump(vectorizer, Path(args.model_dir) / "vectorizer.pkl", compress=3)
        joblib.dump(svd, Path(args.model_dir) / "svd.pkl", compress=3)
        joblib.dump(dense_encoder, Path(args.model_dir) / "dense_encoder.pkl", compress=3)
        shutil.copy2(args.advanced_model, Path(args.model_dir) / "advanced_router.pkl")
        write_json(
            Path(args.model_dir) / "config.json",
            {
                "class_order": ALL_CLASSES,
                "serializer": args.serializer,
                "feature_set": selected["feature_set"],
                "w_student": best_blend["w_student"],
                "w_advanced": 1.0 - best_blend["w_student"],
                "bias_by_class": bias_payload["bias_by_class"] if bias_payload["adopted"] else {c: 0.0 for c in ALL_CLASSES},
                "selected_config": serializable_cfg(selected),
            },
        )
        submit_zip = None

    write_summary(
        Path(args.report_dir) / "SUMMARY.md",
        teacher_metrics,
        adv_metrics,
        fast_results,
        mlp_results,
        best_blend,
        bias_payload,
        adopted,
        submit_zip,
    )
    status.update(
        "finished",
        f"final_f1={final_metrics['macro_f1']:.6f} adopted={adopted}",
        done_units=10.0,
        final_f1=f"{final_metrics['macro_f1']:.6f}",
        adopted=adopted,
    )


if __name__ == "__main__":
    main()
