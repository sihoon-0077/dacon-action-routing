import argparse
import shutil
import zipfile
from pathlib import Path


DISTILL_SCRIPT = r'''
import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

import router_base
from pipeline_v4.serialize import (
    action_sequence,
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
    workflow_state_v22,
)


ALL_CLASSES = router_base.ALL_CLASSES
LABEL2ID = {c: i for i, c in enumerate(ALL_CLASSES)}
GROUP_NAMES = ["inspect", "modify", "execute", "communicate"]


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bucket_num(value, bins, labels):
    try:
        value = int(value)
    except Exception:
        return "unknown"
    for upper, label in zip(bins, labels):
        if value <= upper:
            return label
    return labels[-1]


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
        "budget_bucket": bucket_num(meta.get("budget_tokens_remaining"), [5000, 20000, 80000], ["b0", "b1", "b2", "b3"]),
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


def predict_advanced_with_scores(samples, artifact, batch_size=2048):
    all_scores = []
    all_preds = []
    all_groups = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        coarse_texts = [router_base.compact_flags_text(sample) for sample in batch]
        coarse_x = artifact["coarse_vectorizer"].transform(coarse_texts)
        group_pred = artifact["coarse_model"].predict(coarse_x)
        scores = np.full((len(batch), len(ALL_CLASSES)), -1e9, dtype=np.float32)
        for group, actions in router_base.ADVANCED_GROUP_TO_ACTIONS.items():
            idx = np.where(group_pred == group)[0]
            if not len(idx):
                continue
            texts = [router_base.advanced_group_text(batch[i], group) for i in idx]
            x = artifact["group_vectorizers"][group].transform(texts)
            group_scores = router_base.advanced_aligned_log_proba(artifact["group_models"][group], x, actions)
            for j, action in enumerate(actions):
                scores[idx, ALL_CLASSES.index(action)] = group_scores[:, j]
        prior = router_base.advanced_transition_prior_matrix(
            batch,
            artifact["transition_last2"],
            artifact["global_counts"],
            artifact["config"].get("prior_smooth", 1.0),
        )
        alpha = float(artifact["config"].get("prior_alpha", 0.3))
        for i, group in enumerate(group_pred):
            for action in router_base.ADVANCED_GROUP_TO_ACTIONS[str(group)]:
                scores[i, ALL_CLASSES.index(action)] += alpha * prior[i, ALL_CLASSES.index(action)]
        prob_like = np.exp(np.clip(scores, -50, 50))
        order = np.argsort(prob_like, axis=1)
        top1 = np.array([ALL_CLASSES[i] for i in order[:, -1]], dtype=object)
        top2 = np.array([ALL_CLASSES[i] for i in order[:, -2]], dtype=object)
        margin = prob_like[np.arange(len(batch)), order[:, -1]] - prob_like[np.arange(len(batch)), order[:, -2]]
        preds = top1.copy()
        pair_thr = float(artifact["config"].get("pair_threshold", 0.08))
        for i, (a, b, m) in enumerate(zip(top1, top2, margin)):
            pair = tuple(sorted((str(a), str(b))))
            resolver = artifact.get("pair_resolvers", {}).get(pair)
            if resolver is None or m > pair_thr:
                continue
            text = router_base.advanced_pair_text(batch[i], pair)
            x = resolver["vectorizer"].transform([text])
            preds[i] = str(resolver["model"].predict(x)[0])
            scores[i, ALL_CLASSES.index(str(preds[i]))] += 0.2
        all_scores.append(scores)
        all_preds.extend(str(x) for x in preds)
        all_groups.extend(str(x) for x in group_pred)
    scores = np.vstack(all_scores)
    probs = softmax_np(scores, axis=1).astype(np.float32)
    pred_ids = np.asarray([LABEL2ID[p] for p in all_preds], dtype=np.int64)
    group_ids = np.asarray([GROUP_NAMES.index(g) for g in all_groups], dtype=np.int64)
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


class DistillMLP(nn.Module):
    def __init__(self, input_dim, hidden=(256, 128), dropout=(0.1, 0.1)):
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


def first_existing(paths, required=True):
    for path in paths:
        if path and os.path.exists(path):
            return str(path)
    if required:
        raise FileNotFoundError("none exist: " + ", ".join(str(p) for p in paths))
    return None


def runtime_paths():
    base = Path(__file__).resolve().parent
    cwd = Path.cwd()
    test_path = first_existing([
        base / "data" / "test.jsonl",
        cwd / "data" / "test.jsonl",
        base / "open" / "test.jsonl",
        cwd / "open" / "test.jsonl",
        Path("/data/test.jsonl"),
        Path("/open/test.jsonl"),
    ])
    sub_path = first_existing([
        base / "data" / "sample_submission.csv",
        cwd / "data" / "sample_submission.csv",
        base / "open" / "sample_submission.csv",
        cwd / "open" / "sample_submission.csv",
        Path("/data/sample_submission.csv"),
        Path("/open/sample_submission.csv"),
    ], required=False)
    model_dir = first_existing([base / "model", cwd / "model"])
    return Path(test_path), Path(sub_path) if sub_path else None, Path(model_dir), base / "output" / "submission.csv"


def load_submission_rows(path, ids):
    if path and path.exists():
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return reader.fieldnames, list(reader)
    return ["id", "action"], [{"id": sample_id, "action": "respond_only"} for sample_id in ids]


def save_submission(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def distill_predict(samples, model_dir):
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vectorizer = joblib.load(model_dir / "vectorizer.pkl")
    svd = joblib.load(model_dir / "svd.pkl")
    dense_encoder = joblib.load(model_dir / "dense_encoder.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    advanced_router = joblib.load(model_dir / "advanced_router.pkl")

    texts = [serialize(sample, config.get("serializer", "v2_2")) for sample in samples]
    x_svd = svd.transform(vectorizer.transform(texts)).astype(np.float32)
    x_dense = dense_encoder.transform([dense_dict(sample) for sample in samples]).astype(np.float32)
    _, adv_probs, adv_pred, adv_group = predict_advanced_with_scores(samples, advanced_router)
    adv_features = advanced_feature_matrix(adv_probs, adv_pred, adv_group)
    feature_set = config.get("feature_set", "x_adv_512")
    if feature_set == "x_adv_512":
        x = np.hstack([x_svd[:, : min(512, x_svd.shape[1])], x_dense, adv_features]).astype(np.float32)
    elif feature_set == "x_adv_768":
        x = np.hstack([x_svd, x_dense, adv_features]).astype(np.float32)
    else:
        x = np.hstack([x_svd, x_dense]).astype(np.float32)
    x = scaler.transform(x).astype(np.float32)

    cfg = config.get("selected_config", {})
    hidden = tuple(cfg.get("hidden", [256, 128]))
    dropout = tuple(cfg.get("dropout", [0.1, 0.1]))
    payload = torch.load(model_dir / "student.pt", map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        hidden = tuple(payload.get("config", {}).get("hidden", list(hidden)))
        dropout = tuple(payload.get("config", {}).get("dropout", list(dropout)))
        expected_dim = int(payload.get("input_dim", x.shape[1]))
        if expected_dim != x.shape[1]:
            raise ValueError(f"feature dim mismatch: expected {expected_dim}, got {x.shape[1]}")
    else:
        state = payload
    model = DistillMLP(x.shape[1], hidden, dropout)
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logits = []
    batch_size = 8192
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            logits.append(model(xb).detach().float().cpu().numpy())
    student_probs = softmax_np(np.vstack(logits), axis=1).astype(np.float32)
    w_student = float(config.get("w_student", 0.5))
    w_advanced = float(config.get("w_advanced", 1.0 - w_student))
    probs = w_student * student_probs + w_advanced * adv_probs
    bias_by_class = config.get("bias_by_class", {})
    bias = np.asarray([float(bias_by_class.get(cls, 0.0)) for cls in ALL_CLASSES], dtype=np.float32)
    scores = np.log(np.clip(probs, 1e-12, 1.0)) + bias[None, :]
    return [ALL_CLASSES[int(i)] for i in scores.argmax(axis=1)]


def fallback_predict(samples, model_dir):
    artifact = joblib.load(model_dir / "advanced_router.pkl")
    return router_base.predict_advanced_router(samples, artifact)


def main():
    test_path, sub_path, model_dir, out_path = runtime_paths()
    samples = load_jsonl(test_path)
    ids = [sample.get("id", "") for sample in samples]
    try:
        preds = distill_predict(samples, model_dir)
        print(f"distill_student: rows={len(preds)}")
    except Exception as exc:
        print(f"warning: distill failed, fallback advanced router: {exc}")
        preds = fallback_predict(samples, model_dir)
    pred_by_id = dict(zip(ids, preds))
    fieldnames, rows = load_submission_rows(sub_path, ids)
    for row in rows:
        if row["id"] in pred_by_id:
            row["action"] = pred_by_id[row["id"]]
    save_submission(out_path, fieldnames, rows)
    print(f"Saved: {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
'''


def zip_dir(src_dir, zip_path):
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir).as_posix())
    return zip_path.stat().st_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-dir", default="model/distill_student_strict")
    parser.add_argument("--out-dir", default="sub_distill")
    parser.add_argument("--zip-path", default="sub_distill.zip")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "model").mkdir(parents=True)
    (out_dir / "pipeline_v4").mkdir(parents=True)
    (out_dir / "pipeline_v4" / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2("pipeline_v4/serialize.py", out_dir / "pipeline_v4" / "serialize.py")
    shutil.copy2("script.py", out_dir / "router_base.py")
    (out_dir / "script.py").write_text(DISTILL_SCRIPT.strip() + "\n", encoding="utf-8")
    (out_dir / "requirements.txt").write_text("", encoding="utf-8")

    student_dir = Path(args.student_dir)
    for name in [
        "advanced_router.pkl",
        "config.json",
        "dense_encoder.pkl",
        "scaler.pkl",
        "student.pt",
        "svd.pkl",
        "vectorizer.pkl",
    ]:
        shutil.copy2(student_dir / name, out_dir / "model" / name)

    size = zip_dir(out_dir, args.zip_path)
    unpacked = sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())
    print(f"out_dir={out_dir}")
    print(f"zip={args.zip_path} zip_bytes={size} unpacked_bytes={unpacked}")


if __name__ == "__main__":
    main()
