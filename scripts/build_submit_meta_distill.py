import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.feature_extraction import DictVectorizer

from build_submit_distill import DISTILL_SCRIPT, zip_dir
from run_meta_router_autoresearch import (
    ACTIONS,
    COMMUNICATE,
    EXECUTE,
    INSPECT,
    cat_features,
    entropy,
    load_labels,
    margin,
    read_json,
    read_jsonl,
)
from sklearn.linear_model import SGDClassifier


ROOT = Path(__file__).resolve().parents[1]


META_EXTENSION = r'''
from scipy import sparse


def meta_bucket_num(x, cuts, names):
    try:
        x = float(x)
    except Exception:
        return "unk"
    for cut, name in zip(cuts, names):
        if x <= cut:
            return name
    return names[-1]


def meta_entropy(probs):
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def meta_margin(probs):
    ordered = np.sort(probs)
    return float(ordered[-1] - ordered[-2])


def meta_extract_files(text):
    return set(re.findall(
        r"(?i)(?:[\w.-]+[/\\])+[\w.-]+\.[a-z0-9]{1,12}\b|[\w.-]+\.(?:py|js|ts|tsx|jsx|json|md|yaml|yml|txt|csv|sql|ipynb|toml|rs|go|java|cpp|c|h)\b",
        text or "",
    ))


def meta_last_actions(sample, n=6):
    out = []
    for turn in reversed(sample.get("history") or []):
        if turn.get("role") == "assistant_action":
            out.append(str(turn.get("name") or "none"))
            if len(out) >= n:
                break
    return list(reversed(out))


def meta_result_bucket(text):
    low = router_base.safe_text(text, 600).lower()
    if not low:
        return "none"
    if any(x in low for x in ["traceback", "exception", "error", "failed", "fail", "permission denied"]):
        return "fail"
    if any(x in low for x in ["no matches", "0 matches", "not found", "zero match"]):
        return "zero_match"
    m = re.search(r"(\d+)\s*(matches?|files?|items?|occurrences?|results?)", low)
    if m:
        n = int(m.group(1))
        if n == 0:
            return "zero_match"
        if n <= 3:
            return "few"
        if n <= 20:
            return "some"
        return "many"
    if any(x in low for x in ["match", "found", "occurrence", "result"]):
        return "matches"
    if any(x in low for x in ["read", "opened", "lines"]):
        return "read_ok"
    if any(x in low for x in ["listed", "entries", "directory", "files"]):
        return "listed"
    return "other"


def meta_last_result(sample):
    for turn in reversed(sample.get("history") or []):
        if turn.get("role") == "assistant_action":
            return meta_result_bucket(turn.get("result_summary"))
    return "none"


def meta_inspect_streak(sample):
    inspect = set(["read_file", "grep_search", "list_directory", "glob_pattern"])
    streak = 0
    for act in reversed(meta_last_actions(sample, 12)):
        if act in inspect:
            streak += 1
        else:
            break
    return meta_bucket_num(streak, [0, 1, 2, 4], ["s0", "s1", "s2", "s3_4", "s5p"])


def meta_open_profile(sample):
    ws = ((sample.get("session_meta") or {}).get("workspace") or {})
    files = ws.get("open_files") or []
    if not files:
        return "none"
    if len(files) >= 3:
        return "many3p"
    exts = {Path(str(p).lower()).suffix.lstrip(".") for p in files if Path(str(p)).suffix}
    if exts and exts <= {"py"}:
        return "py_only"
    if exts and exts <= {"js", "jsx", "ts", "tsx"}:
        return "js_only"
    if len(exts) == 1:
        return "one_" + next(iter(exts))
    return "mixed"


def meta_prompt_file_rel(sample):
    prompt_files = meta_extract_files(sample.get("current_prompt") or "")
    if not prompt_files:
        return "no_file"
    ws = ((sample.get("session_meta") or {}).get("workspace") or {})
    open_names = set()
    for p in ws.get("open_files") or []:
        p = str(p).replace("\\", "/").lower()
        open_names.add(p)
        open_names.add(p.rsplit("/", 1)[-1])
    prompt_names = {p.replace("\\", "/").lower() for p in prompt_files}
    prompt_names.update(x.rsplit("/", 1)[-1] for x in list(prompt_names))
    return "open" if (prompt_names & open_names) else "not_open"


def meta_prompt_intent(sample):
    t = router_base.safe_text(sample.get("current_prompt"), 1200).lower()
    if any(x in t for x in ["find", "search", "grep", "where"]):
        return "find"
    if any(x in t for x in ["open", "read", "show", "check"]):
        return "read"
    if any(x in t for x in ["list", "ls", "tree", "directory"]):
        return "list"
    if any(x in t for x in ["glob", "pattern"]):
        return "glob"
    if any(x in t for x in ["test", "pytest"]):
        return "test"
    if any(x in t for x in ["lint", "typecheck", "tsc", "eslint"]):
        return "lint"
    return "other"


def meta_cat_features(sample, adv_pred, d2_pred, base_pred):
    meta = sample.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    acts = meta_last_actions(sample, 4)
    group = router_base.ADVANCED_ACTION_TO_GROUP
    return {
        "adv_pred": adv_pred,
        "teacher_pred": d2_pred,
        "d2_pred": d2_pred,
        "base_pred": base_pred,
        "base_group": group.get(base_pred, "unknown"),
        "teacher_group": group.get(d2_pred, "unknown"),
        "last1": acts[-1] if acts else "none",
        "last2": ">".join(acts[-2:]) if acts else "none",
        "last_result": meta_last_result(sample),
        "inspect_streak": meta_inspect_streak(sample),
        "open_profile": meta_open_profile(sample),
        "prompt_file_rel": meta_prompt_file_rel(sample),
        "prompt_intent": meta_prompt_intent(sample),
        "ci": str(ws.get("last_ci_status", "none")),
        "dirty": str(int(bool(ws.get("git_dirty", False)))),
        "lang": str(meta.get("language_pref", "none")),
        "turn": meta_bucket_num(meta.get("turn_index"), [1, 3, 8, 12], ["t1", "t2_3", "t4_8", "t9_12", "t13p"]),
        "budget": meta_bucket_num(meta.get("budget_tokens_remaining"), [5000, 20000, 80000], ["b0", "b1", "b2", "b3"]),
        "open_n": meta_bucket_num(len(ws.get("open_files") or []), [0, 1, 2, 4], ["o0", "o1", "o2", "o3_4", "o5p"]),
    }


def meta_make_features(samples, adv_probs, d2_probs, blend_probs, base_probs):
    cat_rows = []
    numeric = []
    adv_pred = adv_probs.argmax(axis=1)
    d2_pred = d2_probs.argmax(axis=1)
    base_pred = base_probs.argmax(axis=1)
    for i, sample in enumerate(samples):
        cat_rows.append(meta_cat_features(
            sample,
            ALL_CLASSES[int(adv_pred[i])],
            ALL_CLASSES[int(d2_pred[i])],
            ALL_CLASSES[int(base_pred[i])],
        ))
        row = []
        for arr in [adv_probs, d2_probs, d2_probs, blend_probs, base_probs]:
            row.extend(arr[i].tolist())
            row.append(float(arr[i].max()))
            row.append(meta_margin(arr[i]))
            row.append(meta_entropy(arr[i]))
        for action in ["read_file", "grep_search", "list_directory", "glob_pattern", "run_bash", "run_tests", "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only"]:
            j = LABEL2ID[action]
            row.append(float(d2_probs[i, j] - base_probs[i, j]))
            row.append(float(d2_probs[i, j] - base_probs[i, j]))
            row.append(float(adv_probs[i, j] - base_probs[i, j]))
        numeric.append(row)
    return cat_rows, np.asarray(numeric, dtype=np.float32), np.array([ALL_CLASSES[i] for i in base_pred], dtype=object)


def meta_scope_mask(name, base_probs):
    if name == "base_margin_lte_020":
        return np.asarray([meta_margin(row) <= 0.20 for row in base_probs], dtype=bool)
    if name == "base_margin_lte_022":
        return np.asarray([meta_margin(row) <= 0.22 for row in base_probs], dtype=bool)
    if name == "base_margin_lte_018":
        return np.asarray([meta_margin(row) <= 0.18 for row in base_probs], dtype=bool)
    return np.ones(base_probs.shape[0], dtype=bool)


def apply_meta_router(samples, base_preds, context, model_dir):
    meta_path = model_dir / "meta_router.pkl"
    if not meta_path.exists():
        return base_preds
    payload = joblib.load(meta_path)
    cat_rows, numeric, inferred_base = meta_make_features(
        samples,
        context["adv_probs"],
        context["student_probs"],
        context["blend_probs"],
        context["base_probs"],
    )
    x_cat = payload["vectorizer"].transform(cat_rows)
    x = sparse.hstack([x_cat, sparse.csr_matrix(numeric)], format="csr")
    proba = payload["model"].predict_proba(x)
    classes = np.asarray(payload["model"].classes_, dtype=object)
    top = proba.argmax(axis=1)
    cand = classes[top]
    conf = proba[np.arange(len(top)), top]
    threshold = float(payload.get("threshold", 0.42))
    scope = meta_scope_mask(str(payload.get("scope", "all")), context["base_probs"])
    out = list(base_preds)
    changed = 0
    applied = 0
    for i, (action, score) in enumerate(zip(cand, conf)):
        if bool(scope[i]) and float(score) >= threshold:
            applied += 1
            if out[i] != str(action):
                changed += 1
            out[i] = str(action)
    print(f"meta_router_sgdl2: applied={applied}/{len(samples)} changed={changed} threshold={threshold} scope={payload.get('scope', 'all')}")
    return out
'''


def train_meta_router(out_dir):
    samples = read_jsonl(ROOT / "data" / "train.jsonl")
    labels = load_labels(ROOT / "data" / "train_labels.csv")
    y = np.array([labels[s["id"]] for s in samples], dtype=object)
    adv = np.load(ROOT / "artifacts" / "advanced_oof_strict" / "advanced_oof_probs.npy").astype(np.float32)
    d2 = np.load(ROOT / "reports" / "distill_step2_strict" / "mlp_oof" / "D2-M5" / "oof_probs.npy").astype(np.float32)
    cfg = read_json(ROOT / "reports" / "distill_step2_strict" / "blends" / "best_config.json")
    bias = np.array([float(cfg["bias"]["bias_by_class"].get(a, 0.0)) for a in ACTIONS], dtype=np.float32)
    prob_cfg_path = ROOT / "reports" / "prob_blend_autoresearch" / "best_config.json"
    if prob_cfg_path.exists():
        prob_cfg = read_json(prob_cfg_path)
        w_adv = float(prob_cfg.get("w_adv", 0.44) or 0.44)
        # The OOF research blend can use a separate teacher probability table.
        # Submit cannot ship that teacher, so teacher weight is folded into the
        # deployable D2 student proxy.
        w_d2 = float(prob_cfg.get("w_teacher", 0.48) or 0.48) + float(prob_cfg.get("w_d2", 0.08) or 0.08)
        bias_scale = float(prob_cfg.get("bias_scale", 1.25) or 1.25)
    else:
        w_adv, w_d2, bias_scale = 0.44, 0.56, 1.25
    raw_scores = (
        w_adv * np.log(np.clip(adv, 1e-12, 1.0))
        + w_d2 * np.log(np.clip(d2, 1e-12, 1.0))
    )
    blend = softmax(raw_scores, axis=1).astype(np.float32)
    base_probs = softmax(raw_scores + bias_scale * bias[None, :], axis=1).astype(np.float32)

    cat_rows = []
    numeric = []
    adv_pred = adv.argmax(axis=1)
    d2_pred = d2.argmax(axis=1)
    base_pred = base_probs.argmax(axis=1)
    for i, sample in enumerate(samples):
        cat_rows.append(
            cat_features(
                sample,
                ACTIONS[int(adv_pred[i])],
                ACTIONS[int(d2_pred[i])],
                ACTIONS[int(d2_pred[i])],
                ACTIONS[int(base_pred[i])],
            )
        )
        row = []
        for arr in [adv, d2, d2, blend, base_probs]:
            row.extend(arr[i].tolist())
            row.append(float(arr[i].max()))
            row.append(margin(arr[i]))
            row.append(entropy(arr[i]))
        for a in INSPECT + EXECUTE + COMMUNICATE:
            j = ACTIONS.index(a)
            row.append(float(d2[i, j] - base_probs[i, j]))
            row.append(float(d2[i, j] - base_probs[i, j]))
            row.append(float(adv[i, j] - base_probs[i, j]))
        numeric.append(row)
    vec = DictVectorizer(sparse=True)
    x_cat = vec.fit_transform(cat_rows)
    x = sparse.hstack([x_cat, sparse.csr_matrix(np.asarray(numeric, dtype=np.float32))], format="csr")
    model = SGDClassifier(
        loss="log_loss",
        alpha=0.00008,
        penalty="l2",
        class_weight="balanced",
        max_iter=100,
        tol=1e-4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)
    payload = {
        "name": "sgdl2_0.00008_all_all_thr0.42_deployable_distill_proxy",
        "threshold": 0.42,
        "scope": "base_margin_lte_020",
        "class_order": ACTIONS,
        "vectorizer": vec,
        "model": model,
        "feature_note": "teacher features replaced by D2 student probabilities for deployable inference; applies only when base margin <= 0.20",
        "deploy_blend": {"w_adv": w_adv, "w_d2": w_d2, "bias_scale": bias_scale},
        "oof_reference": {
            "strict_best_name": "sgdl2_0.00008_all_base_margin_lte_020_thr0.42",
            "strict_macro_f1": 0.7451695146266804,
            "strict_delta": 0.0021176850582127482,
        },
    }
    joblib.dump(payload, out_dir / "model" / "meta_router.pkl", compress=3)


def build_script_text():
    script = DISTILL_SCRIPT.strip()
    script = script.replace("import router_base", "import router_base\nimport re")
    script = script.replace(
        "return [ALL_CLASSES[int(i)] for i in scores.argmax(axis=1)]",
        "\n    deploy_w_adv = float(config.get(\"deploy_w_adv\", 0.44))\n"
        "    deploy_w_student = float(config.get(\"deploy_w_student\", 0.56))\n"
        "    deploy_bias_scale = float(config.get(\"deploy_bias_scale\", 1.25))\n"
        "    deploy_raw_scores = (\n"
        "        deploy_w_adv * np.log(np.clip(adv_probs, 1e-12, 1.0))\n"
        "        + deploy_w_student * np.log(np.clip(student_probs, 1e-12, 1.0))\n"
        "    )\n"
        "    deploy_blend = softmax_np(deploy_raw_scores, axis=1).astype(np.float32)\n"
        "    deploy_scores = deploy_raw_scores + deploy_bias_scale * bias[None, :]\n"
        "    base_probs = softmax_np(deploy_scores, axis=1).astype(np.float32)\n"
        "    preds = [ALL_CLASSES[int(i)] for i in scores.argmax(axis=1)]\n"
        "    context = {\n"
        "        \"adv_probs\": adv_probs.astype(np.float32),\n"
        "        \"student_probs\": student_probs.astype(np.float32),\n"
        "        \"blend_probs\": deploy_blend.astype(np.float32),\n"
        "        \"base_probs\": base_probs,\n"
        "    }\n"
        "    preds = [ALL_CLASSES[int(i)] for i in deploy_scores.argmax(axis=1)]\n"
        "    return preds, context",
    )
    script = script.replace(
        "def main():",
        META_EXTENSION.strip() + "\n\n\ndef main():",
    )
    script = script.replace(
        "preds = distill_predict(samples, model_dir)\n        print(f\"distill_student: rows={len(preds)}\")",
        "preds, meta_context = distill_predict(samples, model_dir)\n"
        "        print(f\"distill_student: rows={len(preds)}\")\n"
        "        preds = apply_meta_router(samples, preds, meta_context, model_dir)",
    )
    return script + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-dir", default="model/distill_student_strict")
    parser.add_argument("--out-dir", default="meta_sgdl2_08")
    parser.add_argument("--zip-path", default="meta_sgdl2_08.zip")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "model").mkdir(parents=True)
    (out_dir / "pipeline_v4").mkdir(parents=True)
    (out_dir / "pipeline_v4" / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(ROOT / "pipeline_v4" / "serialize.py", out_dir / "pipeline_v4" / "serialize.py")
    shutil.copy2(ROOT / "script.py", out_dir / "router_base.py")
    (out_dir / "script.py").write_text(build_script_text(), encoding="utf-8")
    (out_dir / "requirements.txt").write_text("", encoding="utf-8")

    student_dir = ROOT / args.student_dir
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
    cfg_path = out_dir / "model" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    prob_cfg_path = ROOT / "reports" / "prob_blend_autoresearch" / "best_config.json"
    if prob_cfg_path.exists():
        prob_cfg = read_json(prob_cfg_path)
        cfg["deploy_w_adv"] = float(prob_cfg.get("w_adv", 0.44) or 0.44)
        cfg["deploy_w_student"] = float(prob_cfg.get("w_teacher", 0.48) or 0.48) + float(prob_cfg.get("w_d2", 0.08) or 0.08)
        cfg["deploy_bias_scale"] = float(prob_cfg.get("bias_scale", 1.25) or 1.25)
    else:
        cfg["deploy_w_adv"] = 0.44
        cfg["deploy_w_student"] = 0.56
        cfg["deploy_bias_scale"] = 1.25
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    train_meta_router(out_dir)
    size = zip_dir(out_dir, args.zip_path)
    unpacked = sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())
    print(f"out_dir={out_dir}")
    print(f"zip={args.zip_path} zip_bytes={size} unpacked_bytes={unpacked}")


if __name__ == "__main__":
    main()
