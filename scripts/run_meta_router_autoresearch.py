import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "meta_router_autoresearch"

ACTIONS = [
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
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
INSPECT = ["read_file", "grep_search", "list_directory", "glob_pattern"]
EXECUTE = ["run_bash", "run_tests", "lint_or_typecheck"]
COMMUNICATE = ["ask_user", "plan_task", "web_search", "respond_only"]
MODIFY = ["edit_file", "write_file", "apply_patch"]
GROUPS = {"inspect": INSPECT, "execute": EXECUTE, "communicate": COMMUNICATE, "modify": MODIFY}
ACTION_TO_GROUP = {a: g for g, arr in GROUPS.items() for a in arr}

THRESHOLDS_COARSE = [0.0, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
THRESHOLDS_FINE = [0.0, 0.30, 0.35, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60, 0.65, 0.75, 0.85]
THRESHOLDS_PAIR = [0.35, 0.42, 0.45, 0.48, 0.50, 0.55, 0.65]
PAIRWISE_PROBE_KINDS = {"sgd_0.00003", "sgd_0.00005"}

TARGET_PAIRS = [
    ("grep_search", "read_file"),
    ("read_file", "list_directory"),
    ("read_file", "grep_search"),
    ("grep_search", "list_directory"),
    ("list_directory", "read_file"),
]

FILE_RE = re.compile(
    r"(?i)(?:[\w.-]+[/\\])+[\w.-]+\.[a-z0-9]{1,12}\b|[\w.-]+\.(?:py|js|ts|tsx|jsx|json|md|yaml|yml|txt|csv|sql|ipynb|toml|rs|go|java|cpp|c|h)\b"
)
SPACE_RE = re.compile(r"\s+")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def safe_text(value, limit=700):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return SPACE_RE.sub(" ", value).strip()[:limit]


def bucket_num(x, cuts, names):
    try:
        x = float(x)
    except Exception:
        return "unk"
    for cut, name in zip(cuts, names):
        if x <= cut:
            return name
    return names[-1]


def extract_files(text):
    return {m.group(0).replace("\\", "/").lower() for m in FILE_RE.finditer(text or "")}


def last_actions(sample, n=6):
    out = []
    for turn in reversed(sample.get("history") or []):
        if turn.get("role") == "assistant_action":
            out.append(str(turn.get("name") or "none"))
            if len(out) >= n:
                break
    return list(reversed(out))


def result_bucket(text):
    low = safe_text(text, 600).lower()
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


def last_result(sample):
    for turn in reversed(sample.get("history") or []):
        if turn.get("role") == "assistant_action":
            return result_bucket(turn.get("result_summary"))
    return "none"


def inspect_streak(sample):
    streak = 0
    for act in reversed(last_actions(sample, 12)):
        if act in INSPECT:
            streak += 1
        else:
            break
    return bucket_num(streak, [0, 1, 2, 4], ["s0", "s1", "s2", "s3_4", "s5p"])


def open_profile(sample):
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


def prompt_file_rel(sample):
    prompt_files = extract_files(sample.get("current_prompt") or "")
    if not prompt_files:
        return "no_file"
    ws = ((sample.get("session_meta") or {}).get("workspace") or {})
    open_names = set()
    for p in ws.get("open_files") or []:
        p = str(p).replace("\\", "/").lower()
        open_names.add(p)
        open_names.add(p.rsplit("/", 1)[-1])
    prompt_names = set(prompt_files)
    prompt_names.update(x.rsplit("/", 1)[-1] for x in prompt_files)
    return "open" if (prompt_names & open_names) else "not_open"


def prompt_intent(sample):
    t = safe_text(sample.get("current_prompt"), 1200).lower()
    if any(x in t for x in ["find", "search", "grep", "where", "어디", "찾", "검색"]):
        return "find"
    if any(x in t for x in ["open", "read", "show", "열", "읽", "봐", "확인"]):
        return "read"
    if any(x in t for x in ["list", "ls", "tree", "목록", "구조", "파일들"]):
        return "list"
    if any(x in t for x in ["glob", "pattern", "패턴"]):
        return "glob"
    if any(x in t for x in ["test", "pytest", "테스트", "돌려"]):
        return "test"
    if any(x in t for x in ["lint", "typecheck", "tsc", "eslint", "린트"]):
        return "lint"
    return "other"


def entropy(probs):
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def margin(probs):
    ordered = np.sort(probs)
    return float(ordered[-1] - ordered[-2])


def cat_features(sample, adv_pred, teacher_pred, d2_pred, base_pred):
    meta = sample.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    acts = last_actions(sample, 4)
    base_group = ACTION_TO_GROUP[base_pred]
    teacher_group = ACTION_TO_GROUP[teacher_pred]
    d2_group = ACTION_TO_GROUP[d2_pred]
    adv_group = ACTION_TO_GROUP[adv_pred]
    return {
        "adv_pred": adv_pred,
        "teacher_pred": teacher_pred,
        "d2_pred": d2_pred,
        "base_pred": base_pred,
        "base_group": base_group,
        "teacher_group": teacher_group,
        "d2_group": d2_group,
        "adv_group": adv_group,
        "base_teacher_pair": f"{base_pred}->{teacher_pred}",
        "base_d2_pair": f"{base_pred}->{d2_pred}",
        "base_adv_pair": f"{base_pred}->{adv_pred}",
        "teacher_d2_pair": f"{teacher_pred}|{d2_pred}",
        "base_teacher_group_pair": f"{base_group}->{teacher_group}",
        "base_d2_group_pair": f"{base_group}->{d2_group}",
        "base_adv_group_pair": f"{base_group}->{adv_group}",
        "teacher_d2_agree": str(int(teacher_pred == d2_pred)),
        "base_teacher_agree": str(int(base_pred == teacher_pred)),
        "base_d2_agree": str(int(base_pred == d2_pred)),
        "all_models_agree": str(int(base_pred == teacher_pred == d2_pred == adv_pred)),
        "last1": acts[-1] if acts else "none",
        "last2": ">".join(acts[-2:]) if acts else "none",
        "last_result": last_result(sample),
        "inspect_streak": inspect_streak(sample),
        "open_profile": open_profile(sample),
        "prompt_file_rel": prompt_file_rel(sample),
        "prompt_intent": prompt_intent(sample),
        "ci": str(ws.get("last_ci_status", "none")),
        "dirty": str(int(bool(ws.get("git_dirty", False)))),
        "lang": str(meta.get("language_pref", "none")),
        "turn": bucket_num(meta.get("turn_index"), [1, 3, 8, 12], ["t1", "t2_3", "t4_8", "t9_12", "t13p"]),
        "budget": bucket_num(meta.get("budget_tokens_remaining"), [5000, 20000, 80000], ["b0", "b1", "b2", "b3"]),
        "open_n": bucket_num(len(ws.get("open_files") or []), [0, 1, 2, 4], ["o0", "o1", "o2", "o3_4", "o5p"]),
    }


def load_arrays():
    adv = np.load(ROOT / "artifacts" / "advanced_oof_strict" / "advanced_oof_probs.npy").astype(np.float32)
    teacher = np.load(ROOT / "artifacts" / "distill_step2_strict" / "teacher_oof" / "teacher_oof_probs.npy").astype(np.float32)
    d2 = np.load(ROOT / "reports" / "distill_step2_strict" / "mlp_oof" / "D2-M5" / "oof_probs.npy").astype(np.float32)
    cfg = read_json(ROOT / "reports" / "distill_step2_strict" / "blends" / "best_config.json")
    blend = 0.5 * adv + 0.5 * d2
    bias = np.array([float(cfg["bias"]["bias_by_class"].get(a, 0.0)) for a in ACTIONS], dtype=np.float32)
    base_scores = np.log(np.clip(blend, 1e-12, 1.0)) + bias[None, :]
    base_probs = softmax(base_scores, axis=1)
    return adv, teacher, d2, blend, base_probs


def make_features(samples, adv, teacher, d2, blend, base_probs):
    cat_rows = []
    numeric = []
    adv_pred = adv.argmax(axis=1)
    teacher_pred = teacher.argmax(axis=1)
    d2_pred = d2.argmax(axis=1)
    base_pred = base_probs.argmax(axis=1)
    for i, sample in enumerate(samples):
        cat_rows.append(
            cat_features(
                sample,
                ACTIONS[int(adv_pred[i])],
                ACTIONS[int(teacher_pred[i])],
                ACTIONS[int(d2_pred[i])],
                ACTIONS[int(base_pred[i])],
            )
        )
        row = []
        for arr in [adv, teacher, d2, blend, base_probs]:
            row.extend(arr[i].tolist())
            row.append(float(arr[i].max()))
            row.append(margin(arr[i]))
            row.append(entropy(arr[i]))
        for a in INSPECT + EXECUTE + COMMUNICATE:
            j = ACTION_TO_ID[a]
            row.append(float(teacher[i, j] - base_probs[i, j]))
            row.append(float(d2[i, j] - base_probs[i, j]))
            row.append(float(adv[i, j] - base_probs[i, j]))
        numeric.append(row)
    return cat_rows, np.asarray(numeric, dtype=np.float32), np.array([ACTIONS[i] for i in base_pred], dtype=object)


def macro(y, pred):
    return float(f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0))


def group_macro(y, pred, group):
    return float(f1_score(y, pred, labels=GROUPS[group], average="macro", zero_division=0))


def fold_values(y, pred, folds):
    out = []
    for fold in sorted(set(folds.tolist())):
        m = folds == fold
        out.append(macro(y[m], pred[m]))
    return out


def pair_fixed(y, base, pred):
    total = 0
    for true_action, base_action in TARGET_PAIRS:
        total += int(((y == true_action) & (base == base_action) & (pred == true_action)).sum())
    return total


def build_model(kind):
    if kind.startswith("sgdlong"):
        alpha = float(kind.rsplit("_", 1)[1])
        return SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            penalty="elasticnet",
            l1_ratio=0.05,
            class_weight="balanced",
            max_iter=220,
            tol=5e-5,
            random_state=42,
            n_jobs=-1,
        )
    if kind.startswith("sgd"):
        alpha = float(kind.rsplit("_", 1)[1])
        return SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            penalty="elasticnet",
            l1_ratio=0.05,
            class_weight="balanced",
            max_iter=80,
            tol=1e-4,
            random_state=42,
            n_jobs=-1,
        )
    if kind.startswith("et"):
        leaf = int(kind.rsplit("_", 1)[1])
        return ExtraTreesClassifier(
            n_estimators=320,
            min_samples_leaf=leaf,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(kind)


def predict_oof(kind, cat_rows, numeric, y, folds, scope_name):
    pred = np.empty(len(y), dtype=object)
    conf = np.zeros(len(y), dtype=np.float32)
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        val = folds == fold
        if scope_name == "inspect":
            train &= np.isin(y, INSPECT)
        elif scope_name == "execute":
            train &= np.isin(y, EXECUTE)
        elif scope_name == "communicate":
            train &= np.isin(y, COMMUNICATE)
        elif scope_name == "all_non_modify":
            train &= ~np.isin(y, MODIFY)
        vec = DictVectorizer(sparse=True)
        x_cat_train = vec.fit_transform([cat_rows[i] for i in np.where(train)[0]])
        x_cat_val = vec.transform([cat_rows[i] for i in np.where(val)[0]])
        x_train = sparse.hstack([x_cat_train, sparse.csr_matrix(numeric[train])], format="csr")
        x_val = sparse.hstack([x_cat_val, sparse.csr_matrix(numeric[val])], format="csr")
        model = build_model(kind)
        model.fit(x_train, y[train])
        proba = model.predict_proba(x_val)
        classes = np.array(model.classes_, dtype=object)
        top = proba.argmax(axis=1)
        idx = np.where(val)[0]
        pred[idx] = classes[top]
        conf[idx] = proba[np.arange(len(idx)), top]
    return pred, conf


def evaluate_variant(name, y, base_pred, candidate_pred, candidate_conf, folds, scope, thresholds):
    rows = []
    for thr in thresholds:
        pred = base_pred.copy()
        mask = scope & (candidate_conf >= thr)
        pred[mask] = candidate_pred[mask]
        fvals = fold_values(y, pred, folds)
        rows.append(
            {
                "name": f"{name}_thr{thr:.2f}",
                "macro_f1": macro(y, pred),
                "delta": macro(y, pred) - macro(y, base_pred),
                "inspect_f1": group_macro(y, pred, "inspect"),
                "inspect_delta": group_macro(y, pred, "inspect") - group_macro(y, base_pred, "inspect"),
                "execute_f1": group_macro(y, pred, "execute"),
                "communicate_f1": group_macro(y, pred, "communicate"),
                "modify_f1": group_macro(y, pred, "modify"),
                "changed": int((pred != base_pred).sum()),
                "fixed_target_errors": pair_fixed(y, base_pred, pred),
                "min_fold_delta": min(fvals) - min(fold_values(y, base_pred, folds)),
                "folds": ";".join(f"{v:.6f}" for v in fvals),
            }
        )
    return rows


def evaluate_transition_probes(name, y, base_pred, candidate_pred, candidate_conf, folds, thresholds, min_count=120):
    rows = []
    base_score = macro(y, base_pred)
    base_min_fold = min(fold_values(y, base_pred, folds))
    candidates = []
    for base_action in ACTIONS:
        base_mask = base_pred == base_action
        if int(base_mask.sum()) < min_count:
            continue
        for cand_action in ACTIONS:
            if cand_action == base_action:
                continue
            pair_mask_base = base_mask & (candidate_pred == cand_action)
            if int(pair_mask_base.sum()) < min_count:
                continue
            for thr in thresholds:
                mask = pair_mask_base & (candidate_conf >= thr)
                changed = int(mask.sum())
                if changed < min_count:
                    continue
                pred = base_pred.copy()
                pred[mask] = cand_action
                score = macro(y, pred)
                fvals = fold_values(y, pred, folds)
                row = {
                    "name": f"{name}_pair_{base_action}_to_{cand_action}_thr{thr:.2f}",
                    "macro_f1": score,
                    "delta": score - base_score,
                    "inspect_f1": group_macro(y, pred, "inspect"),
                    "inspect_delta": group_macro(y, pred, "inspect") - group_macro(y, base_pred, "inspect"),
                    "execute_f1": group_macro(y, pred, "execute"),
                    "communicate_f1": group_macro(y, pred, "communicate"),
                    "modify_f1": group_macro(y, pred, "modify"),
                    "changed": changed,
                    "fixed_target_errors": pair_fixed(y, base_pred, pred),
                    "min_fold_delta": min(fvals) - base_min_fold,
                    "folds": ";".join(f"{v:.6f}" for v in fvals),
                }
                rows.append(row)
                if row["delta"] > 0 and row["min_fold_delta"] >= -0.0005:
                    candidates.append((row["delta"], mask.copy(), cand_action, row["name"]))

    current = base_pred.copy()
    selected = []
    for _, mask, cand_action, label in sorted(candidates, key=lambda x: x[0], reverse=True):
        trial = current.copy()
        trial[mask] = cand_action
        trial_score = macro(y, trial)
        current_score = macro(y, current)
        trial_min_fold = min(fold_values(y, trial, folds))
        if trial_score > current_score + 1e-6 and trial_min_fold >= base_min_fold - 0.0005:
            current = trial
            selected.append(label)
        if len(selected) >= 20:
            break

    if selected:
        fvals = fold_values(y, current, folds)
        rows.append(
            {
                "name": f"{name}_pair_greedy_top{len(selected)}",
                "macro_f1": macro(y, current),
                "delta": macro(y, current) - base_score,
                "inspect_f1": group_macro(y, current, "inspect"),
                "inspect_delta": group_macro(y, current, "inspect") - group_macro(y, base_pred, "inspect"),
                "execute_f1": group_macro(y, current, "execute"),
                "communicate_f1": group_macro(y, current, "communicate"),
                "modify_f1": group_macro(y, current, "modify"),
                "changed": int((current != base_pred).sum()),
                "fixed_target_errors": pair_fixed(y, base_pred, current),
                "min_fold_delta": min(fvals) - base_min_fold,
                "folds": ";".join(f"{v:.6f}" for v in fvals),
            }
        )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(ROOT / "data" / "train.jsonl")
    labels = load_labels(ROOT / "data" / "train_labels.csv")
    y = np.array([labels[s["id"]] for s in samples], dtype=object)
    folds = np.load(ROOT / "artifacts" / "distill_step2_strict" / "fold_ids.npy")
    adv, teacher, d2, blend, base_probs = load_arrays()
    cat_rows, numeric, base_pred = make_features(samples, adv, teacher, d2, blend, base_probs)

    rows = []
    base_row = {
        "name": "base_strict_distill_bias",
        "macro_f1": macro(y, base_pred),
        "delta": 0.0,
        "inspect_f1": group_macro(y, base_pred, "inspect"),
        "inspect_delta": 0.0,
        "execute_f1": group_macro(y, base_pred, "execute"),
        "communicate_f1": group_macro(y, base_pred, "communicate"),
        "modify_f1": group_macro(y, base_pred, "modify"),
        "changed": 0,
        "fixed_target_errors": 0,
        "min_fold_delta": 0.0,
        "folds": ";".join(f"{v:.6f}" for v in fold_values(y, base_pred, folds)),
    }
    rows.append(base_row)

    specs = [
        ("sgd_0.00002", "all", THRESHOLDS_FINE),
        ("sgd_0.00003", "all", THRESHOLDS_FINE),
        ("sgd_0.00005", "all", THRESHOLDS_FINE),
        ("sgd_0.00007", "all", THRESHOLDS_FINE),
        ("sgd_0.0001", "all", THRESHOLDS_FINE),
        ("sgdlong_0.00003", "all", THRESHOLDS_FINE),
        ("sgdlong_0.00005", "all", THRESHOLDS_FINE),
        ("sgd_0.00003", "inspect", THRESHOLDS_FINE),
        ("sgd_0.0001", "inspect", THRESHOLDS_FINE),
        ("sgd_0.00003", "execute", THRESHOLDS_FINE),
        ("sgd_0.0001", "execute", THRESHOLDS_FINE),
        ("sgd_0.00003", "communicate", THRESHOLDS_FINE),
        ("sgd_0.0001", "communicate", THRESHOLDS_FINE),
        ("et_4", "inspect", THRESHOLDS_COARSE),
        ("et_8", "inspect", THRESHOLDS_COARSE),
        ("et_8", "all_non_modify", THRESHOLDS_COARSE),
    ]
    for kind, scope_name, thresholds in specs:
        print(f"running {kind} scope={scope_name}", flush=True)
        cand_pred, cand_conf = predict_oof(kind, cat_rows, numeric, y, folds, scope_name)
        if scope_name == "all":
            same_group = np.array([ACTION_TO_GROUP.get(a) == ACTION_TO_GROUP.get(b) for a, b in zip(base_pred, cand_pred)])
            scopes = {
                "all": np.ones(len(y), dtype=bool),
                "base_inspect": np.isin(base_pred, INSPECT),
                "base_execute": np.isin(base_pred, EXECUTE),
                "base_communicate": np.isin(base_pred, COMMUNICATE),
                "base_non_modify": ~np.isin(base_pred, MODIFY),
                "same_group": same_group,
                "same_group_non_modify": same_group & (~np.isin(base_pred, MODIFY)),
            }
        elif scope_name == "inspect":
            scopes = {
                "base_inspect": np.isin(base_pred, INSPECT),
                "inspect_to_inspect": np.isin(base_pred, INSPECT) & np.isin(cand_pred, INSPECT),
            }
        elif scope_name == "execute":
            scopes = {
                "base_execute": np.isin(base_pred, EXECUTE),
                "execute_to_execute": np.isin(base_pred, EXECUTE) & np.isin(cand_pred, EXECUTE),
                "same_group": np.array([ACTION_TO_GROUP.get(a) == ACTION_TO_GROUP.get(b) for a, b in zip(base_pred, cand_pred)]),
            }
        elif scope_name == "communicate":
            scopes = {
                "base_communicate": np.isin(base_pred, COMMUNICATE),
                "communicate_to_communicate": np.isin(base_pred, COMMUNICATE) & np.isin(cand_pred, COMMUNICATE),
                "same_group": np.array([ACTION_TO_GROUP.get(a) == ACTION_TO_GROUP.get(b) for a, b in zip(base_pred, cand_pred)]),
            }
        else:
            scopes = {
                "base_non_modify": ~np.isin(base_pred, MODIFY),
                "same_group_non_modify": (~np.isin(base_pred, MODIFY))
                & np.array([ACTION_TO_GROUP.get(a) == ACTION_TO_GROUP.get(b) for a, b in zip(base_pred, cand_pred)]),
            }
        for scope_label, scope in scopes.items():
            rows.extend(
                evaluate_variant(
                    f"{kind}_{scope_name}_{scope_label}",
                    y,
                    base_pred,
                    cand_pred,
                    cand_conf,
                    folds,
                    scope,
                    thresholds=thresholds,
                )
            )
        if scope_name == "all" and kind in PAIRWISE_PROBE_KINDS:
            rows.extend(
                evaluate_transition_probes(
                    f"{kind}_{scope_name}",
                    y,
                    base_pred,
                    cand_pred,
                    cand_conf,
                    folds,
                    thresholds=THRESHOLDS_PAIR,
                )
            )

    rows = sorted(rows, key=lambda r: (r["macro_f1"], r["inspect_f1"]), reverse=True)
    write_csv(OUT / "results.csv", rows, list(rows[0].keys()))
    best = rows[0]
    lines = [
        "# Meta Router Autoresearch",
        "",
        f"- base Macro-F1: `{base_row['macro_f1']:.6f}`",
        f"- best: `{best['name']}`",
        f"- best Macro-F1: `{best['macro_f1']:.6f}`",
        f"- best delta: `{best['delta']:.6f}`",
        f"- best inspect delta: `{best['inspect_delta']:.6f}`",
        f"- changed: `{best['changed']}`",
        f"- fixed target inspect errors: `{best['fixed_target_errors']}`",
        "",
        "## Top Variants",
        "",
        "| name | Macro-F1 | delta | inspect_delta | changed | fixed_target_errors | min_fold_delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:20]:
        lines.append(
            f"| `{row['name']}` | `{row['macro_f1']:.6f}` | `{row['delta']:.6f}` | "
            f"`{row['inspect_delta']:.6f}` | `{row['changed']}` | `{row['fixed_target_errors']}` | `{row['min_fold_delta']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- A deployable candidate needs strict positive Macro-F1, positive group lift, and no unstable fold damage.",
            "- If best delta crosses `+0.03`, expand the same meta-router to execute and communicate bottlenecks.",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]), flush=True)


if __name__ == "__main__":
    main()
