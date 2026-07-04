import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.constants import ALL_CLASSES, ID2LABEL, LABEL2ID
from pipeline_v4.common.data_io import load_fold_rows, load_train_samples
from script import compact_flags_text


RUN_NAME = "mdeberta384_v2_384_5e"
OOF_DIR = ROOT / "pipeline_v4" / "artifacts" / "oof" / RUN_NAME
REPORT_DIR = ROOT / "reports" / "supcon_intent_probe"

CONFUSION_PAIRS = [
    ("read_file", "grep_search"),
    ("read_file", "list_directory"),
    ("grep_search", "list_directory"),
    ("glob_pattern", "read_file"),
    ("glob_pattern", "grep_search"),
    ("ask_user", "plan_task"),
    ("plan_task", "web_search"),
    ("ask_user", "web_search"),
    ("run_tests", "lint_or_typecheck"),
    ("run_bash", "run_tests"),
]

GROUPS = {
    "inspect4": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "communicate4": ["ask_user", "plan_task", "web_search", "respond_only"],
    "execute3": ["run_bash", "run_tests", "lint_or_typecheck"],
    "modify3": ["edit_file", "write_file", "apply_patch"],
}

INTENT_PATTERNS = {
    "time": [
        r"latest",
        r"deprecated",
        r"version",
        r"recent",
        r"update",
        r"최신",
        r"요즘",
        r"버전",
        r"릴리즈",
    ],
    "know": [
        r"official",
        r"docs?",
        r"documentation",
        r"best practice",
        r"recommended",
        r"reference",
        r"paper",
        r"공식",
        r"문서",
        r"권장",
        r"자료",
        r"레퍼런스",
    ],
    "choice": [
        r"which",
        r"choose",
        r"pick",
        r"should i",
        r"better",
        r"between",
        r"어떤",
        r"뭐가",
        r"고르",
        r"선택",
        r"할까",
        r"나을",
    ],
    "explain": [
        r"explain",
        r"why",
        r"how",
        r"summar",
        r"recap",
        r"wrap",
        r"brief",
        r"요약",
        r"정리",
        r"설명",
        r"알려",
        r"어떻게",
        r"마무리",
    ],
    "planning": [
        r"plan",
        r"roadmap",
        r"step",
        r"steps",
        r"approach",
        r"strategy",
        r"break down",
        r"where to start",
        r"계획",
        r"단계",
        r"순서",
        r"로드맵",
        r"방향",
        r"진행",
        r"쪼개",
    ],
}


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def safe_text(value, max_chars=1200):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(text.split())[:max_chars]


def load_fold0():
    samples = load_train_samples(ROOT / "data")
    sample_by_id = {s["id"]: s for s in samples}
    ids = (OOF_DIR / "fold_0_ids.txt").read_text(encoding="utf-8").splitlines()
    logits = np.load(OOF_DIR / "fold_0_logits.npy")
    probs = np.load(OOF_DIR / "fold_0_probs.npy") if (OOF_DIR / "fold_0_probs.npy").exists() else softmax(logits)
    y = np.load(OOF_DIR / "fold_0_y.npy")
    fold0_samples = [sample_by_id[sid] for sid in ids]
    pred = probs.argmax(axis=1)
    return samples, fold0_samples, ids, logits, probs, y, pred


def pair_key(a, b):
    return tuple(sorted((a, b)))


def group_macro(y, pred, labels):
    return f1_score(y, pred, labels=[LABEL2ID[x] for x in labels], average="macro", zero_division=0)


def compute_pair_errors(y, pred, probs):
    rows = []
    top2 = np.argsort(probs, axis=1)[:, -2:]
    margins = probs[np.arange(len(probs)), top2[:, 1]] - probs[np.arange(len(probs)), top2[:, 0]]
    for a, b in CONFUSION_PAIRS:
        ai, bi = LABEL2ID[a], LABEL2ID[b]
        denom = int(np.isin(y, [ai, bi]).sum())
        err_mask = ((y == ai) & (pred == bi)) | ((y == bi) & (pred == ai))
        pair_total_mask = np.isin(y, [ai, bi])
        rows.append(
            {
                "pair": f"{a}<->{b}",
                "support_pair_true": denom,
                "pair_error_count": int(err_mask.sum()),
                "pair_error_rate": float(err_mask.sum() / max(denom, 1)),
                "pair_all_mean_margin": float(margins[pair_total_mask].mean()) if pair_total_mask.any() else 0.0,
                "pair_error_mean_margin": float(margins[err_mask].mean()) if err_mask.any() else 0.0,
                "pair_error_low_margin_lt_0_1": float((margins[err_mask] < 0.1).mean()) if err_mask.any() else 0.0,
                "pair_error_high_margin_gt_0_3": float((margins[err_mask] > 0.3).mean()) if err_mask.any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_group_metrics(y, pred):
    rows = []
    for name, labels in GROUPS.items():
        rows.append({"group": name, "macro_f1": group_macro(y, pred, labels), "support": int(np.isin(y, [LABEL2ID[x] for x in labels]).sum())})
    return pd.DataFrame(rows)


def centroid_proxy(logits, y):
    rows = []
    z = logits.astype(np.float64)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    centroids = {}
    for cls in ALL_CLASSES:
        idx = LABEL2ID[cls]
        mask = y == idx
        if mask.any():
            c = z[mask].mean(axis=0)
            centroids[cls] = c / max(np.linalg.norm(c), 1e-12)
    for group, labels in GROUPS.items():
        dists = []
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                if a not in centroids or b not in centroids:
                    continue
                cos = float(np.dot(centroids[a], centroids[b]))
                dists.append(1.0 - cos)
                rows.append({"scope": group, "a": a, "b": b, "cosine_distance": 1.0 - cos, "proxy": "logit_centroid"})
        if dists:
            rows.append({"scope": group, "a": "__average__", "b": "__average__", "cosine_distance": float(np.mean(dists)), "proxy": "logit_centroid"})
    return pd.DataFrame(rows)


def intent_flags(sample):
    blob_parts = [safe_text(sample.get("current_prompt"), 1200)]
    for turn in (sample.get("history", []) or [])[-8:]:
        if turn.get("role") == "user":
            blob_parts.append(safe_text(turn.get("content"), 500))
        elif turn.get("role") == "assistant_action":
            blob_parts.append(safe_text(turn.get("name"), 80))
            blob_parts.append(safe_text(turn.get("args"), 300))
            blob_parts.append(safe_text(turn.get("result_summary"), 500))
    blob = " ".join(blob_parts).lower()
    flags = {}
    for name, patterns in INTENT_PATTERNS.items():
        flags[name] = int(any(re.search(p, blob, flags=re.IGNORECASE) for p in patterns))
    return flags


def intent_token_text(sample):
    flags = intent_flags(sample)
    tokens = [f"INTENT_{name}_{value}" for name, value in flags.items()]
    tokens.extend([f"INTENT_{name}" for name, value in flags.items() if value])
    return " [INTENT] " + " ".join(tokens)


def build_vectorizer(max_features=160_000):
    half = max_features // 2
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=half,
                    sublinear_tf=True,
                    lowercase=True,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=half,
                    sublinear_tf=True,
                    lowercase=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def load_split(samples):
    fold_by_id = {}
    with open(ROOT / "pipeline_v4" / "folds" / "fold_assignments.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fold_by_id[row["id"]] = int(row["fold"])
    train = [s for s in samples if fold_by_id[s["id"]] != 0]
    val = [s for s in samples if fold_by_id[s["id"]] == 0]
    return train, val


def train_logreg(train_texts, y_train, val_texts):
    vectorizer = build_vectorizer()
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    model = LogisticRegression(max_iter=700, C=2.0, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    return pred


def run_intent_tier_b(samples):
    train, val = load_split(samples)
    y_train = np.array([s["action"] for s in train], dtype=object)
    y_val = np.array([s["action"] for s in val], dtype=object)
    base_train = [compact_flags_text(s) for s in train]
    base_val = [compact_flags_text(s) for s in val]
    intent_train = [compact_flags_text(s) + intent_token_text(s) for s in train]
    intent_val = [compact_flags_text(s) + intent_token_text(s) for s in val]
    base_pred = train_logreg(base_train, y_train, base_val)
    intent_pred = train_logreg(intent_train, y_train, intent_val)
    labels_comm = GROUPS["communicate4"]
    rows = [
        {
            "model": "base_compact_flags_logreg",
            "overall_macro_f1": f1_score(y_val, base_pred, labels=ALL_CLASSES, average="macro", zero_division=0),
            "communicate4_macro_f1": f1_score(y_val, base_pred, labels=labels_comm, average="macro", zero_division=0),
            "web_search_f1": f1_score(y_val, base_pred, labels=["web_search"], average="macro", zero_division=0),
            "ask_user_f1": f1_score(y_val, base_pred, labels=["ask_user"], average="macro", zero_division=0),
            "plan_task_f1": f1_score(y_val, base_pred, labels=["plan_task"], average="macro", zero_division=0),
            "respond_only_f1": f1_score(y_val, base_pred, labels=["respond_only"], average="macro", zero_division=0),
        },
        {
            "model": "intent_v2_1_logreg",
            "overall_macro_f1": f1_score(y_val, intent_pred, labels=ALL_CLASSES, average="macro", zero_division=0),
            "communicate4_macro_f1": f1_score(y_val, intent_pred, labels=labels_comm, average="macro", zero_division=0),
            "web_search_f1": f1_score(y_val, intent_pred, labels=["web_search"], average="macro", zero_division=0),
            "ask_user_f1": f1_score(y_val, intent_pred, labels=["ask_user"], average="macro", zero_division=0),
            "plan_task_f1": f1_score(y_val, intent_pred, labels=["plan_task"], average="macro", zero_division=0),
            "respond_only_f1": f1_score(y_val, intent_pred, labels=["respond_only"], average="macro", zero_division=0),
        },
    ]
    out = pd.DataFrame(rows)
    delta = out.iloc[1].copy()
    delta["model"] = "delta_intent_minus_base"
    for col in out.columns:
        if col != "model":
            delta[col] = out.iloc[1][col] - out.iloc[0][col]
    out = pd.concat([out, pd.DataFrame([delta])], ignore_index=True)
    report = classification_report(y_val, intent_pred, labels=ALL_CLASSES, output_dict=True, zero_division=0)
    class_rows = []
    for cls in ALL_CLASSES:
        r = report.get(cls, {})
        class_rows.append({"class": cls, "precision": r.get("precision", 0.0), "recall": r.get("recall", 0.0), "f1": r.get("f1-score", 0.0), "support": r.get("support", 0)})
    return out, pd.DataFrame(class_rows)


def write_summary(pair_df, group_df, centroid_df, intent_df):
    inspect_pairs = [
        "read_file<->grep_search",
        "read_file<->list_directory",
        "grep_search<->list_directory",
        "glob_pattern<->read_file",
        "glob_pattern<->grep_search",
    ]
    comm_pairs = ["ask_user<->plan_task", "plan_task<->web_search", "ask_user<->web_search"]
    inspect_pair_mean = pair_df[pair_df["pair"].isin(inspect_pairs)]["pair_error_rate"].mean()
    comm_pair_mean = pair_df[pair_df["pair"].isin(comm_pairs)]["pair_error_rate"].mean()
    intent_delta = intent_df[intent_df["model"] == "delta_intent_minus_base"].iloc[0]
    lines = [
        "# SupCon/LCL + INTENT v2.1 Probe",
        "",
        "## S1 Metrics",
        f"- inspect pair error mean: `{inspect_pair_mean:.6f}`",
        f"- communicate pair error mean: `{comm_pair_mean:.6f}`",
        f"- inspect4 Macro-F1: `{float(group_df[group_df['group'] == 'inspect4']['macro_f1'].iloc[0]):.6f}`",
        f"- communicate4 Macro-F1: `{float(group_df[group_df['group'] == 'communicate4']['macro_f1'].iloc[0]):.6f}`",
        f"- execute3 Macro-F1: `{float(group_df[group_df['group'] == 'execute3']['macro_f1'].iloc[0]):.6f}`",
        f"- modify3 Macro-F1: `{float(group_df[group_df['group'] == 'modify3']['macro_f1'].iloc[0]):.6f}`",
        "",
        "M2 note: centroid separation is computed from saved fold0 logits as a cheap proxy. True pooled-embedding centroid requires a separate model forward pass.",
        "",
        "## S2 Margin Profile",
        "- Low margin error share means pair-bias/calibration may help.",
        "- High margin error share means representation/loss changes such as SupCon/LCL are more relevant.",
        "",
        "## S3 INTENT Tier-B",
        f"- overall Macro-F1 delta: `{float(intent_delta['overall_macro_f1']):.6f}`",
        f"- communicate4 Macro-F1 delta: `{float(intent_delta['communicate4_macro_f1']):.6f}`",
        f"- web_search F1 delta: `{float(intent_delta['web_search_f1']):.6f}`",
        f"- ask_user F1 delta: `{float(intent_delta['ask_user_f1']):.6f}`",
        f"- plan_task F1 delta: `{float(intent_delta['plan_task_f1']):.6f}`",
        f"- respond_only F1 delta: `{float(intent_delta['respond_only_f1']):.6f}`",
        "",
        "## Decision",
    ]
    if float(intent_delta["communicate4_macro_f1"]) >= 0.004:
        lines.append("- INTENT passes the Tier-B communicate gate. Add `[INTENT]` to the next transformer serializer or student training bundle.")
    else:
        lines.append("- INTENT does not pass the `+0.004` communicate4 Tier-B gate as a standalone LogReg feature.")
    if inspect_pair_mean > 0.10:
        lines.append("- Inspect confusion remains high enough to justify SupCon/LCL if GPU budget is available.")
    else:
        lines.append("- Inspect pair confusion is not high enough by itself to justify immediate SupCon/LCL.")
    return "\n".join(lines) + "\n"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    samples, fold0_samples, ids, logits, probs, y, pred = load_fold0()
    pair_df = compute_pair_errors(y, pred, probs)
    group_df = compute_group_metrics(y, pred)
    centroid_df = centroid_proxy(logits, y)
    intent_df, intent_class_df = run_intent_tier_b(samples)

    pair_df.to_csv(REPORT_DIR / "pair_errors.csv", index=False)
    group_df.to_csv(REPORT_DIR / "group_macro_f1.csv", index=False)
    centroid_df.to_csv(REPORT_DIR / "logit_centroid_proxy.csv", index=False)
    intent_df.to_csv(REPORT_DIR / "intent_tier_b.csv", index=False)
    intent_class_df.to_csv(REPORT_DIR / "intent_class_report.csv", index=False)
    pd.DataFrame(confusion_matrix(y, pred, labels=list(range(len(ALL_CLASSES)))), index=ALL_CLASSES, columns=ALL_CLASSES).to_csv(REPORT_DIR / "fold0_confusion_matrix.csv")
    summary = write_summary(pair_df, group_df, centroid_df, intent_df)
    (REPORT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
