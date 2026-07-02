import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup


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

LABEL_TO_ID = {label: i for i, label in enumerate(ALL_CLASSES)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def session_id(sample_id):
    return sample_id.rsplit("-step_", 1)[0] if "-step_" in sample_id else sample_id


def clean(value, max_chars=900):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(text.split())[:max_chars]


def bucket_number(value, bins):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for name, upper in bins:
        if value <= upper:
            return name
    return "huge"


def summarize_args(args, max_chars=260):
    if not isinstance(args, dict) or not args:
        return ""
    keep = []
    useful_keys = [
        "path",
        "file",
        "filename",
        "target",
        "pattern",
        "query",
        "glob",
        "command",
        "cmd",
        "args",
        "cwd",
    ]
    for key in useful_keys:
        if key in args and args[key] not in (None, "", []):
            keep.append(f"{key}={clean(args[key], 120)}")
    if not keep:
        for key, value in list(args.items())[:4]:
            keep.append(f"{key}={clean(value, 80)}")
    return " ".join(keep)[:max_chars]


def history_pairs(sample, max_pairs=6):
    pairs = []
    last_user = None
    for turn in sample.get("history", []) or []:
        role = turn.get("role")
        if role == "user":
            last_user = clean(turn.get("content"), 500)
        elif role == "assistant_action":
            action = clean(turn.get("name"), 80)
            args = summarize_args(turn.get("args"), 240)
            result = clean(turn.get("result_summary"), 500)
            pairs.append((last_user or "", action, args, result))
            last_user = None
    return pairs[-max_pairs:]


def serialize_transformer(sample, max_pairs=6):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    open_files = ws.get("open_files", []) or []
    language_mix = ws.get("language_mix", {}) or {}
    mix_items = sorted(language_mix.items(), key=lambda x: -float(x[1]))[:3]

    budget = bucket_number(
        meta.get("budget_tokens_remaining"),
        [("lt5k", 5000), ("lt20k", 20000), ("lt80k", 80000), ("lt200k", 200000)],
    )
    loc = bucket_number(ws.get("loc"), [("lt5k", 5000), ("lt15k", 15000), ("lt40k", 40000), ("lt100k", 100000)])
    turn = bucket_number(meta.get("turn_index"), [("early", 2), ("mid", 7), ("late", 12)])
    elapsed = bucket_number(meta.get("elapsed_session_sec"), [("short", 60), ("mid", 300), ("long", 1200)])

    chunks = [
        "[META] "
        f"tier={clean(meta.get('user_tier'), 40)} "
        f"lang={clean(meta.get('language_pref'), 40)} "
        f"ci={clean(ws.get('last_ci_status'), 40)} "
        f"dirty={ws.get('git_dirty', 'unknown')} "
        f"turn={turn} budget={budget} elapsed={elapsed} loc={loc}",
        "[OPEN] " + (" ".join(clean(path, 120).replace("\\", "/") for path in open_files[:8]) or "none"),
        "[MIX] " + (" ".join(f"{clean(k, 20)}:{float(v):.2f}" for k, v in mix_items) or "none"),
    ]

    for idx, (user_text, action, args, result) in enumerate(history_pairs(sample, max_pairs=max_pairs), 1):
        chunks.append(f"[H{idx}] U: {user_text} >> A: {action} {args} => {result}")

    chunks.append("[NOW] " + clean(sample.get("current_prompt"), 900))
    return "\n".join(chunks)


def length_report(tokenizer, texts, batch_size=256):
    lengths = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    arr = np.array(lengths, dtype=np.int32)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "over_320": int((arr > 320).sum()),
        "over_384": int((arr > 384).sum()),
        "over_512": int((arr > 512).sum()),
    }


def tokenize_to_tensors(tokenizer, texts, max_len, batch_size=512):
    input_ids = []
    attention_mask = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids.append(enc["input_ids"])
        attention_mask.append(enc["attention_mask"])
    return {
        "input_ids": torch.cat(input_ids, dim=0),
        "attention_mask": torch.cat(attention_mask, dim=0),
    }


class EncodedDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def class_weights(labels, mode):
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=len(ALL_CLASSES)).astype(np.float32)
    weights = len(labels) / (len(ALL_CLASSES) * np.maximum(counts, 1.0))
    if mode == "sqrt-balanced":
        weights = np.sqrt(weights)
    elif mode != "balanced":
        raise ValueError(f"unknown loss weight mode: {mode}")
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(model, loader, device):
    model.eval()
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits_all.append(out.logits.detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())
    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    pred = logits.argmax(axis=1)
    return logits, labels, pred


def save_class_report(path, labels, pred):
    report = classification_report(
        labels,
        pred,
        labels=list(range(len(ALL_CLASSES))),
        target_names=ALL_CLASSES,
        zero_division=0,
        output_dict=True,
    )
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["class", "precision", "recall", "f1-score", "support"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in ALL_CLASSES:
            row = {"class": label, **report[label]}
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-name", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--run-id", default="B-run1")
    parser.add_argument("--out-dir", default=os.path.join("reports", "transformer"))
    parser.add_argument("--model-out-dir", default=os.path.join("models", "transformer"))
    parser.add_argument("--max-len", type=int, default=320)
    parser.add_argument("--history-pairs", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--loss-weight", choices=["none", "balanced", "sqrt-balanced"], default="balanced")
    parser.add_argument("--amp", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--model-dtype", choices=["float32", "float16", "auto"], default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--length-report-only", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    run_dir = Path(args.out_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_out_dir) / args.run_id

    samples = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    labels_map = load_labels(os.path.join(args.data_dir, "train_labels.csv"))
    y = np.array([LABEL_TO_ID[labels_map[s["id"]]] for s in samples], dtype=np.int64)

    if args.sample_limit and args.sample_limit < len(samples):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(np.arange(len(samples)), size=args.sample_limit, replace=False)
        idx.sort()
        samples = [samples[i] for i in idx]
        y = y[idx]

    texts = [serialize_transformer(sample, max_pairs=args.history_pairs) for sample in samples]
    groups = np.array([session_id(sample["id"]) for sample in samples])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    lengths = length_report(tokenizer, texts)
    with open(run_dir / "length_report.json", "w", encoding="utf-8") as f:
        json.dump(lengths, f, ensure_ascii=False, indent=2)
    print("length_report", json.dumps(lengths, ensure_ascii=False))
    if args.length_report_only:
        return

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(np.arange(len(samples)), y, groups=groups))
    x_train = [texts[i] for i in train_idx]
    x_val = [texts[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    print(f"tokenizing train={len(x_train)} val={len(x_val)} max_len={args.max_len}")
    train_enc = tokenize_to_tensors(tokenizer, x_train, args.max_len)
    val_enc = tokenize_to_tensors(tokenizer, x_val, args.max_len)
    train_loader = DataLoader(
        EncodedDataset(train_enc, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        EncodedDataset(val_enc, y_val),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "auto": None,
    }
    model_kwargs = {
        "num_labels": len(ALL_CLASSES),
        "id2label": ID_TO_LABEL,
        "label2id": LABEL_TO_ID,
    }
    if dtype_map[args.model_dtype] is not None:
        model_kwargs["torch_dtype"] = dtype_map[args.model_dtype]
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        **model_kwargs,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_update_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)
    weights = class_weights(y_train, args.loss_weight)
    if weights is not None:
        weights = weights.to(device)
    use_amp = device.type == "cuda" and args.amp != "none"
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp == "fp16")

    best = {"macro_f1": -1.0, "epoch": 0}
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        steps = 0
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(**batch)
                step_weights = weights.to(dtype=out.logits.dtype) if weights is not None else None
                loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=step_weights)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu()) * args.grad_accum
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            steps += 1
            if step % max(1, len(train_loader) // 10) == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={total_loss / steps:.4f}")

        logits, labels, pred = evaluate_model(model, val_loader, device)
        macro = f1_score(labels, pred, labels=list(range(len(ALL_CLASSES))), average="macro", zero_division=0)
        acc = accuracy_score(labels, pred)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "macro_f1": float(macro),
            "accuracy": float(acc),
            "elapsed_sec": time.time() - start_time,
        }
        history.append(row)
        print("eval", json.dumps(row, ensure_ascii=False))

        if macro > best["macro_f1"]:
            best = row | {"val_logits_path": str(run_dir / "val_logits.npy")}
            np.save(run_dir / "val_logits.npy", logits)
            np.save(run_dir / "val_labels.npy", labels)
            np.save(run_dir / "val_pred.npy", pred)
            save_class_report(run_dir / "class_report.csv", labels, pred)
            cm = confusion_matrix(labels, pred, labels=list(range(len(ALL_CLASSES))))
            np.save(run_dir / "confusion_matrix.npy", cm)
            if args.save_model:
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
                with open(model_dir / "classes.json", "w", encoding="utf-8") as f:
                    json.dump(ALL_CLASSES, f, ensure_ascii=False, indent=2)

    metrics = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "max_len": args.max_len,
        "history_pairs": args.history_pairs,
        "loss_weight": args.loss_weight,
        "amp": args.amp,
        "model_dtype": args.model_dtype,
        "sample_limit": args.sample_limit,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "length_report": lengths,
        "best": best,
        "history": history,
        "class_distribution": dict(Counter(ID_TO_LABEL[int(i)] for i in y)),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("best", json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
