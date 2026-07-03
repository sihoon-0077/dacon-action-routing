import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, log_loss, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.constants import ALL_CLASSES, COARSE_OF, ID2LABEL, LABEL2ID, SEED
from pipeline_v4.common.data_io import load_fold_rows, load_train_samples
from pipeline_v4.serialize import serialize_for_tokenizer


class EncodedDataset(Dataset):
    def __init__(self, encodings, y):
        self.encodings = encodings
        self.y = torch.tensor(y, dtype=torch.long)
        self.y_coarse = torch.tensor(COARSE_OF[y], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["y"] = self.y[idx]
        item["y_coarse"] = self.y_coarse[idx]
        return item


class MultiTaskClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_fine = nn.Linear(h, 14)
        self.head_coarse = nn.Linear(h, 4)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        pooled = self.dropout(pooled)
        return self.head_fine(pooled), self.head_coarse(pooled)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, overrides):
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value: {item}")
        key, value = item.split("=", 1)
        if value.lower() in {"true", "false"}:
            parsed = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        cfg[key] = parsed
    return cfg


def split_samples(samples, fold_rows, fold):
    fold_of_id = {row["id"]: int(row["fold"]) for row in fold_rows}
    train, val = [], []
    for sample in samples:
        if fold_of_id[sample["id"]] == fold:
            val.append(sample)
        else:
            train.append(sample)
    return train, val


def tokenize_samples(samples, tokenizer, max_len, serializer="v1", batch_size=512):
    texts = [serialize_for_tokenizer(sample, tokenizer, max_len, serializer) for sample in samples]
    encoded = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def parameter_groups(model, cfg):
    head_params = list(model.head_fine.parameters()) + list(model.head_coarse.parameters())
    head_ids = {id(p) for p in head_params}
    encoder_params = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": encoder_params, "lr": float(cfg["lr_encoder"]), "weight_decay": float(cfg["weight_decay"])},
        {"params": head_params, "lr": float(cfg["lr_head"]), "weight_decay": float(cfg["weight_decay"])},
    ]


def evaluate(model, loader, device):
    model.eval()
    fine_all, coarse_all, y_all = [], [], []
    loss_sum, n = 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for batch in loader:
            y = batch.pop("y").to(device)
            batch.pop("y_coarse")
            batch = {k: v.to(device) for k, v in batch.items()}
            fine, coarse = model(**batch)
            loss_sum += float(ce(fine.float(), y).detach().cpu())
            n += int(y.numel())
            fine_all.append(fine.detach().float().cpu().numpy())
            coarse_all.append(coarse.detach().float().cpu().numpy())
            y_all.append(y.detach().cpu().numpy())
    fine = np.concatenate(fine_all, axis=0)
    coarse = np.concatenate(coarse_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    pred = fine.argmax(axis=1)
    probs = torch.softmax(torch.tensor(fine), dim=1).numpy()
    return {
        "fine_logits": fine,
        "coarse_logits": coarse,
        "y": y,
        "pred": pred,
        "nll": float(log_loss(y, probs, labels=list(range(len(ALL_CLASSES))))),
        "torch_nll": loss_sum / max(n, 1),
        "macro_f1": float(f1_score(y, pred, labels=list(range(len(ALL_CLASSES))), average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
    }


def save_class_report(path, y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(len(ALL_CLASSES))), zero_division=0)
    with open(path, "w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for i, cls in enumerate(ALL_CLASSES):
            writer.writerow({"class": cls, "precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])})


def save_checkpoint(model, tokenizer, model_dir, cfg, fold, epoch, metrics):
    model_dir.mkdir(parents=True, exist_ok=True)
    state = {}
    for key, value in model.state_dict().items():
        state[key] = value.detach().half().cpu() if value.is_floating_point() else value.detach().cpu()
    torch.save(state, model_dir / "model.pt")
    if hasattr(model, "encoder") and hasattr(model.encoder, "config"):
        model.encoder.config.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    if not (model_dir / "tokenizer.json").exists():
        raise FileNotFoundError(f"tokenizer.json missing after save_pretrained: {model_dir}")
    payload = {
        "backbone": cfg["backbone"],
        "fold": fold,
        "epoch": epoch,
        "classes": ALL_CLASSES,
        "metrics": metrics,
        "config": cfg,
    }
    (model_dir / "model_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", default="pipeline_v4/configs/mdeberta_a.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--fold-file", default="pipeline_v4/folds/fold_assignments.csv")
    parser.add_argument("--artifact-dir", default="pipeline_v4/artifacts")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    set_seed(int(cfg.get("seed", SEED)))

    run = cfg["run_name"]
    artifact = Path(args.artifact_dir)
    report_dir = artifact / "reports" / run
    model_dir = artifact / "models" / run / f"fold_{args.fold}"
    oof_dir = artifact / "oof" / run
    report_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    samples = load_train_samples(args.data_dir)
    folds = load_fold_rows(args.fold_file)
    train_samples, val_samples = split_samples(samples, folds, args.fold)
    y_train = np.array([LABEL2ID[s["action"]] for s in train_samples], dtype=np.int64)
    y_val = np.array([LABEL2ID[s["action"]] for s in val_samples], dtype=np.int64)

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"], use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("fast tokenizer required by v4 spec")

    print(json.dumps({
        "event": "data_loaded",
        "run": run,
        "fold": args.fold,
        "train": len(train_samples),
        "val": len(val_samples),
        "train_counts": dict(Counter(ID2LABEL[int(i)] for i in y_train)),
        "val_counts": dict(Counter(ID2LABEL[int(i)] for i in y_val)),
    }, ensure_ascii=False))

    serializer = str(cfg.get("serializer", "v1"))
    print(f"tokenizing train serializer={serializer}")
    train_enc = tokenize_samples(train_samples, tokenizer, int(cfg["max_len"]), serializer)
    print(f"tokenizing val serializer={serializer}")
    val_enc = tokenize_samples(val_samples, tokenizer, int(cfg["max_len"]), serializer)

    train_loader = DataLoader(
        EncodedDataset(train_enc, y_train),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        EncodedDataset(val_enc, y_val),
        batch_size=max(8, int(cfg["batch_size"]) * 4),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskClassifier(cfg["backbone"])
    # Keep trainable weights in fp32; autocast handles fp16 activations. Some
    # local HF configs record fp16 dtype and would otherwise create fp16 grads.
    model.float()
    model.to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, cfg))
    total_steps = math.ceil(len(train_loader) / int(cfg["grad_accum"])) * int(cfg["epochs"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(total_steps * float(cfg["warmup_ratio"])),
        total_steps,
    )
    use_amp = device.type == "cuda" and bool(cfg.get("fp16", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    class_weight = None
    if str(cfg.get("class_weight", "none")) == "sqrt_inv_freq":
        counts = np.bincount(y_train, minlength=len(ALL_CLASSES)).astype(np.float64)
        weights = np.sqrt(counts.mean() / np.maximum(counts, 1.0))
        weights = weights / weights.mean()
        class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
        print("class_weight=sqrt_inv_freq " + json.dumps({ALL_CLASSES[i]: float(weights[i]) for i in range(len(ALL_CLASSES))}, ensure_ascii=False))
    ce_fine = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=float(cfg["label_smoothing"]))
    ce_coarse = nn.CrossEntropyLoss()

    best = {"fold_val_nll": float("inf"), "epoch": 0}
    history = []
    start = time.time()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, steps = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            y = batch.pop("y").to(device)
            y_coarse = batch.pop("y_coarse").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                fine, coarse = model(**batch)
                loss = ce_fine(fine.float(), y) + float(cfg["coarse_loss_weight"]) * ce_coarse(coarse.float(), y_coarse)
                loss = loss / int(cfg["grad_accum"])
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu()) * int(cfg["grad_accum"])
            steps += 1
            if step % int(cfg["grad_accum"]) == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % max(1, len(train_loader) // 10) == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={total_loss / steps:.4f}", flush=True)

        metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "fold_val_nll": metrics["nll"],
            "fold_val_macro_f1": metrics["macro_f1"],
            "fold_val_accuracy": metrics["accuracy"],
            "elapsed_sec": time.time() - start,
        }
        history.append(row)
        print("eval " + json.dumps(row, ensure_ascii=False), flush=True)
        if row["fold_val_nll"] < best["fold_val_nll"]:
            best = row.copy()
            np.save(oof_dir / f"fold_{args.fold}_logits.npy", metrics["fine_logits"])
            np.save(oof_dir / f"fold_{args.fold}_coarse.npy", metrics["coarse_logits"])
            np.save(oof_dir / f"fold_{args.fold}_y.npy", metrics["y"])
            (oof_dir / f"fold_{args.fold}_ids.txt").write_text("\n".join(s["id"] for s in val_samples) + "\n", encoding="utf-8")
            save_class_report(report_dir / f"fold_{args.fold}_class_report.csv", metrics["y"], metrics["pred"])
            save_checkpoint(model, tokenizer, model_dir, cfg, args.fold, epoch, row)

    report = {
        "run": run,
        "fold": args.fold,
        "config": cfg,
        "best": best,
        "history": history,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
    }
    (report_dir / f"fold_{args.fold}_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("best " + json.dumps(best, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
