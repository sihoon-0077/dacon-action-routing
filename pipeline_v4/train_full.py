import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.constants import ALL_CLASSES, ID2LABEL, LABEL2ID, SEED
from pipeline_v4.common.data_io import load_train_samples
from pipeline_v4.train_fold import (
    EncodedDataset,
    MultiTaskClassifier,
    apply_overrides,
    load_config,
    parameter_groups,
    save_checkpoint,
    set_seed,
    tokenize_samples,
)


def parse_save_epochs(value):
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    if isinstance(value, (list, tuple, set)):
        return {int(x) for x in value}
    text = str(value).strip()
    if not text:
        return set()
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pipeline_v4/configs/mdeberta_a_local8gb.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="pipeline_v4/artifacts")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(int(cfg.get("seed", SEED)))
    run = str(cfg["run_name"])
    save_epochs = parse_save_epochs(cfg.get("save_epochs"))
    artifact = Path(args.artifact_dir)
    report_dir = artifact / "reports" / run
    model_dir = artifact / "models" / run / "full"
    report_dir.mkdir(parents=True, exist_ok=True)

    samples = load_train_samples(args.data_dir)
    y = np.array([LABEL2ID[s["action"]] for s in samples], dtype=np.int64)
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"], use_fast=True)
    serializer = str(cfg.get("serializer", "v1"))
    print(json.dumps({
        "event": "data_loaded",
        "run": run,
        "train": len(samples),
        "counts": dict(Counter(ID2LABEL[int(i)] for i in y)),
        "serializer": serializer,
        "max_len": int(cfg["max_len"]),
        "epochs": int(cfg["epochs"]),
    }, ensure_ascii=False), flush=True)

    print(f"tokenizing full train serializer={serializer}", flush=True)
    enc = tokenize_samples(samples, tokenizer, int(cfg["max_len"]), serializer)
    loader = DataLoader(
        EncodedDataset(enc, y),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskClassifier(cfg["backbone"])
    model.float()
    model.to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, cfg))
    total_steps = math.ceil(len(loader) / int(cfg["grad_accum"])) * int(cfg["epochs"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(total_steps * float(cfg["warmup_ratio"])),
        total_steps,
    )
    use_amp = device.type == "cuda" and bool(cfg.get("fp16", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce_fine = nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))
    ce_coarse = nn.CrossEntropyLoss()

    history = []
    start = time.time()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, steps = 0.0, 0
        for step, batch in enumerate(loader, 1):
            labels = batch.pop("y").to(device)
            y_coarse = batch.pop("y_coarse").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                fine, coarse = model(**batch)
                loss = ce_fine(fine.float(), labels) + float(cfg["coarse_loss_weight"]) * ce_coarse(coarse.float(), y_coarse)
                loss = loss / int(cfg["grad_accum"])
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu()) * int(cfg["grad_accum"])
            steps += 1
            if step % int(cfg["grad_accum"]) == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % max(1, len(loader) // 10) == 0:
                print(f"epoch={epoch} step={step}/{len(loader)} loss={total_loss / steps:.4f}", flush=True)

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "elapsed_sec": time.time() - start,
        }
        history.append(row)
        print("epoch_done " + json.dumps(row, ensure_ascii=False), flush=True)
        save_checkpoint(model, tokenizer, model_dir, cfg, "full", epoch, row)
        if epoch in save_epochs:
            epoch_dir = artifact / "models" / run / f"full_epoch_{epoch}"
            save_checkpoint(model, tokenizer, epoch_dir, cfg, f"full_epoch_{epoch}", epoch, row)
            print(f"saved_epoch_checkpoint {epoch_dir}", flush=True)

    payload = {
        "run": run,
        "config": cfg,
        "history": history,
        "train_size": len(samples),
        "model_dir": str(model_dir),
    }
    (report_dir / "full_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved " + json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
