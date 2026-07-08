import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.data_io import load_train_samples
from pipeline_v4.serialize import serialize_for_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--serializer", default="v2_2")
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-limit", type=int, default=2000)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload = {
        "model": args.model,
        "serializer": args.serializer,
        "max_len": args.max_len,
        "audit_limit": args.audit_limit,
        "cuda": torch.cuda.is_available(),
        "ok": False,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=True,
            trust_remote_code=args.trust_remote_code,
        )
        samples = load_train_samples("data")
        subset = samples[: max(1, min(args.audit_limit, len(samples)))]
        lengths = []
        for sample in subset:
            text = serialize_for_tokenizer(sample, tokenizer, args.max_len, args.serializer)
            lengths.append(len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]))
        payload["token_lengths"] = {
            "n": len(lengths),
            "mean": float(np.mean(lengths)),
            "p50": float(np.percentile(lengths, 50)),
            "p90": float(np.percentile(lengths, 90)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(np.max(lengths)),
        }

        model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        texts = [serialize_for_tokenizer(sample, tokenizer, args.max_len, args.serializer) for sample in samples[:2]]
        enc = tokenizer(texts, max_length=args.max_len, padding="max_length", truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            outputs = model(**enc)
        hidden = getattr(outputs, "last_hidden_state", None)
        payload["forward"] = {
            "device": str(device),
            "hidden_shape": list(hidden.shape) if hidden is not None else None,
            "peak_memory_mb": int(torch.cuda.max_memory_allocated() / 1024 / 1024) if device.type == "cuda" else None,
        }
        payload["ok"] = hidden is not None
    except Exception as exc:
        payload["error"] = repr(exc)
    finally:
        payload["elapsed_sec"] = time.time() - started
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if payload.get("ok") else 1)


if __name__ == "__main__":
    main()

