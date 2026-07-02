import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import get_session_id, load_test, load_train, write_json
from src.replay_lookup import build_replay_lookup, replay_key


def evaluate(samples, lookup, scoped=True):
    covered = 0
    correct = 0
    changed_rows = []
    for sample in samples:
        pred = lookup.get(replay_key(sample, scoped=scoped))
        if pred:
            covered += 1
            if sample.get("action") == pred:
                correct += 1
            if len(changed_rows) < 10:
                changed_rows.append({"id": sample["id"], "action": sample.get("action"), "lookup": pred, "prompt": sample.get("current_prompt", "")[:160]})
    return {"coverage": covered, "total": len(samples), "coverage_rate": covered / max(len(samples), 1), "precision": correct / max(covered, 1), "examples": changed_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="artifacts/reports/replay_audit")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_train(args.data_dir)
    try:
        test_samples = load_test(args.data_dir)
    except FileNotFoundError:
        test_samples = []

    groups = np.array([get_session_id(sample["id"]) for sample in samples])
    idx = np.arange(len(samples))
    train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(idx, [s["action"] for s in samples], groups))
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    payload = {}
    for scoped in [True, False]:
        name = "session_scoped" if scoped else "global_prompt"
        lookup_all, _, stats_all = build_replay_lookup(samples, scoped=scoped, policy="unique")
        lookup_train, _, stats_train = build_replay_lookup(train_samples, scoped=scoped, policy="unique")
        lookup_train_plus_val, _, stats_tv = build_replay_lookup(train_samples + val_samples, scoped=scoped, policy="unique")
        payload[name] = {
            "all_train_stats": stats_all,
            "split_train_stats": stats_train,
            "train_to_valid": evaluate(val_samples, lookup_train, scoped=scoped),
            "valid_self_transductive": evaluate(val_samples, lookup_train_plus_val, scoped=scoped),
            "placeholder_test_hits": sum(1 for sample in test_samples if lookup_all.get(replay_key(sample, scoped=scoped))),
            "placeholder_test_total": len(test_samples),
            "train_internal": evaluate(samples, lookup_all, scoped=scoped),
        }

    write_json(out_dir / "replay_audit.json", payload)
    lines = ["# Replay Lookup Audit", ""]
    for name, row in payload.items():
        lines += [
            f"## {name}",
            "",
            f"- train internal coverage: `{row['train_internal']['coverage']}/{row['train_internal']['total']}`",
            f"- train internal precision: `{row['train_internal']['precision']:.6f}`",
            f"- GroupSplit train-to-valid coverage: `{row['train_to_valid']['coverage']}/{row['train_to_valid']['total']}`",
            f"- GroupSplit train-to-valid precision: `{row['train_to_valid']['precision']:.6f}`",
            f"- transductive valid coverage: `{row['valid_self_transductive']['coverage']}/{row['valid_self_transductive']['total']}`",
            f"- transductive valid precision: `{row['valid_self_transductive']['precision']:.6f}`",
            f"- placeholder test hits: `{row['placeholder_test_hits']}/{row['placeholder_test_total']}`",
            "",
        ]
    lines += [
        "## Recommendation",
        "",
        "- Keep replay variants separated from safe submissions.",
        "- Session-scoped test self-history replay is diagnostic/transductive and should only be used after rule confirmation.",
    ]
    (out_dir / "replay_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {out_dir / 'replay_audit.md'}")


if __name__ == "__main__":
    main()
