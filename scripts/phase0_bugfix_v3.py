import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import ACTIONS
from src.io_utils import load_train
from src.serialize_policy import serialize_policy_state, serialize_policy_v3


def parse_transformer_classes(path):
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"ALL_CLASSES\s*=\s*(\[[\s\S]*?\])", text)
    if not match:
        return None
    return ast.literal_eval(match.group(1))


def parse_loss_history(stdout_path):
    path = Path(stdout_path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("eval "):
            payload = json.loads(line.split(" ", 1)[1])
            rows.append(payload)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--transformer-script", default="transformer_action_routing.py")
    parser.add_argument("--stdout-log", default="reports/transformer/B-full-mdeberta-70k-nowfirst-lr5e5-none-3e/stdout.log")
    parser.add_argument("--out-md", default="artifacts_v3/reports/phase0_bugfix.md")
    args = parser.parse_args()

    classes = parse_transformer_classes(args.transformer_script)
    label_order_ok = classes == ACTIONS
    history = parse_loss_history(args.stdout_log)
    losses = [row.get("train_loss") for row in history]
    loss_decreases = bool(losses) and losses[-1] < losses[0]

    samples = load_train(args.data_dir)[:5]
    legacy_examples = [serialize_policy_state(sample, layout="legacy") for sample in samples]
    now_first_examples = [serialize_policy_state(sample, layout="now_first") for sample in samples]
    v3_examples = [serialize_policy_v3(sample) for sample in samples[:3]]

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 0 Bugfix Report",
        "",
        "## Verdict",
        "",
        "- Primary root cause found earlier: `[NOW] current_prompt` was placed at the tail in the legacy serializer, so long samples could lose the target prompt under head-only truncation.",
        "- v3 mitigation: `serialize_policy_v3` exposes `[META]`, `[OPEN]`, `[FLAG]`, and `[NOW]` explicitly; training code must preserve `[NOW]` during truncation.",
        "",
        "## P0-1 Label Order",
        "",
        f"- transformer script label order matches `ACTIONS`: `{label_order_ok}`",
        f"- transformer classes: `{classes}`",
        "",
        "## P0-2 Loss Curve",
        "",
        f"- parsed epochs: `{len(history)}`",
        f"- train losses: `{[round(x, 6) for x in losses]}`",
        f"- loss decreases: `{loss_decreases}`",
        "",
        "## P0-3 Serialization Samples",
        "",
        "Legacy serializer puts `[NOW]` at the end:",
        "",
        "```text",
        legacy_examples[0][:1800],
        "```",
        "",
        "Fixed now-first serializer puts `[NOW]` first:",
        "",
        "```text",
        now_first_examples[0][:1800],
        "```",
        "",
        "v3 serializer sample:",
        "",
        "```text",
        v3_examples[0][:1800],
        "```",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
