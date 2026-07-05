import argparse
import json
import shutil
import textwrap
import zipfile
from pathlib import Path


PROBE_SUFFIX = r'''
import json
import sys
from collections import Counter, defaultdict


def session_id(sample_id):
    return sample_id.rsplit("-step_", 1)[0] if "-step_" in sample_id else sample_id


def iter_prompt_action_pairs(sample):
    sid = session_id(sample.get("id", ""))
    history = sample.get("history") or []
    for i, turn in enumerate(history):
        if turn.get("role") != "user":
            continue
        prompt = turn.get("content", "")
        if not prompt:
            continue
        next_action = None
        for later in history[i + 1 :]:
            if later.get("role") == "assistant_action":
                next_action = later.get("name")
                break
            if later.get("role") == "user":
                break
        if next_action:
            yield (sid, prompt), str(next_action)


def build_strict_self_lookup(samples):
    votes = defaultdict(Counter)
    pair_count = 0
    for sample in samples:
        for key, action in iter_prompt_action_pairs(sample):
            pair_count += 1
            votes[key][action] += 1
    lookup = {}
    conflict_examples = []
    for key, counter in votes.items():
        if len(counter) == 1:
            lookup[key] = next(iter(counter))
            continue
        if len(conflict_examples) < 12:
            conflict_examples.append(
                {
                    "session": key[0],
                    "prompt": key[1][:220],
                    "actions": dict(counter),
                }
            )
    return lookup, {
        "source": "test_jsonl_self_history_only",
        "source_samples": len(samples),
        "history_pairs": pair_count,
        "raw_keys": len(votes),
        "usable_keys": len(lookup),
        "conflict_keys": len(votes) - len(lookup),
        "conflict_examples": conflict_examples,
    }


def predict_base(samples, model, feature_mode):
    if feature_mode == "advanced_router":
        return predict_advanced_router(samples, model)
    if feature_mode == "routing_margin_router":
        return predict_routing_margin_router(samples, model)
    if feature_mode == "compact_flags_router":
        return [str(pred) for pred in predict_compact_flags_router(samples, model)]
    texts = [serialize_sample(sample, feature_mode) for sample in samples]
    return [str(pred) for pred in model.predict(texts)] if texts else []


def apply_strict_self_lookup(samples, preds):
    lookup, stats = build_strict_self_lookup(samples)
    out = []
    hits = 0
    changed = 0
    hit_by_base = Counter()
    override_by_action = Counter()
    examples = []
    for sample, pred in zip(samples, preds):
        key = (session_id(sample.get("id", "")), sample.get("current_prompt", ""))
        override = lookup.get(key)
        if override:
            hits += 1
            hit_by_base[str(pred)] += 1
            override_by_action[str(override)] += 1
            if override != pred:
                changed += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "id": sample.get("id"),
                            "base": str(pred),
                            "override": str(override),
                            "prompt": sample.get("current_prompt", "")[:220],
                        }
                    )
            out.append(str(override))
        else:
            out.append(str(pred))
    stats.update(
        {
            "hit_count": hits,
            "hit_rate": hits / max(len(samples), 1),
            "changed_count": changed,
            "changed_rate": changed / max(len(samples), 1),
            "hit_by_base": dict(hit_by_base),
            "override_by_action": dict(override_by_action),
            "changed_examples": examples,
        }
    )
    print("strict_test_self_lookup: " + json.dumps(stats, ensure_ascii=False), file=sys.stderr)
    return out


def probe_main():
    test_path, sample_submission_path, model_dir, output_path = runtime_paths()
    model, feature_mode = load_model_and_config(model_dir)
    samples = load_jsonl(test_path)
    ids = [sample.get("id", "") for sample in samples]
    preds = predict_base(samples, model, feature_mode)
    preds = apply_strict_self_lookup(samples, preds)
    pred_by_id = dict(zip(ids, preds))
    if sample_submission_path:
        fieldnames, rows = load_sample_submission(sample_submission_path)
    else:
        fieldnames, rows = submission_rows_from_ids(ids)
    for row in rows:
        if row["id"] in pred_by_id:
            row["action"] = pred_by_id[row["id"]]
    save_submission(output_path, fieldnames, rows)
    print(f"Saved: {output_path} rows={len(rows)}")


if __name__ == "__main__":
    probe_main()
'''


def build_self_contained_script(base_script_path):
    base_text = Path(base_script_path).read_text(encoding="utf-8")
    entrypoint = '\nif __name__ == "__main__":\n    main()\n'
    if entrypoint not in base_text:
        raise ValueError(f"Could not find standard main entrypoint in {base_script_path}")
    base_without_entrypoint = base_text.rsplit(entrypoint, 1)[0].rstrip()
    return base_without_entrypoint + "\n\n\n" + textwrap.dedent(PROBE_SUFFIX).lstrip()


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
    parser.add_argument("--base-script", default="script.py")
    parser.add_argument("--advanced-router", default="model/advanced_router.pkl")
    parser.add_argument("--out-dir", default="submit_probe_test_self_lookup_strict")
    parser.add_argument("--zip-path", default="submit_probe_test_self_lookup_strict.zip")
    args = parser.parse_args()

    advanced_router = Path(args.advanced_router)
    if not advanced_router.exists():
        raise FileNotFoundError(
            f"missing {advanced_router}. Train or stage advanced_router.pkl before building this probe."
        )

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "model").mkdir(parents=True)
    script_text = build_self_contained_script(args.base_script)
    (out_dir / "script.py").write_text(script_text, encoding="utf-8")
    shutil.copy2(advanced_router, out_dir / "model" / "advanced_router.pkl")
    (out_dir / "requirements.txt").write_text("", encoding="utf-8")
    size = zip_dir(out_dir, args.zip_path)
    print(f"out_dir={out_dir}")
    print(f"zip={args.zip_path} zip_bytes={size}")


if __name__ == "__main__":
    main()
