import csv
import os
from pathlib import Path


def write_submission(path, ids, preds, sample_submission_path=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_by_id = dict(zip(ids, preds))
    if sample_submission_path and os.path.exists(sample_submission_path):
        with open(sample_submission_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["action"] = pred_by_id.get(row["id"], row.get("action", "respond_only"))
    else:
        rows = [{"id": sample_id, "action": pred_by_id[sample_id]} for sample_id in ids]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "action"])
        writer.writeheader()
        writer.writerows(rows)
