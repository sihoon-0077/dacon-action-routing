import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.constants import ALL_CLASSES, LABEL2ID
from pipeline_v4.common.data_io import load_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="pipeline_v4/artifacts")
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()

    artifact = Path(args.artifact_dir)
    oof_dir = artifact / "oof" / args.run
    temps = json.loads((artifact / "calib" / args.run / "temperatures.json").read_text(encoding="utf-8"))["temperatures"]
    labels = load_labels(Path(args.data_dir) / "train_labels.csv")

    probs_rows, y_rows, id_rows = [], [], []
    for fold in [int(x) for x in args.folds.split(",") if x.strip()]:
        logits_path = oof_dir / f"fold_{fold}_logits.npy"
        ids_path = oof_dir / f"fold_{fold}_ids.txt"
        if not logits_path.exists():
            print(f"skip missing fold {fold}: {logits_path}")
            continue
        logits = np.load(logits_path)
        ids = ids_path.read_text(encoding="utf-8").splitlines()
        if len(ids) != logits.shape[0]:
            raise ValueError(f"fold {fold} id/logit length mismatch")
        temp = float(temps[f"fold_{fold}"])
        probs_rows.append(softmax(logits / temp, axis=1))
        id_rows.extend(ids)
        y_rows.extend(LABEL2ID[labels[sample_id]] for sample_id in ids)

    if len(set(id_rows)) != len(id_rows):
        raise RuntimeError("duplicate ids in OOF assembly")
    probs = np.vstack(probs_rows)
    y = np.asarray(y_rows, dtype=np.int64)
    np.save(oof_dir / "oof_probs.npy", probs)
    np.save(oof_dir / "oof_y.npy", y)
    (oof_dir / "oof_ids.txt").write_text("\n".join(id_rows) + "\n", encoding="utf-8")
    pred = probs.argmax(axis=1)
    payload = {
        "run": args.run,
        "n": int(len(y)),
        "nll": float(log_loss(y, probs, labels=list(range(len(ALL_CLASSES))))),
        "argmax_macro_f1": float(f1_score(y, pred, labels=list(range(len(ALL_CLASSES))), average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "folds_included": [int(x) for x in args.folds.split(",") if x.strip() and (oof_dir / f"fold_{int(x)}_logits.npy").exists()],
    }
    (oof_dir / "oof_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
