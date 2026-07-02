import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.constants import ALL_CLASSES


def fit_temperature(logits, y):
    def objective(temp):
        probs = softmax(logits / temp, axis=1)
        return log_loss(y, probs, labels=list(range(len(ALL_CLASSES))))

    before = objective(1.0)
    result = minimize_scalar(objective, bounds=(0.5, 5.0), method="bounded")
    temp = float(result.x)
    after = objective(temp)
    return temp, before, after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--artifact-dir", default="pipeline_v4/artifacts")
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()

    artifact = Path(args.artifact_dir)
    oof_dir = artifact / "oof" / args.run
    out_dir = artifact / "calib" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    temps = {}
    rows = []
    for fold in [int(x) for x in args.folds.split(",") if x.strip()]:
        logits_path = oof_dir / f"fold_{fold}_logits.npy"
        y_path = oof_dir / f"fold_{fold}_y.npy"
        if not logits_path.exists():
            print(f"skip missing fold {fold}: {logits_path}")
            continue
        logits = np.load(logits_path)
        y = np.load(y_path).astype(int)
        temp, before, after = fit_temperature(logits, y)
        if after > before + 1e-10:
            raise RuntimeError(f"fold {fold} calibration worsened: {before} -> {after}")
        if not (0.8 <= temp <= 3.0):
            raise RuntimeError(f"fold {fold} suspicious temperature: {temp}")
        temps[f"fold_{fold}"] = temp
        rows.append({"fold": fold, "temperature": temp, "nll_before": before, "nll_after": after})
    if len(rows) >= 2:
        std = float(np.std([r["temperature"] for r in rows]))
        if std >= 0.3:
            raise RuntimeError(f"temperature std too high: {std}")
    payload = {
        "run": args.run,
        "temperatures": temps,
        "rows": rows,
        "temperature_std": float(np.std([r["temperature"] for r in rows])) if rows else None,
    }
    (out_dir / "temperatures.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
