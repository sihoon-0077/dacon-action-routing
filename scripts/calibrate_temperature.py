import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import fit_temperature, logits_to_probs
from src.constants import ACTIONS
from src.io_utils import write_json
from src.metrics import expected_calibration_error, safe_log_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logits = np.load(args.logits)
    labels = np.load(args.labels).astype(int)
    raw_probs = logits_to_probs(logits)
    raw_ll = safe_log_loss(labels, raw_probs, labels=list(range(len(ACTIONS))))
    raw_ece, raw_buckets = expected_calibration_error(labels, raw_probs)
    temp, calibrated_probs, cal_ll = fit_temperature(logits, labels, labels=list(range(len(ACTIONS))))
    cal_ece, cal_buckets = expected_calibration_error(labels, calibrated_probs)
    np.save(out_dir / "calibrated_probs.npy", calibrated_probs)
    payload = {
        "temperature": temp,
        "raw_log_loss": raw_ll,
        "calibrated_log_loss": cal_ll,
        "raw_ece": raw_ece,
        "calibrated_ece": cal_ece,
        "raw_buckets": raw_buckets,
        "calibrated_buckets": cal_buckets,
    }
    write_json(out_dir / "temperature.json", payload)
    md = [
        "# Calibration Report",
        "",
        f"- temperature: `{temp:.6f}`",
        f"- raw log-loss: `{raw_ll:.6f}`",
        f"- calibrated log-loss: `{cal_ll:.6f}`",
        f"- raw ECE: `{raw_ece:.6f}`",
        f"- calibrated ECE: `{cal_ece:.6f}`",
    ]
    (out_dir / "calibration_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"saved: {out_dir / 'calibration_report.md'}")


if __name__ == "__main__":
    main()
