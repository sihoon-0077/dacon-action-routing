# Clone Runbook

This repository includes the current DACON submit-ready artifacts through Git LFS.

## 1. Clone

```powershell
git clone https://github.com/sihoon-0077/dacon-action-routing.git
cd dacon-action-routing
git lfs pull
```

## 2. Submit-Ready ZIPs

| File | Purpose | Public note |
|---|---|---|
| `cand_distill.zip` | stable strict-distill baseline | public `0.7174979343`, runtime about `2m58s` |
| `distill_ib.zip` | latest inspect-bias candidate | smoke passed; same runtime shape as distill |

Upload one of these zip files directly to DACON.

## 3. Local Smoke Test

The competition data is not committed. Place official DACON files under `data/`:

```text
data/
  test.jsonl
  sample_submission.csv
```

Then smoke-test an extracted package:

```powershell
Expand-Archive distill_ib.zip smoke_distill_ib -Force
Copy-Item data\test.jsonl smoke_distill_ib\data\test.jsonl -Force
Copy-Item data\sample_submission.csv smoke_distill_ib\data\sample_submission.csv -Force
Push-Location smoke_distill_ib
python script.py
Pop-Location
```

Expected stdout should include:

```text
distill_student: rows=...
Saved: ...\output\submission.csv
```

## 4. Rebuild Latest Inspect-Bias ZIP

```powershell
python scripts\build_submit_distill_inspect_bias.py --out-dir distill_ib --zip-path distill_ib.zip
```

## 5. Rebuild Stable Distill ZIP

```powershell
python scripts\build_submit_distill.py --student-dir model\distill_student_strict --out-dir sub_distill --zip-path cand_distill.zip
```

Note: rebuilding requires local model artifacts under `model/distill_student_strict/`. The submit-ready zip files already contain the runtime model files.
