# Episode 005 — Baseline experiment

Status: **upcoming**.

Episode 005 begins only after the Episode 003 target and Episode 004 feature set
are frozen.

For the simple teaching version, start with:

```powershell
python examples/ep005_baseline_minimal.py path/to/ep002-run/accepted.csv
```

That example shows only the majority-class baseline.

This directory contains the **full measured benchmark**.

## Baseline ladder

- **B0 — majority class**
- **B1 — previous-hour direction**
- **B2 — Logistic Regression** using the frozen Episode 004 feature set

## Evaluation protocol

The eligible sample sequence is split chronologically:

- 60% train;
- 20% validation;
- 20% locked test.

Samples whose target timestamp crosses from train into validation, or from
validation into test, are purged at the boundary.

Episode 005 reports train diagnostics and validation baseline metrics. The final
test partition is created but is not evaluated during baseline development.

## Recorded run

```bash
python experiments/ep005_baselines/run_baselines.py \
  --input path/to/ep002-run/accepted.csv \
  --output reports/ep005/baseline_report.json
```

The measured report is the source for Episode 005 visuals. Never replace it with
numbers produced by the minimal teaching example or with illustrative results.
