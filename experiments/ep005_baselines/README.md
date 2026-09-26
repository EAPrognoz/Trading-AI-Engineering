# Episode 005 — Baseline experiment

Episode 005 begins only after the Episode 003 target and Episode 004 feature
contracts are frozen.

## Task

At the close of H1 bar `t`, predict the direction of the next one-hour
close-to-close return.

## Baselines

- **B0 — majority class**: the most common UP/DOWN label in the training set;
- **B1 — previous-hour direction**: use the sign of the current H1 return as the
  next-direction prediction;
- **B2 — Logistic Regression**: standardized linear classifier using the frozen
  Episode 004 baseline-v1 feature set.

## Evaluation protocol

The eligible sample sequence is split chronologically:

- 60% train;
- 20% validation;
- 20% locked test.

Samples whose target timestamp crosses from train into validation, or from
validation into test, are purged at the boundary.

Episode 005 reports train diagnostics and **validation baseline metrics**. The
final test partition is created and identified but is not evaluated. This keeps
it available for a later one-time candidate-model comparison instead of turning
it into another tuning set.

## Run

```bash
python experiments/ep005_baselines/run_baselines.py \
  --input path/to/validated_h1_snapshot.csv \
  --output reports/ep005/baseline_report.json
```

The output is intended to become the measured source for Episode 005 visuals.
Do not replace measured values with illustrative numbers.
