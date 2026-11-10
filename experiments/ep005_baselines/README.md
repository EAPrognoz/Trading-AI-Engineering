# Episode 005 — Baseline experiment

Status: **published**.

Video: [Bitcoin Baselines: Can Logistic Regression Beat Always Up?](https://www.youtube.com/watch?v=iEr_WGBUkxQ).

For the published Bitcoin H1/H4/D1 benchmark, use the
[BTC run guide](https://github.com/EAPrognoz/Trading-AI-Engineering/blob/release/btc-mtf-foundation-20261002/experiments/ep005_baselines/README.md) on `release/btc-mtf-foundation-20261002`.
It documents the required bundle and explicit BTC contract; supply your own
authorized broker data. The general H1 examples below on `main` are not a
reproduction of that published BTC run.

Episode 005 begins only after the Episode 003 target and Episode 004 feature set
are frozen.

For the simple teaching version, start with:

```powershell
python examples/ep005_baseline_minimal.py path/to/ep002-run/accepted.csv
```

That example shows only the majority-class baseline.

This directory on `main` contains the general H1 baseline implementation.
The published BTC implementation and its run instructions are linked above.

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

## General H1 run on main

```bash
python experiments/ep005_baselines/run_baselines.py \
  --input path/to/ep002-run/accepted.csv \
  --output reports/ep005/baseline_report.json
```

For the measured values used in the published Episode 005, follow the BTC
release-branch guide above. Do not substitute this general H1 run, the minimal
teaching example, or illustrative results for that published benchmark.
