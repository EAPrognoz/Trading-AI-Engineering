# Episode 004 — Feature set

Published video: https://youtu.be/SNOTNSZoNQY

Start with the small example:

```powershell
python examples/ep004_feature_set_minimal.py path/to/ep002-run/accepted.csv
```

It shows how ordinary OHLCV bars become understandable model inputs.

This directory contains the **full feature-set analysis**.

## Candidate families

- returns at several backward-looking horizons;
- rolling volatility;
- current-bar range and body;
- relative tick volume;
- cyclical hour-of-day and day-of-week encodings.

Every full-pipeline feature is computable at the close of bar `t` using only
bar `t` and earlier observations.

## Selection policy

The Episode 004 feature set is frozen before Episode 005 baseline comparison.

The first selection is structural, not performance-driven. Intermediate nested
return/volatility windows are omitted to reduce obvious overlap without looking
at validation results or the final test period.

Availability diagnostics describe the accepted history. Correlation/redundancy
diagnostics that can inform development use only the exact Episode 005 training
membership; the locked test partition is not part of that diagnostic scope.

## Recorded run

```bash
python experiments/ep004_feature_engineering/analyze_features.py \
  --input path/to/ep002-run/accepted.csv \
  --manifest path/to/ep002-run/manifest.json \
  --contract configs/features/ep004_baseline_features.toml \
  --experiment-contract configs/experiments/ep005_baselines.toml \
  --output reports/ep004/feature_report.json
```

The feature report verifies the accepted CSV against its EP002 manifest, records
descriptive availability, and reports train-only correlation/redundancy
diagnostics. The config freezes the feature set that Episode 005 consumes.
