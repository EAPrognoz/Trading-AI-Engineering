# Episode 004 — Feature engineering and selection

Episode 004 asked what an AI trading model should actually see. This repository
milestone turns that question into a point-in-time feature contract.

## Candidate families

- returns at several backward-looking horizons;
- realized rolling volatility;
- current-bar range and body;
- relative tick volume;
- cyclical hour-of-day and day-of-week encodings.

Every feature is computable at the close of bar `t` using only bar `t` and
earlier observations.

## Selection policy

The baseline-v1 feature set is frozen **before** Episode 005 model comparison.

The first selection is structural, not performance-driven. Intermediate nested
return/volatility windows are omitted to reduce overlapping inputs without
looking at target accuracy, validation results, or the test period.

The feature report may show correlations as diagnostics. Those values do not
change the predeclared baseline-v1 list.

## Run

```bash
python experiments/ep004_feature_engineering/analyze_features.py \
  --input path/to/validated_h1_snapshot.csv \
  --contract configs/features/ep004_baseline_features.toml \
  --output reports/ep004/feature_report.json
```
