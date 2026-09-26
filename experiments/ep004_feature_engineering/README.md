# Episode 004 — Feature set

Published video: https://youtu.be/SNOTNSZoNQY

Episode 004 asks what an AI trading model should actually see. The repository
turns that into an explicit point-in-time candidate universe plus a frozen first
feature set.

## Candidate families

- returns at several backward-looking horizons;
- rolling volatility;
- current-bar range and body;
- relative tick volume;
- cyclical hour-of-day and day-of-week encodings.

Every feature is computable at the close of bar `t` using only bar `t` and
earlier observations.

## Selection policy

The Episode 004 feature set is frozen **before** Episode 005 baseline comparison.

The first selection is structural, not performance-driven. Intermediate nested
return/volatility windows are omitted to reduce obvious overlap without looking
at target accuracy, validation results, or the test period.

The feature report shows availability and correlation/redundancy diagnostics.
Those diagnostics document the feature universe; they do not turn the final test
partition into a selection tool.

## Run

```bash
python experiments/ep004_feature_engineering/analyze_features.py \
  --input path/to/ep002-run/accepted.csv \
  --contract configs/features/ep004_baseline_features.toml \
  --output reports/ep004/feature_report.json
```
