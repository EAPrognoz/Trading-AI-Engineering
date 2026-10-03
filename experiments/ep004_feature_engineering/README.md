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

## Existing EURUSD H1 recorded run

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

## Separate BTC H1/H4/D1 feature report

After EP002 accepts its three native streams, use the same verified bundle and
the feature contract named by the canonical BTC EP005 experiment config:

```powershell
$runDir = '.local/btc-mtf-<unique-run-id>'
python experiments/ep004_feature_engineering/analyze_features.py `
  --bundle-manifest "$runDir/bundle_manifest.json" `
  --contract configs/features/ep004_btc_mtf_features.toml `
  --experiment-contract configs/experiments/ep005_btc_mtf_baselines.toml `
  --output "$runDir/reports/ep004/feature_report.json"
```

The report loader verifies bundle identities and SHA-256 hashes. Feature names
identify H1, H4, or D1; lookbacks count native bars on each source, distinct
from EP003's fixed one-hour target. MT5 UTC timestamps are bar opens, and a
source's nominal close is open plus its native duration. The strict as-of rule
uses only `nominal_close <= H1 decision_timestamp`; D1 need not open at UTC
midnight. Partial/future bars and rolling windows across unresolved native gaps
are excluded.

Availability diagnostics describe the bundle's common interval. Descriptive
correlations use only the exact train decision timestamps returned by canonical
EP005 split preparation. Validation and locked test decisions cannot enter
correlation or selection diagnostics. The selected set is declared in the
contract, whose SHA-256 must match the feature contract named by the supplied
EP005 config. This contract was used for the published BTC EP005 baseline on
`BITCOIN_i`; the reserved test remained unevaluated. Keep broker history local.
