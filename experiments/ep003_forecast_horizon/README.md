# Episode 003 — Target / forecast horizon

Published video: https://youtu.be/VfnH1q76fj4

If you are following the series for the first time, start with:

```powershell
python examples/ep003_target_minimal.py path/to/ep002-run/accepted.csv
```

That short script teaches the target itself.

This directory contains the **full recorded analysis** used downstream.

## Target contract

```text
source timestamp: UTC opening time of H1 bar t
decision time: nominal close = source opening + 1 hour
horizon: exactly 1 hour
r[t+1] = close[t+1] / close[t] - 1

UP    when r[t+1] > 0
DOWN  when r[t+1] < 0
ZERO  when r[t+1] = 0
GAP   when the next row is not exactly one hour later
```

ZERO and GAP observations are reported and excluded from the downstream binary
classification task. The final row is unlabeled.

Future information is allowed to construct the supervised target. It is not
allowed in the feature vector available at the decision timestamp.

## Existing EURUSD H1 recorded run

```bash
python experiments/ep003_forecast_horizon/analyze_target.py \
  --input path/to/ep002-run/accepted.csv \
  --manifest path/to/ep002-run/manifest.json \
  --output reports/ep003/target_report.json
```

The report verifies the accepted CSV against its EP002 manifest and records
dataset identity, timestamp coverage, class balance, zero/gap counts, and the
forward-return distribution.

## Separate BTC verified-bundle report

After EP002 accepts all three streams, use its verified bundle manifest. This
command is a template for a unique local BTC run directory:

```powershell
$runDir = '.local/btc-mtf-<unique-run-id>'
python experiments/ep003_forecast_horizon/analyze_target.py `
  --bundle-manifest "$runDir/bundle_manifest.json" `
  --output "$runDir/reports/ep003/target_report.json"
```

The loader checks the H1/H4/D1 member identities and SHA-256 hashes before
reporting. EP003 calculates labels from H1 alone within the bundle's common
half-open decision interval. The next consecutive H1 close is exactly one hour
after the decision close; H4/D1 are model inputs, not longer target horizons.
Missing H1 adjacency is recorded as `GAP`, never bridged. The report is local
and supports the verified `BITCOIN_i` bundle used by published EP005. The
27-row BTC validation comparison scored 11/27 for Logistic Regression and 13/27
for Always-Up; no reserved-test metric was computed.
