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

## Recorded run

```bash
python experiments/ep003_forecast_horizon/analyze_target.py \
  --input path/to/ep002-run/accepted.csv \
  --manifest path/to/ep002-run/manifest.json \
  --output reports/ep003/target_report.json
```

The report verifies the accepted CSV against its EP002 manifest and records
dataset identity, timestamp coverage, class balance, zero/gap counts, and the
forward-return distribution.
