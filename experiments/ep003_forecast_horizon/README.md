# Episode 003 — Target / forecast horizon

Published video: https://youtu.be/VfnH1q76fj4

Episode 003 defines the prediction problem before any baseline or candidate
model is chosen.

The current repository target is:

```text
decision time: close of completed H1 bar t
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

## Run

```bash
python experiments/ep003_forecast_horizon/analyze_target.py \
  --input path/to/ep002-run/accepted.csv \
  --output reports/ep003/target_report.json
```

The report records dataset identity, timestamp coverage, class balance, zero/gap
counts, and the forward-return distribution.
