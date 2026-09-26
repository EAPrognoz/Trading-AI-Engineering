# Episode 003 — Forecast horizon / target analysis

This experiment codifies the prediction problem used as the foundation for the
Episode 005 baseline work.

## Contract

At the close of completed H1 bar `t`, predict the sign of the next close-to-close
return:

```text
r[t+1] = close[t+1] / close[t] - 1

UP    when r[t+1] > 0
DOWN  when r[t+1] < 0
ZERO  when r[t+1] = 0
```

ZERO observations are reported but excluded from the binary classification task.
The final row is unlabeled because `close[t+1]` is not present.

Future information is allowed to construct the target. It is not allowed in the
feature vector available at the decision timestamp.

## Run

After installing the package in editable mode:

```bash
python experiments/ep003_forecast_horizon/analyze_target.py \
  --input path/to/validated_h1_snapshot.csv \
  --output reports/ep003/target_report.json
```

The report records dataset identity, timestamp coverage, class balance, zero
returns, and the forward-return distribution.
