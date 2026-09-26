# Episode 003 — Target

Published video: https://youtu.be/VfnH1q76fj4

Episode 003 is the repository's **target** stage.

The teaching order is:

```text
calculate the future return
        ↓
turn it into a target
        ↓
make the forecast horizon explicit
        ↓
make time/leakage boundaries explicit
```

## Start here

Run the minimal example on an accepted Episode 002 snapshot:

```powershell
python examples/ep003_target_minimal.py .local/ep002-mt5/accepted.csv
```

The core idea is simply:

```python
future_return = close.shift(-1) / close - 1
```

and then classify the result as UP or DOWN.

## Full Episode 003 contract

The published episode's central engineering question is forecast horizon:
changing how far ahead the model predicts changes the target, noise, market
dynamics, execution assumptions, and the meaning of a useful forecast.

For the current downstream experiment:

- source timeframe: H1;
- decision time: after the close of completed bar `t`;
- forecast horizon: exactly one hour;
- target: sign of `close[t+1] / close[t] - 1`;
- binary classes: UP / DOWN;
- exact zero returns: reported then excluded;
- non-consecutive H1 row pairs: reported as `GAP` then excluded.

Using the next close to construct the supervised label is legitimate. Giving the
same future close, or a transformation depending on it, to the model at decision
time is leakage.

Each full-pipeline sample carries:

- `decision_timestamp`;
- `target_timestamp`.

The target is defined before the feature set, baseline, or model is chosen.

The minimal example teaches the target. The executable analysis under
`experiments/ep003_forecast_horizon/` produces the recorded report.
