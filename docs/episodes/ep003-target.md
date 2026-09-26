# Episode 003 — Target

Published video: https://youtu.be/VfnH1q76fj4

Episode 003 is the repository's **target** stage.

The episode's central engineering question is the forecast horizon: changing how
far ahead the model predicts changes the target, noise, market dynamics,
execution assumptions, and the meaning of a useful forecast.

For the current downstream experiment the target contract is:

- source timeframe: H1;
- decision time: after the close of completed bar `t`;
- forecast horizon: exactly one hour;
- target: sign of `close[t+1] / close[t] - 1`;
- binary classes: UP / DOWN;
- exact zero returns: reported then excluded;
- non-consecutive H1 row pairs: reported as `GAP` then excluded.

## Boundary rule

Using the next close to construct the supervised label is legitimate. Giving the
same future close, or a transformation depending on it, to the model at decision
time is leakage.

Each sample therefore carries:

- `decision_timestamp` — when model inputs are available;
- `target_timestamp` — when the future price required to realize the label occurs.

The target is defined before the baseline/model is chosen.
