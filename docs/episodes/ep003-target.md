# Episode 003 — Forecast horizon and target

The published Episode 003 focused on a central design question: how far ahead
should a trading model predict?

For the first baseline experiment, the repository freezes one concrete task:

- timeframe: H1;
- decision time: immediately after the close of completed bar `t`;
- horizon: exactly one hour;
- target: sign of `close[t+1] / close[t] - 1`;
- task: binary UP / DOWN classification;
- exact zero returns: reported, then excluded;
- non-consecutive H1 row pairs: reported as `GAP`, then excluded.

## Why this comes before the model

Changing the horizon changes the target and the statistical problem. The model
therefore does not define the prediction task; it receives a task that has
already been declared.

## Boundary rule

Using the next close to create the label is legitimate supervised-learning
construction only when the next observation is exactly one hour later for this
contract.

Each sample carries:

- `decision_timestamp` — when the feature vector is available;
- `target_timestamp` — when the future close required to realize the label occurs.

Weekend/session/missing-history gaps are not silently reinterpreted as a
one-hour forecast.

Episode 005 also uses both timestamps to purge samples whose labels would cross
a train/validation/test boundary.
