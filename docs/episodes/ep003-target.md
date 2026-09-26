# Episode 003 — Forecast horizon and target

The published Episode 003 focused on a central design question: how far ahead
should a trading model predict?

For the first baseline experiment, the repository freezes one concrete task:

- timeframe: H1;
- decision time: immediately after the close of completed bar `t`;
- horizon: one bar / one hour;
- target: sign of `close[t+1] / close[t] - 1`;
- task: binary UP / DOWN classification;
- exact zero returns: reported, then excluded from the binary task.

## Why this comes before the model

Changing the horizon changes the target and the statistical problem. The model
therefore does not define the prediction task; it receives a task that has
already been declared.

## Boundary rule

Using `close[t+1]` to create the label is legitimate supervised-learning
construction. Using `close[t+1]`, or any transformation that depends on it, in
the feature vector at decision time `t` is leakage.

The Episode 004 feature layer is responsible for enforcing the input side of
that boundary.
