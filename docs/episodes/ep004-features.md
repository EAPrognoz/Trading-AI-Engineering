# Episode 004 — Point-in-time features

The published Episode 004 focused on model inputs and the fact that more features
do not automatically create a better model.

The repository implements that idea in two separate steps.

## 1. Feature engineering

Candidate features are computed from data available no later than the close of
the current completed H1 bar.

No feature may depend on `close[t+1]`, the Episode 003 target, a centered
rolling window, or any other future observation.

## 2. Feature selection

The first baseline feature set is deliberately frozen before the Episode 005
baseline comparison.

Baseline v1 keeps a sparse set of return horizons and volatility windows while
omitting some nested intermediate windows. This is a structural choice intended
to limit obvious overlap. It is not selected by maximizing accuracy, validation
performance, or backtest returns.

Data-driven feature selection, if introduced later, must occur inside an
explicit training/evaluation protocol and may not inspect the final test period.

## Baseline-v1 selected inputs

- return_1h
- return_6h
- return_24h
- rolling_vol_6h
- rolling_vol_24h
- range_pct
- body_return
- relative_tick_volume_24h
- hour_sin / hour_cos
- dow_sin / dow_cos
