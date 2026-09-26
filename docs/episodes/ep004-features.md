# Episode 004 — Feature set

Published video: https://youtu.be/SNOTNSZoNQY

Episode 004 is the repository's **feature-set** stage: what information should
the trading model actually see?

The teaching order is:

```text
one market table
      ↓
a few understandable transformations
      ↓
candidate feature universe
      ↓
remove obvious redundancy
      ↓
freeze the first feature set
```

## Start here

```powershell
python examples/ep004_feature_set_minimal.py .local/ep002-mt5/accepted.csv
```

The first example shows only a few transparent transformations such as:

```python
df["return_1h"] = df["close"].pct_change()
df["rolling_vol_6h"] = df["return_1h"].rolling(6).std()
df["range_pct"] = (df["high"] - df["low"]) / df["close"]
```

These are model inputs, not trading signals.

## Full Episode 004 feature set

The full candidate universe includes:

- returns over several backward-looking horizons;
- rolling volatility;
- current-bar range and body;
- relative tick volume;
- cyclical hour-of-day and day-of-week encodings.

No full-pipeline feature may depend on the Episode 003 future target, a
centered/future rolling window, or observations after decision time.

Timestamp gaps start a new feature segment so rolling windows do not silently
bridge unresolved data boundaries.

Before Episode 005 baseline fitting, Episode 004 freezes:

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

The first exclusions are structural rather than performance-driven:
`return_3h`, `return_12h`, and `rolling_vol_12h` are omitted to reduce
obvious overlap without inspecting validation or test performance.

The minimal example teaches feature construction. The full implementation under
`experiments/ep004_feature_engineering/` creates the recorded feature report
and frozen feature contract consumed by Episode 005.
