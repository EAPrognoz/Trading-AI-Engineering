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

The selected list remains frozen. The feature report describes availability over
the accepted history, but its correlation/redundancy diagnostics use the exact
Episode 005 training membership only. The locked test partition is not used to
make feature decisions.

The minimal example teaches feature construction. The full implementation under
`experiments/ep004_feature_engineering/` creates the recorded feature report
and frozen feature contract consumed by Episode 005.

## Separate BTC H1/H4/D1 feature set

The listed H1 features and teaching example above belong to the existing
EURUSD route. The BTC contract is
`configs/features/ep004_btc_mtf_features.toml`, with selected features
namespaced by H1, H4, or D1. A 24-bar H4 or D1 lookback counts that many
native source bars; it does not change EP003's one-hour forecast horizon.

MT5 source timestamps are UTC bar opens. EP004 computes each source's nominal
close as open plus native duration and selects only a completed source row with
`nominal_close <= H1 decision_timestamp`. D1 opens need not be UTC midnight.
No partial/future bar or rolling window across an unresolved native gap is used.

The BTC feature report reads the verified EP002 `bundle_manifest.json` and
checks that its feature contract matches the one named by the canonical BTC
EP005 experiment config. Its availability statistics describe the common
interval, while correlation/redundancy diagnostics use exactly EP005's prepared
train decision timestamps. Validation and locked test decisions are excluded
from those selection diagnostics. The selected H1/H4/D1 feature contract fed the
published EP005 BTC baseline on `BITCOIN_i`. Validation was 11/27 for Logistic
Regression and 13/27 for Always-Up; the reserved test remains unevaluated.
