# Episode 004 — Feature set

Published video: https://youtu.be/SNOTNSZoNQY

Episode 004 is the repository's **feature-set** stage: what information should
the trading model actually see?

The implementation follows the episode's central rule: more inputs do not
automatically create more information. Candidate features are explicit,
point-in-time, and inspectable before the baseline comparison.

## Candidate families

The first H1 feature universe contains:

- returns over several backward-looking horizons;
- rolling volatility;
- current-bar range and body;
- relative tick volume;
- cyclical hour-of-day and day-of-week encodings.

These represent price/returns, volatility, volume, and derived-feature families.
Additional indicators can be added as explicit candidates; being available does
not automatically earn a feature a place in the frozen set.

## Point-in-time rule

No feature may depend on the Episode 003 future target, a centered/future rolling
window, or any observation after decision time.

Timestamp gaps start a new feature segment so rolling windows do not silently
bridge unresolved missing-history/session boundaries inherited from the data
layer.

## Frozen first feature set

Before Episode 005 baseline fitting, the repository freezes:

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
obvious overlap. This decision does not inspect target accuracy, validation
performance, or the final test partition.

Feature diagnostics report missing/non-finite values, first availability,
cardinality, and correlation/redundancy information. The frozen feature set is
the output that Episode 005 consumes.
