# Episode 002 — Market-data boundary

Episode 002 established the data layer that later model experiments depend on.

The repository version deliberately separates two concerns:

1. obtaining broker history; and
2. validating and identifying the exact snapshot used by an experiment.

This first repository milestone implements the second concern. A validated H1
snapshot must have explicit UTC timestamps, unique ordered periods, finite OHLC
values, internally consistent highs/lows, and non-negative volume/spread fields.

The repository does **not** commit large or redistributable broker datasets.
Instead, later experiment reports record a dataset hash and timestamp coverage.

This is a retrospective codification of the Episode 002 engineering contract,
not a claim that this exact source tree was present when the video was published.
