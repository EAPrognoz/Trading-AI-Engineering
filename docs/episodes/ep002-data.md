# Episode 002 — Data pipeline

Published video: https://youtu.be/kltuqw7vKrY

Episode 002 starts with a deliberately simple fact:

```text
Python → local MetaTrader 5 terminal → copy_rates_range() → DataFrame
```

The first example is intentionally short. A new viewer should be able to see the
connection, run it, and understand what happened before any validation framework
is introduced.

The rest of the episode builds engineering around that simple fetch.

## Step 1 — Fetch H1 bars

The teaching example is:

`examples/ep002_mt5_minimal.py`

It initializes MT5, calls `copy_rates_range`, shuts MT5 down, and prints the
returned table.

## Step 2 — Make the request explicit

The full pipeline adds:

- one explicitly named broker symbol;
- H1 bars;
- timezone-aware UTC range boundaries aligned to an hour;
- an explicit cutoff;
- internal `[start, end)` range semantics;
- only bars whose nominal one-hour interval is complete by the cutoff.

## Step 3 — Validate without hiding failures

The pipeline checks valid/ordered/unique timestamps, finite fields, OHLC
consistency, requested coverage, and unresolved gaps.

Raw data is preserved before filtering.

## Step 4 — Save evidence

Every run saves request, raw response, validation report, and manifest.
An accepted run also saves `accepted.csv`; a rejected run saves rejection
evidence and no accepted dataset.

The important teaching order is therefore:

```text
first make it work
then make the assumptions explicit
then make failures visible
then make it reproducible
```

A passing report still does not prove that broker prices are true, reconstruct
all historical feed revisions, prevent future ML leakage, or imply profitability.
