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
returned table. The H1 `timestamp` returned by this layer is interpreted as the
bar-opening time in UTC.

## Step 2 — Make the request explicit

The full pipeline adds:

- one explicitly named broker symbol;
- H1 bars;
- timezone-aware UTC range boundaries aligned to an hour;
- an explicit cutoff;
- internal `[start, end)` range semantics;
- only bars whose nominal one-hour interval is complete by the cutoff.

## Step 3 — Validate without hiding failures

The pipeline checks parseable UTC-hour-aligned H1 timestamps, exact requested
boundary coverage, a non-empty eligible range, ordering/uniqueness, finite
fields, OHLC consistency, and unresolved gaps.

Raw data is preserved before filtering.

## Step 4 — Save evidence

Every run saves request, raw response, validation report, and manifest.
An accepted run also saves `accepted.csv`; a rejected run saves rejection
evidence and no accepted dataset. The accepted-dataset hash in the manifest is
the provenance anchor verified by the recorded downstream experiments.

The important teaching order is therefore:

```text
first make it work
then make the assumptions explicit
then make failures visible
then make it reproducible
```

A passing report still does not prove that broker prices are true, reconstruct
all historical feed revisions, prevent future ML leakage, or imply profitability.

## Separate BTC multi-timeframe acquisition

The H1 teaching example and existing accepted snapshot describe the EURUSD
learning path. The BTC experiment uses the historical-only
`experiments/ep002_market_data/run_mt5_bundle.py` with an exact BTC broker symbol
verified in the connected MT5 terminal. It does not infer a symbol from the
asset name. Its required arguments are `--symbol`, `--analysis-start`,
`--analysis-end`, `--cutoff`, `--feature-contract`, and a new `--run-dir` under
`.local/`; the EP002 full-run guide shows the complete command.

The half-open UTC decision interval is `[analysis-start, analysis-end)`. The
runner reads `max_lookback_bars` separately from each H1/H4/D1 table in
`configs/features/ep004_btc_mtf_features.toml` (currently 24 native bars each)
and requests `(N + 2) * native duration` of pre-roll per stream. It requires at
least `N + 1` completed bars before the first decision and recent completed
coverage at both ends of the common interval. MT5 timestamps are UTC bar opens;
nominal close is open plus that stream's native duration. D1 opens need not be
at 00:00 UTC. Future or partial bars are not accepted as known at an H1 decision.

Each timeframe rejects unresolved native-cadence gaps independently. Successful
acquisition writes `H1/`, `H4/`, and `D1/` accepted files and source manifests,
then `bundle_manifest.json` with exact symbol, dataset identities, and SHA-256
hashes. Downstream readers verify those members before reporting. Keep raw and
acquired BTC data in ignored `.local/`; do not commit broker data. The verified
EP005 source was `BITCOIN_i` broker-CFD history on H1/H4/D1. Its validation
result was published; raw broker files remain local and are not included here.
