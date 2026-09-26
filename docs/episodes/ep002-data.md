# Episode 002 — Data pipeline

Published video: https://youtu.be/kltuqw7vKrY

Episode 002 is the repository's **data** stage. Its output is not merely a CSV
that happens to load. It is an observable market-data run with explicit time
semantics and preserved evidence.

## Contract

The narrow first implementation uses:

- one explicitly named broker symbol;
- H1 bars;
- timezone-aware UTC range boundaries aligned to an hour;
- an explicit cutoff;
- internal `[start, end)` range semantics;
- only bars whose nominal one-hour interval is complete by the cutoff.

A MetaTrader 5 source adapter fetches records and provenance but does not place
orders. The raw response is preserved before the repository applies its own
range/cutoff policy.

## Validation

The pipeline checks:

- valid, ordered, unique timestamps;
- finite OHLC/volume/spread fields;
- OHLC internal consistency;
- non-negative volume/spread fields;
- requested coverage;
- unresolved gaps.

A gap larger than one hour is a question, not an invented price. Without a
session calendar it is reported as an unclassified gap and the accepted dataset
is withheld pending review.

## Accepted vs rejected runs

Every run saves:

- `request.json`;
- `raw_response.csv`;
- `validation_report.json`;
- `manifest.json`.

Accepted runs additionally save `accepted.csv`.

Rejected runs save `rejection_report.json` and do not write an accepted
dataset. An existing run directory is never silently overwritten.

The manifest records request semantics, retrieval provenance, environment, code
version when available, and hashes of the stored artifacts.

## Scope boundary

A passing validation report means the declared checks passed. It does not prove
that broker prices are true, reconstruct every historical feed revision, prevent
future ML leakage, or imply profitability. Source validation comes first;
experiment design comes later.
