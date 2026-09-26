# Episode 002 — Python + MetaTrader 5 market-data pipeline

Published video: https://youtu.be/kltuqw7vKrY

This directory mirrors the engineering contract shown in Episode 002.

## Output of a run

Every run preserves:

- the explicit request;
- the raw source response;
- a validation report;
- a manifest with provenance, environment, code version, and file hashes.

An accepted run additionally writes `accepted.csv`.

A rejected run writes `rejection_report.json` and **does not** write an accepted dataset.

## Time policy

- H1 opening timestamps are interpreted in UTC;
- requested range is `[start, end)`;
- only bars whose nominal H1 interval has completed by the cutoff are eligible;
- the source response is preserved before filtering;
- shorter-than-requested coverage is visible and prevents acceptance.

## Gaps

Without a broker-session calendar the pipeline does not guess whether a gap is a
weekend, scheduled break, or missing history. A gap larger than one hour is
reported as `unclassified_gap` and the dataset is withheld pending review.

## Fixture demo

Good fixture:

```bash
python experiments/ep002_market_data/run_fixture.py \
  --input fixtures/ep002/good_h1.csv \
  --symbol EURUSD \
  --start 2026-01-05T08:00:00Z \
  --end 2026-01-05T12:00:00Z \
  --cutoff 2026-01-05T12:00:00Z \
  --run-dir .local/ep002-good
```

Conflict fixture:

```bash
python experiments/ep002_market_data/run_fixture.py \
  --input fixtures/ep002/conflicting_h1.csv \
  --symbol EURUSD \
  --start 2026-01-05T08:00:00Z \
  --end 2026-01-05T12:00:00Z \
  --cutoff 2026-01-05T12:00:00Z \
  --run-dir .local/ep002-conflict
```

The conflict run is expected to be rejected. Do not change the fixture merely to
make its status green.

## Real terminal

`run_mt5.py` uses the optional local MetaTrader5 Python package. Check the
actual broker symbol name and run this first on a demo account. A successful
fixture run is not evidence that a broker terminal integration was tested.
