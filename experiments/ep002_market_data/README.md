# Episode 002 — Python + MetaTrader 5 market-data pipeline

Published video: https://youtu.be/kltuqw7vKrY

## Start here: getting bars from MT5 is simple

Before looking at validation code, prove the connection works.

On Windows, open MetaTrader 5, log in to a demo account, then:

```powershell
pip install MetaTrader5 pandas
python examples/ep002_mt5_minimal.py
```

The core idea is only this:

```python
import MetaTrader5 as mt5
import pandas as pd

mt5.initialize()

rates = mt5.copy_rates_range(
    "EURUSD",
    mt5.TIMEFRAME_H1,
    start,
    end,
)

mt5.shutdown()

df = pd.DataFrame(rates)
```

That is the teaching point: **Python asks the local MT5 terminal for H1 bars and
gets a table back.**

Everything else in Episode 002 answers a different question:

> Can we trust this table enough to use it in the next experiment?

## Then add the engineering layer

The full Episode 002 pipeline preserves:

- the explicit request;
- the raw source response;
- a validation report;
- a manifest with provenance, environment, code version, and file hashes.

An accepted run additionally writes `accepted.csv`.

A rejected run writes `rejection_report.json` and does **not** write an
accepted dataset.

## Full MT5 pipeline

Once the minimal example works:

```powershell
python experiments/ep002_market_data/run_mt5.py ^
  --symbol EURUSD ^
  --start 2026-01-05T08:00:00Z ^
  --end 2026-01-05T12:00:00Z ^
  --cutoff 2026-01-05T12:00:00Z ^
  --run-dir .local/ep002-mt5
```

The MT5 fetch itself remains small. Request semantics, validation, range/cutoff
rules, manifests, and rejection handling are kept in separate modules so the
viewer can learn one idea at a time.

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

The fixture path lets viewers test the pipeline without a broker login:

```powershell
python experiments/ep002_market_data/run_fixture.py ^
  --input fixtures/ep002/good_h1.csv ^
  --symbol EURUSD ^
  --start 2026-01-05T08:00:00Z ^
  --end 2026-01-05T12:00:00Z ^
  --cutoff 2026-01-05T12:00:00Z ^
  --run-dir .local/ep002-good
```

The conflict fixture is intentionally rejected. It exists to show that a bad
input cannot quietly look like a successful export.
