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

## Existing EURUSD H1 full MT5 pipeline

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

- MT5 H1 timestamps are bar-opening timestamps interpreted in UTC;
- requested range is `[start, end)`;
- only bars whose nominal H1 interval has completed by the cutoff are eligible;
- the source response is preserved before filtering;
- the required start and latest completed H1 boundaries must actually be present;
- an empty eligible range, malformed timestamp, or non-hour-aligned H1 timestamp
  prevents acceptance;
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

For recorded downstream EP003–EP005 runs, keep both `accepted.csv` and the
originating `manifest.json`; the later CLIs verify the accepted CSV hash against
that manifest.

## Separate BTC H1/H4/D1 historical bundle

Before acquisition, inspect the connected MT5 terminal and verify the exact
broker BTC symbol and availability of native H1, H4, and D1 history. Do not
assume an alias such as `BTCUSD` is present. The command below is a template:
replace the symbol and UTC timestamps only after that check, and choose a new
run directory under this repository's ignored `.local/` for each attempt.

```powershell
$btcSymbol = '<exact verified broker symbol>'
$analysisStart = '<UTC hour-aligned ISO 8601 decision start>'
$analysisEnd = '<UTC hour-aligned ISO 8601 decision end>'
$cutoff = '<UTC ISO 8601 cutoff at or after analysis end>'
$runDir = '.local/btc-mtf-<unique-run-id>'
python experiments/ep002_market_data/run_mt5_bundle.py `
  --symbol $btcSymbol `
  --analysis-start $analysisStart `
  --analysis-end $analysisEnd `
  --cutoff $cutoff `
  --feature-contract configs/features/ep004_btc_mtf_features.toml `
  --run-dir $runDir
```

This runner requests historical bars only. The analysis interval is the
half-open H1 decision range `[analysis-start, analysis-end)`, with a cutoff at
or after its end. The feature contract currently declares
`max_lookback_bars = 24` separately for H1, H4, and D1. For each stream the
runner requests
`(max_lookback_bars + 2) * native duration` of pre-roll and checks at least
`N + 1` completed bars before the first decision and recent completed-bar
coverage at both interval endpoints. H1/H4/D1 are input series; these native
bar counts are not EP003's one-hour forecast horizon.

MT5 timestamps are UTC bar opens. Nominal close is open plus native timeframe
duration; D1 bars need not open at 00:00 UTC. Only bars with nominal close no
later than the H1 decision may supply features. Partial/future bars are not
eligible. Each stream rejects unresolved gaps against its own native cadence;
no missing history is bridged.

On success, `$runDir` contains `H1/`, `H4/`, and `D1/` directories, each with
raw response, validation evidence, `accepted.csv`, and `manifest.json`, plus a
top-level `bundle_manifest.json`. The bundle binds exact symbol, timeframe,
dataset ID, source-manifest hash, and accepted-CSV SHA-256 for every member.
Downstream reports load and verify that bundle. A rejected stream leaves no
bundle manifest. Keep all raw/acquired BTC data under `.local/` and out of Git.
The verified BTC symbol is `BITCOIN_i`, a broker CFD, with accepted H1/H4/D1
history. The published EP005 BTC comparison reports validation only; keep the
source CSVs and raw broker records under `.local/` and out of Git.
