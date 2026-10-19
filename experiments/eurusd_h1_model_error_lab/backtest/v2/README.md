# Public EURUSD quote replay interface

This code replays already-frozen `long`, `short`, and `ABSTAIN` decisions
against saved EURUSD bid/ask quote rows. It does not fit or update a model,
connect to MetaTrader, read account history, or send orders. Same-side signals
are no-ops and keep the original one-hour target. `ABSTAIN` holds an open
position to that target. Those policies are covered by synthetic unit tests.

The fixed illustrative scenario uses a USD 10,000 starting balance, one
100,000 EUR lot, and hypothetical 1:200 leverage. This does not model actual
broker margin rules, historical swaps, fees, or slippage. It is a quote replay,
not a profitability test or an account-wide return.

## Install and run

Python 3.11+; the replay engine uses only the standard library.

```bash
python experiments/eurusd_h1_model_error_lab/backtest/v2/replay_csv.py \
  --decisions ./decisions.csv \
  --ticks ./ticks.csv \
  --start 2026-07-17T00:00:00Z \
  --end 2026-10-17T00:00:00Z \
  --out ./replay-output
```

The output directory is create-only. The run writes `positions.csv`,
`signal_actions.csv`, `balance_curve.csv`, `equity_curve.csv`, `metrics.json`,
`source_audit.json`, and a `run_manifest.json` with input/output SHA-256 hashes.
The CSV runner checks timestamp order, exact source row order, finite positive
bid/ask values, and the half-open `[start, end)` period.

## CSV schemas

`decisions.csv` must contain:

| Column | Meaning |
|---|---|
| `signal_id` | Unique opaque row identifier; do not put account identifiers here. |
| `decision_time_utc` | ISO-8601 timestamp with `Z` or `+00:00` UTC offset. |
| `action` | Exactly `long`, `short`, or `ABSTAIN`. |

Do not include target labels or any future information. Create this file from
decisions that were available at the stated time.

`ticks.csv` must contain `time_msc,bid,ask`. Optional columns are `flags` and
`source_row_seq`. `time_msc` is Unix epoch milliseconds. If `source_row_seq` is
omitted, CSV row order becomes the sequence key. Equal-millisecond rows are
allowed and must remain in their original order. Prices must be finite and
`0 < bid <= ask`.

## Exporting your own quotes from MT5

Use the included read-only helper on a machine with an already-connected MT5
terminal:

```powershell
python -m pip install MetaTrader5
python examples/eurusd_v2_export_ticks.py `
  --symbol EURUSD `
  --start 2026-07-17T00:00:00Z `
  --end 2026-10-17T00:00:00Z `
  --out .local/eurusd-ticks.csv
```

The helper calls `copy_ticks_range` with `COPY_TICKS_INFO`, does not call
`login`, does not query account history or account identifiers, and does not
submit orders. It distinguishes `None` (terminal error, including
`last_error()`) from a valid empty response. It applies the half-open end rule
locally, retains duplicate timestamps, source order, and tick flags. Keep its
source manifest beside the CSV. Check that all required bid/ask values are
present before using the replay.

MT5 expects UTC datetimes for range exports; see the official
[`copy_ticks_range` reference](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksrange_py).

## Results and limitations

The repository's `public/proof_summary.json` is an aggregate summary
of one private historical demo quote replay. Raw ticks, model artifacts,
predictions, and account identifiers are not published. Those aggregate values
cannot be reproduced without the same authorized private inputs. Synthetic
fixtures in tests prove software behavior only and are never market results.

The same-side and `ABSTAIN` policies, duplicate/ordering checks, boundary
handling, and finite quote validation are exercised in:
`tests/experiments/test_v2_policy_semantics.py` and
`tests/experiments/test_v2_replay_csv.py`.
