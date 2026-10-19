# Read-only MT5 deal cost report

This example builds a period-based report from one connected MetaTrader 5 demo
account. It reads `history_deals_get`; it does not log in, submit orders, or
change terminal state. The public repository contains the code and synthetic
unit tests only. It does not include the saved demo export, account identifiers,
or report outputs.

## Setup

Use a Windows machine where MetaTrader 5 is installed and already connected to
the intended demo account. In PowerShell:

```powershell
python -m pip install MetaTrader5
$env:PYTHONPATH = "src"
python experiments/mt5_cost_report_r1/run_demo.py `
  --start 2026-10-17T00:00:00Z `
  --end 2026-10-18T00:00:00Z `
  --out .local/mt5-cost-report
```

The adapter checks that the connected account is in demo mode before requesting
history. To pin the expected account, add `--expected-login` and
`--expected-server`. These values are local command arguments; do not commit
them. The report requests all symbols so separate commission operations remain
visible.

Supply timezone-aware ISO-8601 timestamps. The reporting period is half-open:
`[start, end)`. The implementation applies that boundary locally after the
history request, so a deal exactly at `start` is included and one exactly at
`end` is excluded.

## Outputs

- `raw_deals.csv`: one row per returned deal, including ticket/order/position
  identifiers, epoch seconds and milliseconds, UTC timestamp, type, entry,
  symbol, and profit/commission/swap/fee. Every record has one partition label.
- `buy_sell_daily.csv`: UTC daily totals for BUY and SELL deals only. Each of
  the four monetary components stays separate; the subtotal adds their
  recorded signed values, including opening-deal commissions.
- `other_operations.csv`: deposits, withdrawals, separate commissions, and
  every other non-BUY/SELL operation for review. The report does not allocate
  these amounts to a symbol, strategy, or position.
- `reconciliation.json`: account and currency scope, interval, row counts,
  partitions, validation checks, and limitations.

`None` from `history_deals_get` is an error and includes `last_error()` in the
exception. An empty tuple is a valid empty response. Duplicate deal tickets,
missing or nonfinite money, timestamp disagreement, and scope mismatch fail
visibly. Swap signs are preserved. The report does not calculate floating P/L,
lifetime position return, or an account-wide net-profit figure.

## Export and timestamp review

For an independent check, select the same dates in the terminal's History tab
and export the deal list. Compare deal tickets and all four money components;
check `time` against `time_msc` at millisecond precision and document the
terminal's displayed timezone. The report uses UTC and the API's epoch fields;
do not claim Python/terminal integration success until a permitted terminal
export confirms the same records and timestamp interpretation.

The official API documents the period request and `None` on error. The deal
properties reference defines the deal ticket, order, time, type, entry,
position, symbol, profit, commission, swap, and fee fields:

- [Python `history_deals_get`](https://www.mql5.com/en/docs/python_metatrader5/mt5historydealsget_py)
- [MQL5 deal properties](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties)

## Tests

Run the repository's report tests with pytest:

```bash
python -m pytest -q tests/reports
```

The synthetic fixtures test accounting edge cases only; they are not market or
terminal results.
