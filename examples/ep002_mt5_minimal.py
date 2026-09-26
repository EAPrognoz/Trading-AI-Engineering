"""Episode 002: the simplest useful MetaTrader 5 fetch.

Run this on Windows with MetaTrader 5 open and logged in:

    pip install MetaTrader5 pandas
    python examples/ep002_mt5_minimal.py
"""

from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

symbol = "EURUSD"
start = datetime(2026, 1, 5, tzinfo=timezone.utc)
end = datetime(2026, 1, 9, tzinfo=timezone.utc)

if not mt5.initialize():
    raise RuntimeError(mt5.last_error())

rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start, end)
mt5.shutdown()

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

print(df.head())
print(f"Rows: {len(df)}")
