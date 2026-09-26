"""Episode 004: turn market bars into model inputs.

Usage:
    python examples/ep004_feature_set_minimal.py path/to/accepted.csv
"""

from pathlib import Path
import sys

import pandas as pd

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".local/ep002-mt5/accepted.csv")
df = pd.read_csv(path, parse_dates=["timestamp"])

df["return_1h"] = df["close"].pct_change()
df["return_6h"] = df["close"].pct_change(6)
df["rolling_vol_6h"] = df["return_1h"].rolling(6).std(ddof=0)
df["range_pct"] = (df["high"] - df["low"]) / df["close"]
df["body_return"] = df["close"] / df["open"] - 1.0
df["relative_tick_volume_24h"] = (
    df["tick_volume"] / df["tick_volume"].rolling(24).mean() - 1.0
)

features = [
    "return_1h",
    "return_6h",
    "rolling_vol_6h",
    "range_pct",
    "body_return",
    "relative_tick_volume_24h",
]

print(df[["timestamp", *features]].tail(12).to_string(index=False))
print()
print("These are model inputs, not trading signals.")
print("The full frozen Episode 004 set is in configs/features/ep004_baseline_features.toml")
