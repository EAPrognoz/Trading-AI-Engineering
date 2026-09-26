"""Episode 003: define the target before choosing a model.

Usage:
    python examples/ep003_target_minimal.py path/to/accepted.csv
"""

from pathlib import Path
import sys

import pandas as pd

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".local/ep002-mt5/accepted.csv")
df = pd.read_csv(path, parse_dates=["timestamp"])

next_time = df["timestamp"].shift(-1)
future_return = df["close"].shift(-1) / df["close"] - 1.0
is_exactly_one_hour = (next_time - df["timestamp"]) == pd.Timedelta(hours=1)

target = pd.Series(pd.NA, index=df.index, dtype="string")
target.loc[is_exactly_one_hour & (future_return > 0)] = "UP"
target.loc[is_exactly_one_hour & (future_return < 0)] = "DOWN"
target.loc[is_exactly_one_hour & (future_return == 0)] = "ZERO"
target.loc[next_time.notna() & ~is_exactly_one_hour] = "GAP"

result = pd.DataFrame(
    {
        "decision_timestamp": df["timestamp"],
        "target_timestamp": next_time,
        "future_return_1h": future_return.where(is_exactly_one_hour),
        "target": target,
    }
)

print(result.tail(12).to_string(index=False))
print()
print(result["target"].value_counts(dropna=False))
