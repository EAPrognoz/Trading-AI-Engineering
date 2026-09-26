"""Episode 005: create the simplest benchmark first.

Usage:
    python examples/ep005_baseline_minimal.py path/to/accepted.csv
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

y = target[target.isin(["UP", "DOWN"])].reset_index(drop=True)

train_end = int(len(y) * 0.60)
validation_end = int(len(y) * 0.80)

y_train = y.iloc[:train_end]
y_validation = y.iloc[train_end:validation_end]

majority_class = y_train.value_counts().idxmax()
prediction = pd.Series(majority_class, index=y_validation.index)
accuracy = (prediction == y_validation).mean()

print(f"Training majority class: {majority_class}")
print(f"Validation rows: {len(y_validation)}")
print(f"B0 majority-class accuracy: {accuracy:.3f}")
print()
print("Now a more complex model has something concrete to beat.")
