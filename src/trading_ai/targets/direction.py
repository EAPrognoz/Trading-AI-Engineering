"""Episode 003 target: direction of the next H1 close-to-close return."""

from __future__ import annotations

import pandas as pd


def build_h1_direction_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct the one-hour-ahead direction label.

    The label is allowed to use close[t+1] because it is the supervised target.
    This function does not create model features.

    Returns a frame aligned to the input rows with:
      - decision_timestamp: timestamp of the completed bar t;
      - target_timestamp: timestamp of bar t+1 used to realize the label;
      - future_return_1h: close[t+1] / close[t] - 1;
      - target_h1_direction: UP, DOWN, ZERO, or <NA> for the final row.
    """
    required = {"timestamp", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)

    target_timestamp = timestamp.shift(-1)
    future_return = close.shift(-1).div(close).sub(1.0)
    target = pd.Series(pd.NA, index=frame.index, dtype="string")
    target.loc[future_return > 0] = "UP"
    target.loc[future_return < 0] = "DOWN"
    target.loc[future_return == 0] = "ZERO"

    return pd.DataFrame(
        {
            "decision_timestamp": timestamp,
            "target_timestamp": target_timestamp,
            "future_return_1h": future_return,
            "target_h1_direction": target,
        },
        index=frame.index,
    )
