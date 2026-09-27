"""Episode 003 target: direction of the next exact H1 close-to-close return."""

from __future__ import annotations

import pandas as pd


def build_h1_direction_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct the exact one-hour-ahead direction label.

    The label is allowed to use close[t+1] because it is the supervised target.
    A row pair is eligible only when the next timestamp is exactly one hour
    later. Non-consecutive pairs are marked GAP and excluded by the binary
    experiment assembly.
    """
    required = {"timestamp", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)

    next_open_timestamp = timestamp.shift(-1)
    exact_h1 = next_open_timestamp.sub(timestamp).eq(pd.Timedelta(hours=1))
    future_return = close.shift(-1).div(close).sub(1.0).where(exact_h1)

    decision_timestamp = timestamp.add(pd.Timedelta(hours=1))
    target_timestamp = next_open_timestamp.add(pd.Timedelta(hours=1))

    target = pd.Series(pd.NA, index=frame.index, dtype="string")
    gap_mask = next_open_timestamp.notna() & ~exact_h1
    target.loc[gap_mask] = "GAP"
    target.loc[exact_h1 & (future_return > 0)] = "UP"
    target.loc[exact_h1 & (future_return < 0)] = "DOWN"
    target.loc[exact_h1 & (future_return == 0)] = "ZERO"

    return pd.DataFrame(
        {
            "decision_timestamp": decision_timestamp,
            "target_timestamp": target_timestamp,
            "future_return_1h": future_return,
            "target_h1_direction": target,
        },
        index=frame.index,
    )
