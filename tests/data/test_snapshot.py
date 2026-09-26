from __future__ import annotations

import pandas as pd
import pytest

from trading_ai.data.snapshot import validate_h1_snapshot


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "open": [1.0, 1.1, 1.2, 1.15],
            "high": [1.2, 1.3, 1.25, 1.2],
            "low": [0.9, 1.0, 1.1, 1.1],
            "close": [1.1, 1.2, 1.15, 1.18],
            "tick_volume": [100, 120, 110, 130],
            "spread": [10, 10, 11, 9],
            "real_volume": [0, 0, 0, 0],
        }
    )


def test_valid_snapshot_passes() -> None:
    validate_h1_snapshot(_frame())


def test_duplicate_timestamp_is_rejected() -> None:
    frame = _frame()
    frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"]
    with pytest.raises(ValueError, match="duplicates"):
        validate_h1_snapshot(frame)


def test_contradictory_high_is_rejected() -> None:
    frame = _frame()
    frame.loc[1, "high"] = 1.0
    with pytest.raises(ValueError, match="high"):
        validate_h1_snapshot(frame)


def test_unresolved_gap_is_rejected() -> None:
    frame = _frame().drop(index=2).reset_index(drop=True)
    with pytest.raises(ValueError, match="gap"):
        validate_h1_snapshot(frame)
