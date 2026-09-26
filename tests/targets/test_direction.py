from __future__ import annotations

import pandas as pd
import pytest

from trading_ai.targets.direction import build_h1_direction_target


def test_h1_direction_target_is_aligned_to_decision_bar() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
            "close": [100.0, 101.0, 100.0, 100.0, 102.0],
        }
    )

    result = build_h1_direction_target(frame)

    assert result["target_h1_direction"].tolist()[:4] == ["UP", "DOWN", "ZERO", "UP"]
    assert pd.isna(result["target_h1_direction"].iloc[-1])
    assert result["future_return_1h"].iloc[0] == pytest.approx(0.01)
    assert result["target_timestamp"].iloc[0] == frame["timestamp"].iloc[1]
    assert pd.isna(result["target_timestamp"].iloc[-1])


def test_changing_t_plus_two_does_not_change_label_for_t() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )
    first = build_h1_direction_target(frame)

    changed = frame.copy()
    changed.loc[2, "close"] = 50.0
    second = build_h1_direction_target(changed)

    # The label at t=0 depends on close[1], not on data further in the future.
    assert first.loc[0, "target_h1_direction"] == second.loc[0, "target_h1_direction"]
    assert first.loc[0, "future_return_1h"] == second.loc[0, "future_return_1h"]
