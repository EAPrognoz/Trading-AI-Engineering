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
    assert result["decision_timestamp"].iloc[0] == (
        frame["timestamp"].iloc[0] + pd.Timedelta(hours=1)
    )
    assert result["target_timestamp"].iloc[0] == (
        frame["timestamp"].iloc[1] + pd.Timedelta(hours=1)
    )
    assert pd.isna(result["target_timestamp"].iloc[-1])


def test_non_consecutive_pair_is_not_labeled_as_one_hour() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-03T00:00:00Z",
                    "2026-01-03T01:00:00Z",
                ]
            ),
            "close": [100.0, 101.0, 110.0, 111.0],
        }
    )

    result = build_h1_direction_target(frame)

    assert result.loc[0, "target_h1_direction"] == "UP"
    assert result.loc[1, "target_h1_direction"] == "GAP"
    assert pd.isna(result.loc[1, "future_return_1h"])
    assert result.loc[2, "target_h1_direction"] == "UP"


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

    assert first.loc[0, "target_h1_direction"] == second.loc[0, "target_h1_direction"]
    assert first.loc[0, "future_return_1h"] == second.loc[0, "future_return_1h"]


def test_timestamp_semantics_do_not_change_labels_or_returns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01T08:00:00Z",
                periods=4,
                freq="h",
            ),
            "close": [100.0, 102.0, 101.0, 104.0],
        }
    )

    result = build_h1_direction_target(frame)

    expected_returns = [
        102.0 / 100.0 - 1.0,
        101.0 / 102.0 - 1.0,
        104.0 / 101.0 - 1.0,
    ]
    assert result["target_h1_direction"].tolist()[:3] == ["UP", "DOWN", "UP"]
    assert result["future_return_1h"].iloc[:3].tolist() == pytest.approx(
        expected_returns
    )
    assert result["decision_timestamp"].iloc[0] == pd.Timestamp(
        "2026-01-01T09:00:00Z"
    )
    assert result["target_timestamp"].iloc[0] == pd.Timestamp(
        "2026-01-01T10:00:00Z"
    )
