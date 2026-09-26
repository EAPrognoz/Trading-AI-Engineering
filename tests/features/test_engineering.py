from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from trading_ai.features.engineering import build_point_in_time_features


def _frame(periods: int = 48) -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    close = 100.0 + np.arange(periods, dtype=float) * 0.1 + np.sin(np.arange(periods) / 3.0)
    open_ = close - 0.05
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
            "tick_volume": 100 + (np.arange(periods) % 10) * 5,
        }
    )


def test_future_bar_change_does_not_change_features_at_t() -> None:
    frame = _frame()
    first = build_point_in_time_features(frame)
    t = 30

    changed = frame.copy()
    changed.loc[t + 1, ["open", "high", "low", "close", "tick_volume"]] = [
        500.0,
        550.0,
        450.0,
        525.0,
        9999.0,
    ]
    second = build_point_in_time_features(changed)

    pdt.assert_series_equal(first.loc[t], second.loc[t], check_names=False)


def test_feature_frame_contains_no_target_column() -> None:
    features = build_point_in_time_features(_frame())
    assert "future_return_1h" not in features.columns
    assert "target_h1_direction" not in features.columns


def test_warmup_is_explicit_not_backfilled() -> None:
    features = build_point_in_time_features(_frame())
    assert pd.isna(features.loc[0, "return_1h"])
    assert pd.isna(features.loc[23, "return_24h"])
    assert pd.notna(features.loc[24, "return_24h"])
