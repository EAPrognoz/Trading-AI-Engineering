"""Episode 004 point-in-time feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_LOOKBACK_BARS: dict[str, int] = {
    "return_1h": 1,
    "return_3h": 3,
    "return_6h": 6,
    "return_12h": 12,
    "return_24h": 24,
    "rolling_vol_6h": 6,
    "rolling_vol_12h": 12,
    "rolling_vol_24h": 24,
    "range_pct": 1,
    "body_return": 1,
    "relative_tick_volume_24h": 24,
    "hour_sin": 1,
    "hour_cos": 1,
    "dow_sin": 1,
    "dow_cos": 1,
}


def build_point_in_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features available no later than the close of bar t.

    No feature uses shift(-1), a future rolling window, or the Episode 003
    supervised target. Warm-up rows are retained as NaN and are handled later
    by the experiment assembly step.
    """
    required = {"timestamp", "open", "high", "low", "close", "tick_volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    open_ = pd.to_numeric(frame["open"], errors="raise").astype(float)
    high = pd.to_numeric(frame["high"], errors="raise").astype(float)
    low = pd.to_numeric(frame["low"], errors="raise").astype(float)
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)
    tick_volume = pd.to_numeric(frame["tick_volume"], errors="raise").astype(float)

    one_hour_return = close.pct_change(fill_method=None)

    result = pd.DataFrame(index=frame.index)
    result["timestamp"] = timestamp
    result["return_1h"] = one_hour_return
    result["return_3h"] = close.pct_change(3, fill_method=None)
    result["return_6h"] = close.pct_change(6, fill_method=None)
    result["return_12h"] = close.pct_change(12, fill_method=None)
    result["return_24h"] = close.pct_change(24, fill_method=None)

    result["rolling_vol_6h"] = one_hour_return.rolling(6, min_periods=6).std(ddof=0)
    result["rolling_vol_12h"] = one_hour_return.rolling(12, min_periods=12).std(ddof=0)
    result["rolling_vol_24h"] = one_hour_return.rolling(24, min_periods=24).std(ddof=0)

    result["range_pct"] = high.sub(low).div(close)
    result["body_return"] = close.div(open_).sub(1.0)

    volume_mean_24h = tick_volume.rolling(24, min_periods=24).mean()
    result["relative_tick_volume_24h"] = tick_volume.div(volume_mean_24h).sub(1.0)

    hour = timestamp.dt.hour.astype(float)
    day_of_week = timestamp.dt.dayofweek.astype(float)
    result["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    result["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    result["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

    numeric_columns = [column for column in result.columns if column != "timestamp"]
    result[numeric_columns] = result[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return result
