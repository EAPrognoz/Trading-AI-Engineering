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


def _segment_ids(timestamp: pd.Series) -> pd.Series:
    delta = timestamp.diff()
    new_segment = delta.ne(pd.Timedelta(hours=1))
    if len(new_segment):
        new_segment.iloc[0] = True
    return new_segment.cumsum()


def _grouped_pct_change(values: pd.Series, segment: pd.Series, periods: int) -> pd.Series:
    return values.groupby(segment, sort=False).pct_change(periods=periods, fill_method=None)


def _grouped_rolling_std(values: pd.Series, segment: pd.Series, window: int) -> pd.Series:
    result = (
        values.groupby(segment, sort=False)
        .rolling(window, min_periods=window)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    return result.sort_index()


def _grouped_rolling_mean(values: pd.Series, segment: pd.Series, window: int) -> pd.Series:
    result = (
        values.groupby(segment, sort=False)
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return result.sort_index()


def build_point_in_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features available no later than the close of bar t.

    No feature uses shift(-1), a future rolling window, or the Episode 003
    supervised target. Any timestamp gap other than exactly one hour starts a
    new segment, so return/volatility/volume lookbacks never bridge a weekend,
    session break, or missing-history gap.
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
    segment = _segment_ids(timestamp)

    one_hour_return = _grouped_pct_change(close, segment, 1)

    result = pd.DataFrame(index=frame.index)
    result["timestamp"] = timestamp
    result["return_1h"] = one_hour_return
    result["return_3h"] = _grouped_pct_change(close, segment, 3)
    result["return_6h"] = _grouped_pct_change(close, segment, 6)
    result["return_12h"] = _grouped_pct_change(close, segment, 12)
    result["return_24h"] = _grouped_pct_change(close, segment, 24)

    result["rolling_vol_6h"] = _grouped_rolling_std(one_hour_return, segment, 6)
    result["rolling_vol_12h"] = _grouped_rolling_std(one_hour_return, segment, 12)
    result["rolling_vol_24h"] = _grouped_rolling_std(one_hour_return, segment, 24)

    result["range_pct"] = high.sub(low).div(close)
    result["body_return"] = close.div(open_).sub(1.0)

    volume_mean_24h = _grouped_rolling_mean(tick_volume, segment, 24)
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
