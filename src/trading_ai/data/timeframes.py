"""Supported MT5 source-bar durations in UTC."""

from __future__ import annotations

import pandas as pd


_DURATIONS = {
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(hours=24),
}


def timeframe_duration(timeframe: str) -> pd.Timedelta:
    """Return the nominal duration of a supported source bar."""
    try:
        return _DURATIONS[timeframe]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported timeframe: {timeframe!r}") from exc
