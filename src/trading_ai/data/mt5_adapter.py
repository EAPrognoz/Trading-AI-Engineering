"""Small MetaTrader 5 adapter used by Episode 002."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def fetch_h1_bars(
    symbol: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch H1 bars from the local MetaTrader 5 terminal.

    Keep this function intentionally small. Episode 002 first teaches the
    simple connection to MT5; validation, time policy, manifests, and rejection
    handling live in separate pipeline modules.
    """
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "MetaTrader5 is not installed. Run: pip install MetaTrader5"
        ) from exc

    if not mt5.initialize():
        raise RuntimeError(f"MetaTrader5 initialize failed: {mt5.last_error()}")

    try:
        rates = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_H1,
            start,
            end,
        )
        if rates is None:
            raise RuntimeError(
                f"MetaTrader5 copy_rates_range failed: {mt5.last_error()}"
            )
    finally:
        mt5.shutdown()

    frame = pd.DataFrame(rates)
    if frame.empty:
        return frame

    frame = frame.rename(columns={"time": "timestamp"})
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        unit="s",
        utc=True,
    )
    return frame
