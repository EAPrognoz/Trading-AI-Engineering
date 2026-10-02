"""Small MetaTrader 5 adapter used by Episode 002."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from trading_ai.data.timeframes import timeframe_duration


def fetch_market_bars(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch historical native bars and attach non-account terminal provenance."""
    timeframe_duration(timeframe)
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "MetaTrader5 is not installed. Run: pip install MetaTrader5"
        ) from exc

    mt5_timeframe = getattr(mt5, f"TIMEFRAME_{timeframe}", None)
    if mt5_timeframe is None:
        raise RuntimeError(f"MetaTrader5 does not expose TIMEFRAME_{timeframe}")
    if not mt5.initialize():
        raise RuntimeError(f"MetaTrader5 initialize failed: {mt5.last_error()}")
    try:
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start, end)
        if rates is None:
            raise RuntimeError(
                f"MetaTrader5 copy_rates_range failed: {mt5.last_error()}"
            )
        terminal_info = mt5.terminal_info()
        terminal_build = getattr(terminal_info, "build", None)
        terminal_version = mt5.version()
        if not isinstance(terminal_build, int):
            raise RuntimeError("MetaTrader5 terminal build is unavailable")
        if not isinstance(terminal_version, (tuple, list)) or not terminal_version:
            raise RuntimeError("MetaTrader5 terminal version is unavailable")
        package_version = getattr(mt5, "__version__", None)
        if not isinstance(package_version, str) or not package_version:
            raise RuntimeError("MetaTrader5 package version is unavailable")
        retrieved_at = datetime.now(timezone.utc).isoformat()
    finally:
        mt5.shutdown()

    frame = pd.DataFrame(rates).rename(columns={"time": "timestamp"})
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    frame.attrs["mt5_provenance"] = {
        "source_type": "MetaTrader5",
        "terminal_build": terminal_build,
        "terminal_version": list(terminal_version),
        "metatrader5_package_version": package_version,
        "retrieved_at_utc": retrieved_at,
    }
    return frame


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
    return fetch_market_bars(symbol, "H1", start, end)
