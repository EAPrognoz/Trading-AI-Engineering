"""Optional MetaTrader 5 source adapter for Episode 002.

The core repository does not require MetaTrader5 to run tests. Import happens
only when a real terminal fetch is requested.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_ai.data.request import MarketDataRequest


def fetch_h1_bars(
    request: MarketDataRequest,
    *,
    mt5_module: Any | None = None,
) -> pd.DataFrame:
    """Fetch raw H1 bars without placing orders; always shuts the terminal down."""
    if mt5_module is None:
        try:
            import MetaTrader5 as mt5_module  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "MetaTrader5 package is not installed. "
                "Use the fixture demo or install the compatible package locally."
            ) from exc

    if not mt5_module.initialize():
        raise RuntimeError(f"MetaTrader5 initialize failed: {mt5_module.last_error()}")

    try:
        if not mt5_module.symbol_select(request.symbol, True):
            raise RuntimeError(
                f"MetaTrader5 could not select symbol {request.symbol!r}: "
                f"{mt5_module.last_error()}"
            )

        rates = mt5_module.copy_rates_range(
            request.symbol,
            mt5_module.TIMEFRAME_H1,
            request.start.to_pydatetime(),
            request.end.to_pydatetime(),
        )
        if rates is None:
            raise RuntimeError(
                f"MetaTrader5 copy_rates_range failed: {mt5_module.last_error()}"
            )

        frame = pd.DataFrame(rates)
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "tick_volume",
                    "spread",
                    "real_volume",
                ]
            )

        if "time" not in frame.columns:
            raise RuntimeError("MetaTrader5 response does not contain time")

        frame = frame.rename(columns={"time": "timestamp"})
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            unit="s",
            utc=True,
        )
        keep = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ]
        missing = [column for column in keep if column not in frame.columns]
        if missing:
            raise RuntimeError(
                f"MetaTrader5 response is missing columns: {missing}"
            )
        return frame[keep].copy()
    finally:
        mt5_module.shutdown()
