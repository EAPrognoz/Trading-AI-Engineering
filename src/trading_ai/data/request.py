"""Explicit market-data request contract from Episode 002."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def _utc_timestamp(value: Any, *, field: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return ts.tz_convert("UTC")


def _require_utc_hour_boundary(ts: pd.Timestamp, *, field: str) -> None:
    if ts.minute or ts.second or ts.microsecond or ts.nanosecond:
        raise ValueError(f"{field} must be aligned to a UTC hour")


@dataclass(frozen=True)
class MarketDataRequest:
    """Narrow H1 request used by the Episode 002 pipeline."""

    symbol: str
    start: Any
    end: Any
    cutoff: Any
    timeframe: str = "H1"

    def __post_init__(self) -> None:
        symbol = self.symbol.strip()
        if not symbol:
            raise ValueError("symbol must be non-empty")
        if self.timeframe != "H1":
            raise ValueError("Episode 002 contract currently supports H1 only")

        start = _utc_timestamp(self.start, field="start")
        end = _utc_timestamp(self.end, field="end")
        cutoff = _utc_timestamp(self.cutoff, field="cutoff")
        _require_utc_hour_boundary(start, field="start")
        _require_utc_hour_boundary(end, field="end")

        if not start < end:
            raise ValueError("start must be earlier than end")
        if cutoff <= start:
            raise ValueError("cutoff must be later than start")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "cutoff", cutoff)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "cutoff": self.cutoff.isoformat(),
            "timezone": "UTC",
            "range_policy": "start_inclusive_end_exclusive",
            "completed_bars_only": True,
        }

    @property
    def last_required_open(self) -> pd.Timestamp | None:
        """Latest H1 opening time required by range + cutoff semantics."""
        completed_boundary = self.cutoff.floor("h")
        effective_end = min(self.end, completed_boundary)
        last_open = effective_end - pd.Timedelta(hours=1)
        if last_open < self.start:
            return None
        return last_open
