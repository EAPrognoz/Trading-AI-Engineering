"""Episode 002 time policy: explicit range, cutoff, and coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trading_ai.data.request import MarketDataRequest


@dataclass(frozen=True)
class TimePolicyResult:
    accepted_range: pd.DataFrame
    exclusions: dict[str, int]
    coverage: dict[str, Any]


def apply_h1_time_policy(
    raw: pd.DataFrame,
    request: MarketDataRequest,
) -> TimePolicyResult:
    """Apply half-open range and completed-bar rules without mutating raw data."""
    if "timestamp" not in raw.columns:
        raise ValueError("raw response is missing timestamp")

    working = raw.copy()
    timestamp = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    invalid_timestamp = timestamp.isna()

    before_start = timestamp < request.start
    at_or_after_end = timestamp >= request.end
    incomplete = timestamp.add(pd.Timedelta(hours=1)) > request.cutoff

    keep = ~(invalid_timestamp | before_start | at_or_after_end | incomplete)

    accepted = working.loc[keep].copy()
    accepted["timestamp"] = timestamp.loc[keep]

    valid_raw_ts = timestamp.dropna()
    first_returned = valid_raw_ts.min() if len(valid_raw_ts) else None
    last_returned = valid_raw_ts.max() if len(valid_raw_ts) else None
    last_required = request.last_required_open

    covers_start = first_returned is not None and first_returned <= request.start
    covers_end = last_required is None or (
        last_returned is not None and last_returned >= last_required
    )

    coverage = {
        "first_returned_timestamp": (
            first_returned.isoformat() if first_returned is not None else None
        ),
        "last_returned_timestamp": (
            last_returned.isoformat() if last_returned is not None else None
        ),
        "last_required_open_timestamp": (
            last_required.isoformat() if last_required is not None else None
        ),
        "covers_requested_start": bool(covers_start),
        "covers_required_completed_end": bool(covers_end),
        "coverage_ok": bool(covers_start and covers_end),
    }

    return TimePolicyResult(
        accepted_range=accepted.reset_index(drop=True),
        exclusions={
            "invalid_timestamp": int(invalid_timestamp.sum()),
            "before_start": int(before_start.fillna(False).sum()),
            "at_or_after_end": int(at_or_after_end.fillna(False).sum()),
            "incomplete_by_cutoff": int(incomplete.fillna(False).sum()),
        },
        coverage=coverage,
    )
