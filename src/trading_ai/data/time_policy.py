"""Episode 002 time policy: explicit range, cutoff, and coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trading_ai.data.request import MarketDataRequest
from trading_ai.data.timeframes import timeframe_duration


@dataclass(frozen=True)
class TimePolicyResult:
    accepted_range: pd.DataFrame
    exclusions: dict[str, int]
    coverage: dict[str, Any]


def apply_market_time_policy(
    raw: pd.DataFrame,
    request: MarketDataRequest,
) -> TimePolicyResult:
    """Apply half-open range and native completed-bar rules without mutating raw data."""
    if "timestamp" not in raw.columns:
        raise ValueError("raw response is missing timestamp")

    working = raw.copy()
    timestamp = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    invalid_timestamp = timestamp.isna()

    before_start = timestamp < request.start
    at_or_after_end = timestamp >= request.end
    incomplete = timestamp.add(timeframe_duration(request.timeframe)) > request.cutoff

    keep = ~(invalid_timestamp | before_start | at_or_after_end | incomplete)

    accepted = working.loc[keep].copy()
    accepted["timestamp"] = timestamp.loc[keep]

    valid_raw_ts = timestamp.dropna()
    eligible_ts = pd.DatetimeIndex(accepted["timestamp"]) if len(accepted) else pd.DatetimeIndex([])

    first_returned = valid_raw_ts.min() if len(valid_raw_ts) else None
    last_returned = valid_raw_ts.max() if len(valid_raw_ts) else None
    first_eligible = eligible_ts.min() if len(eligible_ts) else None
    last_eligible = eligible_ts.max() if len(eligible_ts) else None

    last_required = request.last_required_open
    required_start_present = bool(request.start in eligible_ts)
    required_completed_end_present = bool(
        last_required is None or last_required in eligible_ts
    )
    if request.timeframe == "H1":
        coverage_ok = bool(
            last_required is None
            or (required_start_present and required_completed_end_present)
        )
    else:
        # The first source open may be offset from UTC midnight; alignment to
        # the common H1 decision interval is checked when the bundle is built.
        coverage_ok = bool(len(eligible_ts))

    coverage = {
        "first_returned_timestamp": (
            first_returned.isoformat() if first_returned is not None else None
        ),
        "last_returned_timestamp": (
            last_returned.isoformat() if last_returned is not None else None
        ),
        "first_eligible_timestamp": (
            first_eligible.isoformat() if first_eligible is not None else None
        ),
        "last_eligible_timestamp": (
            last_eligible.isoformat() if last_eligible is not None else None
        ),
        "required_start_timestamp": request.start.isoformat(),
        "required_start_present": required_start_present,
        "last_required_open_timestamp": (
            last_required.isoformat() if last_required is not None else None
        ),
        "required_completed_end_present": required_completed_end_present,
        "coverage_ok": coverage_ok,
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


def apply_h1_time_policy(raw: pd.DataFrame, request: MarketDataRequest) -> TimePolicyResult:
    """Compatibility entry point for Episode 002's H1 pipeline."""
    if request.timeframe != "H1":
        raise ValueError("apply_h1_time_policy requires H1")
    return apply_market_time_policy(raw, request)
