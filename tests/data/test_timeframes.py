from __future__ import annotations

import pandas as pd
import pytest

from trading_ai.data.request import MarketDataRequest


@pytest.mark.parametrize(
    ("timeframe", "expected_hours"),
    [("H1", 1), ("H4", 4), ("D1", 24)],
)
def test_supported_timeframe_durations(timeframe: str, expected_hours: int) -> None:
    from trading_ai.data.timeframes import timeframe_duration

    assert timeframe_duration(timeframe) == pd.Timedelta(hours=expected_hours)


def test_unsupported_timeframe_is_rejected() -> None:
    from trading_ai.data.timeframes import timeframe_duration

    with pytest.raises(ValueError, match="unsupported timeframe"):
        timeframe_duration("M15")
    with pytest.raises(ValueError, match="unsupported timeframe"):
        MarketDataRequest(
            symbol="BTCUSD",
            start="2026-01-01T01:00:00Z",
            end="2026-01-03T01:00:00Z",
            cutoff="2026-01-03T01:00:00Z",
            timeframe="M15",
        )


def test_d1_request_accepts_nonmidnight_utc_boundary() -> None:
    request = MarketDataRequest(
        symbol="BTCUSD",
        start="2026-01-01T01:00:00Z",
        end="2026-01-03T01:00:00Z",
        cutoff="2026-01-03T01:00:00Z",
        timeframe="D1",
    )
    assert request.start == pd.Timestamp("2026-01-01T01:00:00Z")
    assert request.timeframe == "D1"


@pytest.mark.parametrize(
    ("timeframe", "open_time", "cutoff", "expected_accepted"),
    [
        ("H4", "2026-01-01T01:00:00Z", "2026-01-01T04:59:59Z", 0),
        ("H4", "2026-01-01T01:00:00Z", "2026-01-01T05:00:00Z", 1),
        ("D1", "2026-01-01T01:00:00Z", "2026-01-02T00:59:59Z", 0),
        ("D1", "2026-01-01T01:00:00Z", "2026-01-02T01:00:00Z", 1),
    ],
)
def test_cutoff_uses_native_nominal_close(
    timeframe: str, open_time: str, cutoff: str, expected_accepted: int
) -> None:
    from trading_ai.data.time_policy import apply_market_time_policy

    request = MarketDataRequest(
        symbol="BTCUSD",
        start="2026-01-01T01:00:00Z",
        end="2026-01-03T01:00:00Z",
        cutoff=cutoff,
        timeframe=timeframe,
    )
    result = apply_market_time_policy(
        pd.DataFrame({"timestamp": [open_time]}), request
    )
    assert len(result.accepted_range) == expected_accepted
    assert result.exclusions["incomplete_by_cutoff"] == 1 - expected_accepted


def test_generic_coverage_reports_actual_eligible_opens() -> None:
    from trading_ai.data.time_policy import apply_market_time_policy

    request = MarketDataRequest(
        symbol="BTCUSD",
        start="2026-01-01T00:00:00Z",
        end="2026-01-04T00:00:00Z",
        cutoff="2026-01-04T00:00:00Z",
        timeframe="D1",
    )
    result = apply_market_time_policy(
        pd.DataFrame({
            "timestamp": [
                "2026-01-01T01:00:00Z",
                "2026-01-02T01:00:00Z",
            ]
        }),
        request,
    )
    assert result.coverage["first_eligible_timestamp"] == "2026-01-01T01:00:00+00:00"
    assert result.coverage["last_eligible_timestamp"] == "2026-01-02T01:00:00+00:00"
    assert result.coverage["coverage_ok"] is True
