from datetime import datetime, timedelta, timezone
from decimal import Decimal

from experiments.eurusd_h1_model_error_lab.backtest.engine import Tick
from experiments.eurusd_h1_model_error_lab.backtest.v2.engine import (
    ABSTAIN_POLICY,
    SAME_SIDE_POLICY,
    Decision,
    ReplayConfig,
    replay,
)


def test_same_side_is_noop_and_abstain_holds_until_original_target():
    """Synthetic policy fixture; this is a software test, not market data."""
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)
    config = ReplayConfig(
        same_side_policy=SAME_SIDE_POLICY,
        abstain_policy=ABSTAIN_POLICY,
        period_start_utc=start,
        period_end_utc=end,
    )
    decisions = (
        Decision("open-long", start, "long"),
        Decision("same-side", start + timedelta(minutes=30), "long"),
        Decision("abstain", start + timedelta(minutes=45), "ABSTAIN"),
    )
    synthetic_ticks = tuple(
        Tick(
            time_utc=start + offset,
            source_seq=seq,
            bid=Decimal("1.10000"),
            ask=Decimal("1.10020"),
        )
        for seq, offset in enumerate(
            (
                timedelta(0),
                timedelta(seconds=1),
                timedelta(minutes=30),
                timedelta(minutes=45),
                timedelta(hours=1),
                timedelta(hours=1, seconds=1),
            )
        )
    )

    result = replay(decisions, synthetic_ticks, config)

    assert len(result.positions) == 1
    position = result.positions[0]
    assert position.signal_id == "open-long"
    assert position.target_time == start + timedelta(hours=1)
    assert position.exit_time == start + timedelta(hours=1)
    assert position.exit_reason == "target"
    dispositions = {row["signal_id"]: row["disposition"] for row in result.signal_actions}
    assert dispositions["same-side"] == "same_side_noop"
    assert dispositions["abstain"] == "abstain_hold"
