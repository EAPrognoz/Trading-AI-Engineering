"""Read-only EURUSD replay engine for the frozen Model Error Lab outputs.

The engine consumes already-saved classifier predictions and historical quote
rows. It does not import a model, fit a scaler, connect to MetaTrader, or place
orders. All P/L is a virtual replay. USD P/L for EURUSD uses fixed EUR base
units and executable bid/ask prices; spread is therefore already embedded in
the quote-to-quote P/L and is only decomposed for reporting.
"""

from __future__ import annotations

import csv
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


UTC = timezone.utc
ZERO = Decimal("0")
ONE_HUNDRED_K = Decimal("100000")
MONEY_QUANTUM = Decimal("0.01")


class ReplayInputError(ValueError):
    """Input rows or assumptions cannot support an auditable replay."""


def _decimal(value: Any, field_name: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ReplayInputError(f"missing or invalid {field_name}: {value!r}")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ReplayInputError(f"invalid {field_name}: {value!r}") from exc
    if not parsed.is_finite():
        raise ReplayInputError(f"nonfinite {field_name}: {value!r}")
    return parsed


def _utc(value: Any, field_name: str) -> datetime:
    if isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            value = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ReplayInputError(f"invalid {field_name}: {value!r}") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ReplayInputError(f"{field_name} must be a timezone-aware datetime")
    if value.utcoffset() != timedelta(0):
        raise ReplayInputError(f"{field_name} must be UTC")
    return value.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    timespec = (
        "seconds"
        if value.microsecond == 0
        else "milliseconds" if value.microsecond % 1000 == 0 else "microseconds"
    )
    return value.astimezone(UTC).isoformat(timespec=timespec).replace("+00:00", "Z")


def _cent(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANTUM)


@dataclass(frozen=True)
class Signal:
    signal_id: str
    decision_time: datetime
    target_time: datetime
    side: str
    probability_up: Decimal
    target_label: str
    partition: str
    model_sha256: str

    def __post_init__(self) -> None:
        decision = _utc(self.decision_time, "decision_time")
        target = _utc(self.target_time, "target_time")
        object.__setattr__(self, "decision_time", decision)
        object.__setattr__(self, "target_time", target)
        p_up = _decimal(self.probability_up, "probability_up")
        if not ZERO <= p_up <= Decimal("1"):
            raise ReplayInputError("probability_up must lie in [0, 1]")
        object.__setattr__(self, "probability_up", p_up)
        if self.side not in {"long", "short"}:
            raise ReplayInputError("signal side must be long or short")
        if target != decision + timedelta(hours=1):
            raise ReplayInputError("H1 target_time must equal decision_time + 1 hour")
        if self.partition != "final_confirmation":
            raise ReplayInputError("only the frozen final_confirmation partition is permitted")
        if self.target_label not in {"UP", "DOWN"}:
            raise ReplayInputError("target_label must be UP or DOWN")
        if len(self.model_sha256) != 64:
            raise ReplayInputError("model_sha256 must be a 64-character SHA-256")


@dataclass(frozen=True)
class Tick:
    time_utc: datetime
    source_seq: int
    bid: Decimal
    ask: Decimal
    flags: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_utc", _utc(self.time_utc, "tick time"))
        bid = _decimal(self.bid, "bid")
        ask = _decimal(self.ask, "ask")
        if bid <= ZERO or ask <= ZERO or ask < bid:
            raise ReplayInputError("invalid bid/ask quote; require 0 < bid <= ask")
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)
        if isinstance(self.source_seq, bool) or int(self.source_seq) != self.source_seq:
            raise ReplayInputError("source_seq must be an integer row-order key")
        object.__setattr__(self, "source_seq", int(self.source_seq))
        if isinstance(self.flags, bool) or int(self.flags) != self.flags:
            raise ReplayInputError("tick flags must be an integer")
        object.__setattr__(self, "flags", int(self.flags))


@dataclass(frozen=True)
class ReplayConfig:
    starting_balance_usd: Decimal
    base_units: Decimal
    pip_size: Decimal
    expected_model_sha256: str
    lower_threshold: Decimal
    upper_threshold: Decimal
    max_entry_delay_seconds: int
    max_exit_delay_seconds: int
    commission_usd_per_100k_per_side: Decimal
    fee_usd_per_100k_per_side: Decimal
    tax_usd_per_round_turn: Decimal
    slippage_pips_per_side: Decimal
    swap_usd_per_100k_per_rollover_long: Decimal
    swap_usd_per_100k_per_rollover_short: Decimal
    rollover_time_utc: str
    triple_swap_weekday: int
    run_mode: str
    period_start_utc: datetime | None = None
    period_end_utc: datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "starting_balance_usd",
            "base_units",
            "pip_size",
            "lower_threshold",
            "upper_threshold",
            "commission_usd_per_100k_per_side",
            "fee_usd_per_100k_per_side",
            "tax_usd_per_round_turn",
            "slippage_pips_per_side",
            "swap_usd_per_100k_per_rollover_long",
            "swap_usd_per_100k_per_rollover_short",
        ):
            object.__setattr__(self, name, _decimal(getattr(self, name), name))
        if self.starting_balance_usd <= ZERO or self.base_units <= ZERO or self.pip_size <= ZERO:
            raise ReplayInputError("starting balance, base units, and pip size must be positive")
        if not ZERO < self.lower_threshold < self.upper_threshold < Decimal("1"):
            raise ReplayInputError("signal thresholds must satisfy 0 < lower < upper < 1")
        for name in (
            "commission_usd_per_100k_per_side",
            "fee_usd_per_100k_per_side",
            "tax_usd_per_round_turn",
            "slippage_pips_per_side",
        ):
            if getattr(self, name) < ZERO:
                raise ReplayInputError(f"{name} must be nonnegative")
        if self.max_entry_delay_seconds < 0 or self.max_exit_delay_seconds < 0:
            raise ReplayInputError("quote delay limits must be nonnegative")
        if len(self.expected_model_sha256) != 64:
            raise ReplayInputError("expected_model_sha256 must be a 64-character SHA-256")
        try:
            hour_text, minute_text = self.rollover_time_utc.split(":", 1)
            rollover = time(int(hour_text), int(minute_text), tzinfo=UTC)
        except (TypeError, ValueError) as exc:
            raise ReplayInputError("rollover_time_utc must be HH:MM in UTC") from exc
        if rollover.second or rollover.microsecond:
            raise ReplayInputError("rollover_time_utc must have minute precision")
        if not 0 <= self.triple_swap_weekday <= 6:
            raise ReplayInputError("triple_swap_weekday must be Monday=0 through Sunday=6")
        if self.run_mode not in {"synthetic_fixture", "market_history"}:
            raise ReplayInputError("run_mode must be synthetic_fixture or market_history")
        if self.period_start_utc is not None:
            object.__setattr__(self, "period_start_utc", _utc(self.period_start_utc, "period_start_utc"))
        if self.period_end_utc is not None:
            object.__setattr__(self, "period_end_utc", _utc(self.period_end_utc, "period_end_utc"))
        if (
            self.period_start_utc is not None
            and self.period_end_utc is not None
            and self.period_start_utc >= self.period_end_utc
        ):
            raise ReplayInputError("period_start_utc must be earlier than period_end_utc")


@dataclass(frozen=True)
class PredictionAudit:
    total_rows: int
    selected_partition_rows: int
    signal_rows: int
    abstain_rows: int
    model_sha256: str


@dataclass(frozen=True)
class PredictionLoad:
    signals: tuple[Signal, ...]
    audit: PredictionAudit


@dataclass(frozen=True)
class Trade:
    signal_id: str
    side: str
    decision_time: datetime
    target_time: datetime
    entry_time: datetime
    exit_time: datetime
    entry_bid: Decimal
    entry_ask: Decimal
    exit_bid: Decimal
    exit_ask: Decimal
    entry_price: Decimal
    exit_price: Decimal
    base_units: Decimal
    midpoint_pnl_usd: Decimal
    spread_cost_usd: Decimal
    quote_pnl_before_slippage_usd: Decimal
    slippage_cost_usd: Decimal
    commission_cost_usd: Decimal
    fee_cost_usd: Decimal
    tax_cost_usd: Decimal
    swap_cost_usd: Decimal
    net_pnl_usd: Decimal
    hold_seconds: Decimal
    target_label: str
    prediction_correct: bool


@dataclass(frozen=True)
class SkippedSignal:
    signal_id: str
    decision_time: datetime
    reason: str


@dataclass(frozen=True)
class OpenPosition:
    signal_id: str
    side: str
    decision_time: datetime
    target_time: datetime
    entry_time: datetime
    last_mark_time: datetime | None
    entry_bid: Decimal
    entry_ask: Decimal
    last_bid: Decimal | None
    last_ask: Decimal | None
    unrealized_liquidation_pnl_usd: Decimal | None


@dataclass(frozen=True)
class ReplayResult:
    trades: tuple[Trade, ...]
    skipped_signals: tuple[SkippedSignal, ...]
    open_positions: tuple[OpenPosition, ...]
    balance_curve: tuple[dict[str, Any], ...]
    equity_curve: tuple[dict[str, Any], ...]
    ledger: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]


def load_frozen_predictions(
    path: str | Path,
    *,
    expected_model_sha256: str,
    partition: str,
    lower_threshold: Decimal,
    upper_threshold: Decimal,
) -> PredictionLoad:
    """Read saved predictions, select final rows, and verify frozen candidate rules."""
    path = Path(path)
    required = {
        "decision_timestamp",
        "target_available_timestamp",
        "partition",
        "candidate_signal",
        "B2_raw_probability_up_not_calibrated",
        "target_h1_direction",
        "model_sha256",
    }
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ReplayInputError("predictions CSV has no header")
            missing = required - set(reader.fieldnames)
            if missing:
                raise ReplayInputError(f"predictions CSV missing fields {sorted(missing)}")
            rows = list(reader)
    except OSError as exc:
        raise ReplayInputError(f"cannot read predictions CSV: {exc}") from exc

    selected = [row for row in rows if row["partition"] == partition]
    if partition != "final_confirmation":
        raise ReplayInputError("replay is restricted to final_confirmation predictions")
    if not selected:
        raise ReplayInputError("predictions CSV contains no final_confirmation rows")
    lower = _decimal(lower_threshold, "lower_threshold")
    upper = _decimal(upper_threshold, "upper_threshold")
    if not ZERO < lower < upper < Decimal("1"):
        raise ReplayInputError("candidate thresholds must satisfy 0 < lower < upper < 1")

    signals: list[Signal] = []
    abstain_rows = 0
    seen: set[datetime] = set()
    previous: datetime | None = None
    for row_number, row in enumerate(selected, start=2):
        if row["model_sha256"] != expected_model_sha256:
            raise ReplayInputError(
                f"prediction row {row_number}: model hash does not match frozen model"
            )
        decision = _utc(row["decision_timestamp"], "decision_timestamp")
        target = _utc(row["target_available_timestamp"], "target_available_timestamp")
        if decision in seen:
            raise ReplayInputError(f"duplicate final decision timestamp {_iso(decision)}")
        if previous is not None and decision <= previous:
            raise ReplayInputError("final prediction rows are not in chronological order")
        seen.add(decision)
        previous = decision
        probability = _decimal(row["B2_raw_probability_up_not_calibrated"], "raw probability")
        recorded = row["candidate_signal"].strip()
        expected = "DOWN" if probability <= lower else "UP" if probability >= upper else "ABSTAIN"
        if recorded != expected:
            raise ReplayInputError(
                f"prediction row {row_number}: candidate_signal {recorded!r} does not match frozen thresholds"
            )
        if target != decision + timedelta(hours=1):
            raise ReplayInputError(
                f"prediction row {row_number}: target availability is not exactly one H1 later"
            )
        if recorded == "ABSTAIN":
            abstain_rows += 1
            continue
        target_label = row["target_h1_direction"].strip()
        signals.append(
            Signal(
                signal_id=_iso(decision) or "",
                decision_time=decision,
                target_time=target,
                side="long" if recorded == "UP" else "short",
                probability_up=probability,
                target_label=target_label,
                partition=partition,
                model_sha256=expected_model_sha256,
            )
        )
    audit = PredictionAudit(
        total_rows=len(rows),
        selected_partition_rows=len(selected),
        signal_rows=len(signals),
        abstain_rows=abstain_rows,
        model_sha256=expected_model_sha256,
    )
    return PredictionLoad(tuple(signals), audit)


def load_mt5_ticks_csv(path: str | Path) -> tuple[Tick, ...]:
    """Read a raw `copy_ticks_range` CSV; source order and duplicate times survive.

    When present, ``source_row_seq`` remains the Tick ordering key instead of
    being renumbered. This also allows filtered windows to retain their original
    dataset-global sequence numbers.
    """
    path = Path(path)
    required = {"time", "time_msc", "bid", "ask", "last", "flags", "volume", "volume_real"}
    ticks: list[Tick] = []
    previous_ms: int | None = None
    previous_source_seq: int | None = None
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ReplayInputError("tick CSV has no header")
            missing = required - set(reader.fieldnames)
            if missing:
                raise ReplayInputError(f"tick CSV missing fields {sorted(missing)}")
            has_source_row_seq = "source_row_seq" in reader.fieldnames
            for index, row in enumerate(reader):
                try:
                    time_seconds = int(row["time"])
                    time_msc = int(row["time_msc"])
                    flags = int(row["flags"])
                    source_seq = int(row["source_row_seq"]) if has_source_row_seq else index
                except (TypeError, ValueError) as exc:
                    raise ReplayInputError(
                        f"tick row {index + 2}: invalid integer time, flags, or source_row_seq"
                    ) from exc
                if has_source_row_seq and row["source_row_seq"].strip() != str(source_seq):
                    raise ReplayInputError(
                        f"tick row {index + 2}: source_row_seq must be a canonical nonnegative integer"
                    )
                if source_seq < 0 or (
                    previous_source_seq is not None and source_seq <= previous_source_seq
                ):
                    raise ReplayInputError(
                        f"tick row {index + 2}: source_row_seq must be nonnegative and strictly increasing"
                    )
                if time_seconds != time_msc // 1000:
                    raise ReplayInputError(
                        f"tick row {index + 2}: time and time_msc disagree"
                    )
                if previous_ms is not None and time_msc < previous_ms:
                    raise ReplayInputError("tick CSV is not in source timestamp order")
                previous_ms = time_msc
                previous_source_seq = source_seq
                timestamp = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=time_msc)
                # Validate unused MT5 numeric fields too: do not let malformed source rows pass silently.
                _decimal(row["last"], "last")
                _decimal(row["volume"], "volume")
                _decimal(row["volume_real"], "volume_real")
                ticks.append(
                    Tick(
                        time_utc=timestamp,
                        source_seq=source_seq,
                        bid=_decimal(row["bid"], "bid"),
                        ask=_decimal(row["ask"], "ask"),
                        flags=flags,
                    )
                )
    except OSError as exc:
        raise ReplayInputError(f"cannot read tick CSV: {exc}") from exc
    return tuple(ticks)


def _rollovers_between(
    entry: datetime,
    exit_: datetime,
    rollover_time: str,
    triple_weekday: int,
) -> tuple[datetime, ...]:
    hour_text, minute_text = rollover_time.split(":", 1)
    time_of_day = time(int(hour_text), int(minute_text), tzinfo=UTC)
    day = entry.date()
    final_day = exit_.date()
    events: list[datetime] = []
    while day <= final_day:
        if day.weekday() < 5:
            event = datetime.combine(day, time_of_day)
            if entry < event <= exit_:
                events.append(event)
        day += timedelta(days=1)
    return tuple(events)


def _metrics_drawdown(
    points: Sequence[Mapping[str, Any]], period_end: datetime
) -> dict[str, Any]:
    return summarize_drawdown(points, period_end=period_end)


def summarize_drawdown(
    points: Sequence[Mapping[str, Any]], *, period_end: datetime
) -> dict[str, Any]:
    """Return max peak-to-trough drawdown and its recovery/unrecovered duration."""
    end = _utc(period_end, "period_end")
    if not points:
        return {
            "maximum_drawdown_usd": "0",
            "maximum_drawdown_pct": "0",
            "peak_time_utc": None,
            "trough_time_utc": None,
            "recovery_time_utc": None,
            "recovered": True,
            "unrecovered_at_period_end": False,
            "duration_peak_to_trough_seconds": 0,
            "duration_peak_to_recovery_seconds": 0,
            "unrecovered_duration_seconds_at_end": 0,
        }
    normalized = sorted(
        ((_utc(point["time_utc"], "drawdown point time"), _decimal(point["value_usd"], "drawdown value")) for point in points),
        key=lambda item: item[0],
    )
    peak_value = normalized[0][1]
    peak_time = normalized[0][0]
    maximum = ZERO
    max_peak_time: datetime | None = None
    max_trough_time: datetime | None = None
    max_peak_value = ZERO
    for point_time, value in normalized:
        if value > peak_value:
            peak_value = value
            peak_time = point_time
        drawdown = peak_value - value
        if drawdown > maximum:
            maximum = drawdown
            max_peak_time = peak_time
            max_trough_time = point_time
            max_peak_value = peak_value
    if maximum == ZERO or max_peak_time is None or max_trough_time is None:
        return {
            "maximum_drawdown_usd": "0",
            "maximum_drawdown_pct": "0",
            "peak_time_utc": None,
            "trough_time_utc": None,
            "recovery_time_utc": None,
            "recovered": True,
            "unrecovered_at_period_end": False,
            "duration_peak_to_trough_seconds": 0,
            "duration_peak_to_recovery_seconds": 0,
            "unrecovered_duration_seconds_at_end": 0,
        }
    recovery = next(
        (point_time for point_time, value in normalized if point_time > max_trough_time and value >= max_peak_value),
        None,
    )
    duration_to_trough = int((max_trough_time - max_peak_time).total_seconds())
    duration_to_recovery = int((recovery - max_peak_time).total_seconds()) if recovery else None
    last_observation = normalized[-1][0]
    elapsed_end = max(end, last_observation)
    return {
        "maximum_drawdown_usd": str(maximum),
        "maximum_drawdown_pct": str((maximum / max_peak_value * Decimal("100")) if max_peak_value else ZERO),
        "peak_time_utc": _iso(max_peak_time),
        "trough_time_utc": _iso(max_trough_time),
        "recovery_time_utc": _iso(recovery),
        "recovered": recovery is not None,
        "unrecovered_at_period_end": recovery is None,
        "duration_peak_to_trough_seconds": duration_to_trough,
        "duration_peak_to_recovery_seconds": duration_to_recovery,
        "unrecovered_duration_seconds_at_end": int((elapsed_end - max_peak_time).total_seconds()) if recovery is None else 0,
    }


def _curve_gaps(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    times = [_utc(point["time_utc"], "equity time") for point in points]
    gaps = [int((later - earlier).total_seconds()) for earlier, later in zip(times, times[1:])]
    return {
        "snapshot_policy": "initial and period end while flat; every valid bid/ask change while a position is open; no interpolation",
        "observation_count": len(times),
        "median_spacing_seconds": median(gaps) if gaps else None,
        "maximum_gap_seconds": max(gaps) if gaps else 0,
        "gap_count_over_60_seconds": sum(gap > 60 for gap in gaps),
    }


def replay(
    signals: Iterable[Signal],
    ticks: Iterable[Tick],
    config: ReplayConfig,
) -> ReplayResult:
    """Execute a one-hour, single-position virtual strategy on quote updates."""
    signal_rows = tuple(signals)
    tick_rows = tuple(ticks)
    if not tick_rows:
        raise ReplayInputError("no bid/ask ticks: a candidate signal cannot be priced")
    if not signal_rows:
        raise ReplayInputError("no non-abstain final_confirmation signals to replay")
    if len({tick.source_seq for tick in tick_rows}) != len(tick_rows):
        raise ReplayInputError("tick source_seq values must be unique")
    if any(
        a.time_utc > b.time_utc
        or (a.time_utc == b.time_utc and a.source_seq >= b.source_seq)
        for a, b in zip(tick_rows, tick_rows[1:])
    ):
        raise ReplayInputError("tick stream must be chronological and preserve source order for timestamp ties")
    tick_times = tuple(tick.time_utc for tick in tick_rows)
    signal_rows = tuple(sorted(signal_rows, key=lambda signal: signal.decision_time))
    if len({signal.decision_time for signal in signal_rows}) != len(signal_rows):
        raise ReplayInputError("duplicate signal decision times are not permitted")
    for signal in signal_rows:
        if signal.partition != "final_confirmation":
            raise ReplayInputError("only final_confirmation signals are permitted")
        if signal.model_sha256 != config.expected_model_sha256:
            raise ReplayInputError("signal model hash does not match the frozen model")
        expected_side = (
            "short" if signal.probability_up <= config.lower_threshold
            else "long" if signal.probability_up >= config.upper_threshold
            else None
        )
        if expected_side != signal.side:
            raise ReplayInputError("signal side does not match the configured frozen probability thresholds")

    period_start = config.period_start_utc or min(signal.decision_time for signal in signal_rows)
    default_end = max(signal.target_time for signal in signal_rows) + timedelta(seconds=config.max_exit_delay_seconds)
    period_end = config.period_end_utc or default_end
    if period_end <= period_start:
        raise ReplayInputError("replay period must have positive duration")
    if any(signal.decision_time < period_start or signal.decision_time >= period_end for signal in signal_rows):
        raise ReplayInputError("every signal decision must fall inside the half-open replay period")
    if config.period_start_utc is not None and any(tick.time_utc < period_start for tick in tick_rows):
        raise ReplayInputError("tick stream contains rows before the half-open replay period")
    if config.period_end_utc is not None and any(tick.time_utc >= period_end for tick in tick_rows):
        raise ReplayInputError("tick stream contains rows at or after the half-open replay period end")

    balance = config.starting_balance_usd
    balance_curve: list[dict[str, Any]] = [
        {"time_utc": _iso(period_start), "balance_usd": str(balance), "event": "initial_balance"}
    ]
    equity_curve: list[dict[str, Any]] = [
        {"time_utc": _iso(period_start), "equity_usd": str(balance), "event": "initial_balance", "signal_id": ""}
    ]
    ledger: list[dict[str, Any]] = []
    trades: list[Trade] = []
    skipped: list[SkippedSignal] = []
    open_positions: list[OpenPosition] = []
    occupied_until: datetime | None = None
    unresolved_open = False

    for signal in signal_rows:
        if unresolved_open or (occupied_until is not None and occupied_until > signal.decision_time):
            skipped.append(SkippedSignal(signal.signal_id, signal.decision_time, "position_open_at_decision"))
            ledger.append({"time_utc": _iso(signal.decision_time), "event": "signal_skipped", "signal_id": signal.signal_id, "reason": "position_open_at_decision"})
            continue

        entry_index = bisect_right(tick_times, signal.decision_time)
        if entry_index >= len(tick_rows):
            skipped.append(SkippedSignal(signal.signal_id, signal.decision_time, "no_entry_quote_after_decision"))
            ledger.append({"time_utc": _iso(signal.decision_time), "event": "signal_unpriced", "signal_id": signal.signal_id, "reason": "no_entry_quote_after_decision"})
            continue
        entry_tick = tick_rows[entry_index]
        entry_delay = (entry_tick.time_utc - signal.decision_time).total_seconds()
        if entry_tick.time_utc >= signal.target_time or entry_delay > config.max_entry_delay_seconds:
            reason = "no_entry_quote_before_target" if entry_tick.time_utc >= signal.target_time else "entry_quote_exceeds_max_delay"
            skipped.append(SkippedSignal(signal.signal_id, signal.decision_time, reason))
            ledger.append({"time_utc": _iso(signal.decision_time), "event": "signal_unpriced", "signal_id": signal.signal_id, "reason": reason})
            continue

        exit_index = bisect_left(tick_times, signal.target_time, lo=entry_index + 1)
        if exit_index >= len(tick_rows):
            exit_index = None
        if exit_index is None:
            last_tick = tick_rows[-1]
            swap_at_last = _swap_through(signal, entry_tick, last_tick, config)
            open_positions.append(
                _open_position(signal, entry_tick, last_tick, config, balance, swap_at_last)
            )
            for mark in tick_rows[entry_index:]:
                accrued_swap = _swap_through(signal, entry_tick, mark, config)
                equity = _mark_equity(signal, entry_tick, mark, config, balance, accrued_swap, is_exit=False)
                equity_curve.append({"time_utc": _iso(mark.time_utc), "equity_usd": str(equity), "event": "quote_mark", "signal_id": signal.signal_id, "source_seq": mark.source_seq})
            skipped.append(SkippedSignal(signal.signal_id, signal.decision_time, "no_exit_quote_at_or_after_target"))
            ledger.append({"time_utc": _iso(signal.decision_time), "event": "position_open_unresolved", "signal_id": signal.signal_id, "reason": "no_exit_quote_at_or_after_target"})
            unresolved_open = True
            occupied_until = None
            continue

        exit_tick = tick_rows[exit_index]
        exit_delay = (exit_tick.time_utc - signal.target_time).total_seconds()
        if exit_delay > config.max_exit_delay_seconds:
            swap_at_last = _swap_through(signal, entry_tick, tick_rows[exit_index], config)
            open_positions.append(
                _open_position(signal, entry_tick, tick_rows[exit_index], config, balance, swap_at_last)
            )
            for mark in tick_rows[entry_index : exit_index + 1]:
                accrued_swap = _swap_through(signal, entry_tick, mark, config)
                equity = _mark_equity(signal, entry_tick, mark, config, balance, accrued_swap, is_exit=False)
                equity_curve.append({"time_utc": _iso(mark.time_utc), "equity_usd": str(equity), "event": "quote_mark", "signal_id": signal.signal_id, "source_seq": mark.source_seq})
            skipped.append(SkippedSignal(signal.signal_id, signal.decision_time, "exit_quote_exceeds_max_delay"))
            ledger.append({"time_utc": _iso(signal.decision_time), "event": "position_open_unresolved", "signal_id": signal.signal_id, "reason": "exit_quote_exceeds_max_delay"})
            unresolved_open = True
            occupied_until = None
            continue

        trade, rollover_events = _build_trade(signal, entry_tick, exit_tick, config)
        prior_balance = balance
        for mark in tick_rows[entry_index : exit_index + 1]:
            accrued_swap = _swap_through(signal, entry_tick, mark, config)
            is_exit = mark.source_seq == exit_tick.source_seq and mark.time_utc == exit_tick.time_utc
            equity = _mark_equity(signal, entry_tick, mark, config, balance, accrued_swap, is_exit=is_exit, trade=trade if is_exit else None)
            equity_curve.append({"time_utc": _iso(mark.time_utc), "equity_usd": str(equity), "event": "trade_exit" if is_exit else "quote_mark", "signal_id": signal.signal_id, "source_seq": mark.source_seq})
            ledger.append({"time_utc": _iso(mark.time_utc), "event": "trade_exit" if is_exit else "quote_mark", "signal_id": signal.signal_id, "side": signal.side, "balance_usd": str(balance), "equity_usd": str(equity), "source_seq": mark.source_seq})
        balance = prior_balance + trade.net_pnl_usd
        trades.append(trade)
        occupied_until = exit_tick.time_utc
        balance_curve.append({"time_utc": _iso(exit_tick.time_utc), "balance_usd": str(balance), "event": "trade_exit", "signal_id": signal.signal_id})
        # The closing mark uses realized balance exactly, after spread and all explicit costs.
        equity_curve[-1] = {"time_utc": _iso(exit_tick.time_utc), "equity_usd": str(balance), "event": "trade_exit", "signal_id": signal.signal_id, "source_seq": exit_tick.source_seq}
        ledger.append({"time_utc": _iso(exit_tick.time_utc), "event": "balance_update", "signal_id": signal.signal_id, "balance_before_usd": str(prior_balance), "balance_after_usd": str(balance), "net_pnl_usd": str(trade.net_pnl_usd), "rollover_count": len(rollover_events)})

    if not unresolved_open:
        end_time = period_end
        last_equity_time = _utc(equity_curve[-1]["time_utc"], "last equity timestamp")
        if end_time < last_equity_time:
            raise ReplayInputError("quote events extend beyond the configured half-open replay interval")
        equity_curve.append({"time_utc": _iso(end_time), "equity_usd": str(balance), "event": "period_end_flat", "signal_id": ""})
    else:
        # No forward fill to the requested period end for an unresolved open position.
        end_time = _utc(equity_curve[-1]["time_utc"], "last equity timestamp")

    metrics = _summarize(
        trades,
        skipped,
        open_positions,
        balance_curve,
        equity_curve,
        config,
        period_start,
        end_time,
        signal_rows,
    )
    return ReplayResult(
        trades=tuple(trades),
        skipped_signals=tuple(skipped),
        open_positions=tuple(open_positions),
        balance_curve=tuple(balance_curve),
        equity_curve=tuple(equity_curve),
        ledger=tuple(ledger),
        metrics=metrics,
    )


def _build_trade(signal: Signal, entry: Tick, exit_: Tick, config: ReplayConfig) -> tuple[Trade, tuple[datetime, ...]]:
    direction = Decimal("1") if signal.side == "long" else Decimal("-1")
    entry_raw = entry.ask if signal.side == "long" else entry.bid
    exit_raw = exit_.bid if signal.side == "long" else exit_.ask
    midpoint_entry = (entry.bid + entry.ask) / Decimal("2")
    midpoint_exit = (exit_.bid + exit_.ask) / Decimal("2")
    midpoint_pnl = direction * (midpoint_exit - midpoint_entry) * config.base_units
    spread_cost = config.base_units * ((entry.ask - entry.bid) + (exit_.ask - exit_.bid)) / Decimal("2")
    quote_pnl = midpoint_pnl - spread_cost
    expected_quote_pnl = direction * (exit_raw - entry_raw) * config.base_units
    if quote_pnl != expected_quote_pnl:
        raise AssertionError("internal spread decomposition does not reconcile to bid/ask P/L")
    slippage_cost = Decimal("2") * config.base_units * config.pip_size * config.slippage_pips_per_side
    scale = config.base_units / ONE_HUNDRED_K
    commission = Decimal("2") * config.commission_usd_per_100k_per_side * scale
    fee = Decimal("2") * config.fee_usd_per_100k_per_side * scale
    tax = config.tax_usd_per_round_turn
    rollovers = _rollovers_between(entry.time_utc, exit_.time_utc, config.rollover_time_utc, config.triple_swap_weekday)
    swap_rate = config.swap_usd_per_100k_per_rollover_long if signal.side == "long" else config.swap_usd_per_100k_per_rollover_short
    swap = sum(
        (swap_rate * scale * (Decimal("3") if event.weekday() == config.triple_swap_weekday else Decimal("1")) for event in rollovers),
        ZERO,
    )
    net = quote_pnl - slippage_cost - commission - fee - tax - swap
    trade = Trade(
        signal_id=signal.signal_id,
        side=signal.side,
        decision_time=signal.decision_time,
        target_time=signal.target_time,
        entry_time=entry.time_utc,
        exit_time=exit_.time_utc,
        entry_bid=entry.bid,
        entry_ask=entry.ask,
        exit_bid=exit_.bid,
        exit_ask=exit_.ask,
        entry_price=entry_raw,
        exit_price=exit_raw,
        base_units=config.base_units,
        midpoint_pnl_usd=midpoint_pnl,
        spread_cost_usd=spread_cost,
        quote_pnl_before_slippage_usd=quote_pnl,
        slippage_cost_usd=slippage_cost,
        commission_cost_usd=commission,
        fee_cost_usd=fee,
        tax_cost_usd=tax,
        swap_cost_usd=swap,
        net_pnl_usd=net,
        hold_seconds=Decimal(str((exit_.time_utc - entry.time_utc).total_seconds())),
        target_label=signal.target_label,
        prediction_correct=(signal.side == "long" and signal.target_label == "UP") or (signal.side == "short" and signal.target_label == "DOWN"),
    )
    return trade, rollovers


def _swap_through(signal: Signal, entry: Tick, current: Tick, config: ReplayConfig) -> Decimal:
    events = _rollovers_between(entry.time_utc, current.time_utc, config.rollover_time_utc, config.triple_swap_weekday)
    rate = config.swap_usd_per_100k_per_rollover_long if signal.side == "long" else config.swap_usd_per_100k_per_rollover_short
    scale = config.base_units / ONE_HUNDRED_K
    return sum((rate * scale * (Decimal("3") if event.weekday() == config.triple_swap_weekday else Decimal("1")) for event in events), ZERO)


def _mark_equity(
    signal: Signal,
    entry: Tick,
    current: Tick,
    config: ReplayConfig,
    balance: Decimal,
    swap_accrued: Decimal,
    *,
    is_exit: bool,
    trade: Trade | None = None,
) -> Decimal:
    direction = Decimal("1") if signal.side == "long" else Decimal("-1")
    entry_price = entry.ask if signal.side == "long" else entry.bid
    liquidation_price = current.bid if signal.side == "long" else current.ask
    raw_mark = direction * (liquidation_price - entry_price) * config.base_units
    entry_slippage = config.base_units * config.pip_size * config.slippage_pips_per_side
    exit_slippage = config.base_units * config.pip_size * config.slippage_pips_per_side
    scale = config.base_units / ONE_HUNDRED_K
    round_turn_commission = Decimal("2") * config.commission_usd_per_100k_per_side * scale
    round_turn_fee = Decimal("2") * config.fee_usd_per_100k_per_side * scale
    if is_exit and trade is not None:
        return balance + trade.net_pnl_usd
    return balance + raw_mark - entry_slippage - exit_slippage - round_turn_commission - round_turn_fee - config.tax_usd_per_round_turn - swap_accrued


def _open_position(
    signal: Signal,
    entry: Tick,
    last: Tick | None,
    config: ReplayConfig,
    balance: Decimal,
    swap_accrued: Decimal,
) -> OpenPosition:
    if last is None:
        bid = ask = unrealized = None
        mark_time = None
    else:
        bid, ask, mark_time = last.bid, last.ask, last.time_utc
        direction = Decimal("1") if signal.side == "long" else Decimal("-1")
        entry_price = entry.ask if signal.side == "long" else entry.bid
        liquidation = last.bid if signal.side == "long" else last.ask
        unrealized = direction * (liquidation - entry_price) * config.base_units
        unrealized -= config.base_units * config.pip_size * config.slippage_pips_per_side
        unrealized -= config.base_units * config.pip_size * config.slippage_pips_per_side
        scale = config.base_units / ONE_HUNDRED_K
        unrealized -= Decimal("2") * (config.commission_usd_per_100k_per_side + config.fee_usd_per_100k_per_side) * scale
        unrealized -= config.tax_usd_per_round_turn
        unrealized -= swap_accrued
    return OpenPosition(
        signal_id=signal.signal_id,
        side=signal.side,
        decision_time=signal.decision_time,
        target_time=signal.target_time,
        entry_time=entry.time_utc,
        last_mark_time=mark_time,
        entry_bid=entry.bid,
        entry_ask=entry.ask,
        last_bid=bid,
        last_ask=ask,
        unrealized_liquidation_pnl_usd=unrealized,
    )


def _summarize(
    trades: Sequence[Trade],
    skipped: Sequence[SkippedSignal],
    open_positions: Sequence[OpenPosition],
    balance_curve: Sequence[Mapping[str, Any]],
    equity_curve: Sequence[Mapping[str, Any]],
    config: ReplayConfig,
    period_start: datetime,
    period_end: datetime,
    signals: Sequence[Signal],
) -> dict[str, Any]:
    net_values = [trade.net_pnl_usd for trade in trades]
    wins = [value for value in net_values if value > ZERO]
    losses = [value for value in net_values if value < ZERO]
    gross_wins = sum(wins, ZERO)
    gross_losses = sum((abs(value) for value in losses), ZERO)
    if gross_losses > ZERO:
        pf_value = str(gross_wins / gross_losses)
        pf_status = "defined"
    elif gross_wins > ZERO:
        pf_value = None
        pf_status = "no_gross_losses"
    else:
        pf_value = None
        pf_status = "no_closed_trade_pnl"
    win_rate = (Decimal(len(wins)) / Decimal(len(net_values))) if net_values else ZERO
    avg_win = sum(wins, ZERO) / Decimal(len(wins)) if wins else None
    avg_loss = sum(losses, ZERO) / Decimal(len(losses)) if losses else None
    expectancy = sum(net_values, ZERO) / Decimal(len(net_values)) if net_values else None
    streak = max_streak = 0
    for value in net_values:
        if value < ZERO:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    closed_pnl = sum(net_values, ZERO)
    final_balance = config.starting_balance_usd + closed_pnl
    balance_points = [{"time_utc": row["time_utc"], "value_usd": Decimal(row["balance_usd"])} for row in balance_curve]
    equity_points = [{"time_utc": row["time_utc"], "value_usd": Decimal(row["equity_usd"])} for row in equity_curve]
    balance_dd = _metrics_drawdown(balance_points, period_end)
    equity_dd = _metrics_drawdown(equity_points, period_end)
    classifier_rows = [signal for signal in signals if signal.target_label in {"UP", "DOWN"}]
    classifier_correct = sum(1 for signal in classifier_rows if (signal.side == "long" and signal.target_label == "UP") or (signal.side == "short" and signal.target_label == "DOWN"))
    classifier_accuracy = Decimal(classifier_correct) / Decimal(len(classifier_rows)) if classifier_rows else ZERO
    duration = Decimal(str((period_end - period_start).total_seconds()))
    market_seconds = sum((trade.hold_seconds for trade in trades), ZERO)
    if open_positions and open_positions[0].last_mark_time:
        market_seconds += Decimal(str((open_positions[0].last_mark_time - open_positions[0].entry_time).total_seconds()))
    by_side: dict[str, Any] = {}
    for side in ("long", "short"):
        side_trades = [trade for trade in trades if trade.side == side]
        by_side[side] = {
            "closed_trade_count": len(side_trades),
            "net_pnl_usd": str(sum((trade.net_pnl_usd for trade in side_trades), ZERO)),
            "wins": sum(1 for trade in side_trades if trade.net_pnl_usd > ZERO),
            "losses": sum(1 for trade in side_trades if trade.net_pnl_usd < ZERO),
        }
    costs = {
        "midpoint_pnl_usd": str(sum((trade.midpoint_pnl_usd for trade in trades), ZERO)),
        "spread_cost_usd": str(sum((trade.spread_cost_usd for trade in trades), ZERO)),
        "slippage_cost_usd": str(sum((trade.slippage_cost_usd for trade in trades), ZERO)),
        "commission_cost_usd": str(sum((trade.commission_cost_usd for trade in trades), ZERO)),
        "fee_cost_usd": str(sum((trade.fee_cost_usd for trade in trades), ZERO)),
        "tax_cost_usd": str(sum((trade.tax_cost_usd for trade in trades), ZERO)),
        "swap_cost_usd": str(sum((trade.swap_cost_usd for trade in trades), ZERO)),
        "spread_subtracted_once": True,
        "spread_method": "bid/ask execution already embeds spread; spread_cost is a midpoint decomposition, not a second P/L deduction",
    }
    max_eq_start = _utc(equity_dd["peak_time_utc"], "dd peak") if equity_dd["peak_time_utc"] else None
    max_eq_end = _utc(equity_dd["trough_time_utc"], "dd trough") if equity_dd["trough_time_utc"] else None
    dd_trade_ids = [
        trade.signal_id
        for trade in trades
        if max_eq_start and max_eq_end and trade.entry_time <= max_eq_end and trade.exit_time >= max_eq_start
    ]
    last_final_balance = final_balance
    metrics = {
        "run_mode": config.run_mode,
        "strategy_contract": {
            "partition": "final_confirmation",
            "lower_probability_threshold_inclusive": str(config.lower_threshold),
            "upper_probability_threshold_inclusive": str(config.upper_threshold),
            "maximum_open_positions": 1,
            "starting_balance_usd": str(config.starting_balance_usd),
            "base_units_per_position": str(config.base_units),
            "signal_horizon_seconds": 3600,
            "max_entry_delay_seconds": config.max_entry_delay_seconds,
            "max_exit_delay_seconds": config.max_exit_delay_seconds,
            "minimum_hold_seconds": max(0, 3600 - config.max_entry_delay_seconds),
            "maximum_hold_seconds_exclusive": 3600 + config.max_exit_delay_seconds,
            "stop_loss": None,
            "take_profit": None,
            "exit_rule": "first executable quote at or after target availability, within configured delay",
            "market_evaluation_status": "synthetic_fixture_only" if config.run_mode == "synthetic_fixture" else "retrospective_strategy_evaluation_not_independent_confirmation",
        },
        "headline_metrics_ready": not open_positions and not any(
            item.reason != "position_open_at_decision" for item in skipped
        ),
        "period_start_utc": _iso(period_start),
        "period_end_utc": _iso(period_end),
        "starting_balance_usd": str(config.starting_balance_usd),
        "final_balance_usd": str(last_final_balance),
        "account_currency": "USD",
        "base_units_per_position": str(config.base_units),
        "model_sha256": config.expected_model_sha256,
        "classifier": {
            "coverage": "candidate signals read from saved frozen predictions only",
            "covered_signal_count": len(classifier_rows),
            "accuracy_on_frozen_candidate_signals": str(classifier_accuracy),
            "calibration": "not asserted; raw probabilities remain uncalibrated",
        },
        "trading": {
            "closed_trade_count": len(trades),
            "open_trade_count": len(open_positions),
            "skipped_signal_count": len(skipped),
            "net_closed_pnl_usd": str(closed_pnl),
            "return_pct_on_starting_balance": str((closed_pnl / config.starting_balance_usd * Decimal("100")) if config.starting_balance_usd else ZERO),
            "win_rate_closed_net": str(win_rate),
            "gross_wins_usd": str(gross_wins),
            "gross_losses_abs_usd": str(gross_losses),
            "profit_factor": pf_value,
            "profit_factor_status": pf_status,
            "average_win_usd": str(avg_win) if avg_win is not None else None,
            "average_loss_usd": str(avg_loss) if avg_loss is not None else None,
            "expectancy_per_closed_trade_usd": str(expectancy) if expectancy is not None else None,
            "maximum_consecutive_net_losses": max_streak,
            "time_in_market_seconds": str(market_seconds),
            "time_in_market_pct_of_period": str((market_seconds / duration * Decimal("100")) if duration > ZERO else ZERO),
            "long_short": by_side,
            "open_positions_excluded_from_win_rate_and_profit_factor": True,
        },
        "costs": costs,
        "balance_drawdown": balance_dd,
        "equity_drawdown": equity_dd,
        "largest_equity_drawdown_trade_ids": dd_trade_ids,
        "equity_curve_sampling": _curve_gaps(equity_curve),
        "integrity": {
            "missing_quote_signal_count": sum(1 for item in skipped if item.reason != "position_open_at_decision"),
            "unresolved_open_positions": len(open_positions),
            "classification_accuracy_is_separate_from_trade_win_rate": True,
            "final_classifier_not_rerun": True,
            "model_or_scaler_retrained": False,
        },
    }
    return metrics
