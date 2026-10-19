"""Read-only one-lot EURUSD reversal replay from user-supplied CSV decisions and bid/ask ticks.

This public replay engine consumes only already-saved candidate decisions and
validated quote history. It has no model-fitting or MetaTrader connection code
and cannot submit orders. Same-side signals are no-ops that keep the original
target; ABSTAIN holds an open position to that target.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from itertools import groupby
from typing import Any, Callable, Iterable, Literal, Mapping

from ..engine import ReplayInputError, Tick


UTC = timezone.utc
ZERO = Decimal("0")
STARTING_BALANCE_USD = Decimal("10000.00")
BASE_UNITS_EUR = Decimal("100000")
SIMULATION_LEVERAGE = 200
TARGET_DURATION = timedelta(hours=1)
MAX_ENTRY_DELAY_SECONDS = 60
SAME_SIDE_POLICY = "keep_original_target"
ABSTAIN_POLICY = "hold"
MONEY_QUANTUM = Decimal("0.01")


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
    if value.microsecond == 0:
        precision = "seconds"
    elif value.microsecond % 1000 == 0:
        precision = "milliseconds"
    else:
        precision = "microseconds"
    return value.astimezone(UTC).isoformat(timespec=precision).replace("+00:00", "Z")


def _money(value: Any, name: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ReplayInputError(f"missing or invalid {name}: {value!r}")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ReplayInputError(f"invalid {name}: {value!r}") from exc
    if not number.is_finite():
        raise ReplayInputError(f"nonfinite {name}: {value!r}")
    return number


@dataclass(frozen=True)
class Decision:
    """A frozen, observable action only; future target labels are excluded."""

    signal_id: str
    decision_time: datetime
    action: Literal["long", "short", "ABSTAIN"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_time", _utc(self.decision_time, "decision_time"))
        if not self.signal_id:
            raise ReplayInputError("signal_id cannot be empty")
        if self.action not in {"long", "short", "ABSTAIN"}:
            raise ReplayInputError(f"unknown decision action: {self.action!r}")


@dataclass(frozen=True)
class ReplayConfig:
    """The user-approved scenario is fixed; action policies have no defaults."""

    same_side_policy: str
    abstain_policy: str
    period_start_utc: datetime
    period_end_utc: datetime
    starting_balance_usd: Decimal = STARTING_BALANCE_USD
    base_units_eur: Decimal = BASE_UNITS_EUR
    leverage: int = SIMULATION_LEVERAGE
    max_entry_delay_seconds: int = MAX_ENTRY_DELAY_SECONDS

    def __post_init__(self) -> None:
        start = _utc(self.period_start_utc, "period_start_utc")
        end = _utc(self.period_end_utc, "period_end_utc")
        object.__setattr__(self, "period_start_utc", start)
        object.__setattr__(self, "period_end_utc", end)
        object.__setattr__(self, "starting_balance_usd", _money(self.starting_balance_usd, "starting_balance_usd"))
        object.__setattr__(self, "base_units_eur", _money(self.base_units_eur, "base_units_eur"))
        if self.same_side_policy != SAME_SIDE_POLICY:
            raise ReplayInputError(f"same_side_policy must be {SAME_SIDE_POLICY!r}")
        if self.abstain_policy != ABSTAIN_POLICY:
            raise ReplayInputError(f"abstain_policy must be {ABSTAIN_POLICY!r}")
        if start >= end:
            raise ReplayInputError("period must be a positive half-open UTC interval")
        if self.starting_balance_usd != STARTING_BALANCE_USD:
            raise ReplayInputError("V2 starting balance is fixed at USD 10,000")
        if self.base_units_eur != BASE_UNITS_EUR:
            raise ReplayInputError("V2 position size is fixed at 100,000 EUR (one standard lot)")
        if self.leverage != SIMULATION_LEVERAGE:
            raise ReplayInputError("V2 simulation leverage is fixed at 1:200")
        if self.max_entry_delay_seconds < 0:
            raise ReplayInputError("max_entry_delay_seconds must be nonnegative")


@dataclass(frozen=True)
class PositionRecord:
    signal_id: str
    side: str
    decision_time: datetime
    target_time: datetime
    entry_time: datetime
    entry_source_seq: int
    entry_bid: Decimal
    entry_ask: Decimal
    entry_price: Decimal
    base_units: Decimal
    exit_time: datetime | None
    exit_source_seq: int | None
    exit_bid: Decimal | None
    exit_ask: Decimal | None
    exit_price: Decimal | None
    quote_pnl_usd: Decimal | None
    exit_reason: str | None
    hold_seconds: Decimal | None
    target_lateness_seconds: Decimal | None
    utc_midnight_boundaries_crossed: int


@dataclass
class _LivePosition:
    signal_id: str
    side: str
    decision_time: datetime
    target_time: datetime
    entry_tick: Tick


@dataclass
class _Pending:
    kind: Literal["entry", "reversal", "abstain_close"]
    signal_id: str
    action: str
    decision_time: datetime
    target_time: datetime
    action_row: dict[str, Any]


@dataclass(frozen=True)
class ReplayResult:
    positions: tuple[PositionRecord, ...]
    open_position: PositionRecord | None
    signal_actions: tuple[dict[str, Any], ...]
    balance_curve: tuple[dict[str, Any], ...]
    equity_curve: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]


def _liquidation_pnl(position: _LivePosition, tick: Tick, units: Decimal) -> Decimal:
    if position.side == "long":
        return (tick.bid - position.entry_tick.ask) * units
    return (position.entry_tick.bid - tick.ask) * units


def _entry_price(side: str, tick: Tick) -> Decimal:
    return tick.ask if side == "long" else tick.bid


def _exit_price(side: str, tick: Tick) -> Decimal:
    return tick.bid if side == "long" else tick.ask


def _utc_midnights_crossed(start: datetime, end: datetime) -> int:
    start_day = start.date()
    end_day = end.date()
    return max(0, (end_day - start_day).days)


def _seconds(delta: timedelta) -> Decimal:
    return (
        Decimal(delta.days * 86_400 + delta.seconds)
        + Decimal(delta.microseconds) / Decimal("1000000")
    )


def _currency_amount(value: Decimal) -> Decimal:
    """Normalize USD ledger amounts to cents after using source bid/ask prices."""
    return value.quantize(MONEY_QUANTUM)


def _position_record(
    position: _LivePosition,
    exit_tick: Tick | None,
    exit_reason: str | None,
    config: ReplayConfig,
    *,
    mark_tick: Tick | None = None,
) -> PositionRecord:
    end_tick = exit_tick or mark_tick
    if end_tick is None:
        raise AssertionError("a position record requires an exit or mark quote")
    if exit_tick is None:
        pnl = None
        exit_time = exit_seq = exit_bid = exit_ask = exit_price = None
        hold = _seconds(end_tick.time_utc - position.entry_tick.time_utc)
        lateness = None
        reason = "open_at_period_end"
    else:
        pnl = _currency_amount(_liquidation_pnl(position, exit_tick, config.base_units_eur))
        exit_time = exit_tick.time_utc
        exit_seq = exit_tick.source_seq
        exit_bid = exit_tick.bid
        exit_ask = exit_tick.ask
        exit_price = _exit_price(position.side, exit_tick)
        hold = _seconds(exit_tick.time_utc - position.entry_tick.time_utc)
        lateness = max(ZERO, _seconds(exit_tick.time_utc - position.target_time))
        reason = exit_reason
    return PositionRecord(
        signal_id=position.signal_id,
        side=position.side,
        decision_time=position.decision_time,
        target_time=position.target_time,
        entry_time=position.entry_tick.time_utc,
        entry_source_seq=position.entry_tick.source_seq,
        entry_bid=position.entry_tick.bid,
        entry_ask=position.entry_tick.ask,
        entry_price=_entry_price(position.side, position.entry_tick),
        base_units=config.base_units_eur,
        exit_time=exit_time,
        exit_source_seq=exit_seq,
        exit_bid=exit_bid,
        exit_ask=exit_ask,
        exit_price=exit_price,
        quote_pnl_usd=pnl,
        exit_reason=reason,
        hold_seconds=hold,
        target_lateness_seconds=lateness,
        utc_midnight_boundaries_crossed=_utc_midnights_crossed(
            position.entry_tick.time_utc, end_tick.time_utc
        ),
    )


def _append_equity(
    *,
    time_utc: datetime,
    equity: Decimal,
    event: str,
    signal_id: str = "",
    source_seq: int | None = None,
    subevent: int = 0,
) -> dict[str, Any]:
    return {
        "time_utc": _iso(time_utc),
        "equity_usd": str(equity),
        "event": event,
        "signal_id": signal_id,
        "source_seq": source_seq,
        "subevent_order": subevent,
    }


def replay(
    decisions: Iterable[Decision],
    ticks: Iterable[Tick],
    config: ReplayConfig,
    *,
    equity_sink: Callable[[Mapping[str, Any]], None] | None = None,
    capture_equity_curve: bool = True,
) -> ReplayResult:
    """Run the single-position quote replay with no interpolation or lookahead.

    The stream is grouped by timestamp so every source row tied at a decision
    time is processed first. A direction is entered only on a strictly later
    quote. A target exit uses the first quote at or after target_time, even if
    the market was closed for a weekend.
    """

    decision_rows = tuple(decisions)
    if any(not isinstance(row, Decision) for row in decision_rows):
        raise ReplayInputError("decisions must be Decision rows")
    if not decision_rows:
        raise ReplayInputError("frozen decision set is empty")
    if tuple(sorted(decision_rows, key=lambda row: row.decision_time)) != decision_rows:
        raise ReplayInputError("decisions must be in chronological order")
    if len({row.decision_time for row in decision_rows}) != len(decision_rows):
        raise ReplayInputError("duplicate decision timestamp")
    if any(
        row.decision_time < config.period_start_utc or row.decision_time >= config.period_end_utc
        for row in decision_rows
    ):
        raise ReplayInputError("every decision must fall inside [period_start_utc, period_end_utc)")

    balance = config.starting_balance_usd
    balance_curve: list[dict[str, Any]] = [{
        "time_utc": _iso(config.period_start_utc),
        "balance_usd": str(balance),
        "event": "initial_balance",
    }]
    equity_curve: list[dict[str, Any]] = []
    equity_observations = 0

    def emit_equity(row: dict[str, Any]) -> None:
        nonlocal equity_observations
        equity_observations += 1
        if capture_equity_curve:
            equity_curve.append(row)
        if equity_sink is not None:
            equity_sink(row)

    emit_equity(_append_equity(
        time_utc=config.period_start_utc,
        equity=balance,
        event="initial_balance",
    ))

    closed_positions: list[PositionRecord] = []
    active: _LivePosition | None = None
    active_last_tick: Tick | None = None
    pending: _Pending | None = None
    signal_actions: list[dict[str, Any]] = []
    decision_index = 0
    previous_tick_time: datetime | None = None
    previous_source_seq: int | None = None
    source_rows = valid_quotes = 0
    max_active = 0
    peak_equity = balance
    max_drawdown = ZERO
    max_drawdown_time: datetime | None = None
    final_tick: Tick | None = None

    def process_decision(decision: Decision) -> None:
        nonlocal pending, active
        action_row: dict[str, Any] = {
            "signal_id": decision.signal_id,
            "decision_time_utc": _iso(decision.decision_time),
            "action": decision.action,
            "position_side_at_decision": active.side if active else "flat",
            "disposition": "",
            "execution_time_utc": "",
            "execution_source_seq": "",
            "related_position_signal_id": active.signal_id if active else "",
            "policy": "",
        }
        signal_actions.append(action_row)
        if pending is not None:
            raise ReplayInputError(
                "multiple action decisions occurred before the next executable quote; "
                f"unresolved intent {pending.signal_id!r} precedes {decision.signal_id!r}"
            )
        if active is None:
            if decision.action == "ABSTAIN":
                action_row["disposition"] = "abstain_while_flat"
                action_row["policy"] = ABSTAIN_POLICY
            else:
                action_row["disposition"] = "entry_queued"
                action_row["policy"] = "first_strictly_later_quote_within_60_seconds"
                pending = _Pending(
                    "entry",
                    decision.signal_id,
                    decision.action,
                    decision.decision_time,
                    decision.decision_time + TARGET_DURATION,
                    action_row,
                )
            return

        if decision.action == "ABSTAIN":
            action_row["disposition"] = "abstain_hold"
            action_row["policy"] = ABSTAIN_POLICY
        elif decision.action == active.side:
            action_row["disposition"] = "same_side_noop"
            action_row["policy"] = SAME_SIDE_POLICY
        else:
            action_row["disposition"] = "reversal_queued"
            action_row["policy"] = "close_then_open_one_lot_at_same_next_quote"
            pending = _Pending(
                "reversal",
                decision.signal_id,
                decision.action,
                decision.decision_time,
                decision.decision_time + TARGET_DURATION,
                action_row,
            )

    def track_equity(equity: Decimal, at: datetime) -> None:
        nonlocal peak_equity, max_drawdown, max_drawdown_time
        if equity > peak_equity:
            peak_equity = equity
        drawdown = peak_equity - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_time = at

    def append_mark(tick: Tick, event: str, *, subevent: int = 0) -> Decimal:
        if active is None:
            raise AssertionError("equity mark requires an active position")
        equity = _currency_amount(balance + _liquidation_pnl(active, tick, config.base_units_eur))
        emit_equity(_append_equity(
            time_utc=tick.time_utc,
            equity=equity,
            event=event,
            signal_id=active.signal_id,
            source_seq=tick.source_seq,
            subevent=subevent,
        ))
        track_equity(equity, tick.time_utc)
        return equity

    def close_active(tick: Tick, reason: str, *, subevent: int = 0) -> None:
        nonlocal active, active_last_tick, balance
        if active is None:
            raise AssertionError("close event requires an active position")
        prior = balance
        record = _position_record(active, tick, reason, config)
        if record.quote_pnl_usd is None:
            raise AssertionError("closed position must record quote P/L")
        balance += record.quote_pnl_usd
        closed_positions.append(record)
        balance_curve.append({
            "time_utc": _iso(tick.time_utc),
            "balance_usd": str(balance),
            "event": "position_close",
            "signal_id": active.signal_id,
            "source_seq": tick.source_seq,
            "balance_before_usd": str(prior),
            "quote_pnl_usd": str(record.quote_pnl_usd),
        })
        emit_equity(_append_equity(
            time_utc=tick.time_utc,
            equity=balance,
            event=reason,
            signal_id=active.signal_id,
            source_seq=tick.source_seq,
            subevent=subevent,
        ))
        track_equity(balance, tick.time_utc)
        if active_last_tick is None or tick.time_utc >= active_last_tick.time_utc:
            active_last_tick = tick
        active = None

    def open_active(tick: Tick, pending_action: _Pending, *, subevent: int = 0) -> None:
        nonlocal active, active_last_tick, max_active
        delay = _seconds(tick.time_utc - pending_action.decision_time)
        if tick.time_utc <= pending_action.decision_time:
            raise AssertionError("entry quote must be strictly later than its decision")
        if pending_action.kind == "entry" and delay > config.max_entry_delay_seconds:
            pending_action.action_row["disposition"] = "entry_unfilled_delay_over_60s"
            pending_action.action_row["execution_time_utc"] = _iso(tick.time_utc) or ""
            pending_action.action_row["execution_source_seq"] = tick.source_seq
            return
        active = _LivePosition(
            pending_action.signal_id,
            pending_action.action,
            pending_action.decision_time,
            pending_action.target_time,
            tick,
        )
        active_last_tick = tick
        max_active = max(max_active, 1)
        pending_action.action_row["disposition"] = (
            "entered" if pending_action.kind == "entry" else "reversed_into"
        )
        pending_action.action_row["execution_time_utc"] = _iso(tick.time_utc) or ""
        pending_action.action_row["execution_source_seq"] = tick.source_seq
        pending_action.action_row["entry_delay_seconds"] = str(delay)
        append_mark(tick, "position_open_mark", subevent=subevent)

    tick_iterator = iter(ticks)
    for tick_time, grouped_ticks in groupby(tick_iterator, key=lambda row: row.time_utc):
        group = tuple(grouped_ticks)
        if not group:
            continue
        # All quote rows with t < decision time are causally earlier. A signal
        # exactly at t is deliberately handled only after every tied quote row.
        while decision_index < len(decision_rows) and decision_rows[decision_index].decision_time < tick_time:
            process_decision(decision_rows[decision_index])
            decision_index += 1

        for tick in group:
            if not isinstance(tick, Tick):
                raise ReplayInputError("ticks must be Tick rows")
            source_rows += 1
            if previous_tick_time is not None and (
                tick.time_utc < previous_tick_time
                or (tick.time_utc == previous_tick_time and previous_source_seq is not None and tick.source_seq <= previous_source_seq)
            ):
                raise ReplayInputError("tick stream must be chronological and preserve source_seq order")
            if previous_source_seq is not None and tick.source_seq <= previous_source_seq:
                raise ReplayInputError("tick source_seq values must be globally unique and increasing")
            if tick.time_utc < config.period_start_utc or tick.time_utc >= config.period_end_utc:
                raise ReplayInputError("tick stream must be inside [period_start_utc, period_end_utc)")
            previous_tick_time = tick.time_utc
            previous_source_seq = tick.source_seq
            final_tick = tick
            valid_quotes += 1

            current_pending = pending
            if active is not None and current_pending is not None:
                if tick.time_utc <= current_pending.decision_time:
                    continue
                close_active(
                    tick,
                    "reversal" if current_pending.kind == "reversal" else "abstain_close",
                    subevent=0,
                )
                pending = None
                if current_pending.kind == "reversal":
                    open_active(tick, current_pending, subevent=1)
                continue

            if active is not None and tick.time_utc >= active.target_time:
                close_active(tick, "target")
                continue

            if active is None and current_pending is not None:
                if tick.time_utc <= current_pending.decision_time:
                    continue
                pending = None
                open_active(tick, current_pending)
                continue

            if active is not None:
                active_last_tick = tick
                append_mark(tick, "quote_mark")

        while decision_index < len(decision_rows) and decision_rows[decision_index].decision_time == tick_time:
            process_decision(decision_rows[decision_index])
            decision_index += 1

    if source_rows == 0:
        raise ReplayInputError("no valid bid/ask ticks in the half-open replay interval")
    while decision_index < len(decision_rows):
        process_decision(decision_rows[decision_index])
        decision_index += 1

    open_record: PositionRecord | None = None
    if active is not None:
        if active_last_tick is None:
            raise AssertionError("open position has no mark quote")
        open_record = _position_record(active, None, None, config, mark_tick=active_last_tick)
        if active is not None:
            # Keep the latest observed liquidation equity, with no forward fill.
            final_equity = _currency_amount(
                balance + _liquidation_pnl(active, active_last_tick, config.base_units_eur)
            )
        else:
            final_equity = balance
    else:
        final_equity = balance
        emit_equity(_append_equity(
            time_utc=config.period_end_utc,
            equity=balance,
            event="period_end_flat",
        ))

    if active is not None:
        all_positions = tuple((*closed_positions, open_record))
    else:
        all_positions = tuple(closed_positions)
    closed_quote_pnl = sum((row.quote_pnl_usd or ZERO for row in closed_positions), ZERO)
    metrics = {
        "scenario": "EURUSD one standard lot, one position maximum, USD 10,000 starting balance",
        "period_utc_half_open": [_iso(config.period_start_utc), _iso(config.period_end_utc)],
        "starting_balance_usd": str(config.starting_balance_usd),
        "base_units_eur": str(config.base_units_eur),
        "lots": "1.00",
        "simulation_leverage": "1:200 (user-directed scenario override)",
        "leverage_override_note": "1:200 is a hypothetical scenario input; it is not a broker account statement.",
        "margin_status": "not_modeled_historical_margin_rules_unavailable",
        "swap_status": "unknown_historical_schedule_and_rates",
        "swap_charged_usd": None,
        "net_profit_factor": None,
        "commission_fee_tax_assumption": "not included in this simplified quote replay",
        "slippage_assumption": "zero additional slippage in virtual fills",
        "quote_pnl_scope": "closed_legs_pre_swap_quote_only",
        "closed_position_count": len(closed_positions),
        "open_position_at_period_end": active is not None,
        "source_quote_rows_consumed": source_rows,
        "valid_bid_ask_quotes_consumed": valid_quotes,
        "equity_curve_observations": equity_observations,
        "maximum_simultaneous_positions": max_active,
        "quote_pnl_closed_legs_usd": str(closed_quote_pnl),
        "realized_balance_usd": str(balance),
        "ending_equity_at_last_quote_usd": str(final_equity),
        "maximum_observed_equity_drawdown_pre_swap_usd": str(max_drawdown),
        "maximum_observed_equity_drawdown_time_utc": _iso(max_drawdown_time),
        "equity_curve_policy": "initial balance; each valid bid/ask quote while position is open; no interpolation; no forward fill over quote gaps",
        "target_exit_rule": "first valid quote at or after decision_time + 1 hour; weekend gaps are exited on first available quote after reopening",
        "reversal_rule": "at first quote strictly after opposite decision, close old leg then open opposite one-lot leg on that same quote",
        "same_side_policy": SAME_SIDE_POLICY,
        "abstain_policy": ABSTAIN_POLICY,
        "utc_midnight_crossings_are_not_broker_rollovers": True,
        "account_wide_net_profit_claim": False,
        "trading_profitability_test": False,
    }
    return ReplayResult(
        positions=all_positions,
        open_position=open_record,
        signal_actions=tuple(signal_actions),
        balance_curve=tuple(balance_curve),
        equity_curve=tuple(equity_curve),
        metrics=metrics,
    )
