"""Read-only MT5 deal-cost report with explicit account and time scope.

The report is deliberately period-based. It does not reconstruct position
lifetime P/L, include floating P/L, allocate account operations, or claim an
account-wide net result. All recorded money values retain their source signs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


UTC = timezone.utc
MONEY_FIELDS = ("profit", "commission", "swap", "fee")
REQUIRED_FIELDS = (
    "ticket",
    "order",
    "time",
    "time_msc",
    "type",
    "entry",
    "position_id",
    "symbol",
    *MONEY_FIELDS,
)
RAW_FIELDS = (
    "ticket",
    "order",
    "time",
    "time_msc",
    "time_utc",
    "type",
    "deal_type_label",
    "entry",
    "position_id",
    "symbol",
    "profit",
    "commission",
    "swap",
    "fee",
    "magic",
    "reason",
    "external_id",
    "volume",
    "price",
    "source_deal_id",
    "source_order_id",
    "source_position_id",
    "source_type",
    "source_entry_exit",
    "source_timestamp_utc",
    "source_price_open",
    "source_price_close",
    "source_volume_closed",
    "source_contract_size",
    "commission_source",
    "fee_source",
    "account_scope_id",
    "account_login",
    "account_server",
    "account_currency",
    "in_requested_interval",
    "report_partition",
)
DAILY_FIELDS = (
    "account_login",
    "account_server",
    "account_scope_id",
    "account_currency",
    "date_utc",
    "deal_count",
    "profit",
    "commission",
    "swap",
    "fee",
    "buy_sell_recorded_component_subtotal",
)

DEAL_TYPE_LABELS = {
    0: "BUY",
    1: "SELL",
    2: "BALANCE",
    3: "CREDIT",
    4: "CHARGE",
    5: "CORRECTION",
    6: "BONUS",
    7: "COMMISSION",
    8: "COMMISSION_DAILY",
    9: "COMMISSION_MONTHLY",
    10: "COMMISSION_AGENT_DAILY",
    11: "COMMISSION_AGENT_MONTHLY",
    12: "INTEREST",
    13: "BUY_CANCELED",
    14: "SELL_CANCELED",
    15: "DIVIDEND",
    16: "DIVIDEND_FRANKED",
    17: "TAX",
}


class HistoryFetchError(RuntimeError):
    """The terminal reported a failed history request (MT5 returned None)."""


class DealValidationError(ValueError):
    """A returned deal cannot be included without losing accounting evidence."""


@dataclass(frozen=True)
class AccountScope:
    """The one demo account and currency to which a report belongs."""

    login: str | None
    server: str | None
    currency: str
    trade_mode: str = "demo"
    scope_id: str | None = None

    def __post_init__(self) -> None:
        has_login = self.login is not None and bool(str(self.login).strip())
        has_server = self.server is not None and bool(str(self.server).strip())
        if has_login != has_server:
            raise ValueError("account login and server must be supplied together")
        if not has_login and not (self.scope_id and str(self.scope_id).strip()):
            raise ValueError(
                "account identity or an explicit report scope ID is required for isolation"
            )
        if not str(self.currency).strip():
            raise ValueError("account currency is required for isolation")
        if str(self.trade_mode).lower() != "demo":
            raise ValueError("this report only accepts a verified demo account")

    def as_dict(self) -> dict[str, str | None]:
        return {
            "login": str(self.login) if self.login is not None else None,
            "server": str(self.server) if self.server is not None else None,
            "currency": str(self.currency),
            "trade_mode": "demo",
            "scope_id": str(self.scope_id) if self.scope_id is not None else None,
        }


@dataclass(frozen=True)
class ReportBundle:
    raw_deals: tuple[dict[str, Any], ...]
    buy_sell_daily: tuple[dict[str, Any], ...]
    other_operations: tuple[dict[str, Any], ...]
    reconciliation: dict[str, Any]


def require_history_result(
    deals: Sequence[Any] | None,
    last_error: Any,
) -> tuple[Any, ...]:
    """Keep the MT5 error sentinel distinct from a valid empty history."""
    if deals is None:
        raise HistoryFetchError(
            f"history_deals_get failed (MT5 last_error={last_error!r})"
        )
    return tuple(deals)


def _as_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a timezone-aware datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset")
    return value.astimezone(UTC)


def _iso_utc(value: datetime) -> str:
    timespec = "milliseconds" if value.microsecond % 1000 == 0 else "microseconds"
    return value.isoformat(timespec=timespec).replace("+00:00", "Z")


def _source_hash() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _record_dict(record: Any, row_number: int) -> dict[str, Any]:
    if hasattr(record, "_asdict"):
        values = record._asdict()
    elif isinstance(record, Mapping):
        values = dict(record)
    else:
        raise DealValidationError(
            f"deal row {row_number}: expected a mapping or named tuple"
        )
    missing = [field for field in REQUIRED_FIELDS if field not in values]
    if missing:
        raise DealValidationError(
            f"deal row {row_number}: missing required fields {missing}"
        )
    return dict(values)


def _integer(value: Any, field: str, row_number: int) -> int:
    if isinstance(value, bool):
        raise DealValidationError(
            f"deal row {row_number}: {field} must be an integer, got {value!r}"
        )
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DealValidationError(
            f"deal row {row_number}: invalid {field} {value!r}"
        ) from exc
    if str(value).strip() not in (str(converted), f"{converted}.0"):
        raise DealValidationError(
            f"deal row {row_number}: {field} must be integral, got {value!r}"
        )
    return converted


def _money(value: Any, field: str, row_number: int) -> Decimal:
    if value is None or isinstance(value, bool):
        raise DealValidationError(
            f"deal row {row_number}: missing or invalid monetary field {field}={value!r}"
        )
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DealValidationError(
            f"deal row {row_number}: invalid monetary field {field}={value!r}"
        ) from exc
    if not amount.is_finite():
        raise DealValidationError(
            f"deal row {row_number}: nonfinite monetary field {field}={value!r}"
        )
    return amount


def _validate_record_scope(
    values: Mapping[str, Any], scope: AccountScope, row_number: int
) -> None:
    expected = {
        "account_login": scope.login,
        "account_server": scope.server,
        "account_scope_id": scope.scope_id,
        "account_currency": scope.currency,
    }
    for key, expected_value in expected.items():
        if key in values and str(values[key]) != str(expected_value):
            raise DealValidationError(
                f"deal row {row_number}: {key}={values[key]!r} does not match "
                f"the report account scope {expected_value!r}"
            )
    if "trade_mode" in values and str(values["trade_mode"]).lower() != "demo":
        raise DealValidationError(
            f"deal row {row_number}: source row is not marked as demo"
        )


def _normalize_record(
    record: Any,
    row_number: int,
    start_utc: datetime,
    end_utc: datetime,
    scope: AccountScope,
) -> tuple[dict[str, Any], str, bool, Decimal]:
    values = _record_dict(record, row_number)
    _validate_record_scope(values, scope, row_number)

    ticket = _integer(values["ticket"], "ticket", row_number)
    if ticket <= 0:
        raise DealValidationError(
            f"deal row {row_number}: ticket must be a positive deal identifier"
        )
    type_code = _integer(values["type"], "type", row_number)
    epoch_seconds = _integer(values["time"], "time", row_number)
    epoch_milliseconds = _integer(values["time_msc"], "time_msc", row_number)
    if epoch_seconds != epoch_milliseconds // 1000:
        raise DealValidationError(
            f"deal row {row_number}: time and time_msc disagree "
            f"({epoch_seconds} seconds vs {epoch_milliseconds} milliseconds)"
        )
    # Keep conversion integer-based so epoch magnitude cannot lose subsecond bits.
    epoch_seconds_from_ms, millisecond_remainder = divmod(epoch_milliseconds, 1000)
    timestamp = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
        seconds=epoch_seconds_from_ms,
        microseconds=millisecond_remainder * 1000,
    )
    in_interval = start_utc <= timestamp < end_utc
    amounts = {
        field: _money(values[field], field, row_number) for field in MONEY_FIELDS
    }
    component_sum = sum(amounts.values(), Decimal("0"))
    if not in_interval:
        partition = "outside_requested_interval"
    elif type_code in (0, 1):
        partition = "buy_sell_deal"
    else:
        partition = "other_operation"

    raw = {field: values.get(field, "") for field in RAW_FIELDS}
    raw.update(
        {
            "ticket": ticket,
            "order": values["order"],
            "time": epoch_seconds,
            "time_msc": epoch_milliseconds,
            "time_utc": _iso_utc(timestamp),
            "type": type_code,
            "deal_type_label": DEAL_TYPE_LABELS.get(
                type_code, f"UNKNOWN_TYPE_{type_code}"
            ),
            "entry": values["entry"],
            "position_id": values["position_id"],
            "symbol": "" if values["symbol"] is None else values["symbol"],
            "account_login": scope.login or "",
            "account_server": scope.server or "",
            "account_scope_id": scope.scope_id or "",
            "account_currency": scope.currency,
            "in_requested_interval": in_interval,
            "report_partition": partition,
        }
    )
    for field in MONEY_FIELDS:
        # Keep the numeric representation supplied by the source in the raw file.
        raw[field] = values[field]
    return raw, partition, in_interval, component_sum


def build_report(
    deals: Sequence[Any] | None,
    *,
    start: datetime,
    end: datetime,
    account_scope: AccountScope,
    provenance: Mapping[str, Any] | None = None,
) -> ReportBundle:
    """Validate MT5 deal rows and build the four requested report files' data.

    The reporting interval is always the half-open UTC interval ``[start, end)``.
    Source rows outside that interval are kept in ``raw_deals`` and counted as
    excluded, so every record returned by MT5 is accounted for exactly once.
    """
    if deals is None:
        raise HistoryFetchError(
            "history_deals_get returned None; pass the terminal error through "
            "require_history_result instead of treating this as an empty report"
        )
    start_utc = _as_utc(start, "start")
    end_utc = _as_utc(end, "end")
    if start_utc >= end_utc:
        raise ValueError("start must be earlier than end")

    raw_rows: list[dict[str, Any]] = []
    partitions = {
        "buy_sell_deal": 0,
        "other_operation": 0,
        "outside_requested_interval": 0,
    }
    seen_tickets: set[str] = set()
    daily: dict[str, dict[str, Any]] = {}
    other_rows: list[dict[str, Any]] = []
    for index, deal in enumerate(deals, start=1):
        source = _record_dict(deal, index)
        ticket_value = _integer(source.get("ticket"), "ticket", index)
        if ticket_value <= 0:
            raise DealValidationError(
                f"deal row {index}: ticket must be a positive deal identifier"
            )
        ticket_key = str(ticket_value)
        if ticket_key in seen_tickets:
            raise DealValidationError(f"duplicate deal ticket {ticket_key!r}")
        seen_tickets.add(ticket_key)

        raw, partition, in_interval, component_sum = _normalize_record(
            source, index, start_utc, end_utc, account_scope
        )
        raw_rows.append(raw)
        partitions[partition] += 1
        if not in_interval:
            continue
        if partition == "other_operation":
            review = dict(raw)
            label = str(raw["deal_type_label"])
            review["review_reason"] = (
                f"Review {label} operation in terminal history; no allocation applied."
            )
            other_rows.append(review)
            continue

        date_utc = str(raw["time_utc"])[:10]
        if date_utc not in daily:
            daily[date_utc] = {
                "account_login": account_scope.login,
                "account_server": account_scope.server,
                "account_scope_id": account_scope.scope_id,
                "account_currency": account_scope.currency,
                "date_utc": date_utc,
                "deal_count": 0,
                **{field: Decimal("0") for field in MONEY_FIELDS},
                "buy_sell_recorded_component_subtotal": Decimal("0"),
            }
        day = daily[date_utc]
        day["deal_count"] += 1
        for field in MONEY_FIELDS:
            day[field] += _money(source[field], field, index)
        day["buy_sell_recorded_component_subtotal"] += component_sum

    daily_rows = []
    for date_utc in sorted(daily):
        row = daily[date_utc]
        daily_rows.append(
            {
                key: (str(value) if isinstance(value, Decimal) else value)
                for key, value in row.items()
            }
        )

    in_interval_count = partitions["buy_sell_deal"] + partitions["other_operation"]
    all_partitioned = sum(partitions.values()) == len(raw_rows)
    if not all_partitioned:
        raise AssertionError("internal error: source records were not partitioned once")
    totals = {
        field: str(
            sum((Decimal(row[field]) for row in daily_rows), Decimal("0"))
        )
        for field in MONEY_FIELDS
    }
    totals["buy_sell_recorded_component_subtotal"] = str(
        sum(
            (
                Decimal(row["buy_sell_recorded_component_subtotal"])
                for row in daily_rows
            ),
            Decimal("0"),
        )
    )
    source_provenance = dict(provenance or {})
    source_provenance.setdefault("source_type", "unverified_input")
    source_provenance.setdefault(
        "terminal_export_timestamp_check", "not_verified"
    )
    source_provenance.setdefault("integration_claim", "not_established")
    source_provenance.setdefault("report_module_sha256", _source_hash())
    source_time_fields = source_provenance.get(
        "source_timestamp_fields", ["time", "time_msc"]
    )
    timestamp_interpretation = source_provenance.get(
        "normalized_timestamp_interpretation",
        "Unix epoch seconds and milliseconds in UTC",
    )
    export_crosscheck = source_provenance.get(
        "terminal_export_crosscheck", "not_performed"
    )
    result_status = "empty_valid_response" if not raw_rows else "records_received"
    reconciliation = {
        "report_schema": "mt5-cost-report-r1",
        "result_status": result_status,
        "source_record_count": len(raw_rows),
        "in_requested_interval_record_count": in_interval_count,
        "outside_requested_interval_record_count": partitions[
            "outside_requested_interval"
        ],
        "partition_counts": partitions,
        "all_source_records_partitioned_exactly_once": all_partitioned,
        "outside_interval_tickets": [
            row["ticket"]
            for row in raw_rows
            if row["report_partition"] == "outside_requested_interval"
        ],
        "requested_interval": {
            "start_utc_inclusive": _iso_utc(start_utc),
            "end_utc_exclusive": _iso_utc(end_utc),
            "policy": "[start, end)",
        },
        "timestamp_policy": {
            "source_fields": source_time_fields,
            "normalized_fields": ["time", "time_msc"],
            "interpretation": timestamp_interpretation,
            "consistency_check": "time == floor(time_msc / 1000)",
            "terminal_export_crosscheck": export_crosscheck,
        },
        "account_scope": account_scope.as_dict(),
        "money_validation": {
            "fields": list(MONEY_FIELDS),
            "finite_and_present": True,
            "sign_policy": "recorded source signs preserved",
        },
        "buy_sell_daily_totals_only": totals,
        "other_operation_count": len(other_rows),
        "unknown_type_count": sum(
            1
            for row in other_rows
            if str(row["deal_type_label"]).startswith("UNKNOWN_TYPE_")
        ),
        "account_wide_net_profit_claim": False,
        "limitations": [
            "buy_sell_recorded_component_subtotal covers buy and sell deals only",
            "other operations remain reviewable and are not allocated",
            "floating P/L and lifetime position return are not calculated",
            "direct Python history_deals_get and terminal-screen reconciliation are not established by this report",
        ],
        "provenance": source_provenance,
        "output_files": [
            "raw_deals.csv",
            "buy_sell_daily.csv",
            "other_operations.csv",
            "reconciliation.json",
        ],
    }
    return ReportBundle(
        raw_deals=tuple(raw_rows),
        buy_sell_daily=tuple(daily_rows),
        other_operations=tuple(other_rows),
        reconciliation=reconciliation,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report_bundle(bundle: ReportBundle, output_dir: str | Path) -> dict[str, Path]:
    """Write exactly the four report files without replacing existing content."""
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(
            f"output directory is not empty; refusing to replace existing files: {destination}"
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        _write_csv(staging / "raw_deals.csv", bundle.raw_deals, RAW_FIELDS)
        _write_csv(
            staging / "buy_sell_daily.csv", bundle.buy_sell_daily, DAILY_FIELDS
        )
        other_fields = (*RAW_FIELDS, "review_reason")
        _write_csv(
            staging / "other_operations.csv", bundle.other_operations, other_fields
        )
        with (staging / "reconciliation.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(bundle.reconciliation, handle, indent=2, sort_keys=True)
            handle.write("\n")

        if destination.exists():
            if any(destination.iterdir()):
                raise FileExistsError(
                    f"output directory became nonempty; refusing to replace: {destination}"
                )
            destination.rmdir()
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise

    return {name: destination / name for name in bundle.reconciliation["output_files"]}


def _scope_from_terminal(mt5: Any, account_info: Any) -> AccountScope:
    if account_info is None:
        raise RuntimeError(f"MetaTrader5 account_info failed: {mt5.last_error()!r}")
    if hasattr(account_info, "_asdict"):
        fields = account_info._asdict()
    elif isinstance(account_info, Mapping):
        fields = account_info
    else:
        fields = vars(account_info)
    demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", None)
    if demo_mode is None:
        raise RuntimeError("MetaTrader5 package does not expose demo account mode")
    if fields.get("trade_mode") != demo_mode:
        raise RuntimeError(
            "active MetaTrader 5 account is not verified as demo; history request blocked"
        )
    try:
        return AccountScope(
            login=str(fields["login"]),
            server=str(fields["server"]),
            currency=str(fields["currency"]),
            trade_mode="demo",
        )
    except KeyError as exc:
        raise RuntimeError(f"MetaTrader5 account_info missing {exc.args[0]}") from exc


def fetch_demo_history(
    start: datetime,
    end: datetime,
    *,
    expected_login: str | None = None,
    expected_server: str | None = None,
) -> tuple[tuple[Any, ...], AccountScope]:
    """Fetch history from the already configured local terminal, demo accounts only.

    This function never calls ``login`` or any order/trading API. MT5's Python
    package and a connected terminal are optional dependencies of the project.
    """
    start_utc = _as_utc(start, "start")
    end_utc = _as_utc(end, "end")
    if start_utc >= end_utc:
        raise ValueError("start must be earlier than end")
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "MetaTrader5 is not installed in this environment; no terminal history was queried"
        ) from exc

    if not mt5.initialize():
        raise RuntimeError(f"MetaTrader5 initialize failed: {mt5.last_error()!r}")
    try:
        account_info = mt5.account_info()
        scope = _scope_from_terminal(mt5, account_info)
        if expected_login is not None and scope.login != str(expected_login):
            raise RuntimeError(
                f"active demo login {scope.login!r} does not match expected login"
            )
        if expected_server is not None and scope.server != expected_server:
            raise RuntimeError(
                f"active demo server {scope.server!r} does not match expected server"
            )
        result = mt5.history_deals_get(start_utc, end_utc)
        error = mt5.last_error() if result is None else None
        deals = require_history_result(result, error)
        return deals, scope
    finally:
        mt5.shutdown()
