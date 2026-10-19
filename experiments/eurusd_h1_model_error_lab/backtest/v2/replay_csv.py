"""Read-only V2 quote replay from caller-supplied CSV files.

This interface reads only frozen decisions and bid/ask quote rows. It does not
connect to MetaTrader, train a model, read future labels, or submit orders.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator

from ..engine import ReplayInputError, Tick
from .public_engine import (
    ABSTAIN_POLICY,
    BASE_UNITS_EUR,
    SAME_SIDE_POLICY,
    SIMULATION_LEVERAGE,
    STARTING_BALANCE_USD,
    Decision,
    ReplayConfig,
    replay,
)


UTC = timezone.utc
DECISION_FIELDS = {"signal_id", "decision_time_utc", "action"}
TICK_FIELDS = {"time_msc", "bid", "ask"}


def parse_utc(value: str, field: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ReplayInputError(f"invalid {field}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ReplayInputError(f"{field} must include a UTC offset")
    return parsed.astimezone(UTC)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_decisions(path: Path, start: datetime, end: datetime) -> tuple[Decision, ...]:
    decisions: list[Decision] = []
    seen_ids: set[str] = set()
    try:
        with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not DECISION_FIELDS.issubset(reader.fieldnames):
                raise ReplayInputError(
                    f"decisions CSV requires columns {sorted(DECISION_FIELDS)}"
                )
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ReplayInputError(f"decisions row {row_number}: wider than header")
                signal_id = (row.get("signal_id") or "").strip()
                if not signal_id or signal_id in seen_ids:
                    raise ReplayInputError(
                        f"decisions row {row_number}: empty or duplicate signal_id {signal_id!r}"
                    )
                seen_ids.add(signal_id)
                decision_time = parse_utc(row.get("decision_time_utc", ""), "decision_time_utc")
                if not start <= decision_time < end:
                    raise ReplayInputError(
                        f"decisions row {row_number}: time outside [start, end)"
                    )
                action = (row.get("action") or "").strip()
                try:
                    decisions.append(Decision(signal_id, decision_time, action))
                except ReplayInputError as exc:
                    raise ReplayInputError(f"decisions row {row_number}: {exc}") from exc
    except OSError as exc:
        raise ReplayInputError(f"cannot read decisions CSV: {exc}") from exc
    if not decisions:
        raise ReplayInputError("decisions CSV contains no decision rows")
    if tuple(sorted(decisions, key=lambda item: item.decision_time)) != tuple(decisions):
        raise ReplayInputError("decisions must be in chronological order")
    if len({item.decision_time for item in decisions}) != len(decisions):
        raise ReplayInputError("duplicate decision timestamp")
    return tuple(decisions)


def iter_ticks(
    path: Path,
    start: datetime,
    end: datetime,
    audit: dict[str, Any],
) -> Iterator[Tick]:
    """Stream one row at a time, preserving equal-time source order."""
    prior_ms: int | None = None
    prior_seq: int | None = None
    count = 0
    first_ms: int | None = None
    last_ms: int | None = None
    start_ms = int(start.timestamp()) * 1000 + start.microsecond // 1000
    end_ms = int(end.timestamp()) * 1000 + end.microsecond // 1000
    try:
        handle = Path(path).open("r", newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise ReplayInputError(f"cannot read tick CSV: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not TICK_FIELDS.issubset(reader.fieldnames):
            raise ReplayInputError(f"ticks CSV requires columns {sorted(TICK_FIELDS)}")
        has_sequence = "source_row_seq" in reader.fieldnames
        has_flags = "flags" in reader.fieldnames
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ReplayInputError(f"ticks row {row_number}: wider than header")
            try:
                raw_ms = row.get("time_msc", "")
                time_ms = int(raw_ms)
                if str(raw_ms).strip() not in (str(time_ms), f"{time_ms}.0"):
                    raise ValueError("time_msc must be integral")
                source_seq = int(row["source_row_seq"]) if has_sequence else row_number - 2
                flags = int(row["flags"]) if has_flags and row.get("flags", "").strip() else 0
            except (TypeError, ValueError, OverflowError) as exc:
                raise ReplayInputError(f"ticks row {row_number}: invalid time_msc/source sequence/flags") from exc
            if time_ms < start_ms or time_ms >= end_ms:
                raise ReplayInputError(f"ticks row {row_number}: time outside [start, end)")
            if prior_ms is not None and time_ms < prior_ms:
                raise ReplayInputError(f"ticks row {row_number}: timestamp order moved backwards")
            if prior_seq is not None and source_seq <= prior_seq:
                raise ReplayInputError(f"ticks row {row_number}: source_row_seq must strictly increase")
            if prior_ms == time_ms and prior_seq is not None and source_seq <= prior_seq:
                raise ReplayInputError(f"ticks row {row_number}: equal-time source order is not preserved")
            try:
                quote_time = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=time_ms)
                tick = Tick(quote_time, source_seq, Decimal(row.get("bid", "")), Decimal(row.get("ask", "")), flags)
            except (InvalidOperation, ValueError, OverflowError, ReplayInputError) as exc:
                raise ReplayInputError(f"ticks row {row_number}: {exc}") from exc
            count += 1
            if first_ms is None:
                first_ms = time_ms
            last_ms = time_ms
            prior_ms, prior_seq = time_ms, source_seq
            yield tick
    audit.update({"tick_rows":count,"first_time_msc":first_ms,"last_time_msc":last_ms})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if isinstance(value, Decimal):
        return str(value)
    return value


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key:_csv_value(value) for key,value in row.items()})


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")


def run(decisions_path: Path, ticks_path: Path, start: datetime, end: datetime, output: Path) -> Path:
    start, end = parse_utc(start.isoformat(), "start"), parse_utc(end.isoformat(), "end")
    if start >= end:
        raise ReplayInputError("period must be a positive half-open UTC interval")
    if Path(output).exists():
        raise ReplayInputError(f"output directory already exists: {output}")
    decisions = load_decisions(decisions_path, start, end)
    source_audit: dict[str, Any] = {"tick_file_sha256":_sha256(ticks_path)}
    config = ReplayConfig(
        same_side_policy=SAME_SIDE_POLICY,
        abstain_policy=ABSTAIN_POLICY,
        period_start_utc=start,
        period_end_utc=end,
    )
    result = replay(decisions, iter_ticks(ticks_path,start,end,source_audit), config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    positions=[]
    for position in result.positions:
        row=asdict(position)
        for field in ("decision_time","target_time","entry_time","exit_time"):
            row[field+"_utc"]=_csv_value(row.pop(field))
        positions.append({key:_csv_value(value) for key,value in row.items()})
    actions=[{key:_csv_value(value) for key,value in row.items()} for row in result.signal_actions]
    balance=[{key:_csv_value(value) for key,value in row.items()} for row in result.balance_curve]
    equity=[{key:_csv_value(value) for key,value in row.items()} for row in result.equity_curve]
    _write_csv(output/"positions.csv",positions)
    _write_csv(output/"signal_actions.csv",actions)
    _write_csv(output/"balance_curve.csv",balance)
    _write_csv(output/"equity_curve.csv",equity)
    _write_json(output/"metrics.json",result.metrics)
    _write_json(output/"source_audit.json",source_audit)
    output_hashes={}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            output_hashes[path.name]=_sha256(path)
    manifest={
        "artifact_type":"read_only_v2_csv_quote_replay",
        "period_utc_half_open":{"start_inclusive":start.isoformat().replace("+00:00","Z"),"end_exclusive":end.isoformat().replace("+00:00","Z")},
        "inputs":{"decisions_file_sha256":_sha256(decisions_path),"tick_file_sha256":_sha256(ticks_path),"decision_count":len(decisions),**source_audit},
        "scenario":{"symbol":"EURUSD","starting_balance_usd":str(STARTING_BALANCE_USD),"base_units_eur":str(BASE_UNITS_EUR),"simulation_leverage":f"1:{SIMULATION_LEVERAGE}","same_side_policy":SAME_SIDE_POLICY,"abstain_policy":ABSTAIN_POLICY,"historical_swap":"unknown_not_applied","additional_slippage":"0_assumed","margin_call_stopout":"not_simulated"},
        "output_sha256":output_hashes,
        "read_only":True,
        "account_wide_net_profit_claim":False,
        "trading_profitability_test":False,
    }
    _write_json(output/"run_manifest.json",manifest)
    return output


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="Replay frozen long/short/ABSTAIN decisions against saved EURUSD bid/ask CSV ticks.")
    parser.add_argument("--decisions",required=True,type=Path)
    parser.add_argument("--ticks",required=True,type=Path)
    parser.add_argument("--start",required=True)
    parser.add_argument("--end",required=True)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args(argv)
    try:
        output=run(args.decisions,args.ticks,parse_utc(args.start,"start"),parse_utc(args.end,"end"),args.out)
    except (OSError,ReplayInputError) as exc:
        print(f"replay failed: {exc}",file=sys.stderr)
        return 2
    print(f"read-only replay complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
