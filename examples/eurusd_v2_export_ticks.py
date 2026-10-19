"""Read-only EURUSD tick export for the V2 CSV quote replay example."""

from __future__ import annotations

import argparse
import calendar
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


UTC = timezone.utc
FIELDS = ("time_msc", "bid", "ask", "flags", "source_row_seq")


def parse_utc(value: str, name: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {name}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise argparse.ArgumentTypeError(f"{name} must include a UTC offset")
    return parsed.astimezone(UTC)


def epoch_milliseconds(value: datetime) -> int:
    return calendar.timegm(value.utctimetuple()) * 1000 + value.microsecond // 1000


def _field(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (TypeError, IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"MT5 tick is missing {name!r}") from exc


def export(symbol: str, start: datetime, end: datetime, output: Path) -> tuple[Path, Path]:
    if start >= end:
        raise ValueError("period must be a positive half-open UTC interval")
    output = Path(output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("output CSV or manifest already exists; choose a new path")

    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("install MetaTrader5 in the Python environment attached to MT5") from exc

    initialized = False
    try:
        if not mt5.initialize():
            raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()!r}")
        initialized = True
        ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_INFO)
        if ticks is None:
            raise RuntimeError(f"copy_ticks_range failed: {mt5.last_error()!r}")
        terminal_rows = len(ticks)
        start_ms, end_ms = epoch_milliseconds(start), epoch_milliseconds(end)
        previous_ms: int | None = None
        saved_rows = 0
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for source_seq, tick in enumerate(ticks):
                time_ms = int(_field(tick, "time_msc"))
                if previous_ms is not None and time_ms < previous_ms:
                    raise ValueError("MT5 response moved backwards in source time order")
                previous_ms = time_ms
                # Enforce the documented local [start, end) rule, retaining all
                # in-range rows, equal timestamps, and original row ordering.
                if not start_ms <= time_ms < end_ms:
                    continue
                writer.writerow({
                    "time_msc": time_ms,
                    "bid": str(_field(tick, "bid")),
                    "ask": str(_field(tick, "ask")),
                    "flags": int(_field(tick, "flags")),
                    "source_row_seq": source_seq,
                })
                saved_rows += 1
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        manifest = {
            "artifact_type": "read_only_mt5_copy_ticks_range_export",
            "status": "valid_empty_response" if terminal_rows == 0 else "success",
            "symbol": symbol,
            "source_method": "MetaTrader5.copy_ticks_range",
            "flags_requested": "COPY_TICKS_INFO",
            "interval_utc": {
                "start_inclusive": start.isoformat().replace("+00:00", "Z"),
                "end_exclusive": end.isoformat().replace("+00:00", "Z"),
                "local_policy": "[start, end); end boundary excluded after API response",
            },
            "terminal_response_rows": terminal_rows,
            "saved_rows": saved_rows,
            "source_row_order_preserved": True,
            "timestamps_deduplicated": False,
            "account_identifiers_queried": False,
            "credentials_read": False,
            "orders_or_positions_queried": False,
            "output_csv": output.name,
            "output_csv_sha256": digest,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output, manifest_path
    finally:
        if initialized:
            mt5.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export saved EURUSD bid/ask ticks from an already-connected MT5 terminal; read-only.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        csv_path, manifest_path = export(args.symbol, parse_utc(args.start, "start"), parse_utc(args.end, "end"), args.out)
    except Exception as exc:
        print(f"tick export failed: {exc}", file=sys.stderr)
        return 2
    print(f"read-only tick export: {csv_path}")
    print(f"source manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
