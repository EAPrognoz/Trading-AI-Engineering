from __future__ import annotations

import argparse
from datetime import datetime
import sys

from trading_ai.reports.mt5_cost_report import (
    build_report,
    fetch_demo_history,
    write_report_bundle,
)


def _parse_datetime(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use an ISO-8601 datetime with an explicit UTC offset, for example "
            "2026-10-17T00:00:00Z"
        ) from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise argparse.ArgumentTypeError("datetime must include a UTC offset")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a read-only MT5 deal cost report for one demo account."
    )
    parser.add_argument("--start", required=True, type=_parse_datetime)
    parser.add_argument("--end", required=True, type=_parse_datetime)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-login")
    parser.add_argument("--expected-server")
    args = parser.parse_args()

    try:
        deals, scope = fetch_demo_history(
            args.start,
            args.end,
            expected_login=args.expected_login,
            expected_server=args.expected_server,
        )
        bundle = build_report(
            deals,
            start=args.start,
            end=args.end,
            account_scope=scope,
            provenance={
                "source_type": "MetaTrader5 Python history_deals_get",
                "query_scope": "one verified demo account; no symbol filter",
                "terminal_export_timestamp_check": "pending",
                "integration_claim": "pending permitted terminal export crosscheck",
            },
        )
        paths = write_report_bundle(bundle, args.out)
    except Exception as exc:
        print(f"Report failed: {exc}", file=sys.stderr)
        return 2

    print(bundle.reconciliation["result_status"])
    print(f"Source rows: {bundle.reconciliation['source_record_count']}")
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
