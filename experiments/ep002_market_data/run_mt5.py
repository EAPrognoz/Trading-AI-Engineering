from __future__ import annotations

import argparse
import subprocess

from trading_ai.data.mt5_adapter import fetch_h1_bars
from trading_ai.data.pipeline import process_h1_response
from trading_ai.data.request import MarketDataRequest


def _git_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Episode 002 against local MetaTrader 5."
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    request = MarketDataRequest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        cutoff=args.cutoff,
    )

    # The MT5 part is deliberately simple.
    raw = fetch_h1_bars(
        request.symbol,
        request.start.to_pydatetime(),
        request.end.to_pydatetime(),
    )

    # Everything below is engineering around the downloaded table.
    result = process_h1_response(
        raw,
        request=request,
        run_dir=args.run_dir,
        provenance={
            "source_type": "MetaTrader5",
            "symbol": request.symbol,
        },
        code_version=_git_version(),
    )

    print(result["status"])
    print(result["run_dir"])


if __name__ == "__main__":
    main()
