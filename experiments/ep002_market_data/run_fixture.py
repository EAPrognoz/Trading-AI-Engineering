from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import pandas as pd

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
    parser = argparse.ArgumentParser(description="Run Episode 002 on a CSV fixture.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    raw = pd.read_csv(Path(args.input))
    request = MarketDataRequest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        cutoff=args.cutoff,
    )
    result = process_h1_response(
        raw,
        request=request,
        run_dir=args.run_dir,
        provenance={
            "source_type": "synthetic_fixture",
            "source_path": str(Path(args.input)),
        },
        code_version=_git_version(),
    )
    print(result["status"])
    print(result["run_dir"])


if __name__ == "__main__":
    main()
