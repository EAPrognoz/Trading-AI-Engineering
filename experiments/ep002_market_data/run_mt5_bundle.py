"""Acquire historical H1/H4/D1 MT5 bars into one local accepted bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tomllib

import pandas as pd

from trading_ai.data.bundle import TIMEFRAMES, write_market_data_bundle_manifest
from trading_ai.data.manifest import sha256_file
from trading_ai.data.mt5_adapter import fetch_market_bars
from trading_ai.data.pipeline import process_market_response
from trading_ai.data.request import MarketDataRequest
from trading_ai.data.timeframes import timeframe_duration


def _git_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _feature_lookbacks(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    tables = payload.get("timeframes")
    if not isinstance(tables, dict) or set(tables) != set(TIMEFRAMES):
        raise ValueError("feature contract requires exact H1/H4/D1 timeframe tables")
    result = {}
    for timeframe in TIMEFRAMES:
        value = tables[timeframe].get("max_lookback_bars")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{timeframe}: max_lookback_bars must be a positive integer")
        result[timeframe] = value
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--analysis-start", required=True)
    parser.add_argument("--analysis-end", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--feature-contract", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    local_root = Path(__file__).resolve().parents[2] / ".local"
    resolved_run_dir = run_dir.resolve()
    if not resolved_run_dir.is_relative_to(local_root.resolve()) or resolved_run_dir == local_root.resolve():
        raise ValueError("run directory must be a new child of repository .local")
    lookbacks = _feature_lookbacks(Path(args.feature_contract))
    feature_contract_sha256 = sha256_file(args.feature_contract)
    analysis_start = pd.Timestamp(args.analysis_start)
    analysis_end = pd.Timestamp(args.analysis_end)
    cutoff = pd.Timestamp(args.cutoff)
    if any(value.tzinfo is None for value in (analysis_start, analysis_end, cutoff)):
        raise ValueError("analysis interval and cutoff must be timezone-aware")
    analysis_start = analysis_start.tz_convert("UTC")
    analysis_end = analysis_end.tz_convert("UTC")
    cutoff = cutoff.tz_convert("UTC")
    if analysis_start != analysis_start.floor("h") or analysis_end != analysis_end.floor("h"):
        raise ValueError("analysis boundaries must be UTC hour-aligned")
    if analysis_start >= analysis_end:
        raise ValueError("analysis start must precede end")
    if cutoff < analysis_end:
        raise ValueError("cutoff must be at or later than analysis end")
    if not args.symbol.strip():
        raise ValueError("symbol must be non-empty")

    requests = {}
    for timeframe in TIMEFRAMES:
        # One extra native period covers broker bar opens offset from the H1
        # decision hour while retaining N+1 completed bars for an N-bar return.
        pre_roll = (lookbacks[timeframe] + 2) * timeframe_duration(timeframe)
        requests[timeframe] = MarketDataRequest(
            symbol=args.symbol, timeframe=timeframe,
            start=analysis_start - pre_roll, end=analysis_end, cutoff=cutoff,
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    for timeframe in TIMEFRAMES:
        request = requests[timeframe]
        raw = fetch_market_bars(
            request.symbol, timeframe,
            request.start.to_pydatetime(), request.end.to_pydatetime(),
        )
        provenance = dict(raw.attrs.get("mt5_provenance", {}))
        provenance.update({
            "source_type": "MetaTrader5",
            "symbol": request.symbol,
            "timeframe": timeframe,
            "retrieved_at_utc": provenance.get("retrieved_at_utc")
            or datetime.now(timezone.utc).isoformat(),
            "request_start_utc": request.start.isoformat(),
            "request_end_utc": request.end.isoformat(),
            "cutoff_utc": request.cutoff.isoformat(),
            "feature_max_lookback_bars": lookbacks[timeframe],
            "feature_contract_sha256": feature_contract_sha256,
        })
        result = process_market_response(
            raw, request=request, run_dir=run_dir / timeframe,
            provenance=provenance, code_version=_git_version(),
        )
        if result["status"] != "accepted":
            raise ValueError(f"{timeframe}: rejected source stream; bundle not written")

    manifest = write_market_data_bundle_manifest(
        run_dir, bundle_id=run_dir.name,
        analysis_start=analysis_start, analysis_end=analysis_end,
    )
    print(manifest["bundle_id"])
    print(run_dir / "bundle_manifest.json")


if __name__ == "__main__":
    main()
