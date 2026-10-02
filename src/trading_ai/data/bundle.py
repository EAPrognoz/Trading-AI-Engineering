"""Hash-verified binding of three accepted Episode 002 BTC source streams."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pandas as pd

from trading_ai.data.manifest import market_contract_id, sha256_file
from trading_ai.data.snapshot import load_market_snapshot
from trading_ai.data.timeframes import timeframe_duration


BUNDLE_CONTRACT_ID = "ep002-btc-multitimeframe-bundle-v1"
TIMEFRAMES = ("H1", "H4", "D1")


@dataclass(frozen=True)
class MarketDataBundle:
    frames: dict[str, pd.DataFrame]
    manifest: dict[str, Any]
    source_manifests: dict[str, dict[str, Any]]
    manifest_sha256: str


def _utc_hour(value: Any, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("h"):
        raise ValueError(f"{field} must be aligned to a UTC hour")
    return timestamp


def _member_file(bundle_dir: Path, relative: str, timeframe: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{timeframe}: invalid relative artifact path")
    logical = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if logical.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{timeframe}: absolute artifact path")
    if any(part in {"", ".", ".."} for part in logical.parts):
        raise ValueError(f"{timeframe}: unsafe artifact path")
    if not logical.parts or logical.parts[0] != timeframe:
        raise ValueError(f"{timeframe}: artifact path outside stream directory")
    root = bundle_dir.resolve()
    resolved = (bundle_dir / relative).resolve()
    if not resolved.is_relative_to(root / timeframe):
        raise ValueError(f"{timeframe}: artifact path escapes bundle")
    return resolved


def _source_member(
    bundle_dir: Path, timeframe: str
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    source_relative = f"{timeframe}/manifest.json"
    source_path = _member_file(bundle_dir, source_relative, timeframe)
    if not source_path.is_file():
        raise ValueError(f"{timeframe}: source manifest is missing")
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"{timeframe}: source manifest is invalid") from exc
    if not isinstance(source, dict) or source.get("status") != "accepted":
        raise ValueError(f"{timeframe}: source stream is not accepted")
    if source.get("contract_id") != market_contract_id(timeframe):
        raise ValueError(f"{timeframe}: source contract mismatch")
    request = source.get("request")
    if not isinstance(request, dict) or request.get("timeframe") != timeframe:
        raise ValueError(f"{timeframe}: source timeframe mismatch")
    symbol = request.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"{timeframe}: missing exact symbol")
    accepted = source.get("files", {}).get("accepted_dataset")
    if not isinstance(accepted, dict):
        raise ValueError(f"{timeframe}: missing accepted dataset")
    accepted_name = accepted.get("path")
    if not isinstance(accepted_name, str) or accepted_name != "accepted.csv":
        raise ValueError(f"{timeframe}: invalid accepted dataset path")
    accepted_relative = f"{timeframe}/{accepted_name}"
    accepted_path = _member_file(bundle_dir, accepted_relative, timeframe)
    if not accepted_path.is_file():
        raise ValueError(f"{timeframe}: accepted dataset is missing")
    accepted_sha = sha256_file(accepted_path)
    if accepted.get("sha256") != accepted_sha:
        raise ValueError(f"{timeframe}: accepted dataset hash mismatch")
    dataset_id = f"{market_contract_id(timeframe)}:{accepted_sha}"
    if source.get("dataset_id") != dataset_id:
        raise ValueError(f"{timeframe}: source dataset identity mismatch")
    try:
        frame = load_market_snapshot(accepted_path, timeframe)
    except (ValueError, KeyError) as exc:
        raise ValueError(f"{timeframe}: accepted dataset validation failed") from exc
    member = {
        "symbol": symbol,
        "timeframe": timeframe,
        "dataset_id": dataset_id,
        "source_manifest": {
            "path": source_relative,
            "sha256": sha256_file(source_path),
        },
        "accepted_dataset": {
            "path": accepted_relative,
            "sha256": accepted_sha,
        },
    }
    return member, frame, source


def _all_sources(
    bundle_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    members: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, dict[str, Any]] = {}
    for timeframe in TIMEFRAMES:
        members[timeframe], frames[timeframe], sources[timeframe] = _source_member(
            bundle_dir, timeframe
        )
    return members, frames, sources


def _check_interval_coverage(
    frames: dict[str, pd.DataFrame],
    sources: dict[str, dict[str, Any]],
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> None:
    """Require warm-up and a current completed bar at both interval endpoints."""
    last_decision = analysis_end - pd.Timedelta(hours=1)
    for timeframe in TIMEFRAMES:
        provenance = sources[timeframe].get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{timeframe}: missing pre-roll provenance")
        lookback = provenance.get("feature_max_lookback_bars")
        if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback <= 0:
            raise ValueError(f"{timeframe}: missing positive pre-roll lookback")
        duration = timeframe_duration(timeframe)
        nominal_closes = frames[timeframe]["timestamp"] + duration
        completed_before_first_decision = int((nominal_closes <= analysis_start).sum())
        if completed_before_first_decision < lookback + 1:
            raise ValueError(f"{timeframe}: insufficient completed pre-roll bars")
        for decision in (analysis_start, last_decision):
            eligible = nominal_closes[nominal_closes <= decision]
            if eligible.empty or eligible.iloc[-1] <= decision - duration:
                raise ValueError(
                    f"{timeframe}: incomplete common-interval coverage at {decision.isoformat()}"
                )


def write_market_data_bundle_manifest(
    bundle_dir: str | Path,
    *,
    bundle_id: str,
    analysis_start: Any,
    analysis_end: Any,
) -> dict[str, Any]:
    """Write once after every required native stream has passed verification."""
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "bundle_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"bundle manifest already exists: {manifest_path}")
    if not isinstance(bundle_id, str) or not bundle_id.strip():
        raise ValueError("bundle_id must be non-empty")
    start = _utc_hour(analysis_start, "analysis_start")
    end = _utc_hour(analysis_end, "analysis_end")
    if start >= end:
        raise ValueError("analysis_start must precede analysis_end")
    members, frames, sources = _all_sources(bundle_dir)
    symbols = {member["symbol"] for member in members.values()}
    if len(symbols) != 1:
        raise ValueError("source streams must have the same exact symbol")
    for timeframe, source in sources.items():
        request = source["request"]
        if _utc_hour(request["start"], f"{timeframe} request.start") > start:
            raise ValueError(f"{timeframe}: request starts after analysis interval")
        if _utc_hour(request["end"], f"{timeframe} request.end") < end:
            raise ValueError(f"{timeframe}: request ends before analysis interval")
    _check_interval_coverage(frames, sources, start, end)
    manifest = {
        "contract_id": BUNDLE_CONTRACT_ID,
        "bundle_id": bundle_id,
        "symbol": symbols.pop(),
        "analysis_start": start.isoformat(),
        "analysis_end": end.isoformat(),
        "analysis_range_policy": "start_inclusive_end_exclusive",
        "streams": members,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def load_market_data_bundle(bundle_manifest_path: str | Path) -> MarketDataBundle:
    """Return data only after verifying all source identities and file hashes."""
    path = Path(bundle_manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("invalid bundle manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("contract_id") != BUNDLE_CONTRACT_ID:
        raise ValueError("bundle contract mismatch")
    if not isinstance(manifest.get("bundle_id"), str) or not manifest["bundle_id"].strip():
        raise ValueError("missing bundle ID")
    start = _utc_hour(manifest.get("analysis_start"), "analysis_start")
    end = _utc_hour(manifest.get("analysis_end"), "analysis_end")
    if start >= end or manifest.get("analysis_range_policy") != "start_inclusive_end_exclusive":
        raise ValueError("invalid bundle analysis interval")
    members = manifest.get("streams")
    if not isinstance(members, dict) or set(members) != set(TIMEFRAMES):
        raise ValueError("bundle must contain H1, H4, and D1")
    expected, frames, sources = _all_sources(path.parent)
    if members != expected:
        raise ValueError("bundle member hash or identity mismatch")
    symbols = {member["symbol"] for member in expected.values()}
    if len(symbols) != 1 or manifest.get("symbol") != next(iter(symbols)):
        raise ValueError("bundle symbol mismatch")
    for timeframe, source in sources.items():
        request = source["request"]
        if _utc_hour(request["start"], f"{timeframe} request.start") > start:
            raise ValueError(f"{timeframe}: request starts after analysis interval")
        if _utc_hour(request["end"], f"{timeframe} request.end") < end:
            raise ValueError(f"{timeframe}: request ends before analysis interval")
    _check_interval_coverage(frames, sources, start, end)
    return MarketDataBundle(
        frames=frames, manifest=manifest, source_manifests=sources,
        manifest_sha256=sha256_file(path),
    )
