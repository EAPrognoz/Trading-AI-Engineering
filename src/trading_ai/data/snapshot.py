"""Validated, reproducible H1 market-data snapshots.

Accepted snapshots represent the data boundary established by Episode 002.
They are downstream artifacts, not raw broker responses.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_ai.data.timeframes import timeframe_duration

REQUIRED_H1_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
)


def validate_market_snapshot(frame: pd.DataFrame, timeframe: str) -> None:
    """Raise ValueError when an accepted native-bar snapshot violates its contract."""
    duration = timeframe_duration(timeframe)
    missing = [column for column in REQUIRED_H1_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("timestamp contains invalid values")
    if ts.duplicated().any():
        raise ValueError("timestamp contains duplicates")
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamp must be strictly time ordered")
    if ts.dt.floor("h").ne(ts).any():
        raise ValueError(f"timestamp must be hour-aligned for {timeframe}")
    if len(ts) > 1:
        deltas = ts.diff().dropna()
        if (deltas > duration).any():
            raise ValueError(f"timestamp contains an unresolved {timeframe} gap")
        if (deltas < duration).any():
            raise ValueError(f"timestamp violates {timeframe} native cadence")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("numeric market-data fields contain invalid values")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("numeric market-data fields must be finite")

    if (numeric["high"] < numeric[["open", "close"]].max(axis=1)).any():
        raise ValueError("high must contain both open and close")
    if (numeric["low"] > numeric[["open", "close"]].min(axis=1)).any():
        raise ValueError("low must contain both open and close")

    for column in ("tick_volume", "spread", "real_volume"):
        if (numeric[column] < 0).any():
            raise ValueError(f"{column} must be non-negative")


def validate_h1_snapshot(frame: pd.DataFrame) -> None:
    """Compatibility validator for accepted H1 snapshots."""
    validate_market_snapshot(frame, "H1")


def load_market_snapshot(path: str | Path, timeframe: str) -> pd.DataFrame:
    """Load an accepted CSV snapshot without silently repairing violations."""
    path = Path(path)
    frame = pd.read_csv(path)
    validate_market_snapshot(frame, timeframe)
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    for column in REQUIRED_H1_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column])
    return result


def load_snapshot(path: str | Path) -> pd.DataFrame:
    """Compatibility loader for accepted H1 snapshots."""
    return load_market_snapshot(path, "H1")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Describe and optionally verify the accepted snapshot used by an experiment."""
    validate_h1_snapshot(frame)
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    snapshot_path = Path(path)
    snapshot_sha = sha256_file(snapshot_path)

    metadata: dict[str, Any] = {
        "path": snapshot_path.name,
        "sha256": snapshot_sha,
        "rows": int(len(frame)),
        "first_timestamp": ts.iloc[0].isoformat() if len(ts) else None,
        "last_timestamp": ts.iloc[-1].isoformat() if len(ts) else None,
        "columns": list(frame.columns),
    }

    if source_manifest_path is not None:
        manifest_path = Path(source_manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "accepted":
            raise ValueError("source manifest must have accepted status")

        accepted_artifact = manifest.get("files", {}).get("accepted_dataset")
        if not isinstance(accepted_artifact, dict) or not accepted_artifact.get("sha256"):
            raise ValueError("source manifest is missing files.accepted_dataset")

        expected_sha = str(accepted_artifact["sha256"])
        if snapshot_sha != expected_sha:
            raise ValueError("accepted dataset hash does not match source manifest")

        metadata["source_manifest"] = {
            "contract_id": manifest.get("contract_id"),
            "status": "accepted",
            "manifest_sha256": sha256_file(manifest_path),
            "accepted_dataset_sha256": expected_sha,
            "accepted_dataset_artifact": Path(
                str(accepted_artifact.get("path", "accepted.csv"))
            ).name,
        }

    return metadata
