"""Validated, reproducible H1 market-data snapshots.

Accepted snapshots represent the data boundary established by Episode 002.
They are downstream artifacts, not raw broker responses.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def validate_h1_snapshot(frame: pd.DataFrame) -> None:
    """Raise ValueError when an accepted H1 snapshot violates its contract."""
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
    if len(ts) > 1 and (ts.diff().dropna() > pd.Timedelta(hours=1)).any():
        raise ValueError("timestamp contains an unresolved H1 gap")

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


def load_snapshot(path: str | Path) -> pd.DataFrame:
    """Load an accepted CSV snapshot without silently repairing violations."""
    path = Path(path)
    frame = pd.read_csv(path)
    validate_h1_snapshot(frame)
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    for column in REQUIRED_H1_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column])
    return result


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest(frame: pd.DataFrame, path: str | Path) -> dict[str, Any]:
    """Describe the exact accepted snapshot used by an experiment."""
    validate_h1_snapshot(frame)
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    return {
        "path": str(Path(path)),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "first_timestamp": ts.iloc[0].isoformat() if len(ts) else None,
        "last_timestamp": ts.iloc[-1].isoformat() if len(ts) else None,
        "columns": list(frame.columns),
    }
