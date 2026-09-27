"""Auditable Episode 002 record validation and gap detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_ai.data.snapshot import REQUIRED_H1_COLUMNS


def audit_h1_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return all detected validation issues instead of hiding them in one exception."""
    issues: list[dict[str, Any]] = []

    missing = [column for column in REQUIRED_H1_COLUMNS if column not in frame.columns]
    if missing:
        return [
            {
                "code": "missing_columns",
                "severity": "error",
                "details": {"columns": missing},
            }
        ]

    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    invalid_ts_count = int(timestamp.isna().sum())
    if invalid_ts_count:
        issues.append(
            {
                "code": "invalid_timestamp",
                "severity": "error",
                "details": {"rows": invalid_ts_count},
            }
        )

    valid_timestamp = timestamp.dropna()
    misaligned_mask = valid_timestamp.dt.floor("h").ne(valid_timestamp)
    if misaligned_mask.any():
        issues.append(
            {
                "code": "h1_timestamp_misaligned",
                "severity": "error",
                "details": {"rows": int(misaligned_mask.sum())},
            }
        )

    duplicate_mask = timestamp.notna() & timestamp.duplicated(keep=False)
    if duplicate_mask.any():
        duplicated = sorted(
            {
                value.isoformat()
                for value in timestamp.loc[duplicate_mask].dropna()
            }
        )
        issues.append(
            {
                "code": "duplicate_opening_time",
                "severity": "error",
                "details": {
                    "rows": int(duplicate_mask.sum()),
                    "timestamps": duplicated,
                },
            }
        )

    ordered = timestamp.dropna()
    if not ordered.is_monotonic_increasing:
        issues.append(
            {
                "code": "timestamp_not_ordered",
                "severity": "error",
                "details": {},
            }
        )

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
    finite_array = np.isfinite(numeric.to_numpy(dtype=float))
    invalid_by_column = {}
    for index, column in enumerate(numeric_columns):
        invalid = numeric[column].isna().to_numpy() | ~finite_array[:, index]
        count = int(invalid.sum())
        if count:
            invalid_by_column[column] = count
    if invalid_by_column:
        issues.append(
            {
                "code": "invalid_numeric_value",
                "severity": "error",
                "details": invalid_by_column,
            }
        )

    finite_rows = pd.Series(finite_array.all(axis=1), index=frame.index)
    if finite_rows.any():
        finite_numeric = numeric.loc[finite_rows]
        high_bad = finite_numeric["high"] < finite_numeric[["open", "close"]].max(axis=1)
        low_bad = finite_numeric["low"] > finite_numeric[["open", "close"]].min(axis=1)
        if high_bad.any() or low_bad.any():
            issues.append(
                {
                    "code": "ohlc_contradiction",
                    "severity": "error",
                    "details": {
                        "high_rows": int(high_bad.sum()),
                        "low_rows": int(low_bad.sum()),
                    },
                }
            )

        negative = {
            column: int((finite_numeric[column] < 0).sum())
            for column in ("tick_volume", "spread", "real_volume")
            if int((finite_numeric[column] < 0).sum())
        }
        if negative:
            issues.append(
                {
                    "code": "negative_nonnegative_field",
                    "severity": "error",
                    "details": negative,
                }
            )

    unique_ts = timestamp.dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_ts) > 1:
        deltas = unique_ts.diff()
        for position in deltas.index[deltas > pd.Timedelta(hours=1)]:
            right = unique_ts.iloc[position]
            left = unique_ts.iloc[position - 1]
            issues.append(
                {
                    "code": "unclassified_gap",
                    "severity": "review",
                    "details": {
                        "left_timestamp": left.isoformat(),
                        "right_timestamp": right.isoformat(),
                        "delta_hours": float((right - left) / pd.Timedelta(hours=1)),
                    },
                }
            )

    return issues
