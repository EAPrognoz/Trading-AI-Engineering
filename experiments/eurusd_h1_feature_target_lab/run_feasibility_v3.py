"""Verify v3 observed-bar feasibility and coverage before any model fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_target_selection_v3 import (
    HORIZONS, WINDOWS, assess_feasibility, build_forward_targets,
    candidate_feature_schema, validate_ohlc_frame,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".local/h1_mlp_source/EURUSD_Model_Error_Lab_H1_UTC_20241016_20261017.csv"
PREFIX_ROWS = 10_621
PREFIX_SHA256 = "141318d453eb62941f0cf7d03d6a516770cee52a5131fb15d44fb4c5f0e9a270"
FULL_SHA256_PRIOR_VERIFIED = "74a4f9377dff438968301a6cfe673bab43ce9083add994db79c3e2910745c202"
OUTER_FOLDS = {
    "F1": ("2025-07-01T00:00:00Z", "2025-10-01T00:00:00Z"),
    "F2": ("2025-10-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    "F3": ("2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z"),
    "F4": ("2026-04-01T00:00:00Z", "2026-07-04T00:00:00Z"),
}
INNER_FOLDS = {
    "F1": (("I1_2025Q1", "2025-01-01T00:00:00Z", "2025-04-01T00:00:00Z"),
           ("I2_2025Q2", "2025-04-01T00:00:00Z", "2025-07-01T00:00:00Z")),
    "F2": (("I1_2025Q2", "2025-04-01T00:00:00Z", "2025-07-01T00:00:00Z"),
           ("I2_2025Q3", "2025-07-01T00:00:00Z", "2025-10-01T00:00:00Z")),
    "F3": (("I1_2025Q3", "2025-07-01T00:00:00Z", "2025-10-01T00:00:00Z"),
           ("I2_2025Q4", "2025-10-01T00:00:00Z", "2026-01-01T00:00:00Z")),
    "F4": (("I1_2025Q4", "2025-10-01T00:00:00Z", "2026-01-01T00:00:00Z"),
           ("I2_2026Q1", "2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z")),
}


def prefix_digest(path: Path, rows: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for _ in range(rows + 1):
            line = source.readline()
            if not line:
                raise ValueError("CSV ended before the pinned pre-test prefix")
            digest.update(line)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "min": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(len(values)), "min": float(np.min(values)), "median": float(np.median(values)),
        "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)), "max": float(np.max(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    digest = prefix_digest(source, PREFIX_ROWS)
    if digest != PREFIX_SHA256:
        raise ValueError(f"pinned pre-test prefix SHA-256 mismatch: {digest}")
    frame = pd.read_csv(source, nrows=PREFIX_ROWS)
    if len(frame) != PREFIX_ROWS:
        raise ValueError(f"expected {PREFIX_ROWS} prefix rows, got {len(frame)}")
    timestamp, prices = validate_ohlc_frame(frame)
    counts, time_coverage, summary = assess_feasibility(frame, OUTER_FOLDS, {
        name: (start, end)
        for folds in INNER_FOLDS.values()
        for name, start, end in folds
    })
    if not counts["common_W96_H24_origins"].eq(summary["common_origins_total"]).all():
        raise AssertionError("not all W/H pairs share the frozen W96/H24 origin cohort")

    transition_hours = timestamp.diff().dt.total_seconds().to_numpy(dtype=float) / 3600.0
    transition_hours = transition_hours[1:]
    gap_transition = ~np.isclose(transition_hours, 1.0)
    elapsed_rows: list[dict[str, object]] = []
    row = np.arange(len(frame))
    common = (row >= max(WINDOWS)) & (row + max(HORIZONS) < len(frame))
    for window in WINDOWS:
        hours = (timestamp.array.asi8[common] - timestamp.array.asi8[row[common] - window]) / 3.6e12
        elapsed_rows.append({"kind": "input_history", "W": window, "H": "", "cohort": "common_W96_H24", **_summary(hours),
                             "source_non_one_hour_transitions": int(gap_transition.sum()),
                             "interpretation": "audit duration; never a model feature"})
    target_gap_counts = []
    target_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        target = build_forward_targets(frame, horizon)
        mask = common & target["target_valid"].to_numpy(dtype=bool)
        durations = target.loc[mask, "target_elapsed_calendar_hours"].to_numpy(dtype=float)
        gap_counts = target.loc[mask, "target_non_one_hour_transition_count"].to_numpy(dtype=int)
        target_gap_counts.append(gap_counts)
        target_rows.append({"kind": "target_interval", "W": "", "H": horizon, "cohort": "common_W96_H24", **_summary(durations),
                            "origins_with_non_one_hour_transition": int(np.sum(gap_counts > 0)),
                            "observed_non_one_hour_transitions_in_target_paths": int(gap_counts.sum()),
                            "interpretation": "future label/audit metadata only; elapsed time is not a model feature"})
    pd.DataFrame(elapsed_rows + target_rows).to_csv(output / "elapsed_time_audit.csv", index=False)

    feature_schema = {
        "schema_id": "eurusd_h1_feature_candidates_v3_observed_transitions",
        "status": "fixed candidate definitions before v3 fitting",
        "history_semantics": "W observed completed transitions require W+1 source rows/closes; never reset at timestamp gaps",
        "gap_core_feature": "observed_transition_gap_log_open_prev_close = log(open_i / close_(i-1)); one feature, no duplicate same-bar/lagged feature",
        "gap_return_semantics": "all OHLC one-row log differences use adjacent observed rows, including non-one-hour intervals, without calendar-time division",
        "elapsed_time": "audit metadata only; not a model predictor",
        "zero_variance_skewness": "feature=0 and skewness_defined=false",
        "windows": {str(w): candidate_feature_schema(w) for w in WINDOWS},
    }
    (output / "candidate_feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    counts.to_csv(output / "sample_feasibility_by_W_H.csv", index=False)
    time_coverage.to_csv(output / "common_origin_time_coverage.csv", index=False)
    summary.update({
        "source_file": source.name,
        "source_provenance": "MetaTrader 5 demo EURUSD H1 OHLC export, previously source-audited",
        "source_full_sha256_prior_verified_not_rehashed": FULL_SHA256_PRIOR_VERIFIED,
        "pretest_prefix_data_rows": PREFIX_ROWS,
        "pretest_prefix_sha256_recomputed": digest,
        "test_rows_read": 0,
        "bar_time_semantics": "source time_utc is bar open; decision time is origin open + 1h close",
        "transition_gap_policy": "keep every adjacent observed row; no gap exclusion, no reset, no interpolation, no division by calendar duration",
        "ohlc_validation": {"all_positive_finite": True, "bounds_consistent": True, "timestamps_unique_sorted": True,
                            "source_non_one_hour_transition_count": int(gap_transition.sum()),
                            "source_transition_elapsed_hours": _summary(transition_hours)},
        "outer_folds": OUTER_FOLDS,
        "inner_folds": {f: list(v) for f, v in INNER_FOLDS.items()},
        "grid": {"W": list(WINDOWS), "H_observed_transitions": list(HORIZONS), "masks": 6},
        "expected_inner_fits": 4 * 24 * 6 * 2 * 4,
        "maximum_outer_refits": 8,
        "fit_performed": False,
        "coverage_file": "common_origin_time_coverage.csv",
        "elapsed_audit_file": "elapsed_time_audit.csv",
    })
    (output / "sample_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
