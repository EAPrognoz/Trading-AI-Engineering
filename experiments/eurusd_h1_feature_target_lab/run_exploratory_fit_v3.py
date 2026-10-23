"""Bounded, nested, CPU-only exploratory Ridge selection for EURUSD H1.

Reads only the pinned 10,621-row pre-test prefix. Every adjacent observed row
is retained as one transition; elapsed time is audit metadata, not a predictor.
"""

from __future__ import annotations

import csv
import argparse
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from feature_target_selection_v3 import (
    build_feature_matrix, build_forward_targets, candidate_feature_schema,
    purged_split_indices,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
SOURCE = ROOT / ".local/h1_mlp_source/EURUSD_Model_Error_Lab_H1_UTC_20241016_20261017.csv"
PREFIX_ROWS = 10_621
PREFIX_SHA256 = "141318d453eb62941f0cf7d03d6a516770cee52a5131fb15d44fb4c5f0e9a270"
FULL_SHA256_PRIOR_VERIFIED = "74a4f9377dff438968301a6cfe673bab43ce9083add994db79c3e2910745c202"
HOUR = pd.Timedelta(1, unit="h")
WINDOWS = (16, 32, 64, 96)
HORIZONS = tuple(range(1, 25))
MASKS: dict[str, set[str]] = {
    "core": {"own_ohlc_log_ratios", "same_bar_cross_ratios", "observed_transition_gap"},
    "full": {
        "own_ohlc_log_ratios", "same_bar_cross_ratios", "observed_transition_gap", "lagged_cross_ratios",
        "log_price_differences", "single_series_moments", "mixed_central_moments",
    },
    "full_minus_lagged_cross": {
        "own_ohlc_log_ratios", "same_bar_cross_ratios", "observed_transition_gap", "log_price_differences",
        "single_series_moments", "mixed_central_moments",
    },
    "full_minus_log_differences": {
        "own_ohlc_log_ratios", "same_bar_cross_ratios", "observed_transition_gap", "lagged_cross_ratios",
        "single_series_moments", "mixed_central_moments",
    },
    "full_minus_single_series_moments": {
        "own_ohlc_log_ratios", "same_bar_cross_ratios", "observed_transition_gap", "lagged_cross_ratios",
        "log_price_differences", "mixed_central_moments",
    },
    "full_minus_mixed_moments": {
        "own_ohlc_log_ratios", "same_bar_cross_ratios", "observed_transition_gap", "lagged_cross_ratios",
        "log_price_differences", "single_series_moments",
    },
}
MASK_ORDER = tuple(MASKS)
MAX_FEATURES = 256
EXPECTED_INNER_FITS = len(WINDOWS) * len(HORIZONS) * len(MASKS) * 2 * 4
MAX_TOTAL_FITS = EXPECTED_INNER_FITS + 8
CPU_LIMIT_SECONDS = 900.0
CPU_STOP_BEFORE_SECONDS = 855.0  # reserve time to persist a valid partial run
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


def _sha256_prefix(path: Path, rows: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for _ in range(rows + 1):
            line = source.readline()
            if not line:
                raise ValueError("CSV ended before the pinned prefix")
            digest.update(line)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_ns(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series, utc=True).array.asi8


def _target_panel(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(frame)
    mean = np.full((n, 25), np.nan, dtype=np.float64)
    variance = np.full((n, 25), np.nan, dtype=np.float64)
    valid = np.zeros((n, 25), dtype=bool)
    gap_crossing = np.zeros((n, 25), dtype=bool)
    gap_count = np.full((n, 25), -1, dtype=np.int16)
    target_ns = np.full((n, 25), np.iinfo(np.int64).min, dtype=np.int64)
    elapsed_hours = np.full((n, 25), np.nan, dtype=np.float64)
    decision_ns = _as_ns(pd.to_datetime(frame["time_utc"], utc=True) + HOUR)
    for horizon in HORIZONS:
        target = build_forward_targets(frame, horizon)
        mean[:, horizon] = target["y_mean_log_return_per_observed_transition"].to_numpy(dtype=float)
        variance[:, horizon] = target["y_var_log_return_per_observed_transition_ddof0"].to_numpy(dtype=float)
        valid[:, horizon] = target["target_valid"].to_numpy(dtype=bool)
        gap_crossing[:, horizon] = target["target_crosses_non_one_hour_transition"].to_numpy(dtype=bool)
        gap_count[:, horizon] = target["target_non_one_hour_transition_count"].to_numpy(dtype=np.int16)
        target_ns[:, horizon] = _as_ns(target["target_timestamp"])
        elapsed_hours[:, horizon] = target["target_elapsed_calendar_hours"].to_numpy(dtype=float)
    return {
        "mean": mean,
        "variance": variance,
        "valid": valid,
        "gap_crossing": gap_crossing,
        "gap_count": gap_count,
        "target_ns": target_ns,
        "decision_ns": decision_ns,
        "elapsed_hours": elapsed_hours,
    }


def _trailing16_variance(frame: pd.DataFrame) -> np.ndarray:
    close = frame["close"].to_numpy(dtype=float)
    result = np.full(len(frame), np.nan, dtype=np.float64)
    log_close = np.log(close)
    for row in range(16, len(frame)):
        result[row] = float(np.var(np.diff(log_close[row - 16:row + 1]), ddof=0))
    return result


def _split_indices(
    common: np.ndarray,
    decision_ns: np.ndarray,
    target24_ns: np.ndarray,
    start_text: str,
    end_text: str,
) -> tuple[np.ndarray, np.ndarray]:
    return purged_split_indices(common, decision_ns, target24_ns, start_text, end_text)


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _finite_or_none(value: float | int | np.number | None) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _mae(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(y - prediction)))


def _rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - prediction))))


def _skill(model_mae: float, baseline_mae: float) -> float:
    return float("nan") if baseline_mae == 0.0 else 1.0 - model_mae / baseline_mae


def _balanced_nonzero_accuracy(y: np.ndarray, prediction: np.ndarray) -> float:
    nonzero = y != 0.0
    if not np.any(nonzero):
        return float("nan")
    actual_sign = np.sign(y[nonzero])
    predicted_sign = np.sign(prediction[nonzero])
    classes = np.unique(actual_sign)
    recalls = [float(np.mean(predicted_sign[actual_sign == c] == c)) for c in classes]
    return float(np.mean(recalls))


def _metrics(
    y_mean: np.ndarray,
    y_var: np.ndarray,
    pred_mean: np.ndarray,
    pred_var_raw: np.ndarray | None,
    train_mean_baseline: float,
    train_var_mean_baseline: float | None,
    past16_observed_baseline: np.ndarray,
) -> dict[str, float | int | None]:
    zero_mean = np.zeros_like(y_mean)
    mean_model_mae = _mae(y_mean, pred_mean)
    mean_zero_mae = _mae(y_mean, zero_mean)
    mean_train_mae = _mae(y_mean, np.full_like(y_mean, train_mean_baseline))
    result: dict[str, float | int | None] = {
        "mean_mae": mean_model_mae,
        "mean_rmse": _rmse(y_mean, pred_mean),
        "mean_skill_zero": _skill(mean_model_mae, mean_zero_mae),
        "mean_skill_train_mean": _skill(mean_model_mae, mean_train_mae),
        "mean_zero_baseline_mae": mean_zero_mae,
        "mean_train_mean_baseline_mae": mean_train_mae,
        "mean_exact_sign_accuracy": float(np.mean(np.sign(y_mean) == np.sign(pred_mean))),
        "mean_balanced_nonzero_sign_accuracy": _balanced_nonzero_accuracy(y_mean, pred_mean),
        "mean_actual_zero_count": int(np.sum(y_mean == 0.0)),
        "mean_predicted_zero_count": int(np.sum(pred_mean == 0.0)),
        "variance_mae": None,
        "variance_rmse": None,
        "variance_skill_train_mean": None,
        "variance_skill_past16_observed": None,
        "variance_train_mean_baseline_mae": None,
        "variance_past16_observed_baseline_mae": None,
        "variance_negative_raw_fraction": None,
        "variance_negative_raw_count": None,
        "variance_prediction_count": None,
    }
    if pred_var_raw is None or train_var_mean_baseline is None:
        return result
    pred_var = np.maximum(0.0, pred_var_raw)
    model_var_mae = _mae(y_var, pred_var)
    train_var_pred = np.full_like(y_var, train_var_mean_baseline)
    train_var_mae = _mae(y_var, train_var_pred)
    past16_mae = _mae(y_var, past16_observed_baseline)
    result.update({
        "variance_mae": model_var_mae,
        "variance_rmse": _rmse(y_var, pred_var),
        "variance_skill_train_mean": _skill(model_var_mae, train_var_mae),
        "variance_skill_past16_observed": _skill(model_var_mae, past16_mae),
        "variance_train_mean_baseline_mae": train_var_mae,
        "variance_past16_observed_baseline_mae": past16_mae,
        "variance_negative_raw_fraction": float(np.mean(pred_var_raw < 0.0)),
        "variance_negative_raw_count": int(np.sum(pred_var_raw < 0.0)),
        "variance_prediction_count": int(len(pred_var_raw)),
    })
    return result


def _calibration_rows(fit_id: int, info: dict[str, Any], y_true: np.ndarray, pred_raw: np.ndarray) -> list[dict[str, Any]]:
    pred = np.maximum(0.0, pred_raw)
    order = np.argsort(pred, kind="stable")
    groups = np.array_split(order, min(5, len(order)))
    rows = []
    for bin_id, idx in enumerate(groups, start=1):
        rows.append({
            "fit_id": fit_id,
            **info,
            "bin": bin_id,
            "n": int(len(idx)),
            "mean_predicted_variance": float(np.mean(pred[idx])),
            "mean_realized_variance": float(np.mean(y_true[idx])),
        })
    return rows


def _write_prediction_rows(
    writer: csv.writer,
    fit_id: int,
    info: dict[str, Any],
    row_idx: np.ndarray,
    target_panel: dict[str, np.ndarray],
    pred_mean: np.ndarray,
    pred_var_raw: np.ndarray | None,
    train_mean: float,
    train_var_mean: float | None,
    trailing16: np.ndarray,
    input_history_elapsed_by_window: dict[int, np.ndarray],
    input_history_gap_count_by_window: dict[int, np.ndarray],
    timestamp_text: list[str],
    decision_text: list[str],
) -> None:
    horizon = int(info["H"])
    for j, origin in enumerate(row_idx):
        pred_var_value = "" if pred_var_raw is None else float(pred_var_raw[j])
        clipped_var = "" if pred_var_raw is None else max(0.0, float(pred_var_raw[j]))
        var_baseline = "" if train_var_mean is None else float(train_var_mean)
        past16_value = "" if horizon == 1 else float(trailing16[origin])
        target_ns = int(target_panel["target_ns"][origin, horizon])
        target_text = "" if target_ns == np.iinfo(np.int64).min else pd.Timestamp(target_ns, tz="UTC").isoformat()
        writer.writerow([
            fit_id, info["stage"], info["outer_fold"], info.get("inner_fold", ""), info["track"],
            info["W"], horizon, info["mask_id"], int(origin), timestamp_text[origin],
            decision_text[origin], target_text, float(target_panel["mean"][origin, horizon]),
            float(target_panel["variance"][origin, horizon]), float(pred_mean[j]), pred_var_value,
            clipped_var, 0.0, float(train_mean), var_baseline, past16_value,
            float(input_history_elapsed_by_window[int(info["W"])][origin]),
            float(target_panel["elapsed_hours"][origin, horizon]),
            int(target_panel["gap_count"][origin, horizon]),
            bool(target_panel["gap_crossing"][origin, horizon]),
            int(input_history_gap_count_by_window[int(info["W"])][origin]),
        ])


def _record_model(
    store: dict[str, np.ndarray],
    fit_id: int,
    feature_indices: np.ndarray,
    output_count: int,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    model: Ridge,
) -> None:
    store["coef"][fit_id, :output_count][:, feature_indices] = np.asarray(model.coef_).reshape(output_count, -1)
    store["intercept"][fit_id, :output_count] = np.asarray(model.intercept_).reshape(-1)
    store["x_mean"][fit_id, feature_indices] = x_scaler.mean_
    store["x_scale"][fit_id, feature_indices] = x_scaler.scale_
    store["feature_mask"][fit_id, feature_indices] = True
    store["y_mean"][fit_id, :output_count] = y_scaler.mean_
    store["y_scale"][fit_id, :output_count] = y_scaler.scale_
    store["output_count"][fit_id] = output_count


def _fit_model(
    X: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    feature_indices: np.ndarray,
) -> tuple[StandardScaler, StandardScaler, Ridge, np.ndarray]:
    X_train = X[np.ix_(train_idx, feature_indices)]
    X_valid = X[np.ix_(validation_idx, feature_indices)]
    x_scaler = StandardScaler().fit(X_train)
    X_train_scaled = x_scaler.transform(X_train)
    X_valid_scaled = x_scaler.transform(X_valid)
    y_train = target[train_idx]
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    y_scaler = StandardScaler().fit(y_train)
    y_train_scaled = y_scaler.transform(y_train)
    model = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky")
    model.fit(X_train_scaled, y_train_scaled)
    prediction_scaled = model.predict(X_valid_scaled)
    if prediction_scaled.ndim == 1:
        prediction_scaled = prediction_scaled[:, None]
    prediction_raw = y_scaler.inverse_transform(prediction_scaled)
    return x_scaler, y_scaler, model, prediction_raw


def _pareto_frontier(candidates: list[dict[str, Any]], xkey: str, ykey: str | None = None) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if ykey is None:
        best = max(float(c[xkey]) for c in candidates)
        return [c for c in candidates if float(c[xkey]) == best]
    frontier = []
    for candidate in candidates:
        x, y = float(candidate[xkey]), float(candidate[ykey])
        dominated = any(
            (float(other[xkey]) >= x and float(other[ykey]) >= y)
            and (float(other[xkey]) > x or float(other[ykey]) > y)
            for other in candidates
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE,
                        help="authorized EURUSD H1 CSV; the pinned 10,621-row prefix hash is required")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_DIR / "results")
    args = parser.parse_args()
    source = args.source.resolve()
    output_root = args.output_root.resolve()
    # Apply a process-wide one-thread limit before reading or transforming data.
    thread_limit_guard = threadpool_limits(limits=1)
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    run_id = datetime.now(timezone.utc).strftime("eurusd_h1_feature_target_v3_%Y%m%dT%H%M%SZ")
    out = output_root / run_id
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {out}")
    out.mkdir(parents=True, exist_ok=False)
    prefix_hash = _sha256_prefix(source, PREFIX_ROWS)
    if prefix_hash != PREFIX_SHA256:
        raise ValueError(f"pinned source prefix changed: {prefix_hash}")
    protocol_path = EXPERIMENT_DIR / "protocol_feature_target_selection_v3.json"
    protocol_hash = _sha256_file(protocol_path)
    for input_name in (
        "candidate_feature_schema.json",
        "sample_feasibility_by_W_H.csv", "sample_feasibility_summary.json",
        "common_origin_time_coverage.csv", "elapsed_time_audit.csv", "README.md",
    ):
        shutil.copy2(EXPERIMENT_DIR / input_name, out / input_name)
    # Preserve the established downstream artifact name inside this run folder.
    shutil.copy2(protocol_path, out / "protocol_frozen_pre_fit.json")
    for code_name in ("feature_target_selection_v3.py", "run_feasibility_v3.py", "run_exploratory_fit_v3.py"):
        shutil.copy2(EXPERIMENT_DIR / code_name, out / code_name)
    shutil.copy2(EXPERIMENT_DIR / "tests/test_eurusd_feature_target_v3.py", out / "test_eurusd_feature_target_v3.py")
    frame = pd.read_csv(source, nrows=PREFIX_ROWS)
    if len(frame) != PREFIX_ROWS:
        raise ValueError(f"expected {PREFIX_ROWS} prefix rows, read {len(frame)}")
    timestamps = pd.to_datetime(frame["time_utc"], utc=True).reset_index(drop=True)
    timestamp_text = [x.isoformat() for x in timestamps]
    decision_ts = timestamps + HOUR
    decision_text = [x.isoformat() for x in decision_ts]
    target_panel = _target_panel(frame)
    trailing16 = _trailing16_variance(frame)
    timestamp_ns = timestamps.array.asi8
    input_history_elapsed_by_window: dict[int, np.ndarray] = {}
    input_history_gap_count_by_window: dict[int, np.ndarray] = {}
    transition_gap = np.zeros(len(frame), dtype=bool)
    if len(frame) > 1:
        delta_hours = timestamps.diff().dt.total_seconds().to_numpy(dtype=float) / 3600.0
        transition_gap[1:] = ~np.isclose(delta_hours[1:], 1.0)
    gap_prefix = np.r_[0, np.cumsum(transition_gap.astype(np.int32))]
    for window in WINDOWS:
        elapsed = np.full(len(frame), np.nan, dtype=np.float64)
        gap_count = np.full(len(frame), -1, dtype=np.int16)
        rows = np.arange(window, len(frame))
        elapsed[rows] = (timestamp_ns[rows] - timestamp_ns[rows - window]) / 3.6e12
        gap_count[rows] = gap_prefix[rows + 1] - gap_prefix[rows - window + 1]
        input_history_elapsed_by_window[window] = elapsed
        input_history_gap_count_by_window[window] = gap_count

    X_by_window: dict[int, np.ndarray] = {}
    valid_by_window: dict[int, np.ndarray] = {}
    skew_by_window: dict[int, np.ndarray] = {}
    schemas: dict[int, list[dict[str, Any]]] = {}
    feature_names_by_window: dict[int, list[str]] = {}
    feature_indices_by_window_mask: dict[tuple[int, str], np.ndarray] = {}
    for window in WINDOWS:
        feature_frame, skew_mask, valid = build_feature_matrix(frame, window)
        X = feature_frame.to_numpy(dtype=np.float64)
        if not np.isfinite(X[valid.to_numpy()]).all():
            raise ValueError(f"nonfinite feature value on valid W={window} origins")
        schema = candidate_feature_schema(window)
        names = [str(x["feature_name"]) for x in schema]
        groups = [str(x["group"]) for x in schema]
        X_by_window[window] = X
        valid_by_window[window] = valid.to_numpy(dtype=bool)
        skew_by_window[window] = skew_mask.to_numpy(dtype=bool)
        schemas[window] = schema
        feature_names_by_window[window] = names
        for mask_id, included_groups in MASKS.items():
            indices = np.asarray([i for i, group in enumerate(groups) if group in included_groups], dtype=np.int32)
            if not len(indices):
                raise ValueError(f"empty feature mask: {mask_id}")
            feature_indices_by_window_mask[(window, mask_id)] = indices

    common = valid_by_window[96] & target_panel["valid"][:, 24]
    expected_common = int(json.loads((EXPERIMENT_DIR / "sample_feasibility_summary.json").read_text())["common_origins_total"])
    if int(common.sum()) != expected_common:
        raise ValueError(f"common prefix cohort changed from pre-fit feasibility: {int(common.sum())} != {expected_common}")
    target24_ns = target_panel["target_ns"][:, 24]
    decision_ns = target_panel["decision_ns"]
    outer_splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    inner_splits: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    membership_rows: list[dict[str, Any]] = []
    boundary_receipts: list[dict[str, Any]] = []

    def record_split(fold_id: str, split_id: str, train_idx: np.ndarray, valid_idx: np.ndarray,
                     boundary_start: str, boundary_end: str) -> None:
        start_ns, end_ns = pd.Timestamp(boundary_start).value, pd.Timestamp(boundary_end).value
        max_train_target = int(target24_ns[train_idx].max()) if len(train_idx) else None
        max_validation_target = int(target24_ns[valid_idx].max()) if len(valid_idx) else None
        assert len(train_idx) == 0 or max_train_target < start_ns
        assert len(valid_idx) == 0 or max_validation_target < end_ns
        assert len(train_idx) == 0 or np.all(decision_ns[train_idx] < start_ns)
        assert len(valid_idx) == 0 or np.all((decision_ns[valid_idx] >= start_ns) & (decision_ns[valid_idx] < end_ns))
        for horizon in HORIZONS:
            target_h_ns = target_panel["target_ns"][:, horizon]
            assert len(train_idx) == 0 or np.all(target_h_ns[train_idx] < start_ns)
            assert len(valid_idx) == 0 or np.all(target_h_ns[valid_idx] < end_ns)
        for partition, indices in (("train", train_idx), ("validation", valid_idx)):
            for idx in indices:
                membership_rows.append({
                    "outer_fold": fold_id,
                    "split_id": split_id,
                    "partition": partition,
                    "origin_row": int(idx),
                    "origin_bar_open_timestamp": timestamp_text[idx],
                    "decision_timestamp": decision_text[idx],
                    "target24_timestamp": pd.Timestamp(target24_ns[idx], tz="UTC").isoformat(),
                })
        boundary_receipts.append({
            "outer_fold": fold_id,
            "split_id": split_id,
            "start_utc": boundary_start,
            "end_exclusive_utc": boundary_end,
            "train_count": int(len(train_idx)),
            "validation_count": int(len(valid_idx)),
            "train_origin_index_sha256": _array_hash(train_idx),
            "validation_origin_index_sha256": _array_hash(valid_idx),
            "train_max_target_timestamp": None if max_train_target is None else pd.Timestamp(max_train_target, tz="UTC").isoformat(),
            "validation_max_target_timestamp": None if max_validation_target is None else pd.Timestamp(max_validation_target, tz="UTC").isoformat(),
            "checks": {
                "train_target_strictly_before_start": True,
                "validation_decision_inside_interval": True,
                "validation_target_strictly_before_end": True,
                "all_H_train_targets_strictly_before_start": True,
                "all_H_validation_targets_strictly_before_end": True,
                "common_W96_H24_origin_cohort_only": True,
                "no_origin_or_target_excluded_for_gaps": True,
                "gap_crossing_status_is_audit_metadata_only": True,
            },
        })

    for fold, (start, end) in OUTER_FOLDS.items():
        train, valid = _split_indices(common, decision_ns, target24_ns, start, end)
        outer_splits[fold] = (train, valid)
        record_split(fold, "outer", train, valid, start, end)
        for inner_id, inner_start, inner_end in INNER_FOLDS[fold]:
            inner_train, inner_valid = _split_indices(common, decision_ns, target24_ns, inner_start, inner_end)
            inner_splits[(fold, inner_id)] = (inner_train, inner_valid)
            record_split(fold, inner_id, inner_train, inner_valid, inner_start, inner_end)

    # Persist reusable inputs and split/target contracts before any estimator fit.
    feature_arrays: dict[str, np.ndarray] = {}
    skew_arrays: dict[str, np.ndarray] = {}
    feature_catalog: dict[str, list[str]] = {}
    for window in WINDOWS:
        feature_arrays[f"X_W{window}"] = X_by_window[window]
        feature_arrays[f"valid_W{window}"] = valid_by_window[window]
        skew_arrays[f"skew_defined_W{window}"] = skew_by_window[window]
        feature_catalog[str(window)] = feature_names_by_window[window]
    np.savez_compressed(out / "feature_matrices.npz", **feature_arrays)
    np.savez_compressed(out / "skewness_defined_masks.npz", **skew_arrays)
    pd.DataFrame(membership_rows).to_csv(out / "split_membership.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1})
    (out / "boundary_receipts.json").write_text(json.dumps(boundary_receipts, indent=2, allow_nan=False) + "\n")
    history_audit = []
    for window in WINDOWS:
        for origin in np.flatnonzero(common):
            history_audit.append({
                "W": window,
                "origin_row": int(origin),
                "origin_bar_open_timestamp": timestamp_text[origin],
                "decision_timestamp": decision_text[origin],
                "input_history_elapsed_calendar_hours_audit_only": float(input_history_elapsed_by_window[window][origin]),
                "input_history_non_one_hour_transition_count_audit_only": int(input_history_gap_count_by_window[window][origin]),
            })
    pd.DataFrame(history_audit).to_csv(
        out / "history_window_audit.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1})
    target_rows: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        target_rows.append(pd.DataFrame({
            "origin_row": np.arange(len(frame), dtype=np.int32),
            "origin_bar_open_timestamp": timestamp_text,
            "decision_timestamp": decision_text,
            "target_timestamp": [
                "" if value == np.iinfo(np.int64).min else pd.Timestamp(int(value), tz="UTC").isoformat()
                for value in target_panel["target_ns"][:, horizon]
            ],
            "H": horizon,
            "y_mean_log_return_per_observed_transition": target_panel["mean"][:, horizon],
            "y_var_log_return_per_observed_transition_ddof0": target_panel["variance"][:, horizon],
            "target_valid_next_observed_H": target_panel["valid"][:, horizon],
            "target_non_one_hour_transition_count": target_panel["gap_count"][:, horizon],
            "target_crosses_non_one_hour_transition": target_panel["gap_crossing"][:, horizon],
            "target_elapsed_calendar_hours_audit_only": target_panel["elapsed_hours"][:, horizon],
            "variance_evaluable": horizon >= 2,
        }))
    pd.concat(target_rows, ignore_index=True).to_csv(
        out / "target_manifest.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1},
    )
    (out / "feature_catalog.json").write_text(json.dumps({"schema": schemas, "names_by_W": feature_catalog}, indent=2) + "\n")

    model_store = {
        "coef": np.zeros((MAX_TOTAL_FITS, 2, MAX_FEATURES), dtype=np.float64),
        "intercept": np.zeros((MAX_TOTAL_FITS, 2), dtype=np.float64),
        "x_mean": np.zeros((MAX_TOTAL_FITS, MAX_FEATURES), dtype=np.float64),
        "x_scale": np.ones((MAX_TOTAL_FITS, MAX_FEATURES), dtype=np.float64),
        "feature_mask": np.zeros((MAX_TOTAL_FITS, MAX_FEATURES), dtype=bool),
        "y_mean": np.zeros((MAX_TOTAL_FITS, 2), dtype=np.float64),
        "y_scale": np.ones((MAX_TOTAL_FITS, 2), dtype=np.float64),
        "output_count": np.zeros(MAX_TOTAL_FITS, dtype=np.int8),
    }
    fit_metadata: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    candidate_metrics: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    fit_id = 0
    completed_inner = 0
    completed_outer = 0
    stopped_for_budget = False
    progress_path = out / "progress.json"
    prediction_path = out / "predictions.csv.gz"
    prediction_header = [
        "fit_id", "stage", "outer_fold", "inner_fold", "track", "W", "H", "mask_id",
        "origin_row", "origin_bar_open_timestamp", "decision_timestamp", "target_timestamp",
        "y_mean_actual", "y_var_actual", "y_mean_prediction", "y_var_raw_prediction",
        "y_var_clipped_prediction", "y_mean_zero_baseline", "y_mean_train_mean_baseline",
        "y_var_train_mean_baseline", "y_var_past16_observed_baseline",
        "input_history_elapsed_calendar_hours_audit_only", "target_elapsed_calendar_hours_audit_only",
        "target_non_one_hour_transition_count", "target_crosses_non_one_hour_transition",
        "input_history_non_one_hour_transition_count_audit_only",
    ]
    wall_fit_start = time.monotonic()
    cpu_fit_start = time.process_time()
    with gzip.open(prediction_path, "wt", newline="", compresslevel=1) as gz:
        writer = csv.writer(gz)
        writer.writerow(prediction_header)
        with threadpool_limits(limits=1):
            for outer_fold in OUTER_FOLDS:
                for inner_id, inner_start, inner_end in INNER_FOLDS[outer_fold]:
                    train_idx, val_idx = inner_splits[(outer_fold, inner_id)]
                    if not len(train_idx) or not len(val_idx):
                        raise ValueError(f"empty inner split: {outer_fold}/{inner_id}")
                    for window in WINDOWS:
                        X = X_by_window[window]
                        names = feature_names_by_window[window]
                        group_by_name = {str(item["feature_name"]): str(item["group"]) for item in schemas[window]}
                        for mask_id in MASK_ORDER:
                            feature_indices = feature_indices_by_window_mask[(window, mask_id)]
                            if time.process_time() - started_cpu >= CPU_STOP_BEFORE_SECONDS:
                                stopped_for_budget = True
                                break
                            X_train = X[np.ix_(train_idx, feature_indices)]
                            X_valid = X[np.ix_(val_idx, feature_indices)]
                            x_scaler = StandardScaler().fit(X_train)
                            X_train_scaled = x_scaler.transform(X_train)
                            X_valid_scaled = x_scaler.transform(X_valid)
                            for horizon in HORIZONS:
                                if time.process_time() - started_cpu >= CPU_STOP_BEFORE_SECONDS:
                                    stopped_for_budget = True
                                    break
                                output_count = 1 if horizon == 1 else 2
                                if horizon == 1:
                                    y_train = target_panel["mean"][train_idx, horizon, None]
                                    y_valid_mean = target_panel["mean"][val_idx, horizon]
                                    y_valid_var = target_panel["variance"][val_idx, horizon]
                                else:
                                    y_train = np.column_stack((
                                        target_panel["mean"][train_idx, horizon],
                                        target_panel["variance"][train_idx, horizon],
                                    ))
                                    y_valid_mean = target_panel["mean"][val_idx, horizon]
                                    y_valid_var = target_panel["variance"][val_idx, horizon]
                                y_scaler = StandardScaler().fit(y_train)
                                y_train_scaled = y_scaler.transform(y_train)
                                model = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky")
                                model.fit(X_train_scaled, y_train_scaled)
                                pred_scaled = model.predict(X_valid_scaled)
                                if pred_scaled.ndim == 1:
                                    pred_scaled = pred_scaled[:, None]
                                pred_raw = y_scaler.inverse_transform(pred_scaled)
                                pred_mean = pred_raw[:, 0]
                                pred_var_raw = None if horizon == 1 else pred_raw[:, 1]
                                train_mean = float(np.mean(target_panel["mean"][train_idx, horizon]))
                                train_var_mean = None if horizon == 1 else float(np.mean(target_panel["variance"][train_idx, horizon]))
                                info = {
                                    "stage": "inner",
                                    "outer_fold": outer_fold,
                                    "inner_fold": inner_id,
                                    "track": "mean_only" if horizon == 1 else "mean_and_variance",
                                    "W": window,
                                    "H": horizon,
                                    "mask_id": mask_id,
                                }
                                metric = _metrics(
                                    y_valid_mean, y_valid_var, pred_mean, pred_var_raw, train_mean,
                                    train_var_mean, trailing16[val_idx],
                                )
                                fit_record = {
                                    "fit_id": fit_id,
                                    **info,
                                    "n_train": int(len(train_idx)),
                                    "n_validation": int(len(val_idx)),
                                    "n_features": int(len(feature_indices)),
                                    "alpha": 1.0,
                                    "output_count": output_count,
                                    "train_origin_index_sha256": _array_hash(train_idx),
                                    "validation_origin_index_sha256": _array_hash(val_idx),
                                }
                                fit_metadata.append(fit_record)
                                _record_model(model_store, fit_id, feature_indices, output_count, x_scaler, y_scaler, model)
                                metric_row = {**fit_record, **metric}
                                metric_rows.append(metric_row)
                                candidate_metrics[(outer_fold, inner_id, window, horizon, mask_id)] = metric_row
                                _write_prediction_rows(
                                    writer, fit_id, info, val_idx, target_panel, pred_mean, pred_var_raw,
                                    train_mean, train_var_mean, trailing16, input_history_elapsed_by_window,
                                    input_history_gap_count_by_window,
                                    timestamp_text, decision_text,
                                )
                                if pred_var_raw is not None:
                                    calibration_rows.extend(_calibration_rows(
                                        fit_id, info, y_valid_var, pred_var_raw,
                                    ))
                                fit_id += 1
                                completed_inner += 1
                                if completed_inner % 128 == 0:
                                    progress = {
                                        "status": "running",
                                        "completed_inner_fits": completed_inner,
                                        "expected_inner_fits": EXPECTED_INNER_FITS,
                                        "completed_outer_fits": completed_outer,
                                        "process_cpu_seconds": time.process_time() - started_cpu,
                                        "fit_loop_cpu_seconds": time.process_time() - cpu_fit_start,
                                    }
                                    progress_path.write_text(json.dumps(progress, indent=2) + "\n")
                                    print(json.dumps(progress), flush=True)
                            if stopped_for_budget:
                                break
                        if stopped_for_budget:
                            break
                    if stopped_for_budget:
                        break
                if stopped_for_budget:
                    break

    # Write complete inner results even if the CPU stop guard ended the grid early.
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out / "inner_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(out / "variance_calibration_bins.csv", index=False)

    # Paired full-vs-drop deltas are conditional Ridge evidence, not universal
    # feature importance. Preserve each fold, horizon, and target head.
    drop_mask_for_group = {
        "lagged_cross_ratios": "full_minus_lagged_cross",
        "log_price_differences": "full_minus_log_differences",
        "single_series_moments": "full_minus_single_series_moments",
        "mixed_central_moments": "full_minus_mixed_moments",
    }
    group_drop_rows: list[dict[str, Any]] = []
    if completed_inner:
        for (outer_fold, inner_id, window, horizon, mask_id), full_metric in list(candidate_metrics.items()):
            if mask_id != "full":
                continue
            for group, dropped_mask in drop_mask_for_group.items():
                dropped = candidate_metrics.get((outer_fold, inner_id, window, horizon, dropped_mask))
                if dropped is None:
                    continue
                group_drop_rows.append({
                    "outer_fold": outer_fold, "inner_fold": inner_id, "W": window, "H": horizon,
                    "group": group,
                    "delta_mean_skill_full_minus_drop": (
                        None if full_metric["mean_skill_zero"] is None or dropped["mean_skill_zero"] is None
                        else full_metric["mean_skill_zero"] - dropped["mean_skill_zero"]
                    ),
                    "delta_variance_skill_full_minus_drop": (
                        None if horizon == 1 or full_metric["variance_skill_train_mean"] is None
                        or dropped["variance_skill_train_mean"] is None
                        else full_metric["variance_skill_train_mean"] - dropped["variance_skill_train_mean"]
                    ),
                    "interpretation": "conditional usefulness for fixed-alpha Ridge and this registered mask set",
                })
    group_drop_df = pd.DataFrame(group_drop_rows)
    group_drop_df.to_csv(out / "group_drop_deltas.csv", index=False)
    if not group_drop_df.empty:
        group_drop_summary = group_drop_df.groupby(["outer_fold", "group"], dropna=False).agg(
            mean_skill_delta_median=("delta_mean_skill_full_minus_drop", "median"),
            mean_skill_delta_positive_fraction=("delta_mean_skill_full_minus_drop", lambda s: float(np.mean(s.dropna() > 0))),
            variance_skill_delta_median=("delta_variance_skill_full_minus_drop", "median"),
            variance_skill_delta_positive_fraction=("delta_variance_skill_full_minus_drop", lambda s: float(np.mean(s.dropna() > 0))),
            comparisons=("H", "count"),
        ).reset_index()
    else:
        group_drop_summary = pd.DataFrame()
    group_drop_summary.to_csv(out / "group_drop_summary.csv", index=False)
    overall_group_rows = []
    if not group_drop_df.empty:
        for group, subset in group_drop_df.groupby("group", sort=True):
            for target_name, column in (
                ("mean", "delta_mean_skill_full_minus_drop"),
                ("variance", "delta_variance_skill_full_minus_drop"),
            ):
                values = pd.to_numeric(subset[column], errors="coerce").dropna()
                overall_group_rows.append({
                    "group": group,
                    "target": target_name,
                    "delta_definition": "skill(full)-skill(full_without_group); positive favors retaining group",
                    "mean_delta_skill": float(values.mean()) if len(values) else None,
                    "median_delta_skill": float(values.median()) if len(values) else None,
                    "positive_delta_count": int((values > 0).sum()),
                    "comparison_count": int(len(values)),
                    "positive_delta_fraction": float((values > 0).mean()) if len(values) else None,
                })
    pd.DataFrame(overall_group_rows).to_csv(out / "group_drop_overall_summary.csv", index=False)

    frontier_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selected_by_fold: dict[tuple[str, str], dict[str, Any] | None] = {}
    if completed_inner == EXPECTED_INNER_FITS:
        for outer_fold in OUTER_FOLDS:
            inner_ids = [x[0] for x in INNER_FOLDS[outer_fold]]
            # Build candidate aggregates and retain every per-inner measurement.
            candidates_h1: list[dict[str, Any]] = []
            candidates_hmulti: list[dict[str, Any]] = []
            for window in WINDOWS:
                for horizon in HORIZONS:
                    for mask_id in MASK_ORDER:
                        pair = [candidate_metrics[(outer_fold, inner_id, window, horizon, mask_id)] for inner_id in inner_ids]
                        n_features = int(pair[0]["n_features"])
                        if horizon == 1:
                            skills = [float(item["mean_skill_zero"]) for item in pair]
                            eligible = all(np.isfinite(x) and x > 0 for x in skills)
                            candidate = {
                                "outer_fold": outer_fold, "track": "mean_only", "W": window, "H": 1,
                                "mask_id": mask_id, "n_features": n_features,
                                "mean_skill_inner_1": skills[0], "mean_skill_inner_2": skills[1],
                                "median_mean_skill": float(np.median(skills)), "eligible": eligible,
                            }
                            candidates_h1.append(candidate)
                        else:
                            mean_skills = [float(item["mean_skill_zero"]) for item in pair]
                            var_skills = [float(item["variance_skill_train_mean"]) for item in pair]
                            eligible = all(np.isfinite(x) and x > 0 for x in mean_skills + var_skills)
                            candidate = {
                                "outer_fold": outer_fold, "track": "mean_and_variance", "W": window,
                                "H": horizon, "mask_id": mask_id, "n_features": n_features,
                                "mean_skill_inner_1": mean_skills[0], "mean_skill_inner_2": mean_skills[1],
                                "variance_skill_inner_1": var_skills[0], "variance_skill_inner_2": var_skills[1],
                                "median_mean_skill": float(np.median(mean_skills)),
                                "median_variance_skill": float(np.median(var_skills)),
                                "min_median_skill": float(min(np.median(mean_skills), np.median(var_skills))),
                                "eligible": eligible,
                            }
                            candidates_hmulti.append(candidate)
            h1_finite = [c for c in candidates_h1 if np.isfinite(c["median_mean_skill"])]
            hm_finite = [c for c in candidates_hmulti if np.isfinite(c["median_mean_skill"])
                         and np.isfinite(c["median_variance_skill"])]
            front_h1_all = _pareto_frontier(h1_finite, "median_mean_skill")
            front_hm_all = _pareto_frontier(hm_finite, "median_mean_skill", "median_variance_skill")
            h1_eligible = [c for c in candidates_h1 if c["eligible"]]
            hm_eligible = [c for c in candidates_hmulti if c["eligible"]]
            front_h1_eligible = _pareto_frontier(h1_eligible, "median_mean_skill")
            front_hm_eligible = _pareto_frontier(hm_eligible, "median_mean_skill", "median_variance_skill")
            for track, full_frontier, eligible_frontier in (
                ("mean_only", front_h1_all, front_h1_eligible),
                ("mean_and_variance", front_hm_all, front_hm_eligible),
            ):
                eligible_keys = {(c["W"], c["H"], c["mask_id"]) for c in eligible_frontier}
                for candidate in full_frontier:
                    key = (candidate["W"], candidate["H"], candidate["mask_id"])
                    frontier_rows.append({
                        **candidate,
                        "is_frontier": True,
                        "is_eligible_frontier": key in eligible_keys,
                    })
            for track, candidates, frontier in (
                ("mean_only", h1_eligible, front_h1_eligible),
                ("mean_and_variance", hm_eligible, front_hm_eligible),
            ):
                if not candidates:
                    selected_by_fold[(outer_fold, track)] = None
                    selection_rows.append({
                        "outer_fold": outer_fold, "track": track, "status": "abstain",
                        "reason": "no candidate had positive required skill in both inner folds",
                        "eligible_candidate_count": 0,
                        "all_candidate_frontier_count": len(front_h1_all if track == "mean_only" else front_hm_all),
                        "eligible_frontier_count": 0,
                    })
                    continue
                if track == "mean_only":
                    order = sorted(candidates, key=lambda c: (-c["median_mean_skill"], c["n_features"], c["W"], c["mask_id"]))
                else:
                    # The predeclared max-min tie-break selects from all eligible
                    # candidates; the full non-dominated frontier is separately saved.
                    order = sorted(candidates, key=lambda c: (-c["min_median_skill"], c["n_features"], c["W"], c["H"], c["mask_id"]))
                chosen = order[0]
                selected_by_fold[(outer_fold, track)] = chosen
                selection_rows.append({"outer_fold": outer_fold, "track": track, "status": "selected", **chosen,
                                       "eligible_candidate_count": len(candidates),
                                       "all_candidate_frontier_count": len(front_h1_all if track == "mean_only" else front_hm_all),
                                       "eligible_frontier_count": len(frontier)})
    else:
        for outer_fold in OUTER_FOLDS:
            for track in ("mean_only", "mean_and_variance"):
                selected_by_fold[(outer_fold, track)] = None
                selection_rows.append({
                    "outer_fold": outer_fold, "track": track, "status": "partial_inner_grid",
                    "reason": "CPU stop occurred before all inner candidates were evaluated",
                })
    pd.DataFrame(frontier_rows).to_csv(out / "pareto_frontiers.csv", index=False)
    selection_df = pd.DataFrame(selection_rows)

    # Outer validation uses only candidates selected by inner scores. Every
    # fold/track remains in selection_by_outer_fold.csv, including abstentions.
    if completed_inner == EXPECTED_INNER_FITS and not stopped_for_budget:
        outer_prediction_path = out / "outer_predictions.csv.gz"
        outer_metric_rows: list[dict[str, Any]] = []
        outer_calibration_rows: list[dict[str, Any]] = []
        with gzip.open(outer_prediction_path, "wt", newline="", compresslevel=1) as gz:
            writer = csv.writer(gz)
            writer.writerow(prediction_header)
            with threadpool_limits(limits=1):
                for outer_fold, (start, end) in OUTER_FOLDS.items():
                    train_idx, val_idx = outer_splits[outer_fold]
                    for track in ("mean_only", "mean_and_variance"):
                        chosen = selected_by_fold[(outer_fold, track)]
                        if chosen is None:
                            continue
                        if time.process_time() - started_cpu >= CPU_STOP_BEFORE_SECONDS:
                            stopped_for_budget = True
                            break
                        window, horizon, mask_id = int(chosen["W"]), int(chosen["H"]), str(chosen["mask_id"])
                        feature_indices = feature_indices_by_window_mask[(window, mask_id)]
                        if horizon == 1:
                            target_full = target_panel["mean"][:, horizon, None]
                        else:
                            target_full = np.column_stack((target_panel["mean"][:, horizon], target_panel["variance"][:, horizon]))
                        x_scaler, y_scaler, model, pred_raw = _fit_model(
                            X_by_window[window], target_full, train_idx, val_idx, feature_indices,
                        )
                        output_count = 1 if horizon == 1 else 2
                        _record_model(model_store, fit_id, feature_indices, output_count, x_scaler, y_scaler, model)
                        info = {
                            "stage": "outer", "outer_fold": outer_fold, "inner_fold": "",
                            "track": track, "W": window, "H": horizon, "mask_id": mask_id,
                        }
                        fit_record = {
                            "fit_id": fit_id, **info, "n_train": int(len(train_idx)),
                            "n_validation": int(len(val_idx)), "n_features": int(len(feature_indices)),
                            "alpha": 1.0, "output_count": output_count,
                            "train_origin_index_sha256": _array_hash(train_idx),
                            "validation_origin_index_sha256": _array_hash(val_idx),
                        }
                        fit_metadata.append(fit_record)
                        pred_mean = pred_raw[:, 0]
                        pred_var_raw = None if horizon == 1 else pred_raw[:, 1]
                        train_mean = float(np.mean(target_panel["mean"][train_idx, horizon]))
                        train_var_mean = None if horizon == 1 else float(np.mean(target_panel["variance"][train_idx, horizon]))
                        metric = _metrics(
                            target_panel["mean"][val_idx, horizon], target_panel["variance"][val_idx, horizon],
                            pred_mean, pred_var_raw, train_mean, train_var_mean, trailing16[val_idx],
                        )
                        outer_metric_rows.append({**fit_record, **metric})
                        if pred_var_raw is not None:
                            outer_calibration_rows.extend(_calibration_rows(
                                fit_id, info, target_panel["variance"][val_idx, horizon], pred_var_raw,
                            ))
                        _write_prediction_rows(
                            writer, fit_id, info, val_idx, target_panel, pred_mean, pred_var_raw,
                            train_mean, train_var_mean, trailing16, input_history_elapsed_by_window,
                            input_history_gap_count_by_window,
                            timestamp_text, decision_text,
                        )
                        fit_id += 1
                        completed_outer += 1
                    if stopped_for_budget:
                        break
        pd.DataFrame(outer_metric_rows).to_csv(out / "outer_metrics.csv", index=False)
        pd.DataFrame(outer_calibration_rows).to_csv(out / "outer_variance_calibration_bins.csv", index=False)
    else:
        pd.DataFrame(columns=["fit_id", "stage", "outer_fold", "track"]).to_csv(out / "outer_metrics.csv", index=False)
        pd.DataFrame().to_csv(out / "outer_variance_calibration_bins.csv", index=False)

    selection_df.to_csv(out / "selection_by_outer_fold.csv", index=False)
    pd.DataFrame(fit_metadata).to_csv(out / "model_fit_index.csv", index=False)
    np.savez_compressed(out / "model_coefficients_and_scalers.npz", **model_store)

    run_wall = time.monotonic() - started_wall
    run_cpu = time.process_time() - started_cpu
    fit_wall = time.monotonic() - wall_fit_start
    fit_cpu = time.process_time() - cpu_fit_start
    try:
        import psutil
        cpu_info = {"logical_cpu_count": psutil.cpu_count(logical=True), "physical_cpu_count": psutil.cpu_count(logical=False)}
    except Exception:
        cpu_info = {"logical_cpu_count": None, "physical_cpu_count": None}
    feasibility_summary = json.loads((out / "sample_feasibility_summary.json").read_text())
    summary = {
        "run_id": run_id,
        "status": "complete" if completed_inner == EXPECTED_INNER_FITS and completed_outer == int((selection_df.status == "selected").sum()) else "partial",
        "model_fit_performed": completed_inner > 0 or completed_outer > 0,
        "protocol_frozen_file": "protocol_frozen_pre_fit.json",
        "protocol_frozen_sha256": protocol_hash,
        "source_csv": source.name,
        "source_full_sha256_prior_verified": FULL_SHA256_PRIOR_VERIFIED,
        "source_full_sha256_rehashed": False,
        "read_rows": len(frame),
        "test_rows_read": 0,
        "prefix_sha256_verified": prefix_hash,
        "common_origin_counts_by_split": feasibility_summary.get("common_origins_by_split"),
        "process_cpu_seconds_total": run_cpu,
        "wall_seconds_total": run_wall,
        "process_cpu_seconds_fit_and_selection": fit_cpu,
        "wall_seconds_fit_and_selection": fit_wall,
        "cpu_budget_seconds": CPU_LIMIT_SECONDS,
        "cpu_stop_guard_seconds": CPU_STOP_BEFORE_SECONDS,
        "completed_inner_fits": completed_inner,
        "expected_inner_fits": EXPECTED_INNER_FITS,
        "completed_outer_refits": completed_outer,
        "maximum_outer_refits": 8,
        "completed_total_fits": completed_inner + completed_outer,
        "maximum_total_fits": MAX_TOTAL_FITS,
        "stopped_for_cpu_budget": stopped_for_budget,
        "alpha": 1.0,
        "thread_count": 1,
        "device": {"type": "CPU", "processor": platform.processor(), "machine": platform.machine(), **cpu_info},
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
        },
        "thread_pools_at_run_end": threadpool_info(),
        "git_head": _git_head(),
        "privacy": {"raw_csv_in_package": False, "library_sharing_requested": False},
        "test_status": "not accessed",
        "claims": {"profitability": False, "statistical_significance": False, "universal_feature_importance": False},
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    progress_path.write_text(json.dumps({
        "status": summary["status"], "completed_inner_fits": completed_inner,
        "expected_inner_fits": EXPECTED_INNER_FITS, "completed_outer_fits": completed_outer,
        "process_cpu_seconds": run_cpu, "wall_seconds": run_wall,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
