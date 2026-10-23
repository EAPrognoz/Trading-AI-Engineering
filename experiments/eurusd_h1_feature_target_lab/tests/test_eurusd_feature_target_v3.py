"""Contract tests for the v3 observed-transition semantics (fixtures only)."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_target_selection_v3 import (
    build_feature_matrix,
    build_forward_targets,
    candidate_feature_schema,
    common_origin_mask,
    purged_split_indices,
    validate_ohlc_frame,
)


def _fixture(n: int = 210, *, gaps: bool = True) -> pd.DataFrame:
    # Deterministic valid fixture for contract checks; this is not market data
    # and is never used in the research run.
    times = pd.date_range("2025-01-01T00:00:00Z", periods=n, freq="h")
    if gaps:
        times = times.to_series().reset_index(drop=True)
        times.iloc[20:] += pd.Timedelta("4h")
        times.iloc[137:] += pd.Timedelta("26h")
        times = pd.DatetimeIndex(times)
    t = np.arange(n, dtype=float)
    close = 1.05 + 0.0002 * t + 0.00001 * np.sin(t / 3)
    open_ = close * (1.0 + 0.00003 * np.cos(t / 5))
    high = np.maximum(open_, close) + 0.0001
    low = np.minimum(open_, close) - 0.0001
    return pd.DataFrame({"time_utc": times, "open": open_, "high": high, "low": low, "close": close})


def test_gap_transition_is_kept_in_feature_history_and_moments():
    frame = _fixture()
    W = 16
    X, skew, valid = build_feature_matrix(frame, W)
    assert valid.sum() == len(frame) - W
    assert valid.iloc[16:].all()
    name = "observed_transition_gap_log_open_prev_close"
    assert name in X
    expected_gap = np.log(frame.loc[20, "open"] / frame.loc[19, "close"])
    assert X.loc[20, name] == pytest.approx(expected_gap)
    expected_transition = np.log(frame.loc[20, "close"] / frame.loc[19, "close"])
    assert X.loc[20, "log_price_difference_order_1_close"] == pytest.approx(expected_transition)
    # Sixteen observed close transitions ending at row 20 include the gap row.
    expected_mean = np.mean(np.diff(np.log(frame.loc[4:20, "close"].to_numpy())))
    got_mean = X.loc[20, "univariate_mean_raw_return_close_w16"]
    assert got_mean == pytest.approx(expected_mean)
    assert np.isfinite(X.loc[valid].to_numpy()).all()
    assert not any("elapsed" in column.lower() or "timestamp" in column.lower() for column in X.columns)
    assert skew.shape == X.shape


def test_targets_use_observed_steps_without_calendar_time_rescaling():
    frame = _fixture()
    targets = build_forward_targets(frame, 2)
    origin = 18
    step_returns = np.diff(np.log(frame.loc[origin:origin + 2, "close"].to_numpy()))
    assert targets.loc[origin, "target_valid"]
    assert targets.loc[origin, "y_mean_log_return_per_observed_transition"] == pytest.approx(step_returns.mean())
    assert targets.loc[origin, "y_var_log_return_per_observed_transition_ddof0"] == pytest.approx(step_returns.var(ddof=0))
    assert targets.loc[origin, "target_non_one_hour_transition_count"] == 1
    expected_elapsed = (frame.loc[origin + 2, "time_utc"] - frame.loc[origin, "time_utc"]).total_seconds() / 3600
    assert targets.loc[origin, "target_elapsed_calendar_hours"] == expected_elapsed
    assert targets.loc[origin, "target_timestamp"] == frame.loc[origin + 2, "time_utc"] + pd.Timedelta("1h")


def test_h1_variance_is_exact_zero_and_not_a_gap_filter():
    frame = _fixture()
    targets = build_forward_targets(frame, 1)
    assert targets.loc[targets.target_valid, "y_var_log_return_per_observed_transition_ddof0"].eq(0.0).all()
    assert targets.loc[19, "target_valid"]
    assert targets.loc[20, "target_valid"]
    assert targets.loc[19, "target_non_one_hour_transition_count"] == 1


def test_schema_has_unique_core_gap_feature_and_no_elapsed_predictor():
    for W in (16, 32, 64, 96):
        schema = candidate_feature_schema(W)
        names = [x["feature_name"] for x in schema]
        assert len(names) == len(set(names))
        gap = [x for x in schema if x["group"] == "observed_transition_gap"]
        assert len(gap) == 1
        assert gap[0]["feature_name"] == "observed_transition_gap_log_open_prev_close"
        assert all(x["uses_elapsed_time_as_predictor"] is False for x in schema)
    assert len(candidate_feature_schema(96)) <= 256


def test_full_w96_candidate_matrix_is_finite_on_paired_origins():
    frame = _fixture(210)
    X, skew, valid = build_feature_matrix(frame, 96)
    assert valid.sum() == len(frame) - 96
    assert np.isfinite(X.loc[valid].to_numpy()).all()
    assert skew.shape == X.shape


def test_duplicate_or_unordered_timestamps_and_invalid_ohlc_are_rejected():
    frame = _fixture(30, gaps=False)
    duplicate = frame.copy()
    duplicate.loc[2, "time_utc"] = duplicate.loc[1, "time_utc"]
    with pytest.raises(ValueError, match="unique"):
        validate_ohlc_frame(duplicate)
    unordered = frame.copy()
    unordered.loc[2, "time_utc"], unordered.loc[3, "time_utc"] = unordered.loc[3, "time_utc"], unordered.loc[2, "time_utc"]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_ohlc_frame(unordered)
    bad = frame.copy()
    bad.loc[5, "high"] = bad.loc[5, "low"] - 0.001
    with pytest.raises(ValueError, match="high violates"):
        validate_ohlc_frame(bad)


def test_zero_variance_skewness_is_guarded():
    n = 30
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2025-01-01T00:00:00Z", periods=n, freq="h"),
        "open": np.ones(n), "high": np.ones(n), "low": np.ones(n), "close": np.ones(n),
    })
    X, skew, valid = build_feature_matrix(frame, 16)
    name = "univariate_skewness_return_close_w16"
    assert valid.iloc[16:].all()
    assert not valid.iloc[:16].any()
    assert X.loc[valid, name].eq(0.0).all()
    assert not skew.loc[valid, name].any()


def test_common_cohort_and_target_end_purging_are_explicit():
    common = common_origin_mask(130, max_window=96, max_horizon=24)
    assert np.flatnonzero(common).tolist() == list(range(96, 106))
    base = pd.Timestamp("2025-01-01T00:00:00Z").value
    decision = base + np.array([8, 9, 10, 11, 12, 13], dtype=np.int64) * 3_600_000_000_000
    target24 = base + np.array([9, 10, 11, 13, 14, 15], dtype=np.int64) * 3_600_000_000_000
    train, validation = purged_split_indices(
        np.ones(6, dtype=bool), decision, target24,
        "2025-01-01T10:00:00Z", "2025-01-01T14:00:00Z",
    )
    assert train.tolist() == [0]
    assert validation.tolist() == [2, 3]
    assert np.all(target24[train] < base + 10 * 3_600_000_000_000)
    assert np.all(target24[validation] < base + 14 * 3_600_000_000_000)
