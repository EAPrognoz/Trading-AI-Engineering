"""Point-in-time EURUSD H1 features and next-observed-transition targets.

This revision treats every adjacent source row as one observed transition, even
when its UTC timestamps are separated by more than one hour. No bars are
interpolated, no history is reset at gaps, and no return is divided by elapsed
calendar time. Elapsed time is audit metadata only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations, permutations

import numpy as np
import pandas as pd

HOUR = pd.Timedelta("1h")
PRICE_NAMES = ("open", "high", "low", "close")
SAME_BAR_PAIRS = (("high", "low"), ("close", "open"), ("high", "close"), ("close", "low"))
WINDOWS = (16, 32, 64, 96)
HORIZONS = tuple(range(1, 25))


def validate_ohlc_frame(frame: pd.DataFrame, *, time_column: str = "time_utc") -> tuple[pd.Series, np.ndarray]:
    """Validate ordered unique timestamps and positive, finite, consistent OHLC."""
    required = {time_column, *PRICE_NAMES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    timestamp = pd.to_datetime(frame[time_column], utc=True, errors="raise").reset_index(drop=True)
    if timestamp.isna().any() or timestamp.duplicated().any() or not timestamp.is_monotonic_increasing:
        raise ValueError("timestamps must be present, unique, and strictly increasing")
    prices = frame[list(PRICE_NAMES)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ValueError("OHLC prices must be finite and positive")
    if np.any(prices[:, 1] < prices[:, [0, 2, 3]].max(axis=1)):
        raise ValueError("high violates OHLC bounds")
    if np.any(prices[:, 2] > prices[:, [0, 1, 3]].min(axis=1)):
        raise ValueError("low violates OHLC bounds")
    return timestamp, prices


def _moment_windows(window: int) -> tuple[int, ...]:
    return tuple(sorted({8, min(24, window), window}))


def _own_lags(window: int) -> tuple[int, ...]:
    return tuple(sorted({lag for lag in (3, 6, 12, 24, 48, 64, 96) if lag <= window} | {window}))


def _cross_lags(window: int) -> tuple[int, ...]:
    return tuple(sorted({1, min(6, window), window}))


def _candidate_names_and_groups(window: int) -> list[tuple[str, str, str, int]]:
    if isinstance(window, bool) or not isinstance(window, int) or window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}")
    out: list[tuple[str, str, str, int]] = []

    def add(name: str, group: str, formula: str, history: int) -> None:
        out.append((name, group, formula, history))

    for price in PRICE_NAMES:
        for lag in _own_lags(window):
            add(f"own_log_ratio_{price}_lag_{lag}", "own_ohlc_log_ratios",
                f"log({price}_i/{price}_i_minus_{lag}) over observed rows", lag)
    for numerator, denominator in SAME_BAR_PAIRS:
        add(f"same_bar_log_ratio_{numerator}_{denominator}", "same_bar_cross_ratios",
            f"log({numerator}_i/{denominator}_i)", 0)
    add("observed_transition_gap_log_open_prev_close", "observed_transition_gap",
        "log(open_i/close_(i-1)); current observed transition, available after close_i", 1)
    for numerator, denominator in permutations(PRICE_NAMES, 2):
        for lag in _cross_lags(window):
            add(f"lagged_cross_log_ratio_{numerator}_{denominator}_lag_{lag}", "lagged_cross_ratios",
                f"log({numerator}_i/{denominator}_(i-{lag})) over observed rows", lag)
    for price in PRICE_NAMES:
        for order in (1, 2, 3):
            formula = f"delta_{order}(log({price})) over observed transitions; no gap reset"
            add(f"log_price_difference_order_{order}_{price}", "log_price_differences", formula, order)

    moment_series = {f"return_{price}": f"log({price}_i/{price}_(i-1))" for price in PRICE_NAMES}
    moment_series.update({f"same_bar_{a}_{b}": f"log({a}_i/{b}_i)" for a, b in SAME_BAR_PAIRS})
    moment_series["gap_open_prev_close"] = "log(open_i/close_(i-1)) over observed transitions"
    moment_names = ("mean_raw", "variance_central_ddof0", "third_central_ddof0", "skewness")
    for series, formula in moment_series.items():
        for width in _moment_windows(window):
            for statistic in moment_names:
                add(f"univariate_{statistic}_{series}_w{width}", "single_series_moments",
                    f"{statistic} over last {width} completed observed-transition values of {formula}", width)

    for first, second in combinations(PRICE_NAMES, 2):
        for width in _moment_windows(window):
            add(f"cross_covariance_{first}_{second}_w{width}", "mixed_central_moments",
                f"population covariance of observed one-row log changes in {first} and {second}, last {width}", width)
            add(f"mixed_third_{first}2_{second}_w{width}", "mixed_central_moments",
                f"mean((x-mean(x))^2*(y-mean(y))) for observed log changes, last {width}", width)
            add(f"mixed_third_{first}_{second}2_w{width}", "mixed_central_moments",
                f"mean((x-mean(x))*(y-mean(y))^2) for observed log changes, last {width}", width)
    return out


def candidate_feature_schema(window: int) -> list[dict[str, object]]:
    """Return the fixed, point-in-time feature definitions for one input width."""
    return [
        {
            "feature_name": name,
            "group": group,
            "formula": formula,
            "max_history_observed_transitions": history,
            "available_at": "after_close_of_origin_row_i",
            "causal": True,
            "uses_elapsed_time_as_predictor": False,
        }
        for name, group, formula, history in _candidate_names_and_groups(window)
    ]


def _rolling_moments(series: np.ndarray, width: int, row_start: int) -> tuple[np.ndarray, ...]:
    windows = np.lib.stride_tricks.sliding_window_view(series, width)[row_start - width + 1:]
    means = windows.mean(axis=1)
    centered = windows - means[:, None]
    variance = np.mean(centered * centered, axis=1)
    third = np.mean(centered * centered * centered, axis=1)
    skew = np.zeros_like(third)
    defined = variance > 0.0
    skew[defined] = third[defined] / np.power(variance[defined], 1.5)
    return means, variance, third, skew, defined


def build_feature_matrix(
    frame: pd.DataFrame,
    window: int,
    *,
    time_column: str = "time_utc",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build causal feature candidates over the last W observed transitions.

    Every feature at row i uses row i or earlier. A row is warm after W
    observed transitions, i.e. W+1 closes/rows; elapsed-time gaps do not reset
    this history. The elapsed duration is not included as an input feature.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}")
    timestamp, prices = validate_ohlc_frame(frame, time_column=time_column)
    n = len(frame)
    logp = np.log(prices)
    price_index = {name: i for i, name in enumerate(PRICE_NAMES)}
    rows = np.arange(window, n)
    valid = np.zeros(n, dtype=bool)
    valid[rows] = True
    if len(rows) == 0:
        schema = candidate_feature_schema(window)
        empty = pd.DataFrame(np.full((n, len(schema)), np.nan), columns=[x["feature_name"] for x in schema])
        return empty, pd.DataFrame(False, index=empty.index, columns=empty.columns), pd.Series(valid)

    returns = np.full((n, 4), np.nan, dtype=np.float64)
    returns[1:] = np.diff(logp, axis=0)
    gap = np.full(n, np.nan, dtype=np.float64)
    gap[1:] = np.log(prices[1:, 0] / prices[:-1, 3])
    same = np.column_stack([logp[:, price_index[a]] - logp[:, price_index[b]] for a, b in SAME_BAR_PAIRS])
    difference_values: dict[tuple[str, int], np.ndarray] = {}
    for p, price in enumerate(PRICE_NAMES):
        first = returns[:, p]
        second = np.full(n, np.nan, dtype=np.float64)
        third = np.full(n, np.nan, dtype=np.float64)
        second[2:] = first[2:] - first[1:-1]
        third[3:] = second[3:] - second[2:-1]
        difference_values[(price, 1)] = first
        difference_values[(price, 2)] = second
        difference_values[(price, 3)] = third

    moment_series: dict[str, np.ndarray] = {f"return_{p}": returns[:, i] for i, p in enumerate(PRICE_NAMES)}
    moment_series.update({f"same_bar_{a}_{b}": same[:, i] for i, (a, b) in enumerate(SAME_BAR_PAIRS)})
    moment_series["gap_open_prev_close"] = gap
    moment_cache: dict[tuple[str, int, str], np.ndarray] = {}
    for series_name, series in moment_series.items():
        for width in _moment_windows(window):
            means, variances, thirds, skews, defined = _rolling_moments(series, width, window)
            for stat, vals in (("mean_raw", means), ("variance_central_ddof0", variances),
                               ("third_central_ddof0", thirds), ("skewness", skews),
                               ("skewness_defined", defined)):
                moment_cache[(series_name, width, stat)] = vals

    mixed_cache: dict[tuple[str, str, int, str], np.ndarray] = {}
    for first, second in combinations(PRICE_NAMES, 2):
        x_all = returns[:, price_index[first]]
        y_all = returns[:, price_index[second]]
        for width in _moment_windows(window):
            x_view = np.lib.stride_tricks.sliding_window_view(x_all, width)[window - width + 1:]
            y_view = np.lib.stride_tricks.sliding_window_view(y_all, width)[window - width + 1:]
            x = x_view - x_view.mean(axis=1, keepdims=True)
            y = y_view - y_view.mean(axis=1, keepdims=True)
            mixed_cache[(first, second, width, "cov")] = np.mean(x * y, axis=1)
            mixed_cache[(first, second, width, "first2")] = np.mean(x * x * y, axis=1)
            mixed_cache[(first, second, width, "second2")] = np.mean(x * y * y, axis=1)

    feature_arrays: dict[str, np.ndarray] = {}
    skew_defined: dict[str, np.ndarray] = {}
    for item in candidate_feature_schema(window):
        name, group = str(item["feature_name"]), str(item["group"])
        if group == "own_ohlc_log_ratios":
            stem, lag_text = name.removeprefix("own_log_ratio_").rsplit("_lag_", 1)
            price, lag = stem, int(lag_text)
            vals = logp[rows, price_index[price]] - logp[rows - lag, price_index[price]]
        elif group == "same_bar_cross_ratios":
            pair = name.removeprefix("same_bar_log_ratio_")
            a, b = pair.split("_")
            vals = logp[rows, price_index[a]] - logp[rows, price_index[b]]
        elif group == "observed_transition_gap":
            vals = gap[rows]
        elif group == "lagged_cross_ratios":
            stem, lag_text = name.rsplit("_lag_", 1)
            pair = stem.removeprefix("lagged_cross_log_ratio_")
            numerator, denominator = pair.split("_")
            lag = int(lag_text)
            vals = logp[rows, price_index[numerator]] - logp[rows - lag, price_index[denominator]]
        elif group == "log_price_differences":
            parts = name.removeprefix("log_price_difference_order_").split("_")
            order, price = int(parts[0]), parts[1]
            vals = difference_values[(price, order)][rows]
        elif group == "single_series_moments":
            rest = name.removeprefix("univariate_")
            stat_prefix = next((prefix for prefix in (
                "mean_raw_", "variance_central_ddof0_", "third_central_ddof0_", "skewness_"
            ) if rest.startswith(prefix)), None)
            if stat_prefix is None:
                raise ValueError(f"unrecognized moment feature: {name}")
            stat = stat_prefix[:-1]
            series_name, width_text = rest.removeprefix(stat_prefix).rsplit("_w", 1)
            width = int(width_text)
            if stat == "skewness":
                cache_stat = "skewness"
                skew_defined[name] = moment_cache[(series_name, width, "skewness_defined")]
            else:
                cache_stat = stat
            vals = moment_cache[(series_name, width, cache_stat)]
        else:  # mixed_central_moments
            if name.startswith("cross_covariance_"):
                stem = name.removeprefix("cross_covariance_")
                pair, width_text = stem.rsplit("_w", 1)
                first, second = pair.split("_")
                key_stat = "cov"
            elif name.startswith("mixed_third_"):
                stem = name.removeprefix("mixed_third_")
                pair, width_text = stem.rsplit("_w", 1)
                width = int(width_text)
                # Exact pair/order parsing; OHLC price names are each one token.
                if "2_" in pair:
                    first, second = pair.split("2_", 1)
                    key_stat = "first2"
                else:
                    first, second = pair.split("_", 1)
                    second = second.removesuffix("2")
                    key_stat = "second2"
                width_text = str(width)
            else:
                raise ValueError(f"unrecognized feature: {name}")
            width = int(width_text)
            vals = mixed_cache[(first, second, width, key_stat)]
        array = np.full(n, np.nan, dtype=np.float64)
        array[rows] = vals
        feature_arrays[name] = array
        if name in skew_defined:
            mask = np.zeros(n, dtype=bool)
            mask[rows] = skew_defined[name]
            skew_defined[name] = mask

    names = [str(item["feature_name"]) for item in candidate_feature_schema(window)]
    feature_frame = pd.DataFrame({name: feature_arrays[name] for name in names}, index=frame.index)
    skew_frame = pd.DataFrame(False, index=frame.index, columns=names)
    for name, values in skew_defined.items():
        skew_frame[name] = values
    return feature_frame, skew_frame, pd.Series(valid, index=frame.index, name="input_history_valid")


def build_forward_targets(frame: pd.DataFrame, horizon: int, *, time_column: str = "time_utc") -> pd.DataFrame:
    """Target the next H observed close-to-close transitions, preserving gaps."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 24:
        raise ValueError("horizon must be an integer in [1, 24]")
    timestamp, prices = validate_ohlc_frame(frame, time_column=time_column)
    close = prices[:, 3]
    n = len(frame)
    returns = np.full(n, np.nan, dtype=np.float64)
    returns[1:] = np.diff(np.log(close))
    time_deltas = timestamp.diff().dt.total_seconds().to_numpy(dtype=float) / 3600.0
    transition_gap = np.zeros(n, dtype=bool)
    if n > 1:
        transition_gap[1:] = ~np.isclose(time_deltas[1:], 1.0)
    y_mean = np.full(n, np.nan, dtype=np.float64)
    y_var = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    elapsed = np.full(n, np.nan, dtype=np.float64)
    gap_count = np.full(n, -1, dtype=np.int16)
    target_open_ns = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
    target_close_ns = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
    for i in range(max(0, n - horizon)):
        end = i + horizon
        steps = returns[i + 1:end + 1]
        mean = float(np.mean(steps))
        y_mean[i] = mean
        y_var[i] = float(np.mean((steps - mean) ** 2))
        elapsed[i] = (timestamp.iloc[end] - timestamp.iloc[i]).total_seconds() / 3600.0
        gap_count[i] = int(transition_gap[i + 1:end + 1].sum())
        target_open_ns[i] = timestamp.iloc[end].value
        target_close_ns[i] = (timestamp.iloc[end] + HOUR).value
        valid[i] = True
    if horizon == 1:
        y_var[valid] = 0.0
    return pd.DataFrame({
        "origin_row": np.arange(n, dtype=np.int32),
        "origin_bar_open_timestamp": timestamp,
        "decision_timestamp": timestamp + HOUR,
        "target_bar_open_timestamp": pd.to_datetime(target_open_ns, utc=True),
        "target_timestamp": pd.to_datetime(target_close_ns, utc=True),
        "horizon_observed_transitions": horizon,
        "target_elapsed_calendar_hours": elapsed,
        "target_non_one_hour_transition_count": gap_count,
        "target_crosses_non_one_hour_transition": gap_count > 0,
        "target_valid": valid,
        "y_mean_log_return_per_observed_transition": y_mean,
        "y_var_log_return_per_observed_transition_ddof0": y_var,
        "variance_evaluable": horizon >= 2,
    }, index=frame.index)


def common_origin_mask(row_count: int, *, max_window: int = 96, max_horizon: int = 24) -> np.ndarray:
    """Origins that permit paired comparison at every registered W/H pair."""
    if row_count < 0 or max_window < 1 or max_horizon < 1:
        raise ValueError("row_count must be nonnegative and window/horizon positive")
    rows = np.arange(row_count)
    return (rows >= max_window) & (rows + max_horizon < row_count)


def purged_split_indices(
    common: np.ndarray,
    decision_ns: np.ndarray,
    max_horizon_target_ns: np.ndarray,
    start_utc: str,
    end_utc: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological train/validation origins purged by the longest target end."""
    if not (len(common) == len(decision_ns) == len(max_horizon_target_ns)):
        raise ValueError("common mask, decisions and target timestamps must have equal length")
    start, end = pd.Timestamp(start_utc).value, pd.Timestamp(end_utc).value
    if start >= end:
        raise ValueError("split start must be before split end")
    nat = np.iinfo(np.int64).min
    has_target = max_horizon_target_ns != nat
    train = np.flatnonzero(common & has_target & (decision_ns < start) & (max_horizon_target_ns < start))
    validation = np.flatnonzero(
        common & has_target & (decision_ns >= start) & (decision_ns < end) & (max_horizon_target_ns < end)
    )
    if len(train) and not np.all(max_horizon_target_ns[train] < start):
        raise AssertionError("training target overlaps validation start")
    if len(validation) and not np.all(max_horizon_target_ns[validation] < end):
        raise AssertionError("validation target reaches or passes fold end")
    return train, validation


def assess_feasibility(
    frame: pd.DataFrame,
    validation_windows: Mapping[str, tuple[str, str]],
    inner_windows: Mapping[str, tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Count all-pair availability and common-origin split coverage before fit."""
    timestamp, _ = validate_ohlc_frame(frame)
    n = len(frame)
    row = np.arange(n)
    decision = (timestamp + HOUR).array.asi8
    target24_idx = row + 24
    common = (row >= max(WINDOWS)) & (target24_idx < n)
    target24_close = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    exists = target24_idx < n
    target24_close[exists] = timestamp.array.asi8[target24_idx[exists]] + HOUR.value
    transition_gap = np.zeros(n, dtype=bool)
    if n > 1:
        delta = timestamp.diff().dt.total_seconds().to_numpy(dtype=float) / 3600.0
        transition_gap[1:] = ~np.isclose(delta[1:], 1.0)
    records = []
    cohort_audit = []
    for window in WINDOWS:
        for horizon in HORIZONS:
            pair_available = (row >= window) & (row + horizon < n)
            records.append({
                "W": window,
                "H": horizon,
                "candidate_feature_count": len(candidate_feature_schema(window)),
                "available_observed_origins": int(pair_available.sum()),
                "common_W96_H24_origins": int((pair_available & common).sum()),
                "common_origins_equal_across_all_pairs": bool(np.array_equal(pair_available[common], np.ones(int(common.sum()), dtype=bool))),
            })
    split_sets = {**{f"outer_{k}": v for k, v in validation_windows.items()}, **{f"inner_{k}": v for k, v in inner_windows.items()}}
    for split_id, (start_text, end_text) in split_sets.items():
        start, end = pd.Timestamp(start_text).value, pd.Timestamp(end_text).value
        train = common & (decision < start) & (target24_close < start)
        valid = common & (decision >= start) & (decision < end) & (target24_close < end)
        for part, mask in (("train", train), ("validation", valid)):
            idx = row[mask]
            if len(idx):
                # A duration summary is metadata, not a candidate predictor.
                hist_hours = (timestamp.array.asi8[idx] - timestamp.array.asi8[idx - max(WINDOWS)]) / 3.6e12
            else:
                hist_hours = np.asarray([], dtype=float)
            dts = pd.to_datetime(decision[mask], utc=True)
            for dimension, values, bins in (("weekday_utc_monday0", dts.dayofweek, range(7)),
                                             ("hour_utc", dts.hour, range(24))):
                for bucket in bins:
                    cohort_audit.append({
                        "split_id": split_id,
                        "partition": part,
                        "dimension": dimension,
                        "bucket": bucket,
                        "count": int(np.sum(values == bucket)),
                        "common_origin_rule": "W96 history and H24 target-end availability; applied to every W/H/mask",
                    })
        cohort_audit.append({
            "split_id": split_id,
            "partition": "common_origin_metadata",
            "dimension": "input_history_elapsed_hours_W96",
            "bucket": "summary",
            "count": int(valid.sum()),
            "min": float(hist_hours.min()) if len(hist_hours) else None,
            "median": float(np.median(hist_hours)) if len(hist_hours) else None,
            "p95": float(np.quantile(hist_hours, .95)) if len(hist_hours) else None,
            "max": float(hist_hours.max()) if len(hist_hours) else None,
            "non_one_hour_transitions_in_source_prefix": int(transition_gap.sum()),
            "common_origin_rule": "audit only; elapsed hours are not model features",
        })
    summary = {
        "rows_in_pinned_prefix": n,
        "first_bar_open_utc": timestamp.iloc[0].isoformat() if n else None,
        "last_bar_open_utc": timestamp.iloc[-1].isoformat() if n else None,
        "non_one_hour_source_transitions": int(transition_gap.sum()),
        "common_origin_rule": "row >= 96 and row + 24 < row_count; applied unchanged to every W/H/mask",
        "common_origins_total": int(common.sum()),
        "common_origins_by_split": {},
        "horizon_target_end_purging": "train target H24 close < split start; validation decision in [start,end) and target H24 close < end",
        "fit_performed": False,
    }
    for split_id, (start_text, end_text) in split_sets.items():
        start, end = pd.Timestamp(start_text).value, pd.Timestamp(end_text).value
        train = common & (decision < start) & (target24_close < start)
        valid = common & (decision >= start) & (decision < end) & (target24_close < end)
        summary["common_origins_by_split"][split_id] = {"train": int(train.sum()), "validation": int(valid.sum())}
    return pd.DataFrame(records), pd.DataFrame(cohort_audit), summary
