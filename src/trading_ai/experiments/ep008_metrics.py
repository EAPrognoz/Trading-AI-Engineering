"""Causal baseline forecasts and task-specific EP008 scoring."""

from __future__ import annotations

from datetime import date
import math

import numpy as np
import pandas as pd

from trading_ai.experiments.ep008_data import DEVELOPMENT_START, HORIZON, _validate_rates


VARIANCE_FLOOR_BPS2 = 1e-4
EWMA_LAMBDA = 0.94
QUANTILE_HISTORY = 252
QUANTILE_LEVELS = (0.05, 0.50, 0.95)
BOOTSTRAP_BLOCK_LENGTH = 20
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 7


def _aligned_vectors(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or pred.ndim != 1 or y.shape != pred.shape or not len(y):
        raise ValueError("actual and predicted values must be non-empty aligned vectors")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("actual and predicted values must be finite")
    return y, pred


def _date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def _return_arrays(rates: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ordered = _validate_rates(rates)
    usd = ordered["USD"].to_numpy(dtype=float)
    returns = np.full(len(usd), np.nan, dtype=float)
    returns[1:] = usd[1:] / usd[:-1] - 1.0
    return ordered, returns, returns**2


def _ewma_levels(
    dates: list[date],
    squared_returns: np.ndarray,
    *,
    training_start: date = DEVELOPMENT_START,
    lambda_: float = EWMA_LAMBDA,
) -> np.ndarray:
    if not 0.0 <= lambda_ < 1.0:
        raise ValueError("EWMA lambda must be in [0, 1)")
    training_positions = [i for i in range(1, len(dates)) if dates[i] >= training_start]
    if len(training_positions) < 60:
        raise ValueError("at least 60 training one-step returns are required to initialize EWMA")
    initial_positions = training_positions[:60]
    initial_end = initial_positions[-1]
    state = np.full(len(dates), np.nan, dtype=float)
    level = float(np.mean(squared_returns[initial_positions]))
    state[initial_end] = level
    for i in range(initial_end + 1, len(dates)):
        level = lambda_ * level + (1.0 - lambda_) * squared_returns[i]
        state[i] = level
    return state


def movement_baselines(
    rates: pd.DataFrame,
    decision_dates: list[object],
    *,
    lambda_: float = EWMA_LAMBDA,
) -> pd.DataFrame:
    """Return trailing-five and EWMA means of known one-step squared returns in bps²."""
    ordered, _, squared = _return_arrays(rates)
    dates = ordered["date"].tolist()
    positions = {value: i for i, value in enumerate(dates)}
    ewma = _ewma_levels(dates, squared, lambda_=lambda_)
    normalized = [_date(value) for value in decision_dates]
    if len(set(normalized)) != len(normalized):
        raise ValueError("decision dates must be unique")
    rows: list[dict[str, object]] = []
    for decision in normalized:
        if decision not in positions:
            raise ValueError(f"decision date is absent from rates: {decision}")
        i = positions[decision]
        if i < 5 or not np.isfinite(ewma[i]):
            raise ValueError(f"movement baselines are not initialized at {decision}")
        rows.append(
            {
                "decision_date": decision,
                "trailing5_bps2": float(np.mean(squared[i - 4 : i + 1]) * 1e8),
                "ewma_bps2": float(ewma[i] * 1e8),
            }
        )
    return pd.DataFrame(rows, columns=["decision_date", "trailing5_bps2", "ewma_bps2"])


def quantile_baselines(
    rates: pd.DataFrame,
    decision_dates: list[object],
    *,
    history_window: int = QUANTILE_HISTORY,
    lambda_: float = EWMA_LAMBDA,
) -> pd.DataFrame:
    """Build rolling empirical and EWMA-scaled quantiles from matured returns only."""
    if history_window <= 0:
        raise ValueError("quantile history window must be positive")
    ordered, returns, squared = _return_arrays(rates)
    dates = ordered["date"].tolist()
    usd = ordered["USD"].to_numpy(dtype=float)
    positions = {value: i for i, value in enumerate(dates)}
    training_positions = [i for i, value in enumerate(dates) if value >= DEVELOPMENT_START]
    if not training_positions:
        raise ValueError("rates contain no development history from 2000 onward")
    history_start = training_positions[0]
    ewma = _ewma_levels(dates, squared, lambda_=lambda_)
    normalized = [_date(value) for value in decision_dates]
    if len(set(normalized)) != len(normalized):
        raise ValueError("decision dates must be unique")
    rows: list[dict[str, object]] = []
    for decision in normalized:
        if decision not in positions:
            raise ValueError(f"decision date is absent from rates: {decision}")
        i = positions[decision]
        last_matured_decision = i - HORIZON
        if last_matured_decision < history_start:
            raise ValueError(f"no matured development returns are available at {decision}")
        first_history_decision = max(history_start, last_matured_decision - history_window + 1)
        history_decisions = np.arange(first_history_decision, last_matured_decision + 1)
        history_bps = (usd[history_decisions + HORIZON] / usd[history_decisions] - 1.0) * 10_000.0
        history_bps = history_bps[-history_window:]
        if len(history_bps) == 0 or not np.isfinite(history_bps).all():
            raise ValueError(f"matured quantile history is invalid at {decision}")
        quantiles = np.quantile(history_bps, QUANTILE_LEVELS, method="linear")
        if not np.isfinite(ewma[i]):
            raise ValueError(f"EWMA baseline is not initialized at {decision}")
        trailing_start = max(1, i - 251)
        trailing_rms = float(np.sqrt(np.mean(squared[trailing_start : i + 1])))
        current_ewma_rms = float(np.sqrt(ewma[i]))
        ratio = current_ewma_rms / trailing_rms if trailing_rms > 0.0 else 1.0
        scaled = quantiles[1] + ratio * (quantiles - quantiles[1])
        row: dict[str, object] = {
            "decision_date": decision,
            "history_n": int(len(history_bps)),
            "scale_ratio": float(ratio),
        }
        for level, value, scaled_value in zip((5, 50, 95), quantiles, scaled, strict=True):
            row[f"rolling_q{level:02d}_bps"] = float(value)
            row[f"ewma_scaled_q{level}_bps"] = float(scaled_value)
        rows.append(row)
    columns = [
        "decision_date",
        "history_n",
        "scale_ratio",
        "rolling_q05_bps",
        "rolling_q50_bps",
        "rolling_q95_bps",
        "ewma_scaled_q5_bps",
        "ewma_scaled_q50_bps",
        "ewma_scaled_q95_bps",
    ]
    return pd.DataFrame(rows, columns=columns)


def score_return(actual_bps: np.ndarray, predicted_bps: np.ndarray) -> dict[str, float | int]:
    y, pred = _aligned_vectors(actual_bps, predicted_bps)
    error = pred - y
    return {
        "n": int(len(y)),
        "mae_bps": float(np.mean(np.abs(error))),
        "rmse_bps": float(np.sqrt(np.mean(error**2))),
    }


def score_movement(
    actual_bps2: np.ndarray,
    raw_prediction_bps2: np.ndarray,
    *,
    floor_bps2: float = VARIANCE_FLOOR_BPS2,
) -> dict[str, float | int]:
    y, raw_prediction = _aligned_vectors(actual_bps2, raw_prediction_bps2)
    if np.any(y < 0.0):
        raise ValueError("mean-squared-return actuals must be nonnegative")
    if not math.isfinite(floor_bps2) or floor_bps2 <= 0.0:
        raise ValueError("movement floor must be positive and finite")
    actual_for_qlike = np.maximum(y, floor_bps2)
    prediction = np.maximum(raw_prediction, floor_bps2)
    ratio = actual_for_qlike / prediction
    qlike = ratio - np.log(ratio) - 1.0
    rms_mae = np.mean(np.abs(np.sqrt(y) - np.sqrt(prediction)))
    clipped = raw_prediction < floor_bps2
    actual_floored = y <= floor_bps2
    return {
        "n": int(len(y)),
        "qlike": float(np.mean(qlike)),
        "rms_mae_bps": float(rms_mae),
        "floor_bps2": float(floor_bps2),
        "prediction_floor_clipped_count": int(np.count_nonzero(clipped)),
        "prediction_floor_clipped_share": float(np.mean(clipped)),
        "actual_at_or_below_floor_count": int(np.count_nonzero(actual_floored)),
        "actual_at_or_below_floor_share": float(np.mean(actual_floored)),
    }


def score_quantiles(
    actual_bps: np.ndarray,
    raw_predictions_bps: np.ndarray,
) -> dict[str, object]:
    y = np.asarray(actual_bps, dtype=float)
    raw = np.asarray(raw_predictions_bps, dtype=float)
    if y.ndim != 1 or raw.shape != (len(y), 3) or not len(y):
        raise ValueError("quantile actuals and predictions must have aligned (n,) and (n, 3) shapes")
    if not np.isfinite(y).all() or not np.isfinite(raw).all():
        raise ValueError("quantile actuals and predictions must be finite")
    crossed = np.any(raw[:, :-1] > raw[:, 1:], axis=1)
    ordered = np.sort(raw, axis=1)
    quantiles = np.asarray(QUANTILE_LEVELS, dtype=float)
    errors = y[:, None] - ordered
    losses = np.maximum(quantiles[None, :] * errors, (quantiles[None, :] - 1.0) * errors)
    names = ("q05", "q50", "q95")
    pinball = {name: float(value) for name, value in zip(names, losses.mean(axis=0), strict=True)}
    return {
        "n": int(len(y)),
        "crossing_count": int(np.count_nonzero(crossed)),
        "crossing_rate": float(np.mean(crossed)),
        "pinball_mean_bps": pinball,
        "mean_pinball_bps": float(losses.mean()),
        "median_mae_bps": float(np.mean(np.abs(y - ordered[:, 1]))),
        "interval_coverage_90": float(np.mean((y >= ordered[:, 0]) & (y <= ordered[:, 2]))),
        "interval_mean_width_bps": float(np.mean(ordered[:, 2] - ordered[:, 0])),
        "sorted_predictions_bps": ordered.tolist(),
    }


def primary_loss_vector(task: str, actual: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
    """Return row-level primary losses in the same units used for model selection."""
    if task == "return":
        y, pred = _aligned_vectors(actual, raw_prediction)
        return np.abs(pred - y)
    if task == "movement":
        y, pred = _aligned_vectors(actual, raw_prediction)
        if np.any(y < 0.0):
            raise ValueError("mean-squared-return actuals must be nonnegative")
        y_safe = np.maximum(y, VARIANCE_FLOOR_BPS2)
        pred_safe = np.maximum(pred, VARIANCE_FLOOR_BPS2)
        ratio = y_safe / pred_safe
        return ratio - np.log(ratio) - 1.0
    if task == "quantiles":
        y = np.asarray(actual, dtype=float)
        raw = np.asarray(raw_prediction, dtype=float)
        if y.ndim != 1 or raw.shape != (len(y), 3) or not len(y):
            raise ValueError("quantile actuals and predictions must have aligned (n,) and (n, 3) shapes")
        ordered = np.sort(raw, axis=1)
        errors = y[:, None] - ordered
        q = np.asarray(QUANTILE_LEVELS)
        return np.maximum(q[None, :] * errors, (q[None, :] - 1.0) * errors).mean(axis=1)
    raise ValueError(f"unknown EP008 task: {task}")


def moving_block_ci(
    model_losses: np.ndarray,
    baseline_losses: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | int]:
    """Paired non-circular moving-block interval for mean(model loss - baseline loss)."""
    model, baseline = _aligned_vectors(model_losses, baseline_losses)
    if block_length <= 0 or len(model) < block_length:
        raise ValueError("moving-block bootstrap requires at least one full positive-length block")
    if n_resamples <= 0:
        raise ValueError("bootstrap resample count must be positive")
    difference = model - baseline
    blocks_per_resample = int(np.ceil(len(difference) / block_length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(difference) - block_length + 1, size=(n_resamples, blocks_per_resample))
    offsets = np.arange(block_length, dtype=int)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_resamples, -1)[:, : len(difference)]
    draws = difference[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    return {
        "n": int(len(difference)),
        "mean_loss_difference": float(np.mean(difference)),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence_level": 0.95,
        "seed": int(seed),
        "block_length": int(block_length),
        "n_resamples": int(n_resamples),
    }
