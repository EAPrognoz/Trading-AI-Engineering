"""Frozen CPU model grids and development-only configuration selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading_ai.experiments.ep008_data import FEATURE_COLUMNS
from trading_ai.experiments.ep008_metrics import QUANTILE_LEVELS


RIDGE_ALPHAS = (0.1, 10.0, 100.0)
QUANTILE_ALPHAS = (0.0, 0.01, 0.1)
HGB_CONFIGS = (
    {"config_id": "H1", "grid_index": 0, "learning_rate": 0.03, "max_iter": 50, "max_leaf_nodes": 7},
    {"config_id": "H2", "grid_index": 1, "learning_rate": 0.03, "max_iter": 100, "max_leaf_nodes": 7},
    {"config_id": "H3", "grid_index": 2, "learning_rate": 0.10, "max_iter": 50, "max_leaf_nodes": 7},
)
HGB_L2_REGULARIZATION = 1.0
RANDOM_SEED = 7


def candidate_configurations(task: str, family: str) -> list[dict[str, Any]]:
    if task not in {"return", "movement", "quantiles"}:
        raise ValueError(f"unknown task: {task}")
    if family not in {"linear", "boosting"}:
        raise ValueError(f"unknown family: {family}")
    if family == "boosting":
        return [dict(config) for config in HGB_CONFIGS]
    alphas = QUANTILE_ALPHAS if task == "quantiles" else RIDGE_ALPHAS
    return [
        {"config_id": f"L{index + 1}", "grid_index": index, "alpha": alpha}
        for index, alpha in enumerate(alphas)
    ]


def _feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise ValueError("model features must be a DataFrame with the frozen 14 columns")
    missing = set(FEATURE_COLUMNS).difference(features.columns)
    if missing or len(features.columns) != len(FEATURE_COLUMNS):
        raise ValueError(f"model features must contain exactly the frozen 14 columns; missing={sorted(missing)}")
    ordered = features.loc[:, FEATURE_COLUMNS].astype(float)
    if not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise ValueError("model features must be finite")
    return ordered


def _target_vector(target: np.ndarray, n_rows: int, task: str) -> np.ndarray:
    y = np.asarray(target, dtype=float)
    if y.ndim != 1 or len(y) != n_rows:
        raise ValueError("model target must be a one-dimensional vector aligned to features")
    if not np.isfinite(y).all():
        raise ValueError("model target must be finite")
    if task == "movement" and np.any(y < 0.0):
        raise ValueError("mean-squared-return training target must be nonnegative")
    return y


def _hgb(config: dict[str, Any], *, quantile: float | None = None) -> HistGradientBoostingRegressor:
    kwargs: dict[str, Any] = {
        "learning_rate": float(config["learning_rate"]),
        "max_iter": int(config["max_iter"]),
        "max_leaf_nodes": int(config["max_leaf_nodes"]),
        "l2_regularization": HGB_L2_REGULARIZATION,
        "early_stopping": False,
        "random_state": RANDOM_SEED,
    }
    if quantile is None:
        return HistGradientBoostingRegressor(loss="squared_error", **kwargs)
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kwargs)


def fit_model(
    task: str,
    family: str,
    config: dict[str, Any],
    features: pd.DataFrame,
    target: np.ndarray,
) -> Pipeline | HistGradientBoostingRegressor | tuple[Pipeline | HistGradientBoostingRegressor, ...]:
    """Fit one prescribed estimator; all scaling is fit only on supplied training rows."""
    if task not in {"return", "movement", "quantiles"}:
        raise ValueError(f"unknown task: {task}")
    if family not in {"linear", "boosting"}:
        raise ValueError(f"unknown family: {family}")
    x = _feature_frame(features)
    y = _target_vector(target, len(x), task)
    if family == "linear" and task == "quantiles":
        models: list[Pipeline] = []
        for quantile in QUANTILE_LEVELS:
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        QuantileRegressor(
                            quantile=quantile,
                            alpha=float(config["alpha"]),
                            solver="highs",
                            fit_intercept=True,
                        ),
                    ),
                ]
            )
            model.fit(x, y)
            models.append(model)
        return tuple(models)
    if family == "linear":
        model = Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=float(config["alpha"]), fit_intercept=True))]
        )
        model.fit(x, y)
        return model
    if task == "quantiles":
        models = []
        for quantile in QUANTILE_LEVELS:
            model = _hgb(config, quantile=quantile)
            model.fit(x, y)
            models.append(model)
        return tuple(models)
    model = _hgb(config)
    model.fit(x, y)
    return model


def predict_model(
    model: Pipeline | HistGradientBoostingRegressor | Sequence[Pipeline | HistGradientBoostingRegressor],
    task: str,
    features: pd.DataFrame,
) -> np.ndarray:
    if task not in {"return", "movement", "quantiles"}:
        raise ValueError(f"unknown task: {task}")
    x = _feature_frame(features)
    if task == "quantiles":
        if not isinstance(model, Sequence) or len(model) != len(QUANTILE_LEVELS):
            raise ValueError("quantile prediction requires exactly three fitted models")
        predictions = np.column_stack([item.predict(x) for item in model])
    else:
        if isinstance(model, Sequence):
            raise ValueError("return and movement predictions require one fitted model")
        predictions = np.asarray(model.predict(x), dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("model predictions must be finite")
    return predictions


def select_candidate(
    candidates: list[dict[str, Any]],
    *,
    primary_metric: str,
    secondary_metric: str | None = None,
) -> dict[str, Any]:
    """Select by equal-fold macro score, then secondary metric and frozen grid order."""
    if not candidates:
        raise ValueError("candidate list cannot be empty")
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        fold_scores = candidate.get("scores_by_fold")
        if not isinstance(fold_scores, list) or len(fold_scores) != 3:
            raise ValueError("each candidate must have scores for exactly three validation folds")
        metrics = [primary_metric] + ([secondary_metric] if secondary_metric else [])
        macro: dict[str, float] = {}
        for metric in metrics:
            values = np.asarray([score[metric] for score in fold_scores], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"candidate scores for {metric} must be finite")
            macro[metric] = float(np.mean(values))
        evaluated.append(
            {
                **{key: value for key, value in candidate.items() if key != "scores_by_fold"},
                "scores_by_fold": fold_scores,
                "macro_scores": macro,
            }
        )
    selected = min(
        evaluated,
        key=lambda item: (
            item["macro_scores"][primary_metric],
            item["macro_scores"][secondary_metric] if secondary_metric else 0.0,
            int(item.get("grid_index", 0)),
        ),
    )
    return {
        **selected,
        "primary_metric": primary_metric,
        "macro_primary": selected["macro_scores"][primary_metric],
        "macro_secondary": selected["macro_scores"].get(secondary_metric) if secondary_metric else None,
    }
