from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.pipeline import Pipeline

from trading_ai.experiments.ep008_data import FEATURE_COLUMNS
from trading_ai.experiments.ep008_models import (
    HGB_CONFIGS,
    candidate_configurations,
    fit_model,
    predict_model,
    select_candidate,
)


def _x(n: int = 40) -> pd.DataFrame:
    values = np.arange(n * len(FEATURE_COLUMNS), dtype=float).reshape(n, len(FEATURE_COLUMNS))
    return pd.DataFrame(values, columns=FEATURE_COLUMNS)


def test_candidate_grids_match_frozen_protocol_order() -> None:
    assert [item["alpha"] for item in candidate_configurations("return", "linear")] == [0.1, 10.0, 100.0]
    assert [item["alpha"] for item in candidate_configurations("movement", "linear")] == [0.1, 10.0, 100.0]
    assert [item["alpha"] for item in candidate_configurations("quantiles", "linear")] == [0.0, 0.01, 0.1]
    assert candidate_configurations("return", "boosting") == [dict(item) for item in HGB_CONFIGS]
    assert candidate_configurations("quantiles", "boosting") == [dict(item) for item in HGB_CONFIGS]


def test_committed_config_matches_runtime_grids_and_protocol_constants() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs/experiments/ep008_joint_five_observation_predictability.toml"
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    assert config["linear"]["ridge_alphas"] == [0.1, 10.0, 100.0]
    assert config["linear"]["quantile_alphas"] == [0.0, 0.01, 0.1]
    assert config["boosting"]["l2_regularization"] == 1.0
    assert config["boosting"]["early_stopping"] is False
    assert config["boosting"]["random_state"] == 7
    assert config["boosting"]["candidates"] == [dict(item) for item in HGB_CONFIGS]
    assert config["movement"]["forecast_floor_bps2"] == 1e-4
    assert {key: config["bootstrap"][key] for key in ("block_length", "resamples", "seed")} == {
        "block_length": 20,
        "resamples": 10000,
        "seed": 7,
    }


def test_linear_scaler_fits_only_training_rows_and_return_prediction_shape() -> None:
    train = _x(10)
    target = train["r_usd_0"].to_numpy() * 0.5
    model = fit_model("return", "linear", {"alpha": 0.1}, train, target)
    validation = _x(2) + 1000
    predictions = predict_model(model, "return", validation)

    assert isinstance(model, Pipeline)
    assert model.named_steps["scale"].mean_[0] == pytest.approx(train["r_usd_0"].mean())
    assert isinstance(model.named_steps["model"], Ridge)
    assert predictions.shape == (2,)


def test_quantile_linear_fits_three_standardized_models_with_alpha_zero() -> None:
    train = _x(20)
    target = train["r_usd_0"].to_numpy() * 0.1 + np.sin(np.arange(20))
    models = fit_model("quantiles", "linear", {"alpha": 0.0}, train, target)
    predictions = predict_model(models, "quantiles", _x(4))

    assert len(models) == 3
    assert all(isinstance(model, Pipeline) for model in models)
    assert all(isinstance(model.named_steps["model"], QuantileRegressor) for model in models)
    assert [model.named_steps["model"].quantile for model in models] == [0.05, 0.5, 0.95]
    assert all(model.named_steps["model"].alpha == 0.0 for model in models)
    assert predictions.shape == (4, 3)


def test_hgb_uses_fixed_seed_regularization_no_early_stopping_and_direct_target() -> None:
    train = _x(30)
    target = np.linspace(1.0, 2.0, len(train))
    model = fit_model("movement", "boosting", dict(HGB_CONFIGS[0]), train, target)

    assert isinstance(model, HistGradientBoostingRegressor)
    assert model.learning_rate == 0.03
    assert model.max_iter == 50
    assert model.max_leaf_nodes == 7
    assert model.l2_regularization == 1.0
    assert model.early_stopping is False
    assert model.random_state == 7
    assert predict_model(model, "movement", train).shape == (30,)


def test_quantile_hgb_fits_all_three_loss_levels() -> None:
    train = _x(50)
    target = train["r_usd_0"].to_numpy() + np.sin(np.arange(50))
    models = fit_model("quantiles", "boosting", dict(HGB_CONFIGS[1]), train, target)
    predictions = predict_model(models, "quantiles", _x(3))

    assert all(isinstance(model, HistGradientBoostingRegressor) for model in models)
    assert [model.quantile for model in models] == [0.05, 0.5, 0.95]
    assert all(model.loss == "quantile" for model in models)
    assert predictions.shape == (3, 3)


def test_selection_uses_equal_fold_macro_metric_and_listed_order_for_ties() -> None:
    candidates = [
        {"config_id": "first", "grid_index": 0, "scores_by_fold": [{"primary": 1}, {"primary": 10}, {"primary": 10}]},
        {"config_id": "second", "grid_index": 1, "scores_by_fold": [{"primary": 6}, {"primary": 6}, {"primary": 6}]},
        {"config_id": "third", "grid_index": 2, "scores_by_fold": [{"primary": 6}, {"primary": 6}, {"primary": 6}]},
    ]

    selected = select_candidate(candidates, primary_metric="primary")

    assert selected["config_id"] == "second"
    assert selected["macro_primary"] == pytest.approx(6.0)
    assert selected["macro_scores"]["primary"] == pytest.approx(6.0)


def test_selection_uses_secondary_metric_then_grid_order_for_ties() -> None:
    candidates = [
        {"config_id": "first", "grid_index": 0, "scores_by_fold": [{"primary": 2, "secondary": 3}] * 3},
        {"config_id": "second", "grid_index": 1, "scores_by_fold": [{"primary": 2, "secondary": 2}] * 3},
    ]

    selected = select_candidate(candidates, primary_metric="primary", secondary_metric="secondary")

    assert selected["config_id"] == "second"
    assert selected["macro_secondary"] == pytest.approx(2.0)


def test_model_factory_rejects_unknown_task_or_family() -> None:
    with pytest.raises(ValueError, match="task"):
        candidate_configurations("other", "linear")
    with pytest.raises(ValueError, match="family"):
        candidate_configurations("return", "other")
