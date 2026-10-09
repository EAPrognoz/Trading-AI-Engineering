"""Walk-forward model evaluation and selection for the EP008 development era."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from trading_ai.experiments.ep008_data import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    FEATURE_COLUMNS,
    build_features,
    build_targets,
    walk_forward_splits,
)
from trading_ai.experiments.ep008_metrics import (
    movement_baselines,
    quantile_baselines,
    score_movement,
    score_quantiles,
    score_return,
)
from trading_ai.experiments.ep008_models import candidate_configurations, fit_model, predict_model, select_candidate


TASK_SPECS = {
    "return": {"target": "cumulative_return_bps", "primary": "mae_bps", "secondary": "rmse_bps"},
    "movement": {"target": "movement_target_bps2", "primary": "qlike", "secondary": "rms_mae_bps"},
    "quantiles": {
        "target": "cumulative_return_bps",
        "primary": "mean_pinball_bps",
        "secondary": "median_mae_bps",
    },
}
BASELINE_ORDER = {
    "return": ("zero", "train_mean", "train_median"),
    "movement": ("trailing5", "ewma"),
    "quantiles": ("rolling_empirical", "ewma_scaled_empirical"),
}


def development_samples(rates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create development rows and labels without computing any 2026 decision labels."""
    features = build_features(rates)
    development_features = features.loc[
        features["decision_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
    ].copy()
    rate_dates = rates["date"].tolist()
    rate_positions = {value: i for i, value in enumerate(rate_dates)}
    eligible_decisions: list[date] = []
    endpoint_by_decision: dict[date, date] = {}
    for decision in development_features["decision_date"]:
        position = rate_positions[decision]
        endpoint_position = position + 5
        if endpoint_position >= len(rate_dates):
            continue
        endpoint = rate_dates[endpoint_position]
        endpoint_by_decision[decision] = endpoint
        if endpoint <= DEVELOPMENT_END:
            eligible_decisions.append(decision)
    targets = build_targets(rates, eligible_decisions)
    samples = development_features.merge(targets, on="decision_date", how="left", validate="one_to_one")
    samples["endpoint_date"] = samples["decision_date"].map(endpoint_by_decision)
    samples["sample_id"] = samples.apply(
        lambda row: f"{row['decision_date'].isoformat()}|{row['endpoint_date'].isoformat()}", axis=1
    )
    return samples, targets


def _score(task: str, actual: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if task == "return":
        return score_return(actual, prediction)
    if task == "movement":
        return score_movement(actual, prediction)
    result = score_quantiles(actual, prediction)
    result.pop("sorted_predictions_bps")
    return result


def _baseline_predictions(task: str, fold_train: pd.DataFrame, fold_validation: pd.DataFrame, rates: pd.DataFrame) -> dict[str, np.ndarray]:
    validation_dates = fold_validation["decision_date"].tolist()
    if task == "return":
        target = fold_train["cumulative_return_bps"].to_numpy(dtype=float)
        return {
            "zero": np.zeros(len(fold_validation), dtype=float),
            "train_mean": np.full(len(fold_validation), float(np.mean(target))),
            "train_median": np.full(len(fold_validation), float(np.median(target))),
        }
    if task == "movement":
        baseline_frame = movement_baselines(rates, validation_dates).set_index("decision_date")
        aligned = baseline_frame.loc[validation_dates]
        return {
            "trailing5": aligned["trailing5_bps2"].to_numpy(dtype=float),
            "ewma": aligned["ewma_bps2"].to_numpy(dtype=float),
        }
    baseline_frame = quantile_baselines(rates, validation_dates).set_index("decision_date")
    aligned = baseline_frame.loc[validation_dates]
    return {
        "rolling_empirical": aligned[["rolling_q05_bps", "rolling_q50_bps", "rolling_q95_bps"]].to_numpy(dtype=float),
        "ewma_scaled_empirical": aligned[["ewma_scaled_q5_bps", "ewma_scaled_q50_bps", "ewma_scaled_q95_bps"]].to_numpy(dtype=float),
    }


def run_development(rates: pd.DataFrame) -> dict[str, Any]:
    """Evaluate all frozen candidates and baselines on the three development folds."""
    samples, eligible_targets = development_samples(rates)
    folds = walk_forward_splits(samples)
    fold_results: list[dict[str, Any]] = []
    candidate_fold_scores: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = {
        task: {family: {} for family in ("linear", "boosting")} for task in TASK_SPECS
    }
    baseline_fold_scores: dict[str, dict[str, list[dict[str, float]]]] = {
        task: {name: [] for name in BASELINE_ORDER[task]} for task in TASK_SPECS
    }

    for fold in folds:
        train = fold.train
        validation = fold.validation
        x_train = train.loc[:, FEATURE_COLUMNS]
        x_validation = validation.loc[:, FEATURE_COLUMNS]
        fold_report: dict[str, Any] = {
            "name": fold.name,
            "train_n": int(len(train)),
            "validation_n": int(len(validation)),
            "train_ids_sha256": _id_hash(train["sample_id"].astype(str).tolist()),
            "validation_ids": validation["sample_id"].astype(str).tolist(),
            "purged_train_ids": list(fold.purged_train_ids),
            "purged_validation_ids": list(fold.purged_validation_ids),
            "tasks": {},
        }
        for task, spec in TASK_SPECS.items():
            target_name = spec["target"]
            y_train = train[target_name].to_numpy(dtype=float)
            y_validation = validation[target_name].to_numpy(dtype=float)
            if not np.isfinite(y_train).all() or not np.isfinite(y_validation).all():
                raise ValueError(f"{fold.name} {task} labels are not finite after endpoint purging")
            task_report: dict[str, Any] = {"baselines": {}, "learned": {}}
            baseline_predictions = _baseline_predictions(task, train, validation, rates)
            for baseline_name in BASELINE_ORDER[task]:
                metrics = _score(task, y_validation, baseline_predictions[baseline_name])
                task_report["baselines"][baseline_name] = metrics
                baseline_fold_scores[task][baseline_name].append(metrics)
            for family in ("linear", "boosting"):
                family_scores: list[dict[str, Any]] = []
                for config in candidate_configurations(task, family):
                    model = fit_model(task, family, config, x_train, y_train)
                    prediction = predict_model(model, task, x_validation)
                    metrics = _score(task, y_validation, prediction)
                    family_scores.append({"config": config, "score": metrics})
                    candidate_fold_scores[task][family].setdefault(config["config_id"], []).append(metrics)
                task_report["learned"][family] = family_scores
            fold_report["tasks"][task] = task_report
        fold_results.append(fold_report)

    selection: dict[str, Any] = {}
    for task, spec in TASK_SPECS.items():
        primary = spec["primary"]
        secondary = spec["secondary"]
        task_selection: dict[str, Any] = {}
        baseline_candidates = [
            {
                "config_id": baseline_name,
                "grid_index": index,
                "config": {"baseline_id": baseline_name},
                "scores_by_fold": baseline_fold_scores[task][baseline_name],
            }
            for index, baseline_name in enumerate(BASELINE_ORDER[task])
        ]
        task_selection["primary_baseline"] = select_candidate(
            baseline_candidates,
            primary_metric=primary,
        )
        for family in ("linear", "boosting"):
            configs = candidate_configurations(task, family)
            candidates = [
                {
                    "config_id": config["config_id"],
                    "grid_index": config["grid_index"],
                    "config": config,
                    "scores_by_fold": candidate_fold_scores[task][family][config["config_id"]],
                }
                for config in configs
            ]
            task_selection[family] = select_candidate(
                candidates,
                primary_metric=primary,
                secondary_metric=secondary,
            )
        selection[task] = task_selection

    return {
        "development_range": [DEVELOPMENT_START.isoformat(), DEVELOPMENT_END.isoformat()],
        "eligible_development_rows": int(len(eligible_targets)),
        "folds": fold_results,
        "selection": selection,
        "cold_scores_computed": False,
        "cold_target_values_read": False,
        "selection_rule": "equal-block macro-average primary score, then specified secondary metric, then listed order",
    }


def _id_hash(values: list[str]) -> str:
    import hashlib

    return hashlib.sha256("".join(value + "\n" for value in values).encode("utf-8")).hexdigest()


def fit_selected_models(rates: pd.DataFrame, selection: dict[str, Any]) -> dict[str, Any]:
    """Refit selected models once on all eligible development rows through 2025."""
    samples, eligible_targets = development_samples(rates)
    samples = samples.loc[samples["endpoint_date"] <= DEVELOPMENT_END].copy()
    if len(samples) != len(eligible_targets):
        raise ValueError("eligible development fit rows changed while preparing final refit")
    x_train = samples.loc[:, FEATURE_COLUMNS]
    models: dict[str, Any] = {}
    for task, spec in TASK_SPECS.items():
        target = samples[spec["target"]].to_numpy(dtype=float)
        models[task] = {}
        for family in ("linear", "boosting"):
            config = selection[task][family]["config"]
            models[task][family] = fit_model(task, family, config, x_train, target)
    return {
        "models": models,
        "training_sample_ids": samples["sample_id"].astype(str).tolist(),
        "training_n": int(len(samples)),
        "return_baselines": {
            "zero": 0.0,
            "train_mean": float(samples["cumulative_return_bps"].mean()),
            "train_median": float(samples["cumulative_return_bps"].median()),
        },
    }
