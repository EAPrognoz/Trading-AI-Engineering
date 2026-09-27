"""Execute B0/B1/B2 on train and validation while keeping test locked."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import tomllib

import pandas as pd

from trading_ai.baselines.models import (
    fit_logistic_baseline,
    fit_majority_label,
    predict_majority,
    predict_previous_hour_direction,
)
from trading_ai.evaluation.metrics import classification_metrics
from trading_ai.evaluation.split import chronological_split
from trading_ai.experiments.baseline_dataset import assemble_episode005_dataset


def _read_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _evaluate_partition(
    frame: pd.DataFrame,
    *,
    selected_features: list[str],
    majority_label: str,
    logistic_model: Any,
) -> dict[str, Any]:
    y = frame["target_h1_direction"]
    majority = predict_majority(frame.index, majority_label)
    persistence = predict_previous_hour_direction(
        frame["return_1h"],
        zero_fallback=majority_label,
    )
    logistic = pd.Series(
        logistic_model.predict(frame[selected_features]),
        index=frame.index,
        dtype="string",
    )

    return {
        "B0_majority_class": classification_metrics(y, majority),
        "B1_previous_hour_direction": classification_metrics(y, persistence),
        "B2_logistic_regression": classification_metrics(y, logistic),
    }


def run_episode005_validation_baselines(
    snapshot_path: str | Path,
    experiment_contract_path: str | Path = "configs/experiments/ep005_baselines.toml",
    *,
    source_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the Episode 005 baseline benchmark without evaluating the test set."""
    contract = _read_toml(experiment_contract_path)
    feature_contract_path = Path(str(contract["feature_contract"]))
    if not feature_contract_path.is_absolute():
        feature_contract_path = Path(experiment_contract_path).resolve().parents[2] / feature_contract_path

    train_fraction = float(contract["train_fraction"])
    validation_fraction = float(contract["validation_fraction"])
    test_fraction = float(contract["test_fraction"])
    if abs((train_fraction + validation_fraction + test_fraction) - 1.0) > 1e-12:
        raise ValueError("train/validation/test fractions must sum to 1")

    dataset = assemble_episode005_dataset(
        snapshot_path,
        feature_contract_path,
        source_manifest_path=source_manifest_path,
    )
    split = chronological_split(
        dataset.frame,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    features = dataset.selected_features
    y_train = split.train["target_h1_direction"]
    majority_label = fit_majority_label(y_train)
    logistic_model = fit_logistic_baseline(split.train[features], y_train)

    train_metrics = _evaluate_partition(
        split.train,
        selected_features=features,
        majority_label=majority_label,
        logistic_model=logistic_model,
    )
    validation_metrics = _evaluate_partition(
        split.validation,
        selected_features=features,
        majority_label=majority_label,
        logistic_model=logistic_model,
    )

    return {
        "contract_id": str(contract["contract_id"]),
        "task": "H1 next-return direction",
        "target_contract_id": dataset.metadata["target_contract_id"],
        "feature_contract_id": dataset.metadata["feature_contract_id"],
        "dataset": dataset.metadata,
        "selected_features": features,
        "split": {
            "requested_fractions": {
                "train": train_fraction,
                "validation": validation_fraction,
                "test": test_fraction,
            },
            "validation_start": split.validation_start.isoformat(),
            "test_start": split.test_start.isoformat(),
            "train_rows": int(len(split.train)),
            "validation_rows": int(len(split.validation)),
            "test_rows": int(len(split.test)),
            "purged_train_boundary_rows": split.purged_train_rows,
            "purged_validation_boundary_rows": split.purged_validation_rows,
        },
        "fit": {
            "majority_label_from_train": majority_label,
            "logistic_scaler_fit_scope": "train_only",
            "logistic_model_fit_scope": "train_only",
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test": {
            "policy": str(contract["test_policy"]),
            "evaluated": False,
            "rows": int(len(split.test)),
            "note": "The final test partition remains locked during Episode 005 baseline development.",
        },
    }
