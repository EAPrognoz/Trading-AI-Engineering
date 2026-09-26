"""Serializable predictive metrics for Episode 005 baselines."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef

LABELS = ["DOWN", "UP"]


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    """Return the fixed Episode 005 binary-classification metric set."""
    true = pd.Series(y_true, dtype="string")
    pred = pd.Series(y_pred, dtype="string")
    matrix = confusion_matrix(true, pred, labels=LABELS)

    return {
        "rows": int(len(true)),
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "mcc": float(matthews_corrcoef(true, pred)),
        "class_counts": {
            label: int((true == label).sum())
            for label in LABELS
        },
        "prediction_counts": {
            label: int((pred == label).sum())
            for label in LABELS
        },
        "confusion_matrix": {
            "labels": LABELS,
            "values": matrix.astype(int).tolist(),
        },
    }
