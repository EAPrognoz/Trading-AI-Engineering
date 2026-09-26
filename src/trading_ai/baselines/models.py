"""Simple baselines that future candidate models must justify beating."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_majority_label(y_train: pd.Series) -> str:
    """Return the training-set majority class with deterministic tie-breaking."""
    counts = pd.Series(y_train, dtype="string").value_counts()
    if counts.empty:
        raise ValueError("cannot fit majority baseline on an empty target")
    maximum = int(counts.max())
    winners = sorted(str(label) for label, count in counts.items() if int(count) == maximum)
    return winners[0]


def predict_majority(index: pd.Index, label: str) -> pd.Series:
    """Predict one fixed label for every row."""
    return pd.Series(label, index=index, dtype="string")


def predict_previous_hour_direction(
    return_1h: pd.Series,
    *,
    zero_fallback: str,
) -> pd.Series:
    """B1: predict the next direction from the sign of the current H1 return."""
    values = pd.to_numeric(return_1h, errors="raise").to_numpy(dtype=float)
    prediction = np.full(len(values), zero_fallback, dtype=object)
    prediction[values > 0] = "UP"
    prediction[values < 0] = "DOWN"
    return pd.Series(prediction, index=return_1h.index, dtype="string")


def fit_logistic_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """B2: standardized Logistic Regression fit on training data only."""
    if x_train.empty:
        raise ValueError("cannot fit Logistic Regression on an empty training set")
    if pd.Series(y_train).nunique() < 2:
        raise ValueError("Logistic Regression requires both target classes in training")

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model
