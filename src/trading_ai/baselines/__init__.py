"""Episode 005 baseline definitions."""

from .models import (
    fit_logistic_baseline,
    fit_majority_label,
    predict_majority,
    predict_previous_hour_direction,
)

__all__ = [
    "fit_logistic_baseline",
    "fit_majority_label",
    "predict_majority",
    "predict_previous_hour_direction",
]
