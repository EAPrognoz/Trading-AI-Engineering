from __future__ import annotations

import pandas as pd

from trading_ai.evaluation.split import chronological_split


def test_boundary_labels_are_purged() -> None:
    decision = pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")
    samples = pd.DataFrame(
        {
            "decision_timestamp": decision,
            "target_timestamp": decision + pd.Timedelta(hours=1),
            "target_h1_direction": ["UP", "DOWN"] * 10,
        }
    )

    split = chronological_split(samples, train_fraction=0.6, validation_fraction=0.2)

    assert (split.train["target_timestamp"] < split.validation_start).all()
    assert (split.validation["target_timestamp"] < split.test_start).all()
    assert (split.test["decision_timestamp"] >= split.test_start).all()
    assert split.purged_train_rows == 1
    assert split.purged_validation_rows == 1
