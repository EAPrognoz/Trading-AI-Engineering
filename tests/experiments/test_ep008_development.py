from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ai.experiments.ep008_development import run_development


def _compact_history() -> pd.DataFrame:
    groups = [
        pd.bdate_range("1998-01-02", periods=200),
        pd.bdate_range("2000-01-03", periods=310),
        pd.bdate_range("2017-01-03", periods=30),
        pd.bdate_range("2020-01-03", periods=30),
        pd.bdate_range("2023-01-03", periods=30),
        pd.bdate_range("2026-01-02", periods=8),
    ]
    dates = groups[0]
    for group in groups[1:]:
        dates = dates.append(group)
    rng = np.random.default_rng(19)
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates.date,
            "USD": 1.1 * np.cumprod(1.0 + rng.normal(0.00001, 0.0005, n)),
            "GBP": 0.85 * np.cumprod(1.0 + rng.normal(0.00001, 0.0006, n)),
            "JPY": 130 * np.cumprod(1.0 + rng.normal(0.00001, 0.0007, n)),
            "CHF": 0.96 * np.cumprod(1.0 + rng.normal(0.00001, 0.0004, n)),
        }
    )


def test_development_scores_every_candidate_and_baseline_and_freezes_selections() -> None:
    result = run_development(_compact_history())

    assert [fold["name"] for fold in result["folds"]] == ["2017-2019", "2020-2022", "2023-2025"]
    assert result["cold_scores_computed"] is False
    assert result["cold_target_values_read"] is False
    assert result["eligible_development_rows"] > 250
    for fold in result["folds"]:
        assert fold["validation_n"] > 0
        assert fold["purged_train_ids"] or fold["purged_validation_ids"]
        assert set(fold["tasks"]) == {"return", "movement", "quantiles"}
        assert len(fold["tasks"]["return"]["learned"]["linear"]) == 3
        assert len(fold["tasks"]["return"]["learned"]["boosting"]) == 3
        assert len(fold["tasks"]["movement"]["learned"]["linear"]) == 3
        assert len(fold["tasks"]["quantiles"]["learned"]["linear"]) == 3
        assert set(fold["tasks"]["return"]["baselines"]) == {"zero", "train_mean", "train_median"}
        assert set(fold["tasks"]["movement"]["baselines"]) == {"trailing5", "ewma"}
        assert set(fold["tasks"]["quantiles"]["baselines"]) == {
            "rolling_empirical",
            "ewma_scaled_empirical",
        }
    for task in ("return", "movement", "quantiles"):
        assert result["selection"][task]["primary_baseline"]["config_id"]
        assert result["selection"][task]["linear"]["config_id"]
        assert result["selection"][task]["boosting"]["config_id"]
