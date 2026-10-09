from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_ai.experiments.ep008_metrics import (
    VARIANCE_FLOOR_BPS2,
    movement_baselines,
    moving_block_ci,
    quantile_baselines,
    score_movement,
    score_quantiles,
    score_return,
)


def _history(n: int = 800, *, flat_usd: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range("1999-10-01", periods=n)
    index = np.arange(n, dtype=float)
    usd = np.full(n, 1.1) if flat_usd else 1.1 * np.cumprod(1.0 + 0.0002 + 0.0001 * np.sin(index / 5))
    return pd.DataFrame(
        {
            "date": dates.date,
            "USD": usd,
            "GBP": 0.85 * np.cumprod(1.0 + 0.0001 * np.cos(index / 7)),
            "JPY": 130 * np.cumprod(1.0 + 0.0001 * np.sin(index / 11)),
            "CHF": 0.96 * np.cumprod(1.0 + 0.0001 * np.cos(index / 13)),
        }
    )


def test_return_metric_uses_bps_mae_and_rmse() -> None:
    report = score_return(np.array([100.0, -100.0]), np.array([50.0, -150.0]))

    assert report["n"] == 2
    assert report["mae_bps"] == pytest.approx(50.0)
    assert report["rmse_bps"] == pytest.approx(50.0)


def test_movement_qlike_applies_bps2_floor_and_reports_rms_mae() -> None:
    actual = np.array([0.0, VARIANCE_FLOOR_BPS2, 4 * VARIANCE_FLOOR_BPS2])
    raw_prediction = np.array([-2 * VARIANCE_FLOOR_BPS2, 4 * VARIANCE_FLOOR_BPS2, 16 * VARIANCE_FLOOR_BPS2])

    report = score_movement(actual, raw_prediction)

    expected_actual = np.maximum(actual, VARIANCE_FLOOR_BPS2)
    expected_prediction = np.maximum(raw_prediction, VARIANCE_FLOOR_BPS2)
    ratio = expected_actual / expected_prediction
    expected_qlike = ratio - np.log(ratio) - 1.0
    expected_rms_mae = np.mean(np.abs(np.sqrt(actual) - np.sqrt(expected_prediction)))
    assert report["floor_bps2"] == pytest.approx(1e-4)
    assert report["qlike"] == pytest.approx(float(np.mean(expected_qlike)))
    assert report["rms_mae_bps"] == pytest.approx(float(expected_rms_mae))
    assert report["prediction_floor_clipped_count"] == 1
    assert report["prediction_floor_clipped_share"] == pytest.approx(1 / 3)
    assert report["actual_at_or_below_floor_count"] == 2
    assert report["qlike"] >= 0


def test_quantile_crossing_is_counted_before_sorting_and_scores_after_sorting() -> None:
    actual = np.array([1.0, 0.0, 10.0])
    raw = np.array([[2.0, 0.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])

    report = score_quantiles(actual, raw)

    sorted_predictions = np.sort(raw, axis=1)
    assert report["crossing_count"] == 1
    assert report["crossing_rate"] == pytest.approx(1 / 3)
    assert report["sorted_predictions_bps"] == sorted_predictions.tolist()
    assert report["interval_coverage_90"] == pytest.approx(1 / 3)
    assert report["interval_mean_width_bps"] == pytest.approx(7 / 3)
    assert report["median_mae_bps"] == pytest.approx(14 / 3)
    assert set(report["pinball_mean_bps"]) == {"q05", "q50", "q95"}


def test_movement_baselines_use_recent_known_returns_including_decision_return() -> None:
    rates = _history()
    decision_index = 500
    decision = rates.iloc[decision_index]["date"]
    output = movement_baselines(rates, [decision]).iloc[0]
    usd = rates["USD"].to_numpy(dtype=float)
    returns = usd[1:] / usd[:-1] - 1.0
    expected_trailing5 = np.mean(returns[decision_index - 5 : decision_index] ** 2) * 1e8
    training_start_index = next(i for i, value in enumerate(rates["date"]) if value >= date(2000, 1, 1))
    training_returns = returns[training_start_index - 1 : training_start_index + 59]
    initial_level = float(np.mean(training_returns**2))
    for position in range(training_start_index + 60, decision_index + 1):
        observed_return = returns[position - 1]
        initial_level = 0.94 * initial_level + 0.06 * observed_return**2

    assert output["trailing5_bps2"] == pytest.approx(expected_trailing5)
    assert output["ewma_bps2"] == pytest.approx(initial_level * 1e8)


def test_rolling_quantiles_use_latest_252_matured_targets_and_ignore_future_rates() -> None:
    rates = _history()
    decision_index = 500
    decision = rates.iloc[decision_index]["date"]
    output = quantile_baselines(rates, [decision]).iloc[0]
    changed = rates.copy()
    changed.loc[changed.index > decision_index, ["USD", "GBP", "JPY", "CHF"]] *= 1.5
    changed_output = quantile_baselines(changed, [decision]).iloc[0]

    usd = rates["USD"].to_numpy(dtype=float)
    start = next(i for i, value in enumerate(rates["date"]) if value >= date(2000, 1, 1))
    last_matured_decision = decision_index - 5
    first_history_decision = max(start, last_matured_decision - 251)
    history = np.array(
        [(usd[i + 5] / usd[i] - 1.0) * 10_000 for i in range(first_history_decision, last_matured_decision + 1)]
    )
    expected = np.quantile(history, [0.05, 0.50, 0.95], method="linear")

    assert output["history_n"] == 252
    assert [output[f"rolling_q{q:02d}_bps"] for q in (5, 50, 95)] == pytest.approx(expected)
    assert output["scale_ratio"] >= 0
    for column in [name for name in output.index if name.endswith("_bps") or name == "scale_ratio"]:
        assert output[column] == pytest.approx(changed_output[column])


def test_zero_scale_denominator_uses_ratio_one_without_changing_quantiles() -> None:
    rates = _history(flat_usd=True)
    decision = rates.iloc[500]["date"]

    output = quantile_baselines(rates, [decision]).iloc[0]

    assert output["scale_ratio"] == pytest.approx(1.0)
    assert [output[f"ewma_scaled_q{q}_bps"] for q in (5, 50, 95)] == pytest.approx(
        [output[f"rolling_q{q:02d}_bps"] for q in (5, 50, 95)]
    )


def test_moving_block_interval_is_paired_reproducible_and_fixed_for_zero_delta() -> None:
    model = np.arange(80, dtype=float) / 10
    baseline = np.zeros(80, dtype=float)
    first = moving_block_ci(model, baseline, n_resamples=500)
    second = moving_block_ci(model, baseline, n_resamples=500)
    zero = moving_block_ci(model, model, n_resamples=500)

    assert first == second
    assert first["mean_loss_difference"] == pytest.approx(float(np.mean(model)))
    assert first["ci_low"] <= first["mean_loss_difference"] <= first["ci_high"]
    assert zero["mean_loss_difference"] == pytest.approx(0.0)
    assert zero["ci_low"] == pytest.approx(0.0)
    assert zero["ci_high"] == pytest.approx(0.0)


def test_metric_functions_reject_nonfinite_or_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        score_return(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        score_movement(np.array([1.0]), np.array([np.nan]))
    with pytest.raises(ValueError, match="shape"):
        score_quantiles(np.array([1.0]), np.zeros((1, 2)))
