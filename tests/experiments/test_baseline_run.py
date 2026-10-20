from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import trading_ai.experiments.baseline_dataset as baseline_dataset
from trading_ai.experiments.baseline_run import run_episode005_validation_baselines


def _write_snapshot(path: Path, periods: int = 240) -> None:
    t = np.arange(periods, dtype=float)
    timestamp = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    close = 100.0 + 0.01 * t + 0.8 * np.sin(t / 3.0) + 0.25 * np.sin(t / 11.0)
    open_ = close - 0.05 * np.cos(t / 5.0)
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_,
            "high": np.maximum(open_, close) + 0.15,
            "low": np.minimum(open_, close) - 0.15,
            "close": close,
            "tick_volume": 100 + (np.arange(periods) % 17) * 4,
            "spread": 10 + (np.arange(periods) % 3),
            "real_volume": np.zeros(periods, dtype=int),
        }
    )
    frame.to_csv(path, index=False)


def test_episode005_run_keeps_test_locked(tmp_path: Path) -> None:
    snapshot = tmp_path / "synthetic_h1.csv"
    _write_snapshot(snapshot)

    report = run_episode005_validation_baselines(snapshot)

    assert report["test"]["evaluated"] is False
    assert report["test"]["policy"] == "locked"
    assert report["split"]["purged_train_boundary_rows"] >= 1
    assert report["split"]["purged_validation_boundary_rows"] >= 1
    assert set(report["validation_metrics"]) == {
        "B0_majority_class",
        "B1_previous_hour_direction",
        "B2_logistic_regression",
    }
    json.dumps(report)


def test_uniform_timestamp_shift_does_not_change_baseline_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = tmp_path / "synthetic_h1.csv"
    _write_snapshot(snapshot)

    corrected = run_episode005_validation_baselines(snapshot)
    real_builder = baseline_dataset.build_h1_direction_target

    def legacy_timestamp_builder(frame: pd.DataFrame) -> pd.DataFrame:
        result = real_builder(frame).copy()
        result["decision_timestamp"] = (
            result["decision_timestamp"] - pd.Timedelta(hours=1)
        )
        result["target_timestamp"] = (
            result["target_timestamp"] - pd.Timedelta(hours=1)
        )
        return result

    monkeypatch.setattr(
        baseline_dataset,
        "build_h1_direction_target",
        legacy_timestamp_builder,
    )
    legacy = run_episode005_validation_baselines(snapshot)

    assert corrected["validation_metrics"] == legacy["validation_metrics"]
    assert corrected["split"]["train_rows"] == legacy["split"]["train_rows"]
    assert corrected["split"]["validation_rows"] == legacy["split"]["validation_rows"]
    assert corrected["split"]["test_rows"] == legacy["split"]["test_rows"]
    assert (
        corrected["split"]["purged_train_boundary_rows"]
        == legacy["split"]["purged_train_boundary_rows"]
    )
    assert (
        corrected["split"]["purged_validation_boundary_rows"]
        == legacy["split"]["purged_validation_boundary_rows"]
    )
    assert pd.Timestamp(corrected["split"]["validation_start"]) == (
        pd.Timestamp(legacy["split"]["validation_start"])
        + pd.Timedelta(hours=1)
    )
    assert pd.Timestamp(corrected["split"]["test_start"]) == (
        pd.Timestamp(legacy["split"]["test_start"])
        + pd.Timedelta(hours=1)
    )


def test_shared_episode005_preparation_matches_full_report(tmp_path: Path) -> None:
    snapshot = tmp_path / "synthetic_h1.csv"
    _write_snapshot(snapshot)

    prepared = baseline_dataset.prepare_episode005_split(snapshot)
    report = run_episode005_validation_baselines(snapshot)

    assert len(prepared.split.train) == report["split"]["train_rows"]
    assert len(prepared.split.validation) == report["split"]["validation_rows"]
    assert len(prepared.split.test) == report["split"]["test_rows"]
    assert (
        prepared.split.purged_train_rows
        == report["split"]["purged_train_boundary_rows"]
    )
    assert (
        prepared.split.purged_validation_rows
        == report["split"]["purged_validation_boundary_rows"]
    )


def test_minimal_b0_matches_full_validation_membership_and_accuracy(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "synthetic_h1.csv"
    _write_snapshot(snapshot)
    report = run_episode005_validation_baselines(snapshot)

    completed = subprocess.run(
        [
            sys.executable,
            "examples/ep005_baseline_minimal.py",
            str(snapshot),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    expected_rows = report["split"]["validation_rows"]
    expected_accuracy = report["validation_metrics"]["B0_majority_class"]["accuracy"]
    assert f"Validation rows: {expected_rows}" in completed.stdout
    assert (
        f"B0 majority-class accuracy: {expected_accuracy:.4f}"
        in completed.stdout
    )
