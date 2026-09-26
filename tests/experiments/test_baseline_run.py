from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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
