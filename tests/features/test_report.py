from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_ai.experiments.baseline_dataset import prepare_episode005_split
from trading_ai.features.report import analyze_feature_contract


FEATURE_CONTRACT = "configs/features/ep004_baseline_features.toml"


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


def test_feature_report_correlations_use_exact_episode005_train_membership(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "synthetic_h1.csv"
    _write_snapshot(snapshot)

    report = analyze_feature_contract(snapshot, FEATURE_CONTRACT)
    prepared = prepare_episode005_split(snapshot)

    assert report["selected_features"] == prepared.dataset.selected_features
    assert report["correlation_scope"]["policy"] == "episode005_train_only"
    assert report["correlation_scope"]["rows"] == len(prepared.split.train)
    assert report["correlation_scope"]["validation_start"] == (
        prepared.split.validation_start.isoformat()
    )
    assert report["correlation_scope"]["test_used"] is False


def test_zero_target_near_boundary_does_not_change_correlation_scope_definition(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "synthetic_with_zero_h1.csv"
    _write_snapshot(snapshot)

    frame = pd.read_csv(snapshot)
    frame.loc[130, "close"] = frame.loc[129, "close"]
    frame.loc[130, "high"] = max(
        float(frame.loc[130, "open"]),
        float(frame.loc[130, "close"]),
    ) + 0.15
    frame.loc[130, "low"] = min(
        float(frame.loc[130, "open"]),
        float(frame.loc[130, "close"]),
    ) - 0.15
    frame.to_csv(snapshot, index=False)

    report = analyze_feature_contract(snapshot, FEATURE_CONTRACT)
    prepared = prepare_episode005_split(snapshot)

    assert prepared.dataset.metadata["zero_target_rows"] >= 1
    assert report["correlation_scope"]["rows"] == len(prepared.split.train)
    assert report["correlation_scope"]["test_used"] is False
