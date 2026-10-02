"""Diagnostics for the Episode 003 target contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trading_ai.data.bundle import load_market_data_bundle
from trading_ai.data.snapshot import dataset_manifest, load_snapshot
from trading_ai.targets.direction import build_h1_direction_target


def _maybe_float(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def analyze_h1_direction_target(
    snapshot_path: str | Path | None = None,
    *,
    source_manifest_path: str | Path | None = None,
    bundle_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze class balance and return distribution for exact H1 targets."""
    if (snapshot_path is None) == (bundle_manifest_path is None):
        raise ValueError("choose exactly one source mode: snapshot or bundle")
    if bundle_manifest_path is not None and source_manifest_path is not None:
        raise ValueError("source manifest cannot be combined with bundle mode")
    if bundle_manifest_path is None:
        assert snapshot_path is not None
        frame = load_snapshot(snapshot_path)
        dataset = dataset_manifest(
            frame, snapshot_path, source_manifest_path=source_manifest_path,
        )
    else:
        bundle = load_market_data_bundle(bundle_manifest_path)
        frame = bundle.frames["H1"]
        members = bundle.manifest["streams"]
        dataset = {
            "bundle": {
                "contract_id": bundle.manifest["contract_id"],
                "bundle_id": bundle.manifest["bundle_id"],
                "manifest_sha256": bundle.manifest_sha256,
                "symbol": bundle.manifest["symbol"],
                "source_manifest_sha256": {
                    "H1": members["H1"]["source_manifest"]["sha256"],
                },
                "accepted_dataset_sha256": {
                    "H1": members["H1"]["accepted_dataset"]["sha256"],
                },
                "analysis_start": bundle.manifest["analysis_start"],
                "analysis_end": bundle.manifest["analysis_end"],
            }
        }
    target = build_h1_direction_target(frame)
    if bundle_manifest_path is not None:
        start = pd.Timestamp(bundle.manifest["analysis_start"])
        end = pd.Timestamp(bundle.manifest["analysis_end"])
        target = target.loc[
            target["decision_timestamp"].ge(start)
            & target["decision_timestamp"].lt(end)
        ]
    labeled = target[target["target_h1_direction"].isin(["UP", "DOWN"])]
    zero_count = int((target["target_h1_direction"] == "ZERO").sum())
    gap_count = int((target["target_h1_direction"] == "GAP").sum())
    unlabeled_count = int(target["target_h1_direction"].isna().sum())

    counts = labeled["target_h1_direction"].value_counts().to_dict()
    up_count = int(counts.get("UP", 0))
    down_count = int(counts.get("DOWN", 0))
    n_labeled = int(len(labeled))

    future = target["future_return_1h"].dropna()
    summary = future.describe(percentiles=[0.25, 0.5, 0.75])

    return {
        "contract_id": "ep003-h1-direction-v1",
        "task": "binary_classification",
        "timeframe": "H1",
        "forecast_horizon": "1h",
        "decision_time": "bar_close",
        "return_definition": "close[t+1] / close[t] - 1",
        "zero_return_policy": "exclude",
        "non_consecutive_pair_policy": "exclude",
        "dataset": dataset,
        "observations": {
            "input_rows": int(len(target)),
            "binary_labeled_rows": n_labeled,
            "zero_return_rows": zero_count,
            "non_consecutive_h1_rows": gap_count,
            "unlabeled_future_rows": unlabeled_count,
        },
        "class_counts": {"UP": up_count, "DOWN": down_count},
        "class_fractions": {
            "UP": (up_count / n_labeled) if n_labeled else None,
            "DOWN": (down_count / n_labeled) if n_labeled else None,
        },
        "future_return_summary": {
            "count": int(summary.get("count", 0)),
            "mean": _maybe_float(summary.get("mean")),
            "std": _maybe_float(summary.get("std")),
            "min": _maybe_float(summary.get("min")),
            "p25": _maybe_float(summary.get("25%")),
            "median": _maybe_float(summary.get("50%")),
            "p75": _maybe_float(summary.get("75%")),
            "max": _maybe_float(summary.get("max")),
        },
    }
