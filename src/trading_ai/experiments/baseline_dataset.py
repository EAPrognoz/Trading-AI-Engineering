"""Assemble the EP005 learning matrix from EP003 and EP004 contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

import numpy as np
import pandas as pd

from trading_ai.data.snapshot import dataset_manifest, load_snapshot
from trading_ai.features.engineering import build_point_in_time_features
from trading_ai.targets.direction import build_h1_direction_target


@dataclass(frozen=True)
class BaselineDataset:
    frame: pd.DataFrame
    selected_features: list[str]
    metadata: dict[str, Any]


def _read_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def assemble_episode005_dataset(
    snapshot_path: str | Path,
    feature_contract_path: str | Path,
) -> BaselineDataset:
    """Join the EP003 target to the frozen EP004 feature set.

    Rows with ZERO targets, the final unlabeled target, feature warm-up NaNs, or
    non-finite selected features are excluded with counts recorded in metadata.
    """
    market = load_snapshot(snapshot_path)
    target = build_h1_direction_target(market)
    features = build_point_in_time_features(market)
    feature_contract = _read_toml(feature_contract_path)
    selected = list(feature_contract["selected_features"])

    missing_features = sorted(set(selected).difference(features.columns))
    if missing_features:
        raise ValueError(f"selected features are not implemented: {missing_features}")

    combined = pd.DataFrame(index=market.index)
    combined["decision_timestamp"] = target["decision_timestamp"]
    combined["target_timestamp"] = target["target_timestamp"]
    combined["future_return_1h"] = target["future_return_1h"]
    combined["target_h1_direction"] = target["target_h1_direction"]
    for name in selected:
        combined[name] = features[name]

    binary_mask = combined["target_h1_direction"].isin(["UP", "DOWN"])
    binary = combined[binary_mask].copy()

    feature_values = binary[selected].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(feature_values.to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
    complete = feature_values.notna().all(axis=1).to_numpy() & finite
    assembled = binary.loc[complete].copy().reset_index(drop=True)

    if assembled.empty:
        raise ValueError("no complete binary samples remain after target and feature filters")

    return BaselineDataset(
        frame=assembled,
        selected_features=selected,
        metadata={
            "dataset": dataset_manifest(market, snapshot_path),
            "target_contract_id": "ep003-h1-direction-v1",
            "feature_contract_id": str(feature_contract["contract_id"]),
            "input_rows": int(len(market)),
            "binary_target_rows": int(binary_mask.sum()),
            "zero_target_rows": int((combined["target_h1_direction"] == "ZERO").sum()),
            "unlabeled_target_rows": int(combined["target_h1_direction"].isna().sum()),
            "rows_removed_for_feature_warmup_or_nonfinite": int(len(binary) - len(assembled)),
            "assembled_rows": int(len(assembled)),
        },
    )
