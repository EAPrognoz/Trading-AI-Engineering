"""Assemble the EP005 learning matrix from EP003 and EP004 contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

import numpy as np
import pandas as pd

from trading_ai.data.bundle import load_market_data_bundle
from trading_ai.data.snapshot import dataset_manifest, load_snapshot
from trading_ai.evaluation.split import ChronologicalSplit, chronological_split
from trading_ai.features.alignment import (
    build_multitimeframe_feature_registry,
    build_multitimeframe_point_in_time_features,
)
from trading_ai.features.engineering import build_point_in_time_features
from trading_ai.targets.direction import build_h1_direction_target


@dataclass(frozen=True)
class BaselineDataset:
    frame: pd.DataFrame
    selected_features: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PreparedEpisode005Split:
    dataset: BaselineDataset
    split: ChronologicalSplit
    contract: dict[str, Any]


def _read_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def assemble_episode005_dataset(
    snapshot_path: str | Path | None,
    feature_contract_path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
    bundle_manifest_path: str | Path | None = None,
) -> BaselineDataset:
    """Join the EP003 target to the frozen EP004 feature set."""
    if (snapshot_path is None) == (bundle_manifest_path is None):
        raise ValueError("choose exactly one source mode: snapshot or bundle")
    if bundle_manifest_path is not None and source_manifest_path is not None:
        raise ValueError("source manifest cannot be combined with bundle mode")

    feature_contract = _read_toml(feature_contract_path)
    if bundle_manifest_path is None:
        assert snapshot_path is not None
        market = load_snapshot(snapshot_path)
        target = build_h1_direction_target(market)
        features = build_point_in_time_features(market)
        selected = list(feature_contract["selected_features"])
        source_metadata = {
            "dataset": dataset_manifest(
                market, snapshot_path, source_manifest_path=source_manifest_path,
            )
        }
    else:
        bundle = load_market_data_bundle(bundle_manifest_path)
        market = bundle.frames["H1"]
        target = build_h1_direction_target(market)
        start = pd.Timestamp(bundle.manifest["analysis_start"])
        end = pd.Timestamp(bundle.manifest["analysis_end"])
        in_analysis = target["decision_timestamp"].ge(start) & target[
            "decision_timestamp"
        ].lt(end)
        target = target.loc[in_analysis].reset_index(drop=True)
        registry = build_multitimeframe_feature_registry(feature_contract)
        selected = [
            entry["output_name"] for entry in registry if entry["selected"]
        ]
        features = build_multitimeframe_point_in_time_features(
            bundle.frames, target["decision_timestamp"], feature_contract,
        )
        members = bundle.manifest["streams"]
        source_metadata = {
            "bundle": {
                "contract_id": bundle.manifest["contract_id"],
                "bundle_id": bundle.manifest["bundle_id"],
                "manifest_sha256": bundle.manifest_sha256,
                "symbol": bundle.manifest["symbol"],
                "source_manifest_sha256": {
                    timeframe: members[timeframe]["source_manifest"]["sha256"]
                    for timeframe in ("H1", "H4", "D1")
                },
                "accepted_dataset_sha256": {
                    timeframe: members[timeframe]["accepted_dataset"]["sha256"]
                    for timeframe in ("H1", "H4", "D1")
                },
                "analysis_start": start.isoformat(),
                "analysis_end": end.isoformat(),
            }
        }

    missing_features = sorted(set(selected).difference(features.columns))
    if missing_features:
        raise ValueError(f"selected features are not implemented: {missing_features}")

    combined = pd.DataFrame(index=target.index)
    combined["decision_timestamp"] = target["decision_timestamp"]
    combined["target_timestamp"] = target["target_timestamp"]
    combined["future_return_1h"] = target["future_return_1h"]
    combined["target_h1_direction"] = target["target_h1_direction"]
    for name in selected:
        combined[name] = features[name]

    binary_mask = combined["target_h1_direction"].isin(["UP", "DOWN"])
    binary = combined[binary_mask].copy()

    feature_values = binary[selected].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(
        feature_values.to_numpy(dtype=float, na_value=np.nan)
    ).all(axis=1)
    complete = feature_values.notna().all(axis=1).to_numpy() & finite
    assembled = binary.loc[complete].copy().reset_index(drop=True)

    if assembled.empty:
        raise ValueError(
            "no complete binary samples remain after target and feature filters"
        )

    return BaselineDataset(
        frame=assembled,
        selected_features=selected,
        metadata={
            **source_metadata,
            "target_contract_id": "ep003-h1-direction-v1",
            "feature_contract_id": str(feature_contract["contract_id"]),
            "input_rows": int(len(market)),
            "binary_target_rows": int(binary_mask.sum()),
            "zero_target_rows": int(
                (combined["target_h1_direction"] == "ZERO").sum()
            ),
            "gap_target_rows": int(
                (combined["target_h1_direction"] == "GAP").sum()
            ),
            "unlabeled_target_rows": int(
                combined["target_h1_direction"].isna().sum()
            ),
            "rows_removed_for_feature_warmup_or_nonfinite": int(
                len(binary) - len(assembled)
            ),
            "assembled_rows": int(len(assembled)),
        },
    )


def prepare_episode005_split(
    snapshot_path: str | Path | None,
    experiment_contract_path: str | Path = "configs/experiments/ep005_baselines.toml",
    *,
    source_manifest_path: str | Path | None = None,
    bundle_manifest_path: str | Path | None = None,
) -> PreparedEpisode005Split:
    """Prepare the canonical EP005 dataset and chronological partitions."""
    contract_path = Path(experiment_contract_path)
    contract = _read_toml(contract_path)
    if bundle_manifest_path is not None and (
        contract.get("contract_id") != "ep005-btc-mtf-baseline-v1"
        or contract.get("test_policy") != "locked"
        or any(
            type(contract.get(field)) not in (int, float)
            or contract[field] != expected
            for field, expected in (
                ("train_fraction", 0.60),
                ("validation_fraction", 0.20),
                ("test_fraction", 0.20),
            )
        )
    ):
        raise ValueError("BTC EP005 contract requires fixed ID, 60/20/20 split, and locked test")

    train_fraction = float(contract["train_fraction"])
    validation_fraction = float(contract["validation_fraction"])
    test_fraction = float(contract["test_fraction"])
    if abs((train_fraction + validation_fraction + test_fraction) - 1.0) > 1e-12:
        raise ValueError("train/validation/test fractions must sum to 1")

    feature_contract_path = Path(str(contract["feature_contract"]))
    if not feature_contract_path.is_absolute():
        feature_contract_path = (
            contract_path.resolve().parents[2] / feature_contract_path
        )

    dataset = assemble_episode005_dataset(
        snapshot_path,
        feature_contract_path,
        source_manifest_path=source_manifest_path,
        bundle_manifest_path=bundle_manifest_path,
    )
    split = chronological_split(
        dataset.frame,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    return PreparedEpisode005Split(
        dataset=dataset,
        split=split,
        contract=contract,
    )
