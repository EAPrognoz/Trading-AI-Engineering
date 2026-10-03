"""Diagnostics for the Episode 004 feature contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import tomllib

import numpy as np
import pandas as pd

from trading_ai.data.bundle import load_market_data_bundle
from trading_ai.data.manifest import sha256_file
from trading_ai.data.snapshot import dataset_manifest, load_snapshot
from trading_ai.features.alignment import (
    build_multitimeframe_feature_registry,
    build_multitimeframe_point_in_time_features,
)
from trading_ai.features.engineering import FEATURE_LOOKBACK_BARS, build_point_in_time_features


def _load_contract(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _top_correlations(features: pd.DataFrame, columns: list[str], limit: int = 10) -> list[dict[str, Any]]:
    corr = features[columns].corr().abs()
    pairs: list[tuple[float, str, str]] = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append((float(value), left, right))
    pairs.sort(reverse=True)
    return [
        {"left": left, "right": right, "abs_correlation": value}
        for value, left, right in pairs[:limit]
    ]


def _bundle_feature_report(
    bundle_manifest_path: str | Path,
    contract_path: str | Path,
    experiment_contract_path: str | Path,
) -> dict[str, Any]:
    from trading_ai.experiments.baseline_dataset import prepare_episode005_split

    experiment_path = Path(experiment_contract_path)
    experiment_contract = _load_contract(experiment_path)
    canonical_contract_path = Path(str(experiment_contract["feature_contract"]))
    if not canonical_contract_path.is_absolute():
        canonical_contract_path = (
            experiment_path.resolve().parents[2] / canonical_contract_path
        )
    feature_contract_sha256 = sha256_file(contract_path)
    if feature_contract_sha256 != sha256_file(canonical_contract_path):
        raise ValueError("feature contract does not match EP005 contract")

    contract = _load_contract(contract_path)
    registry = build_multitimeframe_feature_registry(contract)
    bundle = load_market_data_bundle(bundle_manifest_path)
    decisions = bundle.frames["H1"]["timestamp"] + pd.Timedelta(hours=1)
    start = pd.Timestamp(bundle.manifest["analysis_start"])
    end = pd.Timestamp(bundle.manifest["analysis_end"])
    decisions = decisions.loc[decisions.ge(start) & decisions.lt(end)].reset_index(drop=True)
    features = build_multitimeframe_point_in_time_features(
        bundle.frames, decisions, contract,
    )
    candidates = [row["output_name"] for row in registry]
    selected = [row["output_name"] for row in registry if row["selected"]]
    prepared = prepare_episode005_split(
        None, experiment_contract_path, bundle_manifest_path=bundle_manifest_path,
    )
    if selected != prepared.dataset.selected_features:
        raise ValueError("EP004 selected features differ from the EP005 preparation")
    train_decisions = pd.to_datetime(
        prepared.split.train["decision_timestamp"], utc=True,
    )
    correlation_features = features.loc[
        features["decision_timestamp"].isin(train_decisions), candidates
    ]
    if len(correlation_features) != len(train_decisions):
        raise ValueError("EP004 correlation rows differ from EP005 train membership")

    diagnostics: dict[str, Any] = {}
    for row in registry:
        name = row["output_name"]
        series = pd.to_numeric(features[name], errors="coerce")
        finite = pd.Series(
            np.isfinite(series.to_numpy(dtype=float, na_value=np.nan)),
            index=series.index,
        )
        first_valid = features.loc[series.notna() & finite, "decision_timestamp"]
        diagnostics[name] = {
            "source_timeframe": row["source_timeframe"],
            "native_bar_lookback": row["native_bar_lookback"],
            "elapsed_duration_hours": row["elapsed_duration_hours"],
            "missing_count": int(series.isna().sum()),
            "non_finite_count": int((series.notna() & ~finite).sum()),
            "unique_non_null": int(series.dropna().nunique()),
            "first_valid_timestamp": first_valid.iloc[0].isoformat() if len(first_valid) else None,
            "included_in_baseline_v1": row["selected"],
        }

    availability: dict[str, Any] = {}
    for timeframe in ("H1", "H4", "D1"):
        source_close = features[f"{timeframe.lower()}__source_nominal_close_timestamp"]
        valid = source_close.dropna()
        availability[timeframe] = {
            "source_nominal_close_timestamp": {
                "first": valid.iloc[0].isoformat() if len(valid) else None,
                "last": valid.iloc[-1].isoformat() if len(valid) else None,
                "missing_count": int(source_close.isna().sum()),
            },
            "available_decision_rows": int(source_close.notna().sum()),
        }

    return {
        "contract_id": str(contract["contract_id"]),
        "decision_time": str(contract["decision_time"]),
        "asof_policy": str(contract["asof_policy"]),
        "preprocessing_version": "ep004-mtf-native-asof-v1",
        "feature_contract_sha256": feature_contract_sha256,
        "experiment_contract_sha256": sha256_file(experiment_contract_path),
        "selection_basis": str(contract["selection_basis"]),
        "selection_guards": {
            "uses_target_statistics": bool(contract["uses_target_statistics"]),
            "uses_validation_statistics": bool(contract["uses_validation_statistics"]),
            "uses_test_statistics": bool(contract["uses_test_statistics"]),
        },
        "dataset": {"bundle": prepared.dataset.metadata["bundle"]},
        "observations": {"decision_rows": int(len(features))},
        "candidate_features": candidates,
        "selected_features": selected,
        "feature_diagnostics": diagnostics,
        "timeframe_availability": availability,
        "top_absolute_correlations": _top_correlations(correlation_features, candidates),
        "correlation_scope": {
            "policy": "episode005_train_only",
            "rows": int(len(correlation_features)),
            "validation_start": prepared.split.validation_start.isoformat(),
            "test_used": False,
        },
        "correlation_note": (
            "Baseline selection is predeclared. Correlations use exact Episode 005 "
            "train membership; interval-wide availability is descriptive only."
        ),
    }


def analyze_feature_contract(
    snapshot_path: str | Path | None,
    contract_path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
    bundle_manifest_path: str | Path | None = None,
    experiment_contract_path: str | Path = "configs/experiments/ep005_baselines.toml",
) -> dict[str, Any]:
    """Build Episode 004 features and describe the frozen baseline feature set.

    Correlations are diagnostic only. The selected feature list is read from the
    predeclared contract and is not chosen from target, validation, or test
    performance.
    """
    from trading_ai.experiments.baseline_dataset import prepare_episode005_split

    if (snapshot_path is None) == (bundle_manifest_path is None):
        raise ValueError("choose exactly one source mode: snapshot or bundle")
    if bundle_manifest_path is not None:
        if source_manifest_path is not None:
            raise ValueError("source manifest cannot be combined with bundle mode")
        return _bundle_feature_report(
            bundle_manifest_path, contract_path, experiment_contract_path,
        )

    assert snapshot_path is not None
    frame = load_snapshot(snapshot_path)
    features = build_point_in_time_features(frame)
    contract = _load_contract(contract_path)

    candidates = list(contract["candidate_features"])
    selected = list(contract["selected_features"])
    unknown = sorted(set(candidates).difference(features.columns))
    if unknown:
        raise ValueError(f"candidate features are not implemented: {unknown}")
    if not set(selected).issubset(candidates):
        raise ValueError("selected_features must be a subset of candidate_features")

    exclusion_reasons = dict(contract.get("exclusion_reasons", {}))
    diagnostics: dict[str, Any] = {}

    prepared = prepare_episode005_split(
        snapshot_path,
        experiment_contract_path,
        source_manifest_path=source_manifest_path,
    )
    train_open_timestamp = (
        pd.to_datetime(prepared.split.train["decision_timestamp"], utc=True)
        - pd.Timedelta(hours=1)
    )
    correlation_mask = features["timestamp"].isin(train_open_timestamp)
    correlation_features = features.loc[correlation_mask, candidates].copy()

    for name in candidates:
        series = pd.to_numeric(features[name], errors="coerce")
        finite_mask = np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))
        valid_mask = series.notna() & pd.Series(finite_mask, index=series.index)
        first_valid = features.loc[valid_mask, "timestamp"]
        diagnostics[name] = {
            "lookback_bars": int(FEATURE_LOOKBACK_BARS[name]),
            "missing_count": int(series.isna().sum()),
            "non_finite_count": int((series.notna() & ~pd.Series(finite_mask, index=series.index)).sum()),
            "unique_non_null": int(series.dropna().nunique()),
            "first_valid_timestamp": first_valid.iloc[0].isoformat() if len(first_valid) else None,
            "included_in_baseline_v1": name in selected,
            "exclusion_reason": None if name in selected else exclusion_reasons.get(name),
        }

    return {
        "contract_id": str(contract["contract_id"]),
        "decision_time": str(contract["decision_time"]),
        "selection_basis": str(contract["selection_basis"]),
        "selection_guards": {
            "uses_target_statistics": bool(contract["uses_target_statistics"]),
            "uses_validation_statistics": bool(contract["uses_validation_statistics"]),
            "uses_test_statistics": bool(contract["uses_test_statistics"]),
        },
        "dataset": dataset_manifest(
            frame,
            snapshot_path,
            source_manifest_path=source_manifest_path,
        ),
        "candidate_features": candidates,
        "selected_features": selected,
        "feature_diagnostics": diagnostics,
        "top_absolute_correlations": _top_correlations(
            correlation_features,
            candidates,
        ),
        "correlation_scope": {
            "policy": "episode005_train_only",
            "rows": int(len(correlation_features)),
            "validation_start": prepared.split.validation_start.isoformat(),
            "test_used": False,
        },
        "correlation_note": (
            "Baseline v1 selection is predeclared. Redundancy correlations use "
            "the exact downstream Episode 005 train membership; whole-history "
            "feature availability diagnostics are descriptive only."
        ),
    }
