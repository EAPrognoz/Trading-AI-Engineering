"""Diagnostics for the Episode 004 feature contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import tomllib

import numpy as np
import pandas as pd

from trading_ai.data.snapshot import dataset_manifest, load_snapshot
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


def analyze_feature_contract(
    snapshot_path: str | Path,
    contract_path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build Episode 004 features and describe the frozen baseline feature set.

    Correlations are diagnostic only. The selected feature list is read from the
    predeclared contract and is not chosen from target, validation, or test
    performance.
    """
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
        "top_absolute_correlations": _top_correlations(features, candidates),
        "correlation_note": "Diagnostic only; baseline v1 selection is predeclared and does not use these values.",
    }
