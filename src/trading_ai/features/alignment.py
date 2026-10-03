"""Native-bar EP004 features aligned to completed H1 decisions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from trading_ai.data.timeframes import timeframe_duration
from trading_ai.features.engineering import build_native_timeframe_features


_TIMEFRAMES = ("H1", "H4", "D1")
_FROZEN_METADATA: dict[str, Any] = {
    "contract_id": "ep004-btc-mtf-features-v1",
    "forecast_horizon_hours": 1,
    "decision_time": "h1_nominal_bar_close_utc",
    "asof_policy": "source_nominal_close_lte_decision",
    "gap_policy": "reset_on_non_native_timestamp_delta",
    "selection_basis": "pre_model_structural",
    "uses_target_statistics": False,
    "uses_validation_statistics": False,
    "uses_test_statistics": False,
}
_FROZEN_SOURCE: dict[str, Any] = {
    "max_lookback_bars": 24,
    "return_candidate_bars": [1, 3, 6, 12, 24],
    "return_selected_bars": [1, 6, 24],
    "rolling_vol_candidate_bars": [6, 12, 24],
    "rolling_vol_selected_bars": [6, 24],
    "relative_tick_volume_bars": 24,
}


def _matches_frozen_fields(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if set(actual) != set(expected):
        return False
    for key, frozen in expected.items():
        value = actual[key]
        if type(value) is not type(frozen) or value != frozen:
            return False
        if isinstance(frozen, list) and any(type(item) is not int for item in value):
            return False
    return True


def _validate_frozen_contract(contract: Mapping[str, Any]) -> None:
    if not isinstance(contract, Mapping):
        raise ValueError("frozen BTC feature contract must be a mapping")
    metadata = {key: value for key, value in contract.items() if key != "timeframes"}
    if not _matches_frozen_fields(metadata, _FROZEN_METADATA):
        raise ValueError("frozen BTC feature contract metadata mismatch")
    timeframes = contract.get("timeframes")
    if not isinstance(timeframes, Mapping) or set(timeframes) != set(_TIMEFRAMES):
        raise ValueError("frozen BTC feature contract requires H1, H4, and D1")
    for timeframe in _TIMEFRAMES:
        source = timeframes[timeframe]
        if not isinstance(source, Mapping) or not _matches_frozen_fields(
            source, _FROZEN_SOURCE
        ):
            raise ValueError(f"frozen BTC feature contract {timeframe} windows mismatch")


def _positive_bars(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive native-bar count")
    return value


def _bar_list(source: Mapping[str, Any], field: str, timeframe: str) -> list[int]:
    raw = source.get(field)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{timeframe}.{field} must be a nonempty list")
    bars = [_positive_bars(value, f"{timeframe}.{field}") for value in raw]
    if len(bars) != len(set(bars)) or bars != sorted(bars):
        raise ValueError(f"{timeframe}.{field} must be sorted and unique")
    return bars


def _source_settings(contract: Mapping[str, Any], timeframe: str) -> dict[str, Any]:
    if contract.get("contract_id") != "ep004-btc-mtf-features-v1":
        raise ValueError("unsupported BTC feature contract_id")
    timeframes = contract.get("timeframes")
    if not isinstance(timeframes, Mapping) or set(timeframes) != set(_TIMEFRAMES):
        raise ValueError("feature contract must declare exactly H1, H4, and D1")
    source = timeframes[timeframe]
    if not isinstance(source, Mapping):
        raise ValueError(f"{timeframe} feature settings must be a mapping")
    returns = _bar_list(source, "return_candidate_bars", timeframe)
    return_selected = _bar_list(source, "return_selected_bars", timeframe)
    volatility = _bar_list(source, "rolling_vol_candidate_bars", timeframe)
    volatility_selected = _bar_list(source, "rolling_vol_selected_bars", timeframe)
    volume = _positive_bars(
        source.get("relative_tick_volume_bars"),
        f"{timeframe}.relative_tick_volume_bars",
    )
    maximum = _positive_bars(
        source.get("max_lookback_bars"), f"{timeframe}.max_lookback_bars"
    )
    if not set(return_selected).issubset(returns):
        raise ValueError(f"{timeframe} selected returns must be candidates")
    if not set(volatility_selected).issubset(volatility):
        raise ValueError(f"{timeframe} selected volatility must be candidates")
    if max([*returns, *volatility, volume]) > maximum:
        raise ValueError(f"{timeframe} max_lookback_bars understates feature windows")
    return {
        "returns": returns,
        "return_selected": return_selected,
        "volatility": volatility,
        "volatility_selected": volatility_selected,
        "volume": volume,
    }


def build_multitimeframe_feature_registry(
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Describe candidate outputs and their source-native lookbacks."""
    _validate_frozen_contract(contract)
    registry: list[dict[str, Any]] = []
    for timeframe in _TIMEFRAMES:
        settings = _source_settings(contract, timeframe)
        prefix = timeframe.lower()
        hours = int(timeframe_duration(timeframe) / pd.Timedelta(hours=1))

        def add(name: str, bars: int, selected: bool) -> None:
            registry.append(
                {
                    "output_name": f"{prefix}__{name}",
                    "source_timeframe": timeframe,
                    "native_bar_lookback": bars,
                    "elapsed_duration_hours": bars * hours,
                    "selected": selected,
                }
            )

        for bars in settings["returns"]:
            add(f"return_{bars}bar", bars, bars in settings["return_selected"])
        for bars in settings["volatility"]:
            add(
                f"rolling_vol_{bars}bar",
                bars,
                bars in settings["volatility_selected"],
            )
        add("range_pct", 1, True)
        add("body_return", 1, True)
        add(f"relative_tick_volume_{settings['volume']}bar", settings["volume"], True)
        for name in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
            add(name, 1, True)
    return registry


def build_multitimeframe_point_in_time_features(
    frames: Mapping[str, pd.DataFrame],
    decision_timestamps: pd.Series,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Join each source's latest completed, still-current bar to H1 decisions.

    The source close is metadata, not a model feature. A source row expires at
    its next expected native close; an absent bar therefore cannot be carried
    across an unresolved gap.
    """
    if set(frames) != set(_TIMEFRAMES):
        raise ValueError("feature frames must contain exactly H1, H4, and D1")
    registry = build_multitimeframe_feature_registry(contract)
    decision = pd.to_datetime(decision_timestamps, utc=True, errors="raise")
    if decision.isna().any():
        raise ValueError("decision timestamps must not be missing")
    result = pd.DataFrame(
        {
            "_row": range(len(decision)),
            "decision_timestamp": decision.reset_index(drop=True),
        }
    )
    for timeframe in _TIMEFRAMES:
        settings = _source_settings(contract, timeframe)
        source = build_native_timeframe_features(
            frames[timeframe],
            timeframe,
            return_bars=settings["returns"],
            rolling_vol_bars=settings["volatility"],
            relative_tick_volume_bars=settings["volume"],
        )
        duration = timeframe_duration(timeframe)
        prefix = timeframe.lower()
        close_name = f"{prefix}__source_nominal_close_timestamp"
        source[close_name] = source.pop("timestamp") + duration
        feature_names = [
            row["output_name"]
            for row in registry
            if row["source_timeframe"] == timeframe
        ]
        source.columns = [
            name if name == close_name else f"{prefix}__{name}"
            for name in source.columns
        ]
        available = source[[close_name, *feature_names]]
        result = pd.merge_asof(
            result.sort_values("decision_timestamp"),
            available,
            left_on="decision_timestamp",
            right_on=close_name,
            direction="backward",
            allow_exact_matches=True,
            tolerance=duration - pd.Timedelta(nanoseconds=1),
        ).sort_values("_row")
    return result.drop(columns="_row").reset_index(drop=True)
