"""BTC feature alignment across native H1, H4, and D1 source bars."""

from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest


def _alignment():
    assert importlib.util.find_spec("trading_ai.features.alignment") is not None
    return importlib.import_module("trading_ai.features.alignment")


def _contract() -> dict[str, object]:
    source = {
        "max_lookback_bars": 24,
        "return_candidate_bars": [1, 3, 6, 12, 24],
        "return_selected_bars": [1, 6, 24],
        "rolling_vol_candidate_bars": [6, 12, 24],
        "rolling_vol_selected_bars": [6, 24],
        "relative_tick_volume_bars": 24,
    }
    return {
        "contract_id": "ep004-btc-mtf-features-v1",
        "forecast_horizon_hours": 1,
        "decision_time": "h1_nominal_bar_close_utc",
        "asof_policy": "source_nominal_close_lte_decision",
        "gap_policy": "reset_on_non_native_timestamp_delta",
        "selection_basis": "pre_model_structural",
        "uses_target_statistics": False,
        "uses_validation_statistics": False,
        "uses_test_statistics": False,
        "timeframes": {tf: dict(source) for tf in ("H1", "H4", "D1")},
    }


def _bars(start: str, periods: int, freq: str, base: float) -> pd.DataFrame:
    timestamp = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = base + np.arange(periods, dtype=float) * 0.1
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": close - 0.05,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "tick_volume": np.full(periods, 100.0),
        }
    )


def _frames() -> dict[str, pd.DataFrame]:
    return {
        "H1": _bars("2026-01-01 00:00", 840, "h", 100.0),
        "H4": _bars("2026-01-01 01:00", 210, "4h", 200.0),
        "D1": _bars("2026-01-01 01:00", 35, "24h", 300.0),
    }


def _build(
    frames: dict[str, pd.DataFrame], decisions: pd.Series
) -> pd.DataFrame:
    return _alignment().build_multitimeframe_point_in_time_features(
        frames, decisions, _contract()
    )


def test_asof_includes_equal_close_and_excludes_later_close() -> None:
    frames = _frames()
    decision = pd.Timestamp("2026-01-31 01:00:00", tz="UTC")
    exact = _build(frames, pd.Series([decision])).iloc[0]
    assert exact["d1__source_nominal_close_timestamp"] == decision
    assert exact["h4__source_nominal_close_timestamp"] == decision
    assert exact["d1__range_pct"] == (
        frames["D1"].loc[29, "high"] - frames["D1"].loc[29, "low"]
    ) / frames["D1"].loc[29, "close"]

    later = {key: value.copy() for key, value in frames.items()}
    later["D1"]["timestamp"] += pd.Timedelta(seconds=1)
    shifted = _build(later, pd.Series([decision])).iloc[0]
    assert shifted["d1__source_nominal_close_timestamp"] == decision - pd.Timedelta(days=1) + pd.Timedelta(seconds=1)
    assert shifted["d1__range_pct"] == (
        frames["D1"].loc[28, "high"] - frames["D1"].loc[28, "low"]
    ) / frames["D1"].loc[28, "close"]
    assert shifted["h4__source_nominal_close_timestamp"] == decision


def test_future_source_mutation_does_not_change_prior_features() -> None:
    frames = _frames()
    decision = pd.Timestamp("2026-01-30 01:00", tz="UTC")
    before = _build(frames, pd.Series([decision]))
    changed = {key: value.copy() for key, value in frames.items()}
    for tf, first_future in (("H4", 175), ("D1", 29)):
        changed[tf].loc[first_future:, ["open", "high", "low", "close", "tick_volume"]] *= 5
    after = _build(changed, pd.Series([decision]))
    pdt.assert_frame_equal(before, after)


def test_feature_registry_records_native_bars_and_elapsed_duration() -> None:
    registry = _alignment().build_multitimeframe_feature_registry(_contract())
    by_name = {row["output_name"]: row for row in registry}
    assert len(by_name) == 45
    assert by_name["h1__return_24bar"]["native_bar_lookback"] == 24
    assert by_name["h1__return_24bar"]["elapsed_duration_hours"] == 24
    assert by_name["h4__return_24bar"]["elapsed_duration_hours"] == 96
    assert by_name["d1__return_24bar"]["elapsed_duration_hours"] == 576
    assert by_name["d1__return_24bar"]["source_timeframe"] == "D1"
    assert by_name["d1__return_24bar"]["selected"] is True
    assert by_name["d1__return_3bar"]["selected"] is False
    assert "d1__return_24h" not in by_name
    assert not any("source_nominal_close_timestamp" in name for name in by_name)


def test_gap_and_warmup_are_isolated_per_timeframe() -> None:
    frames = _frames()
    frames["H4"].loc[120:, "timestamp"] += pd.Timedelta(hours=4)
    decision = pd.Timestamp(frames["H4"].loc[120, "timestamp"]) + pd.Timedelta(hours=4)
    result = _build(frames, pd.Series([decision])).iloc[0]
    assert result["h4__source_nominal_close_timestamp"] == decision
    assert pd.isna(result["h4__return_1bar"])
    assert pd.isna(result["h4__rolling_vol_6bar"])
    assert pd.notna(result["h1__return_1bar"])
    assert pd.notna(result["d1__return_1bar"])

    early = _build(_frames(), pd.Series([pd.Timestamp("2026-01-03 01:00", tz="UTC")])).iloc[0]
    assert pd.isna(early["d1__return_24bar"])
    assert pd.notna(early["h1__return_24bar"])
    assert pd.notna(early["h4__return_1bar"])


def test_native_h1_formulas_match_legacy_h1_features() -> None:
    from trading_ai.features.engineering import (
        build_native_timeframe_features,
        build_point_in_time_features,
    )

    frame = _frames()["H1"]
    legacy = build_point_in_time_features(frame)
    renamed = legacy.rename(
        columns={
            **{f"return_{bars}h": f"return_{bars}bar" for bars in (1, 3, 6, 12, 24)},
            **{
                f"rolling_vol_{bars}h": f"rolling_vol_{bars}bar"
                for bars in (6, 12, 24)
            },
            "relative_tick_volume_24h": "relative_tick_volume_24bar",
        }
    )
    native = build_native_timeframe_features(
        frame,
        "H1",
        return_bars=[1, 3, 6, 12, 24],
        rolling_vol_bars=[6, 12, 24],
        relative_tick_volume_bars=24,
    )
    pdt.assert_frame_equal(native, renamed)


_METADATA_DRIFT = [
    ("forecast_horizon_hours", 2),
    ("decision_time", "bar_open"),
    ("asof_policy", "source_open_lte_decision"),
    ("gap_policy", "carry_across_gap"),
    ("selection_basis", "validation_search"),
    ("uses_target_statistics", True),
    ("uses_validation_statistics", True),
    ("uses_test_statistics", True),
]


def _entry_point(entry_point: str, contract: dict[str, object]) -> None:
    module = _alignment()
    if entry_point == "registry":
        module.build_multitimeframe_feature_registry(contract)
    else:
        tiny_frames = {
            "H1": _bars("2026-01-01 00:00", 4, "h", 100.0),
            "H4": _bars("2026-01-01 01:00", 4, "4h", 200.0),
            "D1": _bars("2026-01-01 01:00", 4, "24h", 300.0),
        }
        module.build_multitimeframe_point_in_time_features(
            tiny_frames,
            pd.Series([pd.Timestamp("2026-01-02 01:00", tz="UTC")]),
            contract,
        )


@pytest.mark.parametrize("entry_point", ["registry", "builder"])
@pytest.mark.parametrize("field,changed", _METADATA_DRIFT)
def test_same_id_rejects_frozen_metadata_drift(
    entry_point: str, field: str, changed: object
) -> None:
    contract = _contract()
    contract[field] = changed
    with pytest.raises(ValueError, match="frozen BTC feature contract"):
        _entry_point(entry_point, contract)


_WINDOW_DRIFT = [
    ("max_lookback_bars", 25),
    ("return_candidate_bars", [1, 2, 3, 6, 12, 24]),
    ("return_selected_bars", [1, 3, 6, 24]),
    ("rolling_vol_candidate_bars", [6, 8, 12, 24]),
    ("rolling_vol_selected_bars", [6, 12, 24]),
    ("relative_tick_volume_bars", 23),
]


@pytest.mark.parametrize("entry_point", ["registry", "builder"])
@pytest.mark.parametrize("timeframe", ["H1", "H4", "D1"])
@pytest.mark.parametrize("field,changed", _WINDOW_DRIFT)
def test_same_id_rejects_each_timeframe_window_drift(
    entry_point: str, timeframe: str, field: str, changed: object
) -> None:
    contract = deepcopy(_contract())
    contract["timeframes"][timeframe][field] = changed
    with pytest.raises(ValueError, match="frozen BTC feature contract"):
        _entry_point(entry_point, contract)
