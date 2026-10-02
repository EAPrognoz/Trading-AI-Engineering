from __future__ import annotations

from pathlib import Path
import tomllib


def test_baseline_feature_selection_is_predeclared() -> None:
    path = Path("configs/features/ep004_baseline_features.toml")
    with path.open("rb") as handle:
        contract = tomllib.load(handle)

    candidates = set(contract["candidate_features"])
    selected = set(contract["selected_features"])

    assert selected
    assert selected < candidates
    assert contract["uses_target_statistics"] is False
    assert contract["uses_validation_statistics"] is False
    assert contract["uses_test_statistics"] is False


def test_btc_contract_freezes_native_bar_candidates_and_selection() -> None:
    path = Path("configs/features/ep004_btc_mtf_features.toml")
    with path.open("rb") as handle:
        contract = tomllib.load(handle)

    assert contract["contract_id"] == "ep004-btc-mtf-features-v1"
    assert contract["forecast_horizon_hours"] == 1
    assert contract["uses_target_statistics"] is False
    assert contract["uses_validation_statistics"] is False
    assert contract["uses_test_statistics"] is False
    assert set(contract["timeframes"]) == {"H1", "H4", "D1"}
    for timeframe in ("H1", "H4", "D1"):
        source = contract["timeframes"][timeframe]
        assert source["max_lookback_bars"] == 24
        assert source["return_candidate_bars"] == [1, 3, 6, 12, 24]
        assert source["return_selected_bars"] == [1, 6, 24]
        assert source["rolling_vol_candidate_bars"] == [6, 12, 24]
        assert source["rolling_vol_selected_bars"] == [6, 24]
        assert source["relative_tick_volume_bars"] == 24
