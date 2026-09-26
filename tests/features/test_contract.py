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
