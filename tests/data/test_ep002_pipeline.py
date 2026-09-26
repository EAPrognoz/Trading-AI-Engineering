from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from trading_ai.data.pipeline import process_h1_response
from trading_ai.data.request import MarketDataRequest
from trading_ai.data.time_policy import apply_h1_time_policy


def _request() -> MarketDataRequest:
    return MarketDataRequest(
        symbol="EURUSD",
        start="2026-01-05T08:00:00Z",
        end="2026-01-05T12:00:00Z",
        cutoff="2026-01-05T12:00:00Z",
    )


def _good() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-05T08:00:00Z",
                periods=4,
                freq="h",
            ),
            "open": [1.1000, 1.1005, 1.1015, 1.1010],
            "high": [1.1010, 1.1020, 1.1025, 1.1030],
            "low": [1.0990, 1.1000, 1.1005, 1.1008],
            "close": [1.1005, 1.1015, 1.1010, 1.1020],
            "tick_volume": [100, 120, 110, 130],
            "spread": [10, 10, 11, 9],
            "real_volume": [0, 0, 0, 0],
        }
    )


def test_request_requires_hour_aligned_range_boundaries() -> None:
    with pytest.raises(ValueError, match="aligned"):
        MarketDataRequest(
            symbol="EURUSD",
            start="2026-01-05T08:30:00Z",
            end="2026-01-05T12:00:00Z",
            cutoff="2026-01-05T12:00:00Z",
        )


def test_half_open_range_and_cutoff_are_explicit() -> None:
    raw = _good()
    extra = raw.iloc[[-1]].copy()
    extra["timestamp"] = pd.Timestamp("2026-01-05T12:00:00Z")
    raw = pd.concat([raw, extra], ignore_index=True)

    result = apply_h1_time_policy(raw, _request())

    assert len(result.accepted_range) == 4
    assert result.exclusions["at_or_after_end"] == 1
    assert result.coverage["coverage_ok"] is True


def test_incomplete_current_bar_is_excluded() -> None:
    request = MarketDataRequest(
        symbol="EURUSD",
        start="2026-01-05T08:00:00Z",
        end="2026-01-05T12:00:00Z",
        cutoff="2026-01-05T11:30:00Z",
    )
    result = apply_h1_time_policy(_good(), request)

    assert result.accepted_range["timestamp"].max() == pd.Timestamp(
        "2026-01-05T10:00:00Z"
    )
    assert result.exclusions["incomplete_by_cutoff"] == 1


def test_good_run_preserves_raw_and_writes_accepted_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "good"
    result = process_h1_response(
        _good(),
        request=_request(),
        run_dir=run_dir,
        provenance={"source_type": "synthetic_fixture"},
        code_version="test-commit",
    )

    assert result["status"] == "accepted"
    assert (run_dir / "raw_response.csv").exists()
    assert (run_dir / "accepted.csv").exists()
    assert (run_dir / "validation_report.json").exists()
    assert (run_dir / "manifest.json").exists()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["code_version"] == "test-commit"
    assert manifest["request"]["symbol"] == "EURUSD"
    assert "raw_response" in manifest["files"]
    assert "accepted_dataset" in manifest["files"]


def test_conflicting_duplicate_rejects_without_accepted_dataset(tmp_path: Path) -> None:
    raw = _good()
    conflict = raw.iloc[[1]].copy()
    conflict["close"] = 1.1111
    raw = pd.concat([raw.iloc[:2], conflict, raw.iloc[2:]], ignore_index=True)

    run_dir = tmp_path / "conflict"
    result = process_h1_response(
        raw,
        request=_request(),
        run_dir=run_dir,
        provenance={"source_type": "synthetic_fixture"},
    )

    assert result["status"] == "rejected"
    assert not (run_dir / "accepted.csv").exists()
    assert (run_dir / "rejection_report.json").exists()
    codes = {
        issue["code"]
        for issue in result["validation_report"]["issues"]
    }
    assert "duplicate_opening_time" in codes


def test_unclassified_gap_withholds_dataset(tmp_path: Path) -> None:
    raw = _good().drop(index=2).reset_index(drop=True)
    run_dir = tmp_path / "gap"

    result = process_h1_response(
        raw,
        request=_request(),
        run_dir=run_dir,
        provenance={"source_type": "synthetic_fixture"},
    )

    assert result["status"] == "rejected"
    assert not (run_dir / "accepted.csv").exists()
    codes = {
        issue["code"]
        for issue in result["validation_report"]["issues"]
    }
    assert "unclassified_gap" in codes


def test_existing_run_directory_is_not_overwritten(tmp_path: Path) -> None:
    run_dir = tmp_path / "existing"
    run_dir.mkdir()

    with pytest.raises(FileExistsError):
        process_h1_response(
            _good(),
            request=_request(),
            run_dir=run_dir,
            provenance={"source_type": "synthetic_fixture"},
        )
