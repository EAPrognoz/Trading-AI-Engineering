from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from trading_ai.data.snapshot import (
    dataset_manifest,
    sha256_file,
    validate_h1_snapshot,
)
from trading_ai.data.validation import audit_h1_records


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "open": [1.0, 1.1, 1.2, 1.15],
            "high": [1.2, 1.3, 1.25, 1.2],
            "low": [0.9, 1.0, 1.1, 1.1],
            "close": [1.1, 1.2, 1.15, 1.18],
            "tick_volume": [100, 120, 110, 130],
            "spread": [10, 10, 11, 9],
            "real_volume": [0, 0, 0, 0],
        }
    )


def test_valid_snapshot_passes() -> None:
    validate_h1_snapshot(_frame())


def test_duplicate_timestamp_is_rejected() -> None:
    frame = _frame()
    frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"]
    with pytest.raises(ValueError, match="duplicates"):
        validate_h1_snapshot(frame)


def test_contradictory_high_is_rejected() -> None:
    frame = _frame()
    frame.loc[1, "high"] = 1.0
    with pytest.raises(ValueError, match="high"):
        validate_h1_snapshot(frame)


def test_unresolved_gap_is_rejected() -> None:
    frame = _frame().drop(index=2).reset_index(drop=True)
    with pytest.raises(ValueError, match="gap"):
        validate_h1_snapshot(frame)


def test_half_hour_timestamp_is_rejected_as_not_hour_aligned() -> None:
    frame = _frame()
    frame.loc[1, "timestamp"] = pd.Timestamp("2026-01-01T01:30:00Z")

    with pytest.raises(ValueError, match="hour-aligned"):
        validate_h1_snapshot(frame)


def test_subhour_timestamp_sequence_is_rejected() -> None:
    frame = _frame()
    frame["timestamp"] = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:30:00Z",
            "2026-01-01T01:30:00Z",
            "2026-01-01T02:30:00Z",
        ]
    )

    with pytest.raises(ValueError, match="hour-aligned"):
        validate_h1_snapshot(frame)


@pytest.mark.parametrize(("timeframe", "frequency"), [("H4", "4h"), ("D1", "24h")])
def test_native_cadence_passes_without_midnight_assumption(
    timeframe: str, frequency: str, tmp_path: Path
) -> None:
    from trading_ai.data.snapshot import load_market_snapshot, validate_market_snapshot
    from trading_ai.data.validation import audit_market_records

    frame = _frame()
    frame["timestamp"] = pd.date_range(
        "2026-01-01T01:00:00Z", periods=len(frame), freq=frequency
    )
    assert audit_market_records(frame, timeframe) == []
    validate_market_snapshot(frame, timeframe)
    csv_path = tmp_path / "accepted.csv"
    frame.to_csv(csv_path, index=False)
    loaded = load_market_snapshot(csv_path, timeframe)
    pd.testing.assert_frame_equal(loaded, frame)


def test_d1_nonmidnight_open_is_valid_and_dst_length_delta_is_rejected() -> None:
    from trading_ai.data.snapshot import validate_market_snapshot
    from trading_ai.data.validation import audit_market_records

    frame = _frame().iloc[:2].copy()
    frame["timestamp"] = pd.to_datetime(
        ["2026-03-28T01:00:00Z", "2026-03-29T01:00:00Z"]
    )
    validate_market_snapshot(frame, "D1")

    frame.loc[1, "timestamp"] = pd.Timestamp("2026-03-29T02:00:00Z")
    with pytest.raises(ValueError, match="gap|cadence"):
        validate_market_snapshot(frame, "D1")
    assert "unclassified_gap" in {
        issue["code"] for issue in audit_market_records(frame, "D1")
    }


def test_h4_short_interval_is_rejected_as_wrong_cadence() -> None:
    from trading_ai.data.snapshot import validate_market_snapshot
    from trading_ai.data.validation import audit_market_records

    frame = _frame().iloc[:2].copy()
    frame["timestamp"] = pd.to_datetime(
        ["2026-01-01T01:00:00Z", "2026-01-01T04:00:00Z"]
    )
    with pytest.raises(ValueError, match="cadence"):
        validate_market_snapshot(frame, "H4")
    assert "timeframe_cadence_mismatch" in {
        issue["code"] for issue in audit_market_records(frame, "H4")
    }


def test_h1_generic_audit_and_validator_preserve_legacy_results() -> None:
    from trading_ai.data.snapshot import validate_market_snapshot
    from trading_ai.data.validation import audit_market_records

    frame = _frame()
    assert audit_market_records(frame, "H1") == audit_h1_records(frame)
    validate_market_snapshot(frame, "H1")


def _write_accepted_with_manifest(tmp_path: Path) -> tuple[pd.DataFrame, Path, Path, str]:
    run_dir = tmp_path / "ep002-run"
    run_dir.mkdir()
    frame = _frame()
    csv_path = run_dir / "accepted.csv"
    frame.to_csv(csv_path, index=False)
    csv_sha = sha256_file(csv_path)
    manifest = {
        "contract_id": "ep002-h1-market-data-v1",
        "status": "accepted",
        "files": {
            "accepted_dataset": {
                "path": "accepted.csv",
                "sha256": csv_sha,
            }
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return frame, csv_path, manifest_path, csv_sha


def test_dataset_manifest_verifies_matching_ep002_manifest(tmp_path: Path) -> None:
    frame, csv_path, manifest_path, csv_sha = _write_accepted_with_manifest(tmp_path)

    metadata = dataset_manifest(
        frame,
        csv_path,
        source_manifest_path=manifest_path,
    )

    assert metadata["path"] == "accepted.csv"
    assert metadata["sha256"] == csv_sha
    assert metadata["source_manifest"]["contract_id"] == "ep002-h1-market-data-v1"
    assert metadata["source_manifest"]["status"] == "accepted"
    assert metadata["source_manifest"]["accepted_dataset_sha256"] == csv_sha
    assert metadata["source_manifest"]["accepted_dataset_artifact"] == "accepted.csv"
    assert metadata["source_manifest"]["manifest_sha256"]


def test_tampered_snapshot_fails_manifest_verification(tmp_path: Path) -> None:
    frame, csv_path, manifest_path, _ = _write_accepted_with_manifest(tmp_path)
    tampered = frame.copy()
    tampered.loc[0, "close"] = 1.11
    tampered.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="hash"):
        dataset_manifest(
            tampered,
            csv_path,
            source_manifest_path=manifest_path,
        )


def test_rejected_manifest_cannot_verify_accepted_snapshot(tmp_path: Path) -> None:
    frame, csv_path, manifest_path, _ = _write_accepted_with_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "rejected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="accepted"):
        dataset_manifest(
            frame,
            csv_path,
            source_manifest_path=manifest_path,
        )


def test_manifest_without_accepted_dataset_artifact_is_rejected(tmp_path: Path) -> None:
    frame, csv_path, manifest_path, _ = _write_accepted_with_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="accepted_dataset"):
        dataset_manifest(
            frame,
            csv_path,
            source_manifest_path=manifest_path,
        )


def test_dataset_manifest_does_not_expose_absolute_local_path(tmp_path: Path) -> None:
    frame, csv_path, manifest_path, _ = _write_accepted_with_manifest(tmp_path)

    metadata = dataset_manifest(
        frame,
        csv_path,
        source_manifest_path=manifest_path,
    )

    assert str(tmp_path) not in json.dumps(metadata)


def test_manifest_verification_is_hash_based_not_path_based(tmp_path: Path) -> None:
    frame, csv_path, manifest_path, csv_sha = _write_accepted_with_manifest(tmp_path)
    copied_dir = tmp_path / "copied"
    copied_dir.mkdir()
    copied_path = copied_dir / "renamed.csv"
    shutil.copyfile(csv_path, copied_path)

    metadata = dataset_manifest(
        frame,
        copied_path,
        source_manifest_path=manifest_path,
    )

    assert metadata["path"] == "renamed.csv"
    assert metadata["sha256"] == csv_sha
    assert metadata["source_manifest"]["accepted_dataset_artifact"] == "accepted.csv"
