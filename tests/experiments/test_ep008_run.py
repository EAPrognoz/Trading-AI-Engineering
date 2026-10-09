from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from trading_ai.experiments.ep008_data import ids_sha256
from trading_ai.experiments.ep008_run import (
    begin_score_once,
    file_sha256,
    validate_prediction_gate,
    write_bytes_exclusive,
)


def test_write_bytes_is_exclusive_and_hash_matches_contents(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    digest = write_bytes_exclusive(path, b"frozen prediction bytes")

    assert path.read_bytes() == b"frozen prediction bytes"
    assert digest == hashlib.sha256(b"frozen prediction bytes").hexdigest()
    with pytest.raises(FileExistsError):
        write_bytes_exclusive(path, b"replacement")


def _frozen_run(run_dir: Path, *, source_sha256: str = "a" * 64) -> tuple[str, str]:
    run_dir.mkdir(parents=True)
    ids = pd.DataFrame(
        {
            "sample_id": ["2026-01-02|2026-01-09", "2026-01-05|2026-01-12"],
            "decision_date": ["2026-01-02", "2026-01-05"],
            "endpoint_date": ["2026-01-09", "2026-01-12"],
        }
    )
    ids_bytes = ids.to_csv(index=False, lineterminator="\n").encode("utf-8")
    write_bytes_exclusive(run_dir / "cold_ids.csv", ids_bytes)
    ids_file_sha = file_sha256(run_dir / "cold_ids.csv")
    freeze = {
        "source_files_sha256": source_sha256,
        "source_archive_sha256": "b" * 64,
        "cold_ids_file_sha256": ids_file_sha,
        "cold_ids_sha256": ids_sha256(ids),
        "cold_row_count": len(ids),
    }
    freeze_bytes = (json.dumps(freeze, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_bytes_exclusive(run_dir / "freeze.json", freeze_bytes)
    freeze_sha = file_sha256(run_dir / "freeze.json")
    review = {"status": "approved", "freeze_sha256": freeze_sha, "source_files_sha256": source_sha256}
    (run_dir / "method_review.json").write_text(json.dumps(review), encoding="utf-8")
    return freeze_sha, ids_file_sha


def test_prediction_gate_requires_exact_freeze_review_and_ids_hash(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    freeze_sha, _ = _frozen_run(run_dir)
    freeze, ids = validate_prediction_gate(
        run_dir,
        source_files_sha256="a" * 64,
        source_archive_sha256="b" * 64,
    )

    assert file_sha256(run_dir / "freeze.json") == freeze_sha
    assert freeze["cold_row_count"] == 2
    assert ids["sample_id"].tolist() == ["2026-01-02|2026-01-09", "2026-01-05|2026-01-12"]


def test_prediction_gate_rejects_missing_or_mismatched_method_review(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _frozen_run(run_dir)
    (run_dir / "method_review.json").unlink()
    with pytest.raises(ValueError, match="method review"):
        validate_prediction_gate(
            run_dir,
            source_files_sha256="a" * 64,
            source_archive_sha256="b" * 64,
        )

    _frozen_run(tmp_path / "second")
    second = tmp_path / "second"
    review = json.loads((second / "method_review.json").read_text(encoding="utf-8"))
    review["freeze_sha256"] = "0" * 64
    (second / "method_review.json").write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="freeze"):
        validate_prediction_gate(
            second,
            source_files_sha256="a" * 64,
            source_archive_sha256="b" * 64,
        )


def test_score_once_checks_prediction_hash_and_blocks_a_second_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _frozen_run(run_dir)
    prediction_path = run_dir / "cold_predictions.csv"
    write_bytes_exclusive(prediction_path, b"sample_id,pred\na,1\n")
    pred_sha = file_sha256(prediction_path)
    freeze_sha = file_sha256(run_dir / "freeze.json")
    (run_dir / "prediction_manifest.json").write_text(
        json.dumps({"predictions_sha256": pred_sha, "freeze_sha256": freeze_sha}), encoding="utf-8"
    )

    marker = begin_score_once(run_dir)

    assert marker.is_file()
    with pytest.raises(FileExistsError, match="once"):
        begin_score_once(run_dir)


def test_score_once_rejects_modified_prediction_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _frozen_run(run_dir)
    prediction_path = run_dir / "cold_predictions.csv"
    write_bytes_exclusive(prediction_path, b"sample_id,pred\na,1\n")
    freeze_sha = file_sha256(run_dir / "freeze.json")
    (run_dir / "prediction_manifest.json").write_text(
        json.dumps({"predictions_sha256": "0" * 64, "freeze_sha256": freeze_sha}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="prediction hash"):
        begin_score_once(run_dir)
