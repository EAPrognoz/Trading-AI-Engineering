from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import trading_ai.experiments.baseline_dataset as baseline_dataset
from trading_ai.data.bundle import write_market_data_bundle_manifest
from trading_ai.data.manifest import sha256_file
from trading_ai.data.pipeline import process_market_response
from trading_ai.data.request import MarketDataRequest
from trading_ai.data.timeframes import timeframe_duration
from trading_ai.experiments.baseline_run import run_episode005_validation_baselines


BTC_CONTRACT = "configs/experiments/ep005_btc_mtf_baselines.toml"
ANALYSIS_START = pd.Timestamp("2026-01-01T08:00:00Z")
ANALYSIS_END = pd.Timestamp("2026-01-12T08:00:00Z")


def _write_synthetic_bundle(
    root: Path, *, higher_timeframe_shift: float = 0.0
) -> Path:
    starts = {
        "H1": ANALYSIS_START - pd.Timedelta(hours=48),
        "H4": ANALYSIS_START - pd.Timedelta(hours=4 * 30),
        "D1": pd.Timestamp("2025-12-01T01:00:00Z"),
    }
    for timeframe, start in starts.items():
        duration = timeframe_duration(timeframe)
        timestamps = pd.date_range(
            start, ANALYSIS_END, freq=duration, inclusive="left"
        )
        timestamps = timestamps[timestamps + duration <= ANALYSIS_END]
        x = np.arange(len(timestamps), dtype=float)
        shift = higher_timeframe_shift if timeframe != "H1" else 0.0
        close = 100.0 + 0.03 * x + 0.8 * np.sin(x / 3.0) + shift
        open_ = close - 0.05 * np.cos(x / 5.0)
        raw = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_,
            "high": np.maximum(open_, close) + 0.15,
            "low": np.minimum(open_, close) - 0.15,
            "close": close,
            "tick_volume": 100 + (np.arange(len(x)) % 17) * 4,
            "spread": 10 + (np.arange(len(x)) % 3),
            "real_volume": np.zeros(len(x), dtype=int),
        })
        result = process_market_response(
            raw,
            request=MarketDataRequest(
                symbol="BTCUSD.test", timeframe=timeframe,
                start=start, end=ANALYSIS_END, cutoff=ANALYSIS_END,
            ),
            run_dir=root / timeframe,
            provenance={
                "source_type": "synthetic_fixture",
                "feature_max_lookback_bars": 24,
            },
            code_version="test",
        )
        assert result["status"] == "accepted"
    write_market_data_bundle_manifest(
        root, bundle_id="btc-ep005-synthetic",
        analysis_start=ANALYSIS_START, analysis_end=ANALYSIS_END,
    )
    return root / "bundle_manifest.json"


def _write_snapshot(path: Path, periods: int = 240) -> None:
    t = np.arange(periods, dtype=float)
    timestamp = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    close = 100.0 + 0.01 * t + 0.8 * np.sin(t / 3.0) + 0.25 * np.sin(t / 11.0)
    open_ = close - 0.05 * np.cos(t / 5.0)
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_,
            "high": np.maximum(open_, close) + 0.15,
            "low": np.minimum(open_, close) - 0.15,
            "close": close,
            "tick_volume": 100 + (np.arange(periods) % 17) * 4,
            "spread": 10 + (np.arange(periods) % 3),
            "real_volume": np.zeros(periods, dtype=int),
        }
    )
    frame.to_csv(path, index=False)


def test_episode005_run_keeps_test_locked(tmp_path: Path) -> None:
    snapshot = tmp_path / "synthetic_h1.csv"
    _write_snapshot(snapshot)

    report = run_episode005_validation_baselines(snapshot)

    assert report["test"]["evaluated"] is False
    assert report["test"]["policy"] == "locked"
    assert report["split"]["purged_train_boundary_rows"] >= 1
    assert report["split"]["purged_validation_boundary_rows"] >= 1
    assert set(report["validation_metrics"]) == {
        "B0_majority_class",
        "B1_previous_hour_direction",
        "B2_logistic_regression",
    }
    json.dumps(report)


def test_uniform_timestamp_shift_does_not_change_baseline_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = tmp_path / "synthetic_h1.csv"
    _write_snapshot(snapshot)

    corrected = run_episode005_validation_baselines(snapshot)
    real_builder = baseline_dataset.build_h1_direction_target

    def legacy_timestamp_builder(frame: pd.DataFrame) -> pd.DataFrame:
        result = real_builder(frame).copy()
        result["decision_timestamp"] = (
            result["decision_timestamp"] - pd.Timedelta(hours=1)
        )
        result["target_timestamp"] = (
            result["target_timestamp"] - pd.Timedelta(hours=1)
        )
        return result

    monkeypatch.setattr(
        baseline_dataset,
        "build_h1_direction_target",
        legacy_timestamp_builder,
    )
    legacy = run_episode005_validation_baselines(snapshot)

    assert corrected["validation_metrics"] == legacy["validation_metrics"]
    assert corrected["split"]["train_rows"] == legacy["split"]["train_rows"]
    assert corrected["split"]["validation_rows"] == legacy["split"]["validation_rows"]
    assert corrected["split"]["test_rows"] == legacy["split"]["test_rows"]
    assert (
        corrected["split"]["purged_train_boundary_rows"]
        == legacy["split"]["purged_train_boundary_rows"]
    )
    assert (
        corrected["split"]["purged_validation_boundary_rows"]
        == legacy["split"]["purged_validation_boundary_rows"]
    )
    assert pd.Timestamp(corrected["split"]["validation_start"]) == (
        pd.Timestamp(legacy["split"]["validation_start"])
        + pd.Timedelta(hours=1)
    )
    assert pd.Timestamp(corrected["split"]["test_start"]) == (
        pd.Timestamp(legacy["split"]["test_start"])
        + pd.Timedelta(hours=1)
    )


def test_shared_episode005_preparation_matches_full_report(tmp_path: Path) -> None:
    snapshot = tmp_path / "synthetic_h1.csv"
    _write_snapshot(snapshot)

    prepared = baseline_dataset.prepare_episode005_split(snapshot)
    report = run_episode005_validation_baselines(snapshot)

    assert len(prepared.split.train) == report["split"]["train_rows"]
    assert len(prepared.split.validation) == report["split"]["validation_rows"]
    assert len(prepared.split.test) == report["split"]["test_rows"]
    assert (
        prepared.split.purged_train_rows
        == report["split"]["purged_train_boundary_rows"]
    )
    assert (
        prepared.split.purged_validation_rows
        == report["split"]["purged_validation_boundary_rows"]
    )


def test_minimal_b0_matches_full_validation_membership_and_accuracy(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "synthetic_h1.csv"
    _write_snapshot(snapshot)
    report = run_episode005_validation_baselines(snapshot)

    completed = subprocess.run(
        [
            sys.executable,
            "examples/ep005_baseline_minimal.py",
            str(snapshot),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    expected_rows = report["split"]["validation_rows"]
    expected_accuracy = report["validation_metrics"]["B0_majority_class"]["accuracy"]
    assert f"Validation rows: {expected_rows}" in completed.stdout
    assert (
        f"B0 majority-class accuracy: {expected_accuracy:.4f}"
        in completed.stdout
    )


def test_higher_timeframe_inputs_do_not_change_h1_target(tmp_path: Path) -> None:
    first_bundle = _write_synthetic_bundle(tmp_path / "first")
    changed_bundle = _write_synthetic_bundle(
        tmp_path / "changed", higher_timeframe_shift=25.0
    )
    first = baseline_dataset.prepare_episode005_split(
        None, BTC_CONTRACT, bundle_manifest_path=first_bundle
    ).dataset.frame
    changed = baseline_dataset.prepare_episode005_split(
        None, BTC_CONTRACT, bundle_manifest_path=changed_bundle
    ).dataset.frame
    target_columns = [
        "decision_timestamp", "target_timestamp", "future_return_1h",
        "target_h1_direction",
    ]
    pd.testing.assert_frame_equal(first[target_columns], changed[target_columns])
    assert (first["target_timestamp"] - first["decision_timestamp"]).eq(
        pd.Timedelta(hours=1)
    ).all()


def test_bundle_minimal_and_full_b0_share_samples_and_split(tmp_path: Path) -> None:
    bundle = _write_synthetic_bundle(tmp_path / "bundle")
    prepared = baseline_dataset.prepare_episode005_split(
        None, BTC_CONTRACT, bundle_manifest_path=bundle
    )
    report = run_episode005_validation_baselines(
        None, BTC_CONTRACT, bundle_manifest_path=bundle
    )
    completed = subprocess.run(
        [
            sys.executable, "examples/ep005_baseline_minimal.py",
            "--bundle-manifest", str(bundle), "--contract", BTC_CONTRACT,
        ],
        check=True, capture_output=True, text=True,
    )
    validation = prepared.split.validation
    assert validation["decision_timestamp"].tolist() == sorted(
        validation["decision_timestamp"].tolist()
    )
    assert len(validation) == report["split"]["validation_rows"]
    assert prepared.split.validation_start.isoformat() == report["split"]["validation_start"]
    assert prepared.split.test_start.isoformat() == report["split"]["test_start"]
    accuracy = report["validation_metrics"]["B0_majority_class"]["accuracy"]
    assert f"Validation rows: {len(validation)}" in completed.stdout
    assert f"B0 majority-class accuracy: {accuracy:.4f}" in completed.stdout


def test_bundle_split_purges_target_boundary_crossings(tmp_path: Path) -> None:
    bundle = _write_synthetic_bundle(tmp_path / "bundle")
    prepared = baseline_dataset.prepare_episode005_split(
        None, BTC_CONTRACT, bundle_manifest_path=bundle
    )
    dataset = prepared.dataset.frame
    split = prepared.split
    assert dataset["decision_timestamp"].min() >= ANALYSIS_START
    assert dataset["decision_timestamp"].max() < ANALYSIS_END
    assert split.purged_train_rows >= 1
    assert split.purged_validation_rows >= 1
    assert (split.train["target_timestamp"] < split.validation_start).all()
    assert (split.validation["target_timestamp"] < split.test_start).all()


def test_bundle_test_partition_remains_locked(tmp_path: Path) -> None:
    bundle = _write_synthetic_bundle(tmp_path / "bundle")
    report = run_episode005_validation_baselines(
        None, BTC_CONTRACT, bundle_manifest_path=bundle
    )
    assert report["test"]["status"] == "locked"
    assert report["test"]["evaluated"] is False
    assert not any("metric" in key for key in report["test"])
    assert "train_metrics" not in report
    assert set(report["validation_metrics"]) == {
        "B0_majority_class", "B1_previous_hour_direction",
        "B2_logistic_regression",
    }


def test_bundle_baseline_report_omits_local_paths(tmp_path: Path) -> None:
    bundle = _write_synthetic_bundle(tmp_path / "bundle")
    report = run_episode005_validation_baselines(
        None, BTC_CONTRACT, bundle_manifest_path=bundle
    )
    source_hashes = {
        timeframe: sha256_file(bundle.parent / timeframe / "manifest.json")
        for timeframe in ("H1", "H4", "D1")
    }
    assert report["contract_id"] == "ep005-btc-mtf-baseline-v1"
    assert report["target_contract_id"] == "ep003-h1-direction-v1"
    assert report["feature_contract_id"] == "ep004-btc-mtf-features-v1"
    bundle_report = report["dataset"]["bundle"]
    assert bundle_report["contract_id"] == "ep002-btc-multitimeframe-bundle-v1"
    assert bundle_report["bundle_id"] == "btc-ep005-synthetic"
    assert bundle_report["manifest_sha256"] == sha256_file(bundle)
    assert bundle_report["symbol"] == "BTCUSD.test"
    assert bundle_report["source_manifest_sha256"] == source_hashes
    assert bundle_report["accepted_dataset_sha256"] == {
        timeframe: sha256_file(bundle.parent / timeframe / "accepted.csv")
        for timeframe in ("H1", "H4", "D1")
    }
    assert bundle_report["analysis_start"] == ANALYSIS_START.isoformat()
    assert bundle_report["analysis_end"] == ANALYSIS_END.isoformat()
    serialized = json.dumps(report)
    assert str(tmp_path) not in serialized
    assert "C:\\" not in serialized


@pytest.mark.parametrize(
    "snapshot,bundle,source_manifest",
    [
        (False, False, False),
        (True, True, False),
        (False, True, True),
    ],
)
def test_episode005_rejects_ambiguous_or_missing_source_mode(
    tmp_path: Path, snapshot: bool, bundle: bool, source_manifest: bool
) -> None:
    bundle_path = _write_synthetic_bundle(tmp_path / "bundle")
    snapshot_path = bundle_path.parent / "H1" / "accepted.csv"
    manifest_path = bundle_path.parent / "H1" / "manifest.json"
    kwargs = {
        "source_manifest_path": manifest_path if source_manifest else None,
        "bundle_manifest_path": bundle_path if bundle else None,
    }
    with pytest.raises(ValueError, match="source mode|bundle|snapshot"):
        baseline_dataset.prepare_episode005_split(
            snapshot_path if snapshot else None, BTC_CONTRACT, **kwargs
        )


def test_bundle_hash_tampering_stops_episode005(tmp_path: Path) -> None:
    bundle = _write_synthetic_bundle(tmp_path / "bundle")
    accepted = bundle.parent / "H4" / "accepted.csv"
    with accepted.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="hash|identity"):
        baseline_dataset.prepare_episode005_split(
            None, BTC_CONTRACT, bundle_manifest_path=bundle
        )


def test_full_bundle_cli_writes_validation_report(tmp_path: Path) -> None:
    bundle = _write_synthetic_bundle(tmp_path / "bundle")
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable, "experiments/ep005_baselines/run_baselines.py",
            "--bundle-manifest", str(bundle), "--contract", BTC_CONTRACT,
            "--output", str(output),
        ],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dataset"]["bundle"]["manifest_sha256"] == sha256_file(bundle)
    assert report["test"]["evaluated"] is False
    assert "train_metrics" not in report


@pytest.mark.parametrize(
    "script,extra",
    [
        ("experiments/ep005_baselines/run_baselines.py", ["--output", "report.json"]),
        ("examples/ep005_baseline_minimal.py", []),
    ],
)
def test_bundle_cli_rejects_source_manifest(
    tmp_path: Path, script: str, extra: list[str]
) -> None:
    bundle = _write_synthetic_bundle(tmp_path / "bundle")
    source_manifest = bundle.parent / "H1" / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable, script, "--bundle-manifest", str(bundle),
            "--manifest", str(source_manifest), "--contract", BTC_CONTRACT,
            *extra,
        ],
        capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "cannot be combined" in completed.stderr


@pytest.mark.parametrize(
    "original,replacement",
    [
        ('contract_id = "ep005-btc-mtf-baseline-v1"', 'contract_id = "other"'),
        ('test_policy = "locked"', 'test_policy = "open"'),
        ('train_fraction = 0.60', 'train_fraction = 0.50'),
    ],
)
def test_bundle_requires_frozen_ep005_contract(
    tmp_path: Path, original: str, replacement: str
) -> None:
    bundle = _write_synthetic_bundle(tmp_path / "bundle")
    contract = Path(BTC_CONTRACT).read_text(encoding="utf-8")
    feature_path = Path("configs/features/ep004_btc_mtf_features.toml").resolve()
    contract = contract.replace(
        'feature_contract = "configs/features/ep004_btc_mtf_features.toml"',
        f'feature_contract = "{feature_path.as_posix()}"',
    )
    contract = contract.replace(original, replacement)
    if original == 'train_fraction = 0.60':
        contract = contract.replace('test_fraction = 0.20', 'test_fraction = 0.30')
    altered = tmp_path / "altered.toml"
    altered.write_text(contract, encoding="utf-8")
    with pytest.raises(ValueError, match="BTC EP005 contract"):
        baseline_dataset.prepare_episode005_split(
            None, altered, bundle_manifest_path=bundle
        )
