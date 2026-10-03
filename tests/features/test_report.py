from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np
import pandas as pd
import pytest

from trading_ai.data.bundle import load_market_data_bundle, write_market_data_bundle_manifest
from trading_ai.data.manifest import sha256_file
from trading_ai.data.pipeline import process_market_response
from trading_ai.data.request import MarketDataRequest
from trading_ai.data.timeframes import timeframe_duration
from trading_ai.experiments.baseline_dataset import prepare_episode005_split
from trading_ai.features.alignment import build_multitimeframe_point_in_time_features
from trading_ai.features.report import analyze_feature_contract


FEATURE_CONTRACT = "configs/features/ep004_baseline_features.toml"
BTC_FEATURE_CONTRACT = "configs/features/ep004_btc_mtf_features.toml"
BTC_EXPERIMENT_CONTRACT = "configs/experiments/ep005_btc_mtf_baselines.toml"
BTC_START = pd.Timestamp("2026-01-01T08:00:00Z")
BTC_END = pd.Timestamp("2026-01-12T08:00:00Z")


def _write_btc_bundle(root: Path) -> Path:
    starts = {
        "H1": BTC_START - pd.Timedelta(hours=48),
        "H4": BTC_START - pd.Timedelta(hours=4 * 30),
        "D1": pd.Timestamp("2025-12-01T01:00:00Z"),
    }
    for timeframe, start in starts.items():
        duration = timeframe_duration(timeframe)
        timestamps = pd.date_range(start, BTC_END, freq=duration, inclusive="left")
        timestamps = timestamps[timestamps + duration <= BTC_END]
        position = np.arange(len(timestamps), dtype=float)
        close = 100 + 0.03 * position + 0.8 * np.sin(position / 3)
        open_ = close - 0.05 * np.cos(position / 5)
        bars = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_,
            "high": np.maximum(open_, close) + 0.15,
            "low": np.minimum(open_, close) - 0.15,
            "close": close,
            "tick_volume": 100 + (np.arange(len(position)) % 17) * 4,
            "spread": 10,
            "real_volume": 0,
        })
        result = process_market_response(
            bars,
            request=MarketDataRequest(
                symbol="BTCUSD.test", timeframe=timeframe,
                start=start, end=BTC_END, cutoff=BTC_END,
            ),
            run_dir=root / timeframe,
            provenance={"source_type": "synthetic_fixture", "feature_max_lookback_bars": 24},
            code_version="test",
        )
        assert result["status"] == "accepted"
    write_market_data_bundle_manifest(
        root, bundle_id="btc-ep004-report-fixture",
        analysis_start=BTC_START, analysis_end=BTC_END,
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


def test_feature_report_correlations_use_exact_episode005_train_membership(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "synthetic_h1.csv"
    _write_snapshot(snapshot)

    report = analyze_feature_contract(snapshot, FEATURE_CONTRACT)
    prepared = prepare_episode005_split(snapshot)

    assert report["selected_features"] == prepared.dataset.selected_features
    assert report["correlation_scope"]["policy"] == "episode005_train_only"
    assert report["correlation_scope"]["rows"] == len(prepared.split.train)
    assert report["correlation_scope"]["validation_start"] == (
        prepared.split.validation_start.isoformat()
    )
    assert report["correlation_scope"]["test_used"] is False


def test_zero_target_near_boundary_does_not_change_correlation_scope_definition(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "synthetic_with_zero_h1.csv"
    _write_snapshot(snapshot)

    frame = pd.read_csv(snapshot)
    frame.loc[130, "close"] = frame.loc[129, "close"]
    frame.loc[130, "high"] = max(
        float(frame.loc[130, "open"]),
        float(frame.loc[130, "close"]),
    ) + 0.15
    frame.loc[130, "low"] = min(
        float(frame.loc[130, "open"]),
        float(frame.loc[130, "close"]),
    ) - 0.15
    frame.to_csv(snapshot, index=False)

    report = analyze_feature_contract(snapshot, FEATURE_CONTRACT)
    prepared = prepare_episode005_split(snapshot)

    assert prepared.dataset.metadata["zero_target_rows"] >= 1
    assert report["correlation_scope"]["rows"] == len(prepared.split.train)
    assert report["correlation_scope"]["test_used"] is False


def test_ep004_bundle_report_uses_exact_episode005_train_membership(tmp_path: Path) -> None:
    bundle_path = _write_btc_bundle(tmp_path / "bundle")
    report = analyze_feature_contract(
        None, BTC_FEATURE_CONTRACT, bundle_manifest_path=bundle_path,
        experiment_contract_path=BTC_EXPERIMENT_CONTRACT,
    )
    prepared = prepare_episode005_split(
        None, BTC_EXPERIMENT_CONTRACT, bundle_manifest_path=bundle_path,
    )
    bundle = load_market_data_bundle(bundle_path)
    with Path(BTC_FEATURE_CONTRACT).open("rb") as handle:
        contract = tomllib.load(handle)
    train_features = build_multitimeframe_point_in_time_features(
        bundle.frames, prepared.split.train["decision_timestamp"], contract,
    )
    candidates = report["candidate_features"]
    expected_corr = train_features[candidates].corr().abs()

    assert report["selected_features"] == prepared.dataset.selected_features
    assert report["dataset"]["bundle"]["bundle_id"] == "btc-ep004-report-fixture"
    assert report["feature_contract_sha256"] == sha256_file(BTC_FEATURE_CONTRACT)
    assert report["experiment_contract_sha256"] == sha256_file(BTC_EXPERIMENT_CONTRACT)
    assert report["asof_policy"] == "source_nominal_close_lte_decision"
    assert report["correlation_scope"]["policy"] == "episode005_train_only"
    assert report["correlation_scope"]["rows"] == len(prepared.split.train)
    assert report["correlation_scope"]["validation_start"] == (
        prepared.split.validation_start.isoformat()
    )
    assert report["correlation_scope"]["test_used"] is False
    assert report["feature_diagnostics"]["d1__return_1bar"]["source_timeframe"] == "D1"
    assert report["feature_diagnostics"]["d1__return_1bar"]["native_bar_lookback"] == 1
    assert report["feature_diagnostics"]["d1__return_1bar"]["elapsed_duration_hours"] == 24
    assert report["timeframe_availability"]["D1"]["source_nominal_close_timestamp"]
    assert report["observations"]["decision_rows"] == 264
    for pair in report["top_absolute_correlations"]:
        assert pair["abs_correlation"] == pytest.approx(
            expected_corr.loc[pair["left"], pair["right"]]
        )
    assert str(tmp_path) not in json.dumps(report)


def test_ep004_bundle_report_rejects_tampered_bundle(tmp_path: Path) -> None:
    bundle_path = _write_btc_bundle(tmp_path / "bundle")
    accepted = bundle_path.parent / "H4" / "accepted.csv"
    accepted.write_text(accepted.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        analyze_feature_contract(
            None, BTC_FEATURE_CONTRACT, bundle_manifest_path=bundle_path,
            experiment_contract_path=BTC_EXPERIMENT_CONTRACT,
        )


def test_ep004_bundle_report_requires_canonical_feature_contract_bytes(
    tmp_path: Path,
) -> None:
    bundle_path = _write_btc_bundle(tmp_path / "bundle")
    alternate = tmp_path / "same_settings_different_bytes.toml"
    alternate.write_bytes(Path(BTC_FEATURE_CONTRACT).read_bytes() + b"\n# distinct artifact\n")
    assert sha256_file(alternate) != sha256_file(BTC_FEATURE_CONTRACT)
    with alternate.open("rb") as handle:
        parsed_alternate = tomllib.load(handle)
    with Path(BTC_FEATURE_CONTRACT).open("rb") as handle:
        parsed_canonical = tomllib.load(handle)
    assert parsed_alternate == parsed_canonical

    with pytest.raises(ValueError, match="feature contract.*EP005"):
        analyze_feature_contract(
            None, alternate, bundle_manifest_path=bundle_path,
            experiment_contract_path=BTC_EXPERIMENT_CONTRACT,
        )


def test_ep004_cli_bundle_mode_selects_btc_contract_defaults(tmp_path: Path) -> None:
    bundle_path = _write_btc_bundle(tmp_path / "bundle")
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/ep004_feature_engineering/analyze_features.py",
            "--bundle-manifest", str(bundle_path),
            "--output", str(output),
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["contract_id"] == "ep004-btc-mtf-features-v1"
    assert report["correlation_scope"]["test_used"] is False
