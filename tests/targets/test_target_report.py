from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_ai.data.bundle import write_market_data_bundle_manifest
from trading_ai.data.pipeline import process_market_response
from trading_ai.data.request import MarketDataRequest
from trading_ai.data.timeframes import timeframe_duration
from trading_ai.targets.report import analyze_h1_direction_target


START = pd.Timestamp("2026-01-05T08:00:00Z")
END = START + pd.Timedelta(hours=12)


def _bundle(root: Path, *, higher_shift: float = 0.0) -> Path:
    starts = {
        "H1": START - pd.Timedelta(hours=48),
        "H4": START - pd.Timedelta(hours=4 * 30),
        "D1": pd.Timestamp("2025-12-01T01:00:00Z"),
    }
    for timeframe, first in starts.items():
        duration = timeframe_duration(timeframe)
        timestamps = pd.date_range(first, END, freq=duration, inclusive="left")
        timestamps = timestamps[timestamps + duration <= END]
        position = np.arange(len(timestamps), dtype=float)
        close = 100 + position * 0.03 + np.sin(position / 3)
        if timeframe != "H1":
            close += higher_shift
        bars = pd.DataFrame({
            "timestamp": timestamps,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.3,
            "close": close,
            "tick_volume": 100 + position,
            "spread": 10,
            "real_volume": 0,
        })
        result = process_market_response(
            bars,
            request=MarketDataRequest(
                symbol="BTCUSD.test", timeframe=timeframe,
                start=first, end=END, cutoff=END,
            ),
            run_dir=root / timeframe,
            provenance={"source_type": "synthetic_fixture", "feature_max_lookback_bars": 24},
            code_version="test",
        )
        assert result["status"] == "accepted"
    write_market_data_bundle_manifest(
        root, bundle_id="btc-report-fixture", analysis_start=START, analysis_end=END,
    )
    return root / "bundle_manifest.json"


def test_ep003_bundle_report_uses_h1_target_and_records_bundle_id(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    report = analyze_h1_direction_target(bundle_manifest_path=bundle)
    manifest = json.loads(bundle.read_text(encoding="utf-8"))
    h1 = pd.read_csv(bundle.parent / "H1" / "accepted.csv")
    h1["timestamp"] = pd.to_datetime(h1["timestamp"], utc=True)
    decision = START
    row = h1.loc[h1["timestamp"].eq(decision - pd.Timedelta(hours=1))].iloc[0]
    next_row = h1.loc[h1["timestamp"].eq(decision)].iloc[0]
    expected_decisions = h1["timestamp"] + pd.Timedelta(hours=1)
    in_interval = expected_decisions.ge(START) & expected_decisions.lt(END)
    expected_returns = (
        h1["close"].shift(-1).div(h1["close"]).sub(1).loc[in_interval]
    )

    assert report["contract_id"] == "ep003-h1-direction-v1"
    assert report["forecast_horizon"] == "1h"
    assert report["dataset"]["bundle"]["bundle_id"] == "btc-report-fixture"
    assert report["dataset"]["bundle"]["manifest_sha256"]
    assert report["dataset"]["bundle"]["source_manifest_sha256"]["H1"] == (
        manifest["streams"]["H1"]["source_manifest"]["sha256"]
    )
    assert report["dataset"]["bundle"]["analysis_start"] == START.isoformat()
    assert report["dataset"]["bundle"]["analysis_end"] == END.isoformat()
    assert expected_decisions.loc[in_interval].iloc[0] == START
    assert expected_decisions.loc[in_interval].iloc[-1] == END - pd.Timedelta(hours=1)
    assert report["observations"]["input_rows"] == len(expected_decisions.loc[in_interval])
    assert report["observations"]["binary_labeled_rows"] <= 12
    assert report["future_return_summary"]["count"] == len(expected_returns)
    assert report["future_return_summary"]["mean"] == pytest.approx(expected_returns.mean())
    assert report["future_return_summary"]["min"] == pytest.approx(expected_returns.min())
    assert float(next_row["close"]) / float(row["close"]) - 1 == pytest.approx(
        expected_returns.iloc[0]
    )
    assert str(tmp_path) not in json.dumps(report)


def test_ep003_bundle_target_is_invariant_to_h4_d1_data(tmp_path: Path) -> None:
    first = analyze_h1_direction_target(bundle_manifest_path=_bundle(tmp_path / "first"))
    changed = analyze_h1_direction_target(
        bundle_manifest_path=_bundle(tmp_path / "changed", higher_shift=25)
    )
    for key in ("observations", "class_counts", "class_fractions", "future_return_summary"):
        assert first[key] == changed[key]


def test_downstream_bundle_reports_fail_on_tampered_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    accepted = bundle.parent / "D1" / "accepted.csv"
    accepted.write_text(accepted.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        analyze_h1_direction_target(bundle_manifest_path=bundle)
