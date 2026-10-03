from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import tomllib

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_CONFIG_PATH = Path(
    os.environ.get("EP006_CONFIG_PATH", ".local/ep006_btc_stability.private.toml")
)
if not PRIVATE_CONFIG_PATH.is_absolute():
    PRIVATE_CONFIG_PATH = PROJECT_ROOT / PRIVATE_CONFIG_PATH
_PRIVATE_CONFIG = (
    tomllib.loads(PRIVATE_CONFIG_PATH.read_text(encoding="utf-8"))
    if PRIVATE_CONFIG_PATH.is_file()
    else {}
)
RAW_DIR = Path(
    os.environ.get(
        "EP006_RAW_DIR",
        _PRIVATE_CONFIG.get("source", ".local/ep006-user-data"),
    )
)
if not RAW_DIR.is_absolute():
    RAW_DIR = PROJECT_ROOT / RAW_DIR
FEATURE_CONTRACT = PROJECT_ROOT / "configs/features/ep004_btc_mtf_features.toml"
_HAS_PRIVATE_CONFIG = PRIVATE_CONFIG_PATH.is_file()
_HAS_PRIVATE_INPUTS = _HAS_PRIVATE_CONFIG and RAW_DIR.is_dir()
requires_private_config = pytest.mark.skipif(
    not _HAS_PRIVATE_CONFIG,
    reason="requires owner-authorized local EP006 pins",
)
requires_private_inputs = pytest.mark.skipif(
    not _HAS_PRIVATE_INPUTS,
    reason="requires owner-authorized local EP006 source files",
)


def _api(name: str):
    module = importlib.import_module("trading_ai.experiments.ep006_stability")
    assert hasattr(module, name), f"EP006 research API is missing {name}"
    return getattr(module, name)


def _source():
    return _api("read_ep006_raw_source")(RAW_DIR, config_path=PRIVATE_CONFIG_PATH)


def _dataset(source):
    return _api("build_ep006_dataset")(source, FEATURE_CONTRACT)


def test_ep006_research_module_is_available_separately_from_production() -> None:
    spec = importlib.util.find_spec("trading_ai.experiments.ep006_stability")

    assert spec is not None


@requires_private_inputs
def test_ep006_raw_manifest_and_hashes_are_verified() -> None:
    source = _source()

    assert source.manifest["symbol"] == _PRIVATE_CONFIG["symbol"]
    assert source.metadata["raw_manifest_sha256"] == hashlib.sha256(
        (RAW_DIR / "raw_manifest.json").read_bytes()
    ).hexdigest()
    for timeframe in ("H1", "H4", "D1"):
        assert source.metadata["raw_sha256"][timeframe] == source.manifest["streams"][timeframe]["raw_sha256"]
        assert source.metadata["analysis_rows"][timeframe] <= source.metadata["raw_rows"][timeframe]


@requires_private_inputs
def test_ep006_reader_applies_half_open_analysis_interval() -> None:
    source = _source()
    config = _api("_read_config")(PRIVATE_CONFIG_PATH)
    start = pd.Timestamp(config["analysis_start_utc"])
    end = pd.Timestamp(config["analysis_end_utc_exclusive"])

    for frame in source.frames.values():
        assert frame["timestamp"].is_monotonic_increasing
        assert frame["timestamp"].is_unique
        assert frame["timestamp"].min() >= start
        assert frame["timestamp"].max() < end
    assert all(count >= 0 for count in source.metadata["exclusive_end_rows_removed"].values())
    assert source.frames["H1"]["real_volume"].eq(0).all()


@requires_private_inputs
def test_ep006_reader_preserves_native_gaps() -> None:
    source = _source()
    ledger = source.gap_ledger

    actual_intervals = ledger["timeframe"].value_counts().to_dict()
    assert source.metadata["gap_intervals"] == {
        timeframe: actual_intervals.get(timeframe, 0) for timeframe in ("H1", "H4", "D1")
    }
    assert sum(source.metadata["missing_bar_counts"].values()) == int(
        ledger["missing_bar_count"].sum()
    )
    assert not source.metadata["production_bundle_accepted"]


@requires_private_inputs
def test_ep006_reader_rejects_source_or_schema_mismatch(tmp_path: Path) -> None:
    load = _api("read_ep006_raw_source")
    modified_manifest = tmp_path / "manifest"
    shutil.copytree(RAW_DIR, modified_manifest)
    manifest_path = modified_manifest / "raw_manifest.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace('"symbol": "BITCOIN_i"', '"symbol": "EURUSD"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw manifest SHA-256"):
        load(modified_manifest)

    modified_schema = tmp_path / "schema"
    shutil.copytree(RAW_DIR, modified_schema)
    h1_path = modified_schema / "H1_raw.csv"
    h1_path.write_text(
        h1_path.read_text(encoding="utf-8").replace("real_volume", "real_volume_extra", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="H1 schema"):
        load(modified_schema)


@requires_private_inputs
def test_ep006_target_keeps_gap_and_zero_out_of_binary_rows() -> None:
    source = _source()
    dataset = _dataset(source)

    counts = dataset.metadata["target_counts"]
    assert set(counts) == {"UP", "DOWN", "GAP", "ZERO"}
    assert dataset.metadata["binary_target_rows"] == counts["UP"] + counts["DOWN"]
    assert dataset.metadata["assembled_rows"] <= dataset.metadata["binary_target_rows"]
    assert set(dataset.frame["target_h1_direction"].unique()) == {"UP", "DOWN"}


@requires_private_inputs
def test_ep006_features_reset_at_native_timeframe_gaps() -> None:
    from trading_ai.features.engineering import build_native_timeframe_features

    source = _source()
    h1 = source.frames["H1"]
    first_after_gap = pd.Timestamp(
        source.gap_ledger.loc[source.gap_ledger["timeframe"].eq("H1"), "next_timestamp_utc"].iloc[0]
    )
    registry = build_native_timeframe_features(
        h1,
        "H1",
        return_bars=[1, 3, 6, 12, 24],
        rolling_vol_bars=[6, 12, 24],
        relative_tick_volume_bars=24,
    ).set_index("timestamp")

    assert pd.isna(registry.loc[first_after_gap, "return_1bar"])
    assert pd.isna(registry.loc[first_after_gap, "return_24bar"])
    assert pd.notna(registry.loc[first_after_gap + pd.Timedelta(hours=1), "return_1bar"])


@requires_private_inputs
def test_ep006_features_enforce_point_in_time_close_and_expiry() -> None:
    from trading_ai.features.alignment import build_multitimeframe_point_in_time_features
    from trading_ai.targets.direction import build_h1_direction_target

    source = _source()
    feature_contract = tomllib.loads(FEATURE_CONTRACT.read_text(encoding="utf-8"))
    target = build_h1_direction_target(source.frames["H1"])
    decisions = target["decision_timestamp"]
    features = build_multitimeframe_point_in_time_features(
        source.frames, decisions, feature_contract,
    )
    decisions = pd.to_datetime(decisions, utc=True)
    for timeframe, duration in (("h1", "1h"), ("h4", "4h"), ("d1", "24h")):
        close = pd.to_datetime(features[f"{timeframe}__source_nominal_close_timestamp"], utc=True)
        delta = decisions - close
        available = delta.notna()
        assert available.any()
        assert (delta.loc[available] >= pd.Timedelta(0)).all()
        assert (delta.loc[available] < pd.Timedelta(duration)).all()
    assert (
        features["h1__source_nominal_close_timestamp"].eq(decisions).all()
    )
    assert features["h4__source_nominal_close_timestamp"].eq(decisions).any()
    assert features["d1__source_nominal_close_timestamp"].eq(decisions).any()


@requires_private_inputs
def test_ep006_feature_rows_follow_configured_interval() -> None:
    source = _source()
    dataset = _dataset(source)

    assert dataset.selected_features
    assert len(dataset.frame) == dataset.metadata["assembled_rows"]
    config = _api("_read_config")(PRIVATE_CONFIG_PATH)
    assert dataset.metadata["analysis_interval_utc"] == [
        pd.Timestamp(config["analysis_start_utc"]).isoformat(),
        pd.Timestamp(config["analysis_end_utc_exclusive"]).isoformat(),
    ]


@requires_private_inputs
def test_ep006_windows_match_approved_membership() -> None:
    source = _source()
    dataset = _dataset(source)
    split = _api("build_ep006_folds")(dataset, config_path=PRIVATE_CONFIG_PATH)

    config = _api("_read_config")(PRIVATE_CONFIG_PATH)
    assert len(split.folds) == len(config["validation_windows"])
    assert split.metadata["train_rows"] + split.metadata["validation_rows"] + split.metadata["locked_test_rows"] <= len(dataset.frame)
    assert [
        (fold.window["id"], fold.window["start_utc"], fold.window["end_utc_exclusive"])
        for fold in split.folds
    ] == [
        (window["id"], window["start_utc"], window["end_utc_exclusive"])
        for window in config["validation_windows"]
    ]
    assert all(len(fold.validation) <= len(fold.candidates) for fold in split.folds)


@requires_private_inputs
def test_ep006_expanding_folds_use_strict_target_purge() -> None:
    source = _source()
    dataset = _dataset(source)
    split = _api("build_ep006_folds")(dataset, config_path=PRIVATE_CONFIG_PATH)

    for fold in split.folds:
        start = pd.Timestamp(fold.window["start_utc"])
        end = pd.Timestamp(fold.window["end_utc_exclusive"])
        assert (fold.train["decision_timestamp"] < start).all()
        assert (fold.train["target_timestamp"] < start).all()
        assert (fold.validation["decision_timestamp"] >= start).all()
        assert (fold.validation["decision_timestamp"] < end).all()
        assert (fold.validation["target_timestamp"] < end).all()
    assert [len(fold.train) for fold in split.folds] == sorted(
        len(fold.train) for fold in split.folds
    )


@requires_private_inputs
def test_ep006_train_validation_and_locked_test_are_disjoint() -> None:
    from trading_ai.evaluation.split import chronological_split

    source = _source()
    dataset = _dataset(source)
    split = _api("build_ep006_folds")(dataset, config_path=PRIVATE_CONFIG_PATH)
    outer = chronological_split(dataset.frame, train_fraction=0.60, validation_fraction=0.20)
    test_decisions = set(pd.to_datetime(outer.test["decision_timestamp"], utc=True))
    test_start = pd.Timestamp(outer.test_start)

    for fold in split.folds:
        train_decisions = set(pd.to_datetime(fold.train["decision_timestamp"], utc=True))
        validation_decisions = set(pd.to_datetime(fold.validation["decision_timestamp"], utc=True))
        assert train_decisions.isdisjoint(test_decisions)
        assert validation_decisions.isdisjoint(test_decisions)
        assert pd.to_datetime(fold.train["target_timestamp"], utc=True).lt(test_start).all()
        assert pd.to_datetime(fold.validation["target_timestamp"], utc=True).lt(test_start).all()
        assert fold.validation["decision_timestamp"].is_unique
    assert (outer.validation["target_timestamp"] < outer.test_start).all()
    last_locked = outer.test.iloc[-1]
    config = _api("_read_config")(PRIVATE_CONFIG_PATH)
    assert last_locked["target_timestamp"] == pd.Timestamp(config["analysis_end_utc_exclusive"])


@requires_private_config
def test_ep006_config_rejects_window_and_model_drift(tmp_path: Path) -> None:
    config_text = PRIVATE_CONFIG_PATH.read_text(encoding="utf-8")
    changed_window = tmp_path / "changed-window.toml"
    changed_window.write_text(
        re.sub(r'(?m)^(start_utc\s*=\s*).+$', r'\1"invalid"', config_text, count=1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation windows"):
        _api("_read_config")(changed_window)

    changed_model = tmp_path / "changed-model.toml"
    changed_model.write_text(
        config_text.replace('logistic_regression_max_iter = 1000', 'logistic_regression_max_iter = 500'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="estimator settings"):
        _api("_read_config")(changed_model)


@pytest.fixture(scope="module")
def ep006_completed_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    if not _HAS_PRIVATE_INPUTS:
        pytest.skip("requires owner-authorized local EP006 source files")
    output = tmp_path_factory.mktemp("ep006-run") / "artifacts"
    report = _api("run_episode006_stability")(
        RAW_DIR, output, config_path=PRIVATE_CONFIG_PATH
    )
    return output, report


@requires_private_inputs
def test_ep006_runner_fits_each_fold_only_on_expanding_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_ai.experiments import ep006_stability as module
    from trading_ai.baselines.models import fit_logistic_baseline as real_fit

    run = _api("run_episode006_stability")
    source = module.read_ep006_raw_source(RAW_DIR, config_path=PRIVATE_CONFIG_PATH)
    dataset = module.build_ep006_dataset(source, FEATURE_CONTRACT)
    fold_set = module.build_ep006_folds(dataset, config_path=PRIVATE_CONFIG_PATH)
    seen_fit_indices: list[list[int]] = []
    seen_prediction_indices: list[list[int]] = []

    def capture_fit(x_train: pd.DataFrame, y_train: pd.Series):
        seen_fit_indices.append(x_train.index.tolist())
        estimator = real_fit(x_train, y_train)

        class RecordingEstimator:
            def predict(self, x_test: pd.DataFrame) -> pd.Series:
                seen_prediction_indices.append(x_test.index.tolist())
                return estimator.predict(x_test)

        return RecordingEstimator()

    monkeypatch.setattr(module, "fit_logistic_baseline", capture_fit)
    run(RAW_DIR, tmp_path / "runner-spy", config_path=PRIVATE_CONFIG_PATH)

    assert seen_fit_indices == [fold.train.index.tolist() for fold in fold_set.folds]
    assert seen_prediction_indices == [fold.validation.index.tolist() for fold in fold_set.folds]


def test_ep006_runner_scores_only_four_approved_validation_windows(
    ep006_completed_run: tuple[Path, dict],
) -> None:
    output, report = ep006_completed_run
    predictions = pd.read_csv(output / "predictions.csv")

    expected_ids = {
        row["id"] for row in _api("_read_config")(PRIVATE_CONFIG_PATH)["validation_windows"]
    }
    assert set(predictions["window_id"]) == expected_ids
    assert report["locked_test"]["evaluated"] is False
    assert predictions["decision_timestamp"].notna().all()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    predicted_rows = predictions["window_id"].value_counts().to_dict()
    metric_rows = {
        window_id: next(iter(baselines.values()))["rows"]
        for window_id, baselines in metrics["per_window"].items()
    }
    assert predicted_rows == metric_rows
    assert not set(predictions["window_id"]).intersection({"TEST", "HOLDOUT"})


def test_ep006_metrics_include_four_frozen_baselines(
    ep006_completed_run: tuple[Path, dict],
) -> None:
    output, report = ep006_completed_run
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    baselines = {
        "B0_majority_class", "B1_previous_hour_direction",
        "B2_logistic_regression", "Fixed_UP_reference",
    }

    expected_ids = {
        row["id"] for row in _api("_read_config")(PRIVATE_CONFIG_PATH)["validation_windows"]
    }
    assert set(metrics["per_window"]) == expected_ids
    assert all(set(window) == baselines for window in metrics["per_window"].values())
    assert set(metrics["pooled"]) == baselines
    assert all("accuracy" in row and "balanced_accuracy" in row and "mcc" in row and "confusion_matrix" in row for row in metrics["pooled"].values())
    for name in baselines:
        assert metrics["pooled"][name]["rows"] == sum(
            row[name]["rows"] for row in metrics["per_window"].values()
        )
    assert report["locked_test"]["evaluated"] is False


def test_ep006_report_marks_test_locked_and_unevaluated(
    ep006_completed_run: tuple[Path, dict],
) -> None:
    output, report = ep006_completed_run
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest["locked_test_evaluated"] is False
    assert report["locked_test"]["status"] == "locked"
    assert report["locked_test"]["evaluated"] is False
    assert "metrics" not in report["locked_test"]
    assert (output / "gap_coverage_ledger.csv").is_file()
    assert (output / "window_membership.csv").is_file()
    assert (output / "predictions.csv").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "report.md").is_file()
    report_markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "correct/total" in report_markdown
    assert "balanced accuracy" in report_markdown.lower()
    assert "test fit, tuning, or scoring was run" in report_markdown.lower()
    for window_metrics in [*json.loads((output / "metrics.json").read_text(encoding="utf-8"))["per_window"].values(), json.loads((output / "metrics.json").read_text(encoding="utf-8"))["pooled"]]:
        for metric in window_metrics.values():
            matrix = metric["confusion_matrix"]["values"]
            correct = matrix[0][0] + matrix[1][1]
            assert f"{correct}/{metric['rows']}" in report_markdown


def test_ep006_run_manifest_identifies_executed_implementation(ep006_completed_run: tuple[Path, dict]) -> None:
    output, _ = ep006_completed_run
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    implementation = manifest["implementation_sha256"]

    assert set(implementation) == {
        "src/trading_ai/experiments/ep006_stability.py",
        "experiments/ep006_baselines/run_ep006.py",
        "src/trading_ai/baselines/models.py",
        "src/trading_ai/evaluation/metrics.py",
    }
    for relative, digest in implementation.items():
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)
        assert hashlib.sha256(Path(relative).read_bytes()).hexdigest() == digest
