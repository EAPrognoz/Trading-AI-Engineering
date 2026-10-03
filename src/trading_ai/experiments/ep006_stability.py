"""Research-only EP006 BTC stability experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import platform
import sys
import tomllib
from typing import Any

import numpy as np
import pandas as pd

from trading_ai.data.timeframes import timeframe_duration
from trading_ai.baselines.models import (
    fit_logistic_baseline,
    fit_majority_label,
    predict_majority,
    predict_previous_hour_direction,
)
from trading_ai.evaluation.split import chronological_split
from trading_ai.evaluation.metrics import classification_metrics
from trading_ai.experiments.baseline_dataset import BaselineDataset
from trading_ai.features.alignment import (
    build_multitimeframe_feature_registry,
    build_multitimeframe_point_in_time_features,
)
from trading_ai.targets.direction import build_h1_direction_target


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs/experiments/ep006_btc_stability.toml"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TIMEFRAMES = ("H1", "H4", "D1")
EXPECTED_COLUMNS = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
EXPECTED_MODEL = {
    "logistic_regression_solver": "lbfgs",
    "logistic_regression_max_iter": 1000,
    "scaler": "StandardScaler",
}
EXPECTED_SPLIT = {
    "train_fraction": 0.60,
    "validation_fraction": 0.20,
    "test_fraction": 0.20,
    "purge_rule": "target_timestamp < next_partition_start",
    "test_policy": "locked",
}
EXPECTED_METRICS = ["accuracy", "balanced_accuracy", "mcc", "confusion_matrix"]
IMPLEMENTATION_FILES = (
    "src/trading_ai/experiments/ep006_stability.py",
    "experiments/ep006_baselines/run_ep006.py",
    "src/trading_ai/baselines/models.py",
    "src/trading_ai/evaluation/metrics.py",
)


@dataclass(frozen=True)
class Episode006RawSource:
    """Hash-verified research inputs and a ledger of unfilled native gaps."""

    frames: dict[str, pd.DataFrame]
    manifest: dict[str, Any]
    metadata: dict[str, Any]
    gap_ledger: pd.DataFrame


@dataclass(frozen=True)
class Episode006Fold:
    """One frozen expanding-train validation fold."""

    window: dict[str, Any]
    train: pd.DataFrame
    candidates: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class Episode006FoldSet:
    folds: tuple[Episode006Fold, ...]
    metadata: dict[str, Any]


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_report_markdown(report: dict[str, Any], metrics: dict[str, Any], path: Path) -> None:
    names = (
        "B0_majority_class", "B1_previous_hour_direction",
        "B2_logistic_regression", "Fixed_UP_reference",
    )
    lines = [
        "# Episode 006: Bitcoin Baseline Stability",
        "",
        "This report scores only four frozen chronological validation windows for `BITCOIN_i` H1 direction using H1/H4/D1 features.",
        "The analysis uses broker CFD bars and is exploratory classification evidence, not trading-profit evidence.",
        "",
        "| Window | Rows | B0 majority (correct/total; accuracy; balanced accuracy) | B1 prior H1 (correct/total; accuracy; balanced accuracy) | B2 logistic regression (correct/total; accuracy; balanced accuracy) | Fixed-UP (correct/total; accuracy; balanced accuracy) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    window_metrics = {
        **metrics["per_window"],
        "Pooled validation": metrics["pooled"],
    }
    for window_id, values in window_metrics.items():
        rows = values[names[0]]["rows"]
        cells = []
        for name in names:
            metric = values[name]
            matrix = metric["confusion_matrix"]["values"]
            correct = matrix[0][0] + matrix[1][1]
            cells.append(
                f"{correct}/{metric['rows']} ({metric['accuracy']:.4f}; {metric['balanced_accuracy']:.4f})"
            )
        lines.append(f"| {window_id} | {rows} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Direction class counts",
        "",
        "Counts are DOWN/UP. Actual counts describe each scored window; predicted counts expose each baseline's class balance.",
        "",
        "| Window | Actual DOWN/UP | B0 predicted DOWN/UP | B1 predicted DOWN/UP | B2 predicted DOWN/UP | Fixed-UP predicted DOWN/UP |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for window_id, values in metrics["per_window"].items():
        actual = values[names[0]]["class_counts"]
        predicted = [values[name]["prediction_counts"] for name in names]
        pred_cells = " | ".join(f"{item['DOWN']}/{item['UP']}" for item in predicted)
        lines.append(
            f"| {window_id} | {actual['DOWN']}/{actual['UP']} | {pred_cells} |"
        )
    pooled_actual = metrics["pooled"][names[0]]["class_counts"]
    pooled_predicted = [metrics["pooled"][name]["prediction_counts"] for name in names]
    pooled_pred_cells = " | ".join(
        f"{item['DOWN']}/{item['UP']}" for item in pooled_predicted
    )
    lines.extend([
        f"| Pooled validation | {pooled_actual['DOWN']}/{pooled_actual['UP']} | {pooled_pred_cells} |",
        "",
        f"Actual class counts (DOWN/UP): V1 {metrics['per_window']['V1'][names[0]]['class_counts']['DOWN']}/{metrics['per_window']['V1'][names[0]]['class_counts']['UP']}; V2 {metrics['per_window']['V2'][names[0]]['class_counts']['DOWN']}/{metrics['per_window']['V2'][names[0]]['class_counts']['UP']}; V3 {metrics['per_window']['V3'][names[0]]['class_counts']['DOWN']}/{metrics['per_window']['V3'][names[0]]['class_counts']['UP']}; V4 {metrics['per_window']['V4'][names[0]]['class_counts']['DOWN']}/{metrics['per_window']['V4'][names[0]]['class_counts']['UP']}; pooled {pooled_actual['DOWN']}/{pooled_actual['UP']}.",
        "",
        "## Protocol and limitations",
        "",
        "- The four windows, membership, model, and metrics were frozen before this run.",
        "- Scalers and Logistic Regression were fit separately on each expanding training fold; B0 was recomputed from that fold's train labels.",
        "- H1 direction persistence (B1) uses the prior H1 direction feature. Fixed-UP is a separate reference.",
        "- The outer test partition remains locked and unevaluated. No test fit, tuning, or scoring was run.",
        "- Returns, costs, execution, slippage, and profitability were not evaluated.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    configured = os.environ.get("EP006_CONFIG_PATH")
    return Path(configured) if configured else DEFAULT_CONFIG_PATH


def _read_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = _resolve_config_path(path)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("experiment_id") != "ep006-btc-baseline-stability-v1":
        raise ValueError("unsupported EP006 experiment config")

    local_fields = (
        "dataset_id", "raw_manifest_sha256", "coverage_reconciliation_sha256",
        "source_revision", "source_tree_sha256",
        "normalized_source_snapshot_sha256", "feature_contract_sha256",
        "analysis_start_utc", "analysis_end_utc_exclusive",
    )
    if any(
        not isinstance(config.get(field), str)
        or not config[field].strip()
        or config[field] == "SET_LOCALLY"
        for field in local_fields
    ):
        raise ValueError(
            "EP006 public template is not runnable; copy it to a local config and set local config pins"
        )
    for field in (
        "raw_manifest_sha256", "coverage_reconciliation_sha256",
        "normalized_source_snapshot_sha256", "feature_contract_sha256",
    ):
        if not _SHA256_PATTERN.fullmatch(config[field]):
            raise ValueError(f"EP006 local config {field} must be a lowercase SHA-256")
    for field in ("source_revision", "source_tree_sha256"):
        if not re.fullmatch(r"[0-9a-f]{40}", config[field]):
            raise ValueError(f"EP006 local config {field} must be a Git object ID")

    if not isinstance(config.get("source"), str) or not config["source"].strip():
        raise ValueError("EP006 local config must name a local source directory")
    if not isinstance(config.get("symbol"), str) or not config["symbol"].strip():
        raise ValueError("EP006 local config must name a Bitcoin symbol")
    if config.get("feature_timeframes") != list(TIMEFRAMES):
        raise ValueError("EP006 config source identity mismatch")

    start = _utc_timestamp(config["analysis_start_utc"])
    end = _utc_timestamp(config["analysis_end_utc_exclusive"])
    if not start < end:
        raise ValueError("EP006 analysis interval must be ordered UTC timestamps")
    windows = config.get("validation_windows", [])
    if len(windows) != 4:
        raise ValueError("EP006 requires exactly four validation windows")
    seen_ids: set[str] = set()
    previous_end: pd.Timestamp | None = None
    for row in windows:
        window_id = row.get("id")
        try:
            window_start = _utc_timestamp(row["start_utc"])
            window_end = _utc_timestamp(row["end_utc_exclusive"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("EP006 validation windows need valid UTC boundaries") from exc
        if not isinstance(window_id, str) or not window_id or window_id in seen_ids:
            raise ValueError("EP006 validation window IDs must be non-empty and unique")
        if not window_start < window_end:
            raise ValueError("EP006 validation window boundaries must be ordered")
        if previous_end is not None and window_start != previous_end:
            raise ValueError("EP006 validation windows must be consecutive and nonoverlapping")
        seen_ids.add(window_id)
        previous_end = window_end

    if config.get("model") != EXPECTED_MODEL:
        raise ValueError("EP006 config does not match the frozen estimator settings")
    if config.get("split") != EXPECTED_SPLIT:
        raise ValueError("EP006 config does not match the frozen split policy")
    if config.get("metrics", {}).get("names") != EXPECTED_METRICS:
        raise ValueError("EP006 config does not match the frozen metrics")
    return config


def _project_root_from_config(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if (parent / "src/trading_ai").is_dir() and (parent / "configs").is_dir():
            return parent
    return _PROJECT_ROOT


def _gap_rows(frame: pd.DataFrame, timeframe: str) -> list[dict[str, Any]]:
    duration = timeframe_duration(timeframe)
    delta = frame["timestamp"].diff()
    indices = np.flatnonzero((delta > duration).to_numpy())
    rows: list[dict[str, Any]] = []
    for index in indices:
        previous = frame.iloc[index - 1]["timestamp"]
        following = frame.iloc[index]["timestamp"]
        missing = int(delta.iloc[index] / duration) - 1
        if missing < 1 or delta.iloc[index] % duration != pd.Timedelta(0):
            raise ValueError(f"{timeframe} has a non-native cadence gap")
        rows.append({
            "timeframe": timeframe,
            "previous_timestamp_utc": previous.isoformat(),
            "next_timestamp_utc": following.isoformat(),
            "gap_start_utc": (previous + duration).isoformat(),
            "gap_end_utc_exclusive": following.isoformat(),
            "missing_bar_count": missing,
        })
    return rows


def read_ep006_raw_source(
    raw_dir: str | Path,
    *,
    config_path: str | Path | None = None,
) -> Episode006RawSource:
    """Read only the pinned BITCOIN_i source, preserving gaps and raw bytes."""
    config_path = _resolve_config_path(config_path)
    config = _read_config(config_path)
    root = Path(raw_dir).resolve()
    manifest_path = root / "raw_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    if manifest_sha256 != config["raw_manifest_sha256"]:
        raise ValueError("raw manifest SHA-256 mismatch")
    manifest = json.loads(manifest_bytes)

    start = pd.Timestamp(config["analysis_start_utc"])
    end = pd.Timestamp(config["analysis_end_utc_exclusive"])
    if start.tzinfo is None or end.tzinfo is None or not start < end:
        raise ValueError("EP006 analysis interval must be ordered UTC timestamps")
    if manifest.get("dataset_id") != config["dataset_id"]:
        raise ValueError("raw source dataset identity mismatch")
    if manifest.get("symbol") != config["symbol"]:
        raise ValueError("raw source symbol mismatch")
    project_source = manifest.get("project_source", {})
    if (
        project_source.get("repository_head") != config["source_revision"]
        or project_source.get("normalized_tracked_snapshot_sha256")
        != config["normalized_source_snapshot_sha256"]
    ):
        raise ValueError("raw source project revision mismatch")
    if manifest.get("range_policy", {}).get("analysis_interval") != (
        f"[{start.isoformat().replace('+00:00', 'Z')},"
        f"{end.isoformat().replace('+00:00', 'Z')})"
    ):
        raise ValueError("raw source analysis interval mismatch")
    if set(manifest.get("streams", {})) != set(TIMEFRAMES):
        raise ValueError("raw source must include exactly H1, H4, and D1")

    coverage_path = root / "coverage_reconciliation.json"
    coverage_sha256 = _sha256_file(coverage_path)
    if coverage_sha256 != config["coverage_reconciliation_sha256"]:
        raise ValueError("coverage reconciliation SHA-256 mismatch")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if (
        coverage.get("source_manifest_sha256") != manifest_sha256
        or coverage.get("symbol") != config["symbol"]
        or coverage.get("baseline_metrics_computed") is not False
    ):
        raise ValueError("coverage reconciliation does not match the frozen source")

    # The source snapshot digest is pinned in the raw manifest. These component
    # hashes independently verify the existing modules used by EP006.
    project_root = _project_root_from_config(config_path)
    relevant_hashes = project_source.get("relevant_file_sha256", {})
    if not relevant_hashes:
        raise ValueError("raw source project component hashes are missing")
    for relative, expected in relevant_hashes.items():
        dependency = project_root / relative
        if not dependency.is_file() or _sha256_file(dependency) != expected:
            raise ValueError(f"source component hash mismatch: {relative}")

    frames: dict[str, pd.DataFrame] = {}
    raw_rows: dict[str, int] = {}
    analysis_rows: dict[str, int] = {}
    excluded_end_rows: dict[str, int] = {}
    gap_records: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    for timeframe in TIMEFRAMES:
        stream = manifest["streams"][timeframe]
        if stream.get("columns") != EXPECTED_COLUMNS:
            raise ValueError(f"{timeframe} schema mismatch in raw manifest")
        path_name = stream.get("raw_file")
        if path_name != f"{timeframe}_raw.csv":
            raise ValueError(f"{timeframe} raw filename mismatch")
        path = root / path_name
        header = list(pd.read_csv(path, nrows=0).columns)
        if header != EXPECTED_COLUMNS:
            raise ValueError(f"{timeframe} schema mismatch in raw CSV")
        raw_hash = _sha256_file(path)
        if raw_hash != stream.get("raw_sha256"):
            raise ValueError(f"{timeframe} raw file hash mismatch")
        frame = pd.read_csv(path)
        if len(frame) != stream.get("rows"):
            raise ValueError(f"{timeframe} raw row count mismatch")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        for column in EXPECTED_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        timestamp = frame["timestamp"]
        if timestamp.isna().any() or timestamp.duplicated().any() or not timestamp.is_monotonic_increasing:
            raise ValueError(f"{timeframe} timestamps must be unique and ordered")
        numeric = frame[EXPECTED_COLUMNS[1:]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{timeframe} contains nonfinite market values")
        if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
            raise ValueError(f"{timeframe} high violates OHLC bounds")
        if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
            raise ValueError(f"{timeframe} low violates OHLC bounds")
        in_interval = timestamp.ge(start) & timestamp.lt(end)
        analysis = frame.loc[in_interval].reset_index(drop=True)
        excluded_end_rows[timeframe] = int((timestamp == end).sum())
        if (timestamp.lt(start) | timestamp.gt(end)).any():
            raise ValueError(f"{timeframe} source has rows outside the frozen request")
        raw_rows[timeframe] = int(len(frame))
        analysis_rows[timeframe] = int(len(analysis))
        raw_hashes[timeframe] = raw_hash
        frames[timeframe] = analysis
        gap_records.extend(_gap_rows(analysis, timeframe))

    gap_ledger = pd.DataFrame.from_records(
        gap_records,
        columns=[
            "timeframe", "previous_timestamp_utc", "next_timestamp_utc",
            "gap_start_utc", "gap_end_utc_exclusive", "missing_bar_count",
        ],
    )
    gap_intervals = {
        timeframe: int((gap_ledger["timeframe"] == timeframe).sum())
        for timeframe in TIMEFRAMES
    }
    missing_bar_counts = {
        timeframe: int(
            gap_ledger.loc[gap_ledger["timeframe"] == timeframe, "missing_bar_count"].sum()
        )
        for timeframe in TIMEFRAMES
    }
    metadata = {
        "dataset_id": manifest["dataset_id"],
        "symbol": manifest["symbol"],
        "source_type": manifest["source_type"],
        "source_revision": project_source["repository_head"],
        "source_tree_sha256": config["source_tree_sha256"],
        "normalized_source_snapshot_sha256": project_source[
            "normalized_tracked_snapshot_sha256"
        ],
        "raw_manifest_sha256": manifest_sha256,
        "coverage_reconciliation_sha256": coverage_sha256,
        "feature_contract_sha256": config["feature_contract_sha256"],
        "analysis_start_utc": start.isoformat(),
        "analysis_end_utc_exclusive": end.isoformat(),
        "raw_sha256": raw_hashes,
        "raw_rows": raw_rows,
        "analysis_rows": analysis_rows,
        "exclusive_end_rows_removed": excluded_end_rows,
        "gap_intervals": gap_intervals,
        "missing_bar_counts": missing_bar_counts,
        "production_bundle_accepted": False,
    }
    return Episode006RawSource(frames, manifest, metadata, gap_ledger)


def build_ep006_dataset(
    source: Episode006RawSource,
    feature_contract_path: str | Path,
) -> BaselineDataset:
    """Reuse EP003/EP004 to assemble complete, binary EP006 samples."""
    contract_path = Path(feature_contract_path)
    expected_contract_sha256 = source.metadata["feature_contract_sha256"]
    if _sha256_file(contract_path) != expected_contract_sha256:
        raise ValueError("EP006 feature contract SHA-256 mismatch")
    with contract_path.open("rb") as handle:
        contract = tomllib.load(handle)
    if contract.get("contract_id") != "ep004-btc-mtf-features-v1":
        raise ValueError("EP006 feature contract identity mismatch")

    target = build_h1_direction_target(source.frames["H1"])
    start = pd.Timestamp(source.metadata["analysis_start_utc"])
    end = pd.Timestamp(source.metadata["analysis_end_utc_exclusive"])
    in_interval = (
        target["decision_timestamp"].ge(start)
        & target["decision_timestamp"].lt(end)
    )
    target = target.loc[in_interval].reset_index(drop=True)
    registry = build_multitimeframe_feature_registry(contract)
    selected = [entry["output_name"] for entry in registry if entry["selected"]]
    features = build_multitimeframe_point_in_time_features(
        source.frames, target["decision_timestamp"], contract,
    )
    if not features["decision_timestamp"].equals(target["decision_timestamp"]):
        raise ValueError("EP006 target and feature decision timestamps diverged")

    combined = pd.concat(
        [target.reset_index(drop=True), features.drop(columns="decision_timestamp")],
        axis=1,
    )
    missing = sorted(set(selected).difference(combined.columns))
    if missing:
        raise ValueError(f"EP006 selected features are missing: {missing}")
    target_counts = {
        label: int(target["target_h1_direction"].eq(label).sum())
        for label in ("UP", "DOWN", "GAP", "ZERO")
    }
    binary_mask = combined["target_h1_direction"].isin(["UP", "DOWN"])
    binary = combined.loc[binary_mask].copy()
    values = binary[selected].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
    complete = values.notna().all(axis=1).to_numpy() & finite
    assembled = binary.loc[complete].copy()
    if assembled.empty:
        raise ValueError("no complete binary samples remain for EP006")

    metadata = {
        **source.metadata,
        "target_contract_id": "ep003-h1-direction-v1",
        "feature_contract_id": str(contract["contract_id"]),
        "feature_contract_sha256": expected_contract_sha256,
        "analysis_interval_utc": [start.isoformat(), end.isoformat()],
        "target_rows": int(len(target)),
        "target_counts": target_counts,
        "binary_target_rows": int(binary_mask.sum()),
        "rows_removed_for_feature_warmup_or_nonfinite": int(len(binary) - len(assembled)),
        "assembled_rows": int(len(assembled)),
        "selected_feature_count": int(len(selected)),
    }
    return BaselineDataset(
        frame=assembled.reset_index(drop=True),
        selected_features=selected,
        metadata=metadata,
    )


def _utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("EP006 fold boundaries must include UTC offsets")
    return timestamp.tz_convert("UTC")


def build_ep006_folds(
    dataset: BaselineDataset,
    *,
    config_path: str | Path | None = None,
) -> Episode006FoldSet:
    """Create only the four frozen validation windows and purge at each edge."""
    config_path = _resolve_config_path(config_path)
    config = _read_config(config_path)
    root = _project_root_from_config(config_path)
    outer_path = root / config["outer_experiment_contract"]
    with outer_path.open("rb") as handle:
        outer_contract = tomllib.load(handle)
    expected_split = config["split"]
    if (
        outer_contract.get("contract_id") != "ep005-btc-mtf-baseline-v1"
        or outer_contract.get("test_policy") != "locked"
        or any(
            outer_contract.get(field) != expected_split[key]
            for field, key in (
                ("train_fraction", "train_fraction"),
                ("validation_fraction", "validation_fraction"),
                ("test_fraction", "test_fraction"),
            )
        )
    ):
        raise ValueError("EP006 requires the frozen locked EP005 60/20/20 split")

    outer = chronological_split(
        dataset.frame,
        train_fraction=float(expected_split["train_fraction"]),
        validation_fraction=float(expected_split["validation_fraction"]),
    )
    windows = config.get("validation_windows", [])
    if len(windows) != 4:
        raise ValueError("EP006 requires exactly four frozen validation windows")
    if outer.validation_start != _utc_timestamp(windows[0]["start_utc"]):
        raise ValueError("EP006 first validation window does not start at EP005 validation")
    if outer.test_start != _utc_timestamp(windows[-1]["end_utc_exclusive"]):
        raise ValueError("EP006 final validation window does not end at locked test start")

    decisions = pd.to_datetime(dataset.frame["decision_timestamp"], utc=True, errors="raise")
    targets = pd.to_datetime(dataset.frame["target_timestamp"], utc=True, errors="raise")
    outer_train = outer.train
    # The outer locked-test purge defines the validation rows eligible for the
    # stability experiment. Internal fold edges then purge each window again.
    outer_validation = dataset.frame.loc[
        decisions.ge(outer.validation_start)
        & decisions.lt(outer.test_start)
        & targets.lt(outer.test_start)
    ].copy()
    folds: list[Episode006Fold] = []
    prior_candidates: list[pd.DataFrame] = []
    previous_end: pd.Timestamp | None = None
    for window in windows:
        start = _utc_timestamp(window["start_utc"])
        end = _utc_timestamp(window["end_utc_exclusive"])
        if previous_end is not None and start != previous_end:
            raise ValueError("EP006 validation windows must be consecutive and nonoverlapping")
        previous_end = end
        if not start < end or end > outer.test_start:
            raise ValueError("EP006 validation window crosses the locked-test boundary")
        candidates = outer_validation.loc[
            pd.to_datetime(outer_validation["decision_timestamp"], utc=True).ge(start)
            & pd.to_datetime(outer_validation["decision_timestamp"], utc=True).lt(end)
        ].copy()
        candidate_targets = pd.to_datetime(candidates["target_timestamp"], utc=True)
        validation = candidates.loc[candidate_targets.lt(end)].copy()
        expanded = [
            previous.loc[
                pd.to_datetime(previous["target_timestamp"], utc=True).lt(start)
            ]
            for previous in prior_candidates
        ]
        train = pd.concat([outer_train, *expanded]).sort_values(
            "decision_timestamp"
        )
        if candidates.empty or validation.empty or train.empty:
            raise ValueError(f"EP006 {window['id']} has no eligible training or validation rows")
        if train.index.intersection(validation.index).size:
            raise ValueError(f"EP006 {window['id']} train and validation overlap")
        if not pd.to_datetime(train["target_timestamp"], utc=True).lt(start).all():
            raise ValueError(f"EP006 {window['id']} train target crosses its validation start")
        if not pd.to_datetime(validation["target_timestamp"], utc=True).lt(end).all():
            raise ValueError(f"EP006 {window['id']} validation target crosses its window end")
        folds.append(Episode006Fold(
            window=dict(window), train=train, candidates=candidates, validation=validation,
        ))
        prior_candidates.append(candidates)

    locked_last = outer.test.iloc[-1]
    analysis_end = _utc_timestamp(config["analysis_end_utc_exclusive"])
    last_test_decision = pd.Timestamp(locked_last["decision_timestamp"])
    last_test_target = pd.Timestamp(locked_last["target_timestamp"])
    if last_test_target != analysis_end:
        raise ValueError("EP006 locked-test membership does not end at the exclusive analysis end")
    metadata = {
        "train_rows": len(outer.train),
        "validation_rows": len(outer.validation),
        "locked_test_rows": len(outer.test),
        "validation_start_utc": outer.validation_start.isoformat(),
        "test_start_utc": outer.test_start.isoformat(),
        "purged_train_boundary_rows": outer.purged_train_rows,
        "purged_validation_boundary_rows": outer.purged_validation_rows,
        "locked_test_last_decision_utc": last_test_decision.isoformat(),
        "locked_test_final_target_utc": last_test_target.isoformat(),
        "locked_test_evaluated": False,
    }
    return Episode006FoldSet(tuple(folds), metadata)


def run_episode006_stability(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    command_text: str | None = None,
) -> dict[str, Any]:
    """Fit and score only the four frozen validation folds, then save evidence."""
    config_path = _resolve_config_path(config_path).resolve()
    config = _read_config(config_path)
    root = _project_root_from_config(config_path)
    destination = Path(output_dir).resolve()
    local_root = (root / ".local").resolve()
    if not destination.is_relative_to(local_root):
        raise ValueError("EP006 research outputs must remain under ignored .local/")
    destination.mkdir(parents=True, exist_ok=False)

    source = read_ep006_raw_source(raw_dir, config_path=config_path)
    feature_contract_path = root / config["feature_contract"]
    dataset = build_ep006_dataset(source, feature_contract_path)
    fold_set = build_ep006_folds(dataset, config_path=config_path)
    selected = dataset.selected_features
    persistence_feature = "h1__return_1bar"
    per_window: dict[str, dict[str, Any]] = {}
    prediction_frames: list[pd.DataFrame] = []
    membership_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold in fold_set.folds:
        window_id = str(fold.window["id"])
        train = fold.train
        validation = fold.validation
        y_train = train["target_h1_direction"]
        y_true = validation["target_h1_direction"]
        majority_label = fit_majority_label(y_train)
        model = fit_logistic_baseline(train[selected], y_train)
        predictions = {
            "B0_majority_class": predict_majority(validation.index, majority_label),
            "B1_previous_hour_direction": predict_previous_hour_direction(
                validation[persistence_feature], zero_fallback=majority_label,
            ),
            "B2_logistic_regression": pd.Series(
                model.predict(validation[selected]), index=validation.index, dtype="string",
            ),
            "Fixed_UP_reference": pd.Series("UP", index=validation.index, dtype="string"),
        }
        per_window[window_id] = {
            name: classification_metrics(y_true, prediction)
            for name, prediction in predictions.items()
        }
        prediction_frame = validation[["decision_timestamp", "target_timestamp"]].copy()
        prediction_frame.insert(0, "window_id", window_id)
        prediction_frame["actual_direction"] = y_true
        for name, prediction in predictions.items():
            prediction_frame[name] = prediction
        prediction_frames.append(prediction_frame.reset_index(drop=True))
        scored_keys = set(validation.index)
        for row_index, row in fold.candidates.iterrows():
            membership_rows.append({
                "window_id": window_id,
                "partition_role": "validation_candidate",
                "decision_timestamp": row["decision_timestamp"].isoformat(),
                "target_timestamp": row["target_timestamp"].isoformat(),
                "scored": row_index in scored_keys,
            })
        for _, row in train.iterrows():
            membership_rows.append({
                "window_id": window_id,
                "partition_role": "expanding_train",
                "decision_timestamp": row["decision_timestamp"].isoformat(),
                "target_timestamp": row["target_timestamp"].isoformat(),
                "scored": False,
            })
        fold_summaries.append({
            "window_id": window_id,
            "candidate_rows": len(fold.candidates),
            "scored_validation_rows": len(validation),
            "expanding_train_rows": len(train),
            "start_utc": fold.window["start_utc"],
            "end_utc_exclusive": fold.window["end_utc_exclusive"],
            "first_scored_decision_utc": validation["decision_timestamp"].iloc[0].isoformat(),
            "last_scored_decision_utc": validation["decision_timestamp"].iloc[-1].isoformat(),
            "last_scored_target_utc": validation["target_timestamp"].iloc[-1].isoformat(),
            "b0_majority_label": majority_label,
            "scaler_fit_scope": "expanding_train_only",
            "logistic_fit_scope": "expanding_train_only",
            "scored_partition": "validation_only",
        })

    predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled = {
        name: classification_metrics(
            predictions["actual_direction"], predictions[name],
        )
        for name in (
            "B0_majority_class", "B1_previous_hour_direction",
            "B2_logistic_regression", "Fixed_UP_reference",
        )
    }
    metrics = {"per_window": per_window, "pooled": pooled}
    report = {
        "experiment_id": config["experiment_id"],
        "symbol": config["symbol"],
        "task": "H1 next-hour direction classification",
        "feature_timeframes": config["feature_timeframes"],
        "source": source.metadata,
        "split": fold_set.metadata,
        "windows": fold_summaries,
        "locked_test": {
            "status": "locked",
            "evaluated": False,
            "rows": fold_set.metadata["locked_test_rows"],
        },
        "metric_names": config["metrics"]["names"],
        "baseline_names": [
            "B0_majority_class", "B1_previous_hour_direction",
            "B2_logistic_regression", "Fixed_UP_reference",
        ],
        "pooled_validation_metrics": pooled,
    }
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
    }
    config_copy = destination / "ep006_btc_stability.toml"
    config_copy.write_bytes(config_path.read_bytes())
    source_ledger = source.gap_ledger.to_dict(orient="records")
    gap_rows: list[dict[str, Any]] = []
    for timeframe in TIMEFRAMES:
        gaps = [row for row in source_ledger if row["timeframe"] == timeframe]
        gap_rows.append({
            "record_type": "timeframe_summary",
            "timeframe": timeframe,
            "raw_sha256": source.metadata["raw_sha256"][timeframe],
            "raw_rows": source.metadata["raw_rows"][timeframe],
            "analysis_rows": source.metadata["analysis_rows"][timeframe],
            "exclusive_end_rows_removed": source.metadata["exclusive_end_rows_removed"][timeframe],
            "gap_interval_count": len(gaps),
            "missing_bar_count": source.metadata["missing_bar_counts"][timeframe],
        })
    for gap in source_ledger:
        gap_rows.append({"record_type": "native_gap", **gap})
    pd.DataFrame(gap_rows).to_csv(destination / "gap_coverage_ledger.csv", index=False)
    pd.DataFrame(membership_rows).to_csv(destination / "window_membership.csv", index=False)
    predictions.to_csv(destination / "predictions.csv", index=False)
    _json_dump(destination / "metrics.json", metrics)
    _json_dump(destination / "report.json", report)
    _write_report_markdown(report, metrics, destination / "report.md")

    artifact_names = (
        "ep006_btc_stability.toml", "gap_coverage_ledger.csv",
        "window_membership.csv", "predictions.csv", "metrics.json", "report.json", "report.md",
    )
    artifacts = {
        name: {"sha256": _sha256_file(destination / name), "bytes": (destination / name).stat().st_size}
        for name in artifact_names
    }
    run_manifest = {
        "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command_text or "python experiments/ep006_baselines/run_ep006.py",
        "runtime": runtime,
        "config_sha256": _sha256_file(config_path),
        "feature_contract_sha256": _sha256_file(feature_contract_path),
        "source": source.metadata,
        "source_component_sha256": source.manifest["project_source"]["relevant_file_sha256"],
        "implementation_sha256": {
            relative: _sha256_file(root / relative)
            for relative in IMPLEMENTATION_FILES
        },
        "dataset": dataset.metadata,
        "outer_split": fold_set.metadata,
        "folds": fold_summaries,
        "locked_test_evaluated": False,
        "locked_test_fit_or_scored": False,
        "artifacts": artifacts,
    }
    _json_dump(destination / "run_manifest.json", run_manifest)
    return report
