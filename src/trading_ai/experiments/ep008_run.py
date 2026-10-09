"""Immutable phase execution and artifact gates for the EP008 experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from trading_ai.experiments.ep008_data import (
    COLD_END,
    COLD_START,
    DEVELOPMENT_END,
    FEATURE_COLUMNS,
    build_features,
    build_targets,
    cold_ids,
    ids_sha256,
    load_ecb_zip,
)
from trading_ai.experiments.ep008_development import (
    BASELINE_ORDER,
    TASK_SPECS,
    development_samples,
    fit_selected_models,
    run_development,
)
from trading_ai.experiments.ep008_metrics import (
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    VARIANCE_FLOOR_BPS2,
    movement_baselines,
    moving_block_ci,
    primary_loss_vector,
    quantile_baselines,
    score_movement,
    score_quantiles,
    score_return,
)
from trading_ai.experiments.ep008_models import predict_model


ARCHIVE_RELATIVE_PATH = Path("data/raw/ecb_eurofxref_hist/eurofxref-hist.zip")
CONFIG_RELATIVE_PATH = Path("configs/experiments/ep008_joint_five_observation_predictability.toml")
EXPERIMENT_RELATIVE_PATH = Path("experiments/ep008_joint_five_observation_predictability")
SOURCE_FILES = (
    "configs/experiments/ep008_joint_five_observation_predictability.toml",
    "src/trading_ai/experiments/ep008_data.py",
    "src/trading_ai/experiments/ep008_metrics.py",
    "src/trading_ai/experiments/ep008_models.py",
    "src/trading_ai/experiments/ep008_development.py",
    "src/trading_ai/experiments/ep008_run.py",
    "experiments/ep008_joint_five_observation_predictability/run.py",
    "tests/experiments/test_ep008_data.py",
    "tests/experiments/test_ep008_metrics.py",
    "tests/experiments/test_ep008_models.py",
    "tests/experiments/test_ep008_development.py",
    "tests/experiments/test_ep008_run.py",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files_sha256(repo_root: str | Path) -> str:
    root = Path(repo_root)
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_bytes_exclusive(path: str | Path, payload: bytes) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def write_json_exclusive(path: str | Path, payload: Any) -> str:
    return write_bytes_exclusive(path, _json_bytes(payload))


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def validate_prediction_gate(
    run_dir: str | Path,
    *,
    source_files_sha256: str,
    source_archive_sha256: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Verify approved freeze, source/data identities, review, and common IDs."""
    directory = Path(run_dir)
    freeze_path = directory / "freeze.json"
    ids_path = directory / "cold_ids.csv"
    review_path = directory / "method_review.json"
    for path in (freeze_path, ids_path, review_path):
        if not path.is_file():
            raise ValueError(f"prediction is blocked: required freeze/method review artifact missing: {path.name}")
    freeze = read_json(freeze_path)
    if freeze.get("source_files_sha256") != source_files_sha256:
        raise ValueError("frozen source files hash differs from current implementation")
    if freeze.get("source_archive_sha256") != source_archive_sha256:
        raise ValueError("frozen archive SHA-256 differs from approved ECB snapshot")
    if file_sha256(ids_path) != freeze.get("cold_ids_file_sha256"):
        raise ValueError("frozen cold ID file hash mismatch")
    ids = pd.read_csv(ids_path, dtype={"sample_id": str})
    required = {"sample_id", "decision_date", "endpoint_date"}
    if not required.issubset(ids.columns):
        raise ValueError("frozen cold ID file is missing required columns")
    ids["decision_date"] = pd.to_datetime(ids["decision_date"], format="%Y-%m-%d", errors="raise").dt.date
    ids["endpoint_date"] = pd.to_datetime(ids["endpoint_date"], format="%Y-%m-%d", errors="raise").dt.date
    if ids["sample_id"].duplicated().any() or len(ids) != int(freeze.get("cold_row_count", -1)):
        raise ValueError("frozen cold ID rows are duplicated or have the wrong count")
    if ids_sha256(ids) != freeze.get("cold_ids_sha256"):
        raise ValueError("frozen decision|endpoint ID hash mismatch")
    if not ids["decision_date"].between(COLD_START, COLD_END).all():
        raise ValueError("frozen cold decisions fall outside the approved date window")
    if not ids["endpoint_date"].between(COLD_START, COLD_END).all():
        raise ValueError("frozen cold endpoints fall outside the approved date window")
    if not (ids["endpoint_date"] > ids["decision_date"]).all():
        raise ValueError("cold endpoints must be later than their decisions")
    review = read_json(review_path)
    freeze_sha = file_sha256(freeze_path)
    if review.get("status") != "approved" or review.get("freeze_sha256") != freeze_sha:
        raise ValueError("independent method review is missing, not approved, or refers to another freeze")
    if review.get("source_files_sha256") != source_files_sha256:
        raise ValueError("method review refers to a different source version")
    return freeze, ids


def begin_score_once(run_dir: str | Path) -> Path:
    """Write an irreversible local start marker before loading cold targets."""
    directory = Path(run_dir)
    marker = directory / "cold_score_started.json"
    if marker.exists() or (directory / "cold_scores.json").exists() or (directory / "cold_score_completed.json").exists():
        raise FileExistsError("cold scoring has already started; it is allowed once only")
    freeze_path = directory / "freeze.json"
    predictions_path = directory / "cold_predictions.csv"
    manifest_path = directory / "prediction_manifest.json"
    if not all(path.is_file() for path in (freeze_path, predictions_path, manifest_path)):
        raise ValueError("cold scoring requires a committed freeze and prediction artifact")
    manifest = read_json(manifest_path)
    if file_sha256(predictions_path) != manifest.get("predictions_sha256"):
        raise ValueError("prediction hash mismatch; refusing cold scoring")
    if file_sha256(freeze_path) != manifest.get("freeze_sha256"):
        raise ValueError("prediction manifest refers to a different freeze")
    write_json_exclusive(
        marker,
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "predictions_sha256": manifest["predictions_sha256"],
            "freeze_sha256": manifest["freeze_sha256"],
            "same_row_cold_evaluation_targets_read": False,
        },
    )
    return marker


def _repo_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _dependency_fingerprint() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _run_dir(repo_root: Path, run_id: str) -> Path:
    if not run_id.startswith("ep008-") or "/" in run_id or "\\" in run_id:
        raise ValueError("invalid EP008 run ID")
    return repo_root / EXPERIMENT_RELATIVE_PATH / "runs" / run_id


def _approved_archive(repo_root: Path) -> tuple[Path, str, pd.DataFrame]:
    archive = repo_root / ARCHIVE_RELATIVE_PATH
    actual_sha = file_sha256(archive)
    expected_sha = "e39ffa4c4e8cf3207f2b2589ce5ed5aea56253b35e33c66928d1f7ed76a9d941"
    if actual_sha != expected_sha:
        raise ValueError("ECB archive SHA-256 differs from the approved snapshot")
    return archive, actual_sha, load_ecb_zip(archive)


def _compact_selection(selection: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for task, task_selection in selection.items():
        compact[task] = {}
        for role, chosen in task_selection.items():
            compact[task][role] = {
                "config_id": chosen["config_id"],
                "grid_index": chosen["grid_index"],
                "config": chosen["config"],
                "primary_metric": chosen["primary_metric"],
                "macro_primary": chosen["macro_primary"],
                "macro_secondary": chosen["macro_secondary"],
                "macro_scores": chosen["macro_scores"],
            }
    return compact


def _assert_committed_source(repo_root: Path) -> tuple[str, str]:
    branch = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if branch != "ep008-joint-five-observation":
        raise ValueError(f"EP008 must run from its dedicated branch, got {branch!r}")
    for relative in SOURCE_FILES:
        subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", relative],
            check=True,
            capture_output=True,
            text=True,
        )
    diff = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--quiet", "HEAD", "--", *SOURCE_FILES],
        check=False,
    )
    if diff.returncode != 0:
        raise ValueError("EP008 source/config/test files differ from committed source")
    source_commit = _repo_commit(repo_root)
    return source_commit, source_files_sha256(repo_root)


def _assert_committed_artifacts(repo_root: Path, paths: tuple[Path, ...]) -> None:
    """Require phase inputs to exist in HEAD and match the committed bytes."""
    for path in paths:
        relative = str(path.relative_to(repo_root))
        subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", relative],
            check=True,
            capture_output=True,
            text=True,
        )
        changed = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--quiet", "HEAD", "--", relative],
            check=False,
        )
        if changed.returncode != 0:
            raise ValueError(f"required phase artifact is not committed or differs from HEAD: {relative}")


def _config_sha256(repo_root: Path) -> str:
    return file_sha256(repo_root / CONFIG_RELATIVE_PATH)


def run_validation_phase(repo_root: str | Path) -> str:
    """Run and preserve development-only fits/scores; does not build cold targets."""
    root = Path(repo_root).resolve()
    source_commit, source_hash = _assert_committed_source(root)
    archive_path, archive_hash, rates = _approved_archive(root)
    config_hash = _config_sha256(root)
    dependencies = _dependency_fingerprint()
    identity = {
        "source_commit": source_commit,
        "source_files_sha256": source_hash,
        "config_sha256": config_hash,
        "data_sha256": archive_hash,
        "dependencies": dependencies,
        "seed": BOOTSTRAP_SEED,
    }
    run_id = f"ep008-{hashlib.sha256(_json_bytes(identity)).hexdigest()[:24]}"
    directory = _run_dir(root, run_id)
    if directory.exists():
        raise FileExistsError(f"run directory already exists: {directory}")
    development = run_development(rates)
    if development.get("cold_scores_computed") or development.get("cold_target_values_read"):
        raise ValueError("development phase must not compute or read cold target values/scores")
    directory.mkdir(parents=True, exist_ok=False)
    development_sha = write_json_exclusive(directory / "development_scores.json", development)
    manifest = {
        **identity,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(archive_path.relative_to(root)),
        "archive_member": "eurofxref-hist.csv",
        "archive_bytes": int(archive_path.stat().st_size),
        "development_scores_sha256": development_sha,
        "development_rows": int(development["eligible_development_rows"]),
        "development_phase_same_row_cold_evaluation_targets_read": False,
        "cold_scores_computed": False,
    }
    write_json_exclusive(directory / "run_manifest.json", manifest)
    return run_id


def freeze_phase(repo_root: str | Path, run_id: str) -> dict[str, Any]:
    """Freeze selected development choices and the exact date-only cold ID list."""
    root = Path(repo_root).resolve()
    directory = _run_dir(root, run_id)
    manifest = read_json(directory / "run_manifest.json")
    development_path = directory / "development_scores.json"
    if file_sha256(development_path) != manifest.get("development_scores_sha256"):
        raise ValueError("development score artifact hash differs from run manifest")
    source_commit, source_hash = _assert_committed_source(root)
    if source_hash != manifest.get("source_files_sha256"):
        raise ValueError("implementation source files changed after development run")
    archive_path, archive_hash, rates = _approved_archive(root)
    if archive_hash != manifest.get("data_sha256"):
        raise ValueError("source archive changed after development run")
    features = build_features(rates)
    ids = cold_ids(rates, features)
    if len(ids) != 186:
        raise ValueError(f"approved cold date window should yield 186 shared pairs, got {len(ids)}")
    if not (ids["endpoint_date"] <= COLD_END).all() or not (ids["decision_date"] >= COLD_START).all():
        raise ValueError("cold ID list contains an out-of-window endpoint or decision")
    ids_bytes = ids.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ids_csv_sha = hashlib.sha256(ids_bytes).hexdigest()
    development = read_json(development_path)
    selection = _compact_selection(development["selection"])
    freeze = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": manifest["source_commit"],
        "source_files_sha256": source_hash,
        "source_archive_sha256": archive_hash,
        "source_archive_path": str(archive_path.relative_to(root)),
        "config_sha256": manifest["config_sha256"],
        "development_scores_sha256": manifest["development_scores_sha256"],
        "development_range": ["2000-01-01", "2025-12-31"],
        "eligible_development_rows": int(development["eligible_development_rows"]),
        "selected_configurations_and_baselines": selection,
        "selection": selection,
        "cold_window": [COLD_START.isoformat(), COLD_END.isoformat()],
        "cold_row_count": int(len(ids)),
        "cold_ids_sha256": ids_sha256(ids),
        "cold_ids_file_sha256": ids_csv_sha,
        "features": list(FEATURE_COLUMNS),
        "bootstrap": {
            "type": "non-circular moving-block paired loss difference",
            "block_length": BOOTSTRAP_BLOCK_LENGTH,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "movement_forecast_floor_bps2": VARIANCE_FLOOR_BPS2,
        "same_row_cold_evaluation_targets_read": False,
        "prior_matured_cold_targets_used_by_rolling_baseline": False,
        "cold_scores_computed": False,
        "predictions_must_be_saved_before_cold_scoring": True,
    }
    write_bytes_exclusive(directory / "cold_ids.csv", ids_bytes)
    write_json_exclusive(directory / "freeze.json", freeze)
    freeze_sha = file_sha256(directory / "freeze.json")
    write_bytes_exclusive(directory / "freeze.sha256", (freeze_sha + "\n").encode("ascii"))
    return {"run_id": run_id, "cold_row_count": len(ids), "cold_ids_sha256": ids_sha256(ids), "freeze_sha256": freeze_sha}


def _add_quantile_prediction_columns(
    output: pd.DataFrame,
    name: str,
    predictions: np.ndarray,
) -> None:
    raw = np.asarray(predictions, dtype=float)
    if raw.shape != (len(output), 3) or not np.isfinite(raw).all():
        raise ValueError(f"{name} quantile predictions have the wrong shape or non-finite values")
    ordered = np.sort(raw, axis=1)
    for index, quantile in enumerate(("q05", "q50", "q95")):
        output[f"quantiles_{name}_raw_{quantile}_bps"] = raw[:, index]
        output[f"quantiles_{name}_{quantile}_bps"] = ordered[:, index]


def predict_cold_phase(repo_root: str | Path, run_id: str) -> dict[str, Any]:
    """Refit selected development models and persist predictions without cold labels."""
    root = Path(repo_root).resolve()
    directory = _run_dir(root, run_id)
    source_commit, source_hash = _assert_committed_source(root)
    archive_path, archive_hash, rates = _approved_archive(root)
    if (directory / "cold_predictions.csv").exists() or (directory / "prediction_manifest.json").exists():
        raise FileExistsError("cold predictions already exist; predictions are immutable")
    freeze, ids = validate_prediction_gate(
        directory,
        source_files_sha256=source_hash,
        source_archive_sha256=archive_hash,
    )
    _assert_committed_artifacts(root, (directory / "freeze.json", directory / "cold_ids.csv"))
    if source_hash != freeze.get("source_files_sha256"):
        raise ValueError("current source files differ from the source frozen for this run")
    features = build_features(rates).set_index("decision_date")
    dates = ids["decision_date"].tolist()
    x_cold = features.loc[dates, list(FEATURE_COLUMNS)]
    if len(x_cold) != len(ids) or not np.isfinite(x_cold.to_numpy(dtype=float)).all():
        raise ValueError("cold feature rows do not align with the frozen IDs")
    refit = fit_selected_models(rates, freeze["selection"])
    output = ids.copy()

    return_models = refit["models"]["return"]
    return_baselines = refit["return_baselines"]
    output["return_zero_bps"] = np.zeros(len(ids), dtype=float)
    output["return_train_mean_bps"] = np.full(len(ids), return_baselines["train_mean"])
    output["return_train_median_bps"] = np.full(len(ids), return_baselines["train_median"])
    for family in ("linear", "boosting"):
        output[f"return_{family}_bps"] = predict_model(return_models[family], "return", x_cold)

    movement = movement_baselines(rates, dates).set_index("decision_date").loc[dates]
    movement_raw: dict[str, np.ndarray] = {
        "trailing5": movement["trailing5_bps2"].to_numpy(dtype=float),
        "ewma": movement["ewma_bps2"].to_numpy(dtype=float),
    }
    for family in ("linear", "boosting"):
        movement_raw[family] = predict_model(refit["models"]["movement"][family], "movement", x_cold)
    for name, raw in movement_raw.items():
        output[f"movement_{name}_raw_bps2"] = raw
        output[f"movement_{name}_bps2"] = np.maximum(raw, VARIANCE_FLOOR_BPS2)

    quantile_frame = quantile_baselines(rates, dates).set_index("decision_date").loc[dates]
    quantile_raw: dict[str, np.ndarray] = {
        "rolling_empirical": quantile_frame[["rolling_q05_bps", "rolling_q50_bps", "rolling_q95_bps"]].to_numpy(dtype=float),
        "ewma_scaled_empirical": quantile_frame[["ewma_scaled_q5_bps", "ewma_scaled_q50_bps", "ewma_scaled_q95_bps"]].to_numpy(dtype=float),
    }
    for family in ("linear", "boosting"):
        quantile_raw[family] = predict_model(refit["models"]["quantiles"][family], "quantiles", x_cold)
    for name, raw in quantile_raw.items():
        _add_quantile_prediction_columns(output, name, raw)

    if len(output) != 186 or output["sample_id"].tolist() != ids["sample_id"].tolist():
        raise ValueError("saved prediction rows do not match the exact frozen 186 cold IDs")
    prediction_bytes = output.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    predictions_sha = write_bytes_exclusive(directory / "cold_predictions.csv", prediction_bytes)
    freeze_sha = file_sha256(directory / "freeze.json")
    train_id_hash = hashlib.sha256(
        "".join(value + "\n" for value in refit["training_sample_ids"]).encode("utf-8")
    ).hexdigest()
    prediction_manifest = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": freeze["source_commit"],
        "prediction_commit": source_commit,
        "source_files_sha256": source_hash,
        "source_archive_sha256": archive_hash,
        "freeze_sha256": freeze_sha,
        "cold_ids_sha256": freeze["cold_ids_sha256"],
        "predictions_sha256": predictions_sha,
        "prediction_bytes": len(prediction_bytes),
        "cold_row_count": len(output),
        "development_training_row_count": refit["training_n"],
        "development_training_ids_sha256": train_id_hash,
        "selected_configurations_and_baselines": freeze["selection"],
        "same_row_cold_evaluation_targets_read": False,
        "prior_matured_cold_targets_used_by_rolling_baseline": True,
        "unmatured_cold_targets_used": False,
        "cold_scores_computed": False,
    }
    write_json_exclusive(directory / "prediction_manifest.json", prediction_manifest)
    return {
        "run_id": run_id,
        "cold_row_count": len(output),
        "predictions_sha256": predictions_sha,
        "freeze_sha256": freeze_sha,
        "same_row_cold_evaluation_targets_read": False,
        "predictions_path": str((directory / "cold_predictions.csv").relative_to(root)),
    }


def _prediction_values(frame: pd.DataFrame, task: str, name: str) -> np.ndarray:
    if task == "return":
        column = f"return_{name}_bps"
        return frame[column].to_numpy(dtype=float)
    if task == "movement":
        column = f"movement_{name}_raw_bps2"
        return frame[column].to_numpy(dtype=float)
    columns = [f"quantiles_{name}_raw_{q}_bps" for q in ("q05", "q50", "q95")]
    return frame[columns].to_numpy(dtype=float)


def _score_task(
    task: str,
    actual: np.ndarray,
    predictions: pd.DataFrame,
    frozen_selection: dict[str, Any],
) -> dict[str, Any]:
    baseline_id = frozen_selection["primary_baseline"]["config_id"]
    baseline_scores: dict[str, Any] = {}
    for name in BASELINE_ORDER[task]:
        predicted = _prediction_values(predictions, task, name)
        if task == "return":
            metrics = score_return(actual, predicted)
        elif task == "movement":
            metrics = score_movement(actual, predicted)
        else:
            metrics = score_quantiles(actual, predicted)
            metrics.pop("sorted_predictions_bps")
        baseline_scores[name] = metrics
    best_baseline = _prediction_values(predictions, task, baseline_id)
    baseline_loss = primary_loss_vector(task, actual, best_baseline)
    selected_scores: dict[str, Any] = {}
    for family in ("linear", "boosting"):
        predicted = _prediction_values(predictions, task, family)
        if task == "return":
            metrics = score_return(actual, predicted)
        elif task == "movement":
            metrics = score_movement(actual, predicted)
        else:
            metrics = score_quantiles(actual, predicted)
            metrics.pop("sorted_predictions_bps")
        model_loss = primary_loss_vector(task, actual, predicted)
        ci = moving_block_ci(model_loss, baseline_loss)
        selected_scores[family] = {
            "config_id": frozen_selection[family]["config_id"],
            "config": frozen_selection[family]["config"],
            "metrics": metrics,
            "ci_loss_difference_vs_frozen_best_baseline": ci,
        }
    return {
        "n": int(len(actual)),
        "primary_metric": TASK_SPECS[task]["primary"],
        "frozen_best_baseline": baseline_id,
        "baseline_scores": baseline_scores,
        "selected_model_scores": selected_scores,
        "ci_definition": "model row-level primary loss minus frozen development-best baseline row-level primary loss",
    }


def score_cold_phase(repo_root: str | Path, run_id: str) -> dict[str, Any]:
    """Read cold labels and score exactly once after freeze, review, tests, and predictions."""
    root = Path(repo_root).resolve()
    directory = _run_dir(root, run_id)
    source_commit, source_hash = _assert_committed_source(root)
    archive_path, archive_hash, rates = _approved_archive(root)
    freeze, ids = validate_prediction_gate(
        directory,
        source_files_sha256=source_hash,
        source_archive_sha256=archive_hash,
    )
    _assert_committed_artifacts(
        root,
        (
            directory / "freeze.json",
            directory / "cold_ids.csv",
            directory / "method_review.json",
            directory / "cold_predictions.csv",
            directory / "prediction_manifest.json",
        ),
    )
    if source_hash != freeze.get("source_files_sha256"):
        raise ValueError("current source files differ from the reviewed/frozen implementation")
    predictions_path = directory / "cold_predictions.csv"
    if not predictions_path.is_file():
        raise ValueError("cold predictions must be saved and committed before scoring")
    prediction_frame = pd.read_csv(predictions_path, dtype={"sample_id": str})
    if prediction_frame["sample_id"].tolist() != ids["sample_id"].tolist():
        raise ValueError("prediction sample IDs differ from the frozen 186 cold IDs")
    begin_score_once(directory)

    # This is the first cold-target construction; a start marker already exists.
    targets = build_targets(rates, ids["decision_date"].tolist())
    if targets["sample_id"].tolist() != ids["sample_id"].tolist():
        raise ValueError("cold target IDs differ from the frozen ID list")
    joined = ids.merge(targets, on=["sample_id", "decision_date", "endpoint_date"], how="inner", validate="one_to_one")
    if len(joined) != len(ids):
        raise ValueError("cold target join changed the frozen row count")
    predictions = prediction_frame
    actual_return = joined["cumulative_return_bps"].to_numpy(dtype=float)
    actual_movement = joined["movement_target_bps2"].to_numpy(dtype=float)
    task_actuals = {"return": actual_return, "movement": actual_movement, "quantiles": actual_return}
    tasks: dict[str, Any] = {}
    for task in TASK_SPECS:
        tasks[task] = _score_task(task, task_actuals[task], predictions, freeze["selection"][task])

    result = {
        "run_id": run_id,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": freeze["source_commit"],
        "scoring_commit": source_commit,
        "source_files_sha256": source_hash,
        "source_archive_sha256": archive_hash,
        "freeze_sha256": file_sha256(directory / "freeze.json"),
        "predictions_sha256": file_sha256(predictions_path),
        "cold_window": [COLD_START.isoformat(), COLD_END.isoformat()],
        "cold_row_count": len(joined),
        "cold_decision_min": min(joined["decision_date"]).isoformat(),
        "cold_decision_max": max(joined["decision_date"]).isoformat(),
        "cold_endpoint_max": max(joined["endpoint_date"]).isoformat(),
        "tasks": tasks,
        "bootstrap": {
            "type": "non-circular moving-block paired loss difference",
            "block_length": BOOTSTRAP_BLOCK_LENGTH,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "same_row_cold_evaluation_targets_read": True,
        "prior_matured_cold_targets_used_by_rolling_baseline": True,
        "score_run_count": 1,
        "interpretation": "informational ECB reference rates; educational ML comparison only; no executable prices, P&L, or profit claim",
    }
    scores_sha = write_json_exclusive(directory / "cold_scores.json", result)
    write_json_exclusive(
        directory / "cold_score_completed.json",
        {
            "completed_at_utc": result["scored_at_utc"],
            "cold_scores_sha256": scores_sha,
            "cold_row_count": len(joined),
            "score_run_count": 1,
        },
    )
    return {
        "run_id": run_id,
        "cold_row_count": len(joined),
        "cold_scores_sha256": scores_sha,
        "scores_path": str((directory / "cold_scores.json").relative_to(root)),
    }
