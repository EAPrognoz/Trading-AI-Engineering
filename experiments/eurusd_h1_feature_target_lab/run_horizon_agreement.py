#!/usr/bin/env python3
"""Run the frozen W16/core H4-H8-H24 exploratory outer agreement track."""
from __future__ import annotations

import csv
import argparse
import gzip
import hashlib
import json
import os
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_RUN = REPO / "experiments/eurusd_h1_feature_target_selection_v3/results/eurusd_h1_feature_target_v3_20261022T151237Z"
PROTOCOL_PATH = HERE / "protocol_frozen_pre_fit.json"
HORIZONS = (4, 8, 24)
FILTER_NAMES = ("H8_only", "agreement_H4_H8", "agreement_H8_H24", "unanimous_H4_H8_H24")
CORE_GROUPS = {"own_ohlc_log_ratios", "same_bar_cross_ratios", "observed_transition_gap"}
FOLDS = {
    "F1": ("2025-07-01T00:00:00Z", "2025-10-01T00:00:00Z"),
    "F2": ("2025-10-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    "F3": ("2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z"),
    "F4": ("2026-04-01T00:00:00Z", "2026-07-04T00:00:00Z"),
}
CPU_LIMIT_SECONDS = 900.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def fmt_num(value: Any, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{places}f}"


def balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    labels = set(int(x) for x in actual)
    if labels != {-1, 1}:
        return None
    recalls = []
    for label in (-1, 1):
        mask = actual == label
        recalls.append(float(np.mean(predicted[mask] == label)))
    return float(np.mean(recalls))


def score_direction(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Score nonzero forecasts against nonzero realized H8 directions only."""
    active = (predicted != 0) & (actual != 0)
    n = int(active.sum())
    if n == 0:
        return {"evaluation_n": 0, "hits": 0, "accuracy": None, "balanced_accuracy": None}
    a, p = actual[active], predicted[active]
    return {
        "evaluation_n": n,
        "hits": int((a == p).sum()),
        "accuracy": float(np.mean(a == p)),
        "balanced_accuracy": balanced_accuracy(a, p),
    }


def build_filter_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    s4 = frame["pred_sign_H4"].to_numpy(dtype=np.int8)
    s8 = frame["pred_sign_H8"].to_numpy(dtype=np.int8)
    s24 = frame["pred_sign_H24"].to_numpy(dtype=np.int8)
    return {
        "H8_only": s8 != 0,
        "agreement_H4_H8": (s4 != 0) & (s8 != 0) & (s4 == s8),
        "agreement_H8_H24": (s8 != 0) & (s24 != 0) & (s8 == s24),
        "unanimous_H4_H8_H24": (s4 != 0) & (s8 != 0) & (s24 != 0) & (s4 == s8) & (s8 == s24),
    }


def metrics_row(
    fold_label: str,
    filter_name: str,
    cohort_name: str,
    frame: pd.DataFrame,
    accepted_mask: np.ndarray,
    majority_by_fold: dict[str, int],
) -> dict[str, Any]:
    cohort_mask = accepted_mask if cohort_name == "accepted" else ~accepted_mask
    subset = frame.loc[cohort_mask]
    y = subset["actual_sign_H8"].to_numpy(dtype=np.int8)
    pred = subset["pred_sign_H8"].to_numpy(dtype=np.int8)
    m = len(frame)
    score = score_direction(y, pred)
    h8_forecast_zero = int((pred == 0).sum())
    h8_actual_zero = int((y == 0).sum())
    eval_mask = (pred != 0) & (y != 0)
    weekday = Counter(pd.to_datetime(subset["decision_timestamp"], utc=True).dt.weekday.astype(int))
    gap = subset["H8_target_non_one_hour_transition_count"].to_numpy(dtype=int) > 0
    h8_buy = int((pred == 1).sum())
    h8_sell = int((pred == -1).sum())

    majority_preds = np.asarray([majority_by_fold.get(str(f), 0) for f in subset["outer_fold"]], dtype=np.int8)
    majority_score = score_direction(y, majority_preds)
    always_up = np.ones(len(y), dtype=np.int8)
    always_up_score = score_direction(y, always_up)

    out = {
        "outer_fold": fold_label,
        "filter": filter_name,
        "cohort": cohort_name,
        "joint_validation_origins": m,
        "cohort_origins": len(subset),
        "cohort_coverage": (len(subset) / m) if m else None,
        "filter_accepted_origins": int(accepted_mask.sum()),
        "filter_coverage": (float(accepted_mask.mean()) if m else None),
        "h8_pred_buy_count": h8_buy,
        "h8_pred_sell_count": h8_sell,
        "h8_forecast_zero_count": h8_forecast_zero,
        "h8_actual_zero_count": h8_actual_zero,
        "h8_actual_positive_count": int((y == 1).sum()),
        "h8_actual_negative_count": int((y == -1).sum()),
        "h8_evaluation_n": score["evaluation_n"],
        "h8_hit_count": score["hits"],
        "h8_accuracy": score["accuracy"],
        "h8_balanced_accuracy": score["balanced_accuracy"],
        "h8_gap_crossing_count": int(gap.sum()),
        "h8_gap_crossing_fraction": float(gap.mean()) if len(gap) else None,
        "decision_weekday_mon": int(weekday.get(0, 0)),
        "decision_weekday_tue": int(weekday.get(1, 0)),
        "decision_weekday_wed": int(weekday.get(2, 0)),
        "decision_weekday_thu": int(weekday.get(3, 0)),
        "decision_weekday_fri": int(weekday.get(4, 0)),
        "decision_weekday_sat": int(weekday.get(5, 0)),
        "decision_weekday_sun": int(weekday.get(6, 0)),
        "majority_baseline_eval_n": majority_score["evaluation_n"],
        "majority_baseline_hit_count": majority_score["hits"],
        "majority_baseline_accuracy": majority_score["accuracy"],
        "majority_baseline_balanced_accuracy": majority_score["balanced_accuracy"],
        "always_up_baseline_eval_n": always_up_score["evaluation_n"],
        "always_up_baseline_hit_count": always_up_score["hits"],
        "always_up_baseline_accuracy": always_up_score["accuracy"],
        "always_up_baseline_balanced_accuracy": always_up_score["balanced_accuracy"],
    }
    # A training-majority/always-up baseline is requested only on accepted cohorts.
    if cohort_name != "accepted":
        for key in list(out):
            if key.startswith("majority_baseline_") or key.startswith("always_up_baseline_"):
                out[key] = None
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global SOURCE_RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN,
                        help="completed v3 run directory containing pinned feature/target/split artifacts")
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    args = parser.parse_args()
    SOURCE_RUN = args.source_run.resolve()
    output_root = args.output_root.resolve()
    started = time.perf_counter()
    cpu_started = time.process_time()
    protocol_hash = sha256_file(PROTOCOL_PATH)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_pre_fit":
        raise RuntimeError("Protocol is not frozen; refusing to fit.")

    source_summary = json.loads((SOURCE_RUN / "run_summary.json").read_text(encoding="utf-8"))
    source_protocol = json.loads((SOURCE_RUN / "protocol_frozen_pre_fit.json").read_text(encoding="utf-8"))
    expected_prefix_sha = protocol["source_and_scope"]["source_prefix_sha256"]
    if source_summary["prefix_sha256_verified"] != expected_prefix_sha:
        raise RuntimeError("Source prefix SHA does not match the frozen agreement protocol.")
    if source_summary["read_rows"] != 10621 or source_summary["test_rows_read"] != 0:
        raise RuntimeError("The referenced source prefix metadata differs from the frozen protocol.")
    if source_protocol["feature_schema_sha256"] != protocol["source_and_scope"]["feature_schema_sha256"]:
        raise RuntimeError("Feature schema SHA does not match the frozen agreement protocol.")

    catalog = json.loads((SOURCE_RUN / "feature_catalog.json").read_text(encoding="utf-8"))
    names = catalog["names_by_W"]["16"]
    schema = catalog["schema"]["16"]
    feature_indices = np.asarray([i for i, item in enumerate(schema) if item["group"] in CORE_GROUPS], dtype=int)
    if len(feature_indices) != protocol["fixed_model"]["input_feature_count"]:
        raise RuntimeError("Core feature count differs from the frozen protocol.")
    if [names[i] for i in feature_indices] != [item["feature_name"] for item in schema if item["group"] in CORE_GROUPS]:
        raise RuntimeError("Core feature catalog ordering is inconsistent.")

    with np.load(SOURCE_RUN / "feature_matrices.npz", allow_pickle=False) as z:
        X_all = np.asarray(z["X_W16"], dtype=np.float64)
        valid = np.asarray(z["valid_W16"], dtype=bool)
    n_rows = len(X_all)
    common = (np.arange(n_rows) >= 96) & (np.arange(n_rows) + 24 < n_rows) & valid
    common_rows = set(np.flatnonzero(common).tolist())
    if len(common_rows) != 10501:
        raise RuntimeError(f"Common W96/H24 cohort should contain 10501 origins, got {len(common_rows)}.")
    X_core = X_all[:, feature_indices]
    if not np.isfinite(X_core[list(common_rows)]).all():
        raise RuntimeError("Non-finite W16/core features appear in the common cohort.")

    targets: dict[tuple[int, int], dict[str, Any]] = {}
    with gzip.open(SOURCE_RUN / "target_manifest.csv.gz", "rt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = int(row["H"])
            if h in HORIZONS:
                origin = int(row["origin_row"])
                target_is_valid = row["target_valid_next_observed_H"] == "True"
                target_value = float(row["y_mean_log_return_per_observed_transition"]) if target_is_valid else None
                targets[(origin, h)] = {
                    "origin_bar_open_timestamp": row["origin_bar_open_timestamp"],
                    "decision_timestamp": row["decision_timestamp"],
                    "target_timestamp": row["target_timestamp"],
                    "y": target_value,
                    "valid": target_is_valid,
                    "gap_count": int(row["target_non_one_hour_transition_count"]),
                    "crosses_gap": row["target_crosses_non_one_hour_transition"] == "True",
                }
    for origin in common_rows:
        for h in HORIZONS:
            rec = targets.get((origin, h))
            if rec is None or not rec["valid"] or rec["y"] is None or not np.isfinite(rec["y"]):
                raise RuntimeError(f"Missing or invalid H{h} target at common origin {origin}.")

    split_rows = read_rows(SOURCE_RUN / "split_membership.csv.gz")
    memberships: dict[str, dict[str, list[dict[str, str]]]] = {
        fold: {"train": [], "validation": []} for fold in FOLDS
    }
    for row in split_rows:
        if row["split_id"] != "outer" or row["outer_fold"] not in memberships:
            continue
        if row["partition"] not in ("train", "validation"):
            continue
        memberships[row["outer_fold"]][row["partition"]].append(row)

    fold_membership_summary: dict[str, dict[str, int]] = {}
    for fold, (start_text, end_text) in FOLDS.items():
        sets = memberships[fold]
        train_rows = [int(r["origin_row"]) for r in sets["train"]]
        val_rows = [int(r["origin_row"]) for r in sets["validation"]]
        if len(set(train_rows)) != len(train_rows) or len(set(val_rows)) != len(val_rows):
            raise RuntimeError(f"Duplicate {fold} origin in saved split membership.")
        if not set(train_rows + val_rows).issubset(common_rows):
            raise RuntimeError(f"{fold} split includes origins outside the frozen common cohort.")
        start = pd.Timestamp(start_text).tz_convert("UTC")
        end = pd.Timestamp(end_text).tz_convert("UTC")
        for row in sets["train"]:
            decision = pd.Timestamp(row["decision_timestamp"]).tz_convert("UTC")
            target24 = pd.Timestamp(row["target24_timestamp"]).tz_convert("UTC")
            if not decision < start or not target24 < start:
                raise RuntimeError(f"Train target boundary leak in {fold} at origin {row['origin_row']}.")
        for row in sets["validation"]:
            decision = pd.Timestamp(row["decision_timestamp"]).tz_convert("UTC")
            target24 = pd.Timestamp(row["target24_timestamp"]).tz_convert("UTC")
            if not (start <= decision < end and target24 < end):
                raise RuntimeError(f"Validation target boundary leak in {fold} at origin {row['origin_row']}.")
        for part in ("train", "validation"):
            for row in sets[part]:
                origin = int(row["origin_row"])
                h24 = targets[(origin, 24)]
                if row["decision_timestamp"] != h24["decision_timestamp"] or row["target24_timestamp"] != h24["target_timestamp"]:
                    raise RuntimeError(f"Saved split and H24 manifest disagree in {fold} at origin {origin}.")
        fold_membership_summary[fold] = {"train": len(train_rows), "validation": len(val_rows)}

    run_id = "eurusd_h1_horizon_agreement_v1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    fit_records: list[dict[str, Any]] = []
    coefficient_arrays: dict[str, np.ndarray] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    majority_sign_by_fold: dict[str, int] = {}
    fit_id = 0

    for fold, (start_text, end_text) in FOLDS.items():
        train_idx = np.asarray(sorted(int(r["origin_row"]) for r in memberships[fold]["train"]), dtype=int)
        valid_idx = np.asarray(sorted(int(r["origin_row"]) for r in memberships[fold]["validation"]), dtype=int)
        y8_train = np.asarray([targets[(i, 8)]["y"] for i in train_idx], dtype=np.float64)
        y8_labels = np.asarray([sign(y) for y in y8_train], dtype=np.int8)
        counts = Counter(int(x) for x in y8_labels if x != 0)
        majority_sign = (1 if counts[1] > counts[-1] else (-1 if counts[-1] > counts[1] else 0))
        majority_sign_by_fold[fold] = majority_sign

        predictions: dict[int, np.ndarray] = {}
        for h in HORIZONS:
            y_train = np.asarray([targets[(i, h)]["y"] for i in train_idx], dtype=np.float64).reshape(-1, 1)
            x_scaler = StandardScaler().fit(X_core[train_idx])
            y_scaler = StandardScaler().fit(y_train)
            X_train = x_scaler.transform(X_core[train_idx])
            X_valid = x_scaler.transform(X_core[valid_idx])
            y_scaled = y_scaler.transform(y_train)
            model = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky")
            model.fit(X_train, y_scaled.ravel())
            pred_scaled = np.asarray(model.predict(X_valid), dtype=np.float64).reshape(-1, 1)
            pred_raw = y_scaler.inverse_transform(pred_scaled).ravel()
            if not np.isfinite(pred_raw).all():
                raise RuntimeError(f"Non-finite H{h} predictions in {fold}.")
            predictions[h] = pred_raw

            prefix = f"{fit_id:02d}_{fold}_H{h}"
            coefficient_arrays[f"{prefix}_coef"] = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
            coefficient_arrays[f"{prefix}_intercept"] = np.asarray(model.intercept_, dtype=np.float64).reshape(1)
            coefficient_arrays[f"{prefix}_x_mean"] = np.asarray(x_scaler.mean_, dtype=np.float64)
            coefficient_arrays[f"{prefix}_x_scale"] = np.asarray(x_scaler.scale_, dtype=np.float64)
            coefficient_arrays[f"{prefix}_y_mean"] = np.asarray(y_scaler.mean_, dtype=np.float64).reshape(-1)
            coefficient_arrays[f"{prefix}_y_scale"] = np.asarray(y_scaler.scale_, dtype=np.float64).reshape(-1)
            fit_records.append({
                "fit_id": fit_id,
                "outer_fold": fold,
                "H": h,
                "W": 16,
                "mask_id": "core",
                "n_features": len(feature_indices),
                "n_train": len(train_idx),
                "n_validation": len(valid_idx),
                "alpha": 1.0,
                "solver": "cholesky",
                "x_scaler_fit_rows": len(train_idx),
                "y_scaler_fit_rows": len(train_idx),
                "majority_train_H8_positive": counts.get(1, 0),
                "majority_train_H8_negative": counts.get(-1, 0),
                "majority_H8_prediction_sign": majority_sign,
                "train_origins_sha256": hashlib.sha256(train_idx.astype("<i8").tobytes()).hexdigest(),
                "validation_origins_sha256": hashlib.sha256(valid_idx.astype("<i8").tobytes()).hexdigest(),
            })
            fit_id += 1
            if time.process_time() - cpu_started > CPU_LIMIT_SECONDS:
                raise RuntimeError("CPU runtime cap reached during the fixed 12-fit budget; stop.")

        rows: list[dict[str, Any]] = []
        for j, origin in enumerate(valid_idx.tolist()):
            r8 = targets[(origin, 8)]
            rec: dict[str, Any] = {
                "outer_fold": fold,
                "origin_row": origin,
                "origin_bar_open_timestamp": r8["origin_bar_open_timestamp"],
                "decision_timestamp": r8["decision_timestamp"],
                "majority_baseline_sign": majority_sign,
                "H8_target_non_one_hour_transition_count": r8["gap_count"],
                "H8_target_crosses_non_one_hour_transition": r8["crosses_gap"],
            }
            for h in HORIZONS:
                t = targets[(origin, h)]
                rec[f"target_timestamp_H{h}"] = t["target_timestamp"]
                rec[f"y_mean_actual_H{h}"] = t["y"]
                rec[f"y_mean_prediction_H{h}"] = float(predictions[h][j])
                rec[f"pred_sign_H{h}"] = sign(float(predictions[h][j]))
                rec[f"actual_sign_H{h}"] = sign(t["y"])
                rec[f"target_non_one_hour_transition_count_H{h}"] = t["gap_count"]
                rec[f"target_crosses_non_one_hour_transition_H{h}"] = t["crosses_gap"]
            rows.append(rec)
        prediction_frames[fold] = pd.DataFrame(rows)

    if fit_id != 12:
        raise RuntimeError(f"Expected exactly 12 fits, completed {fit_id}.")
    all_predictions = pd.concat(prediction_frames.values(), ignore_index=True)
    all_predictions = all_predictions.sort_values(["outer_fold", "origin_row"], kind="stable").reset_index(drop=True)
    if all_predictions.duplicated(["outer_fold", "origin_row"]).any():
        raise RuntimeError("Duplicate outer prediction origin.")
    for fold, frame in prediction_frames.items():
        target24_times = pd.to_datetime(frame["target_timestamp_H24"], utc=True)
        if not (target24_times < pd.Timestamp(FOLDS[fold][1])).all():
            raise RuntimeError(f"H24 validation endpoint exceeds fold boundary for {fold}.")
        if not frame[["y_mean_prediction_H4", "y_mean_prediction_H8", "y_mean_prediction_H24"]].notna().all().all():
            raise RuntimeError(f"Incomplete horizon predictions for {fold}.")

    all_predictions["decision_timestamp"] = pd.to_datetime(all_predictions["decision_timestamp"], utc=True).map(lambda x: x.isoformat().replace("+00:00", "Z"))
    for h in HORIZONS:
        all_predictions[f"target_timestamp_H{h}"] = pd.to_datetime(all_predictions[f"target_timestamp_H{h}"], utc=True).map(lambda x: x.isoformat().replace("+00:00", "Z"))
    predictions_path = out_dir / "outer_predictions.csv"
    all_predictions.to_csv(predictions_path, index=False, float_format="%.17g")

    rows_metrics: list[dict[str, Any]] = []
    pooled = all_predictions.copy()
    pooled["outer_fold"] = pooled["outer_fold"].astype(str)
    for fold_label, frame in [(fold, prediction_frames[fold]) for fold in FOLDS] + [("ALL", pooled)]:
        y8_actual = frame["actual_sign_H8"].to_numpy(dtype=np.int8)
        masks = build_filter_masks(frame)
        for filter_name in FILTER_NAMES:
            accepted = masks[filter_name]
            rows_metrics.append(metrics_row(fold_label, filter_name, "accepted", frame, accepted, majority_sign_by_fold))
            rows_metrics.append(metrics_row(fold_label, filter_name, "rejected", frame, accepted, majority_sign_by_fold))
    metrics_path = out_dir / "agreement_metrics_by_fold.csv"
    write_csv(metrics_path, rows_metrics)

    examples: list[dict[str, Any]] = []
    pair_mask = build_filter_masks(all_predictions)["agreement_H4_H8"]
    for label, choose in (
        ("accepted_H4_H8", pair_mask),
        ("rejected_H4_H8_with_active_H8", (~pair_mask) & (all_predictions["pred_sign_H8"].to_numpy(dtype=np.int8) != 0)),
    ):
        candidates = all_predictions.loc[choose].sort_values(["decision_timestamp", "outer_fold"], kind="stable")
        if len(candidates):
            row = candidates.iloc[0]
            examples.append({
                "kind": label,
                "outer_fold": row["outer_fold"],
                "origin_row": int(row["origin_row"]),
                "decision_timestamp": row["decision_timestamp"],
                "target_timestamp_H8": row["target_timestamp_H8"],
                "prediction_signs_H4_H8_H24": [int(row[f"pred_sign_H{h}"]) for h in HORIZONS],
                "H8_prediction": float(row["y_mean_prediction_H8"]),
                "H8_actual": float(row["y_mean_actual_H8"]),
                "H8_actual_sign": int(row["actual_sign_H8"]),
                "H8_target_gap_crossing_audit": bool(row["H8_target_crosses_non_one_hour_transition"]),
            })
    (out_dir / "illustrative_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fit_path = out_dir / "fit_index.csv"
    write_csv(fit_path, fit_records)
    np.savez_compressed(out_dir / "model_coefficients_and_scalers.npz", **coefficient_arrays)

    # Human-readable report: all folds are included in the metrics CSV; this table
    # highlights accepted vs rejected H8 conditional accuracy and baseline.
    report = [
        "# EURUSD H1 horizon agreement — exploratory outer results",
        "",
        f"Run: `{run_id}`. This is an exploratory historical benchmark on a previously examined pre-test prefix, not an untouched confirmation period.",
        "",
        "## Protocol and model",
        "",
        "Frozen W16/core Ridge (`alpha=1`, intercept, Cholesky), with train-only X and target standardization. Three mean-only outputs were fit for H=4, 8, and 24 observed transitions in each of the four fixed outer folds (12 fits total). Training origins were identical across the three horizons within each fold and purged by the H24 endpoint. All outer validation origins had all three forecasts; all gaps were retained. No test rows after the development prefix were read.",
        "",
        "For each accepted subset, the directional prediction is still sign(H8 forecast); the agreement rule only filters origins. Therefore accepted-set H8 accuracy is the conditional accuracy of the same H8 model, not evidence that combining forecasts changed or improved the prediction. Agreement is correlated filtering, not independent confirmation. Exact-zero forecasts abstain and exact-zero H8 outcomes are counted separately and excluded from direction accuracy.",
        "",
        "## Accepted-set and rejected-set H8 results",
        "",
        "Coverage is accepted origins divided by all joint outer-validation origins in that fold. H8 accuracy uses rows with a nonzero H8 forecast and nonzero realized H8 target. Balanced accuracy is omitted when both realized direction classes are not present. Baselines are scored on the accepted subset only; majority is computed from nonzero training H8 labels per fold, with ties abstaining. Weekday and gap counts are descriptive metadata.",
        "",
        "| Fold | Filter | Accepted / joint | Coverage | Accepted H8 hit (n) | Rejected H8 hit (n) | Accepted majority | Accepted always-up | Accepted weekday Mon..Sun | Accepted H8 gap-crossings |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    metric_lookup = {(r["outer_fold"], r["filter"], r["cohort"]): r for r in rows_metrics}
    for fold in ["F1", "F2", "F3", "F4", "ALL"]:
        for filter_name in FILTER_NAMES:
            a = metric_lookup[(fold, filter_name, "accepted")]
            r = metric_lookup[(fold, filter_name, "rejected")]
            acc = f"{fmt_num(a['h8_accuracy'],3)} ({a['h8_hit_count']}/{a['h8_evaluation_n']})"
            rej = f"{fmt_num(r['h8_accuracy'],3)} ({r['h8_hit_count']}/{r['h8_evaluation_n']})"
            maj = f"{fmt_num(a['majority_baseline_accuracy'],3)} ({a['majority_baseline_hit_count']}/{a['majority_baseline_eval_n']})"
            up = f"{fmt_num(a['always_up_baseline_accuracy'],3)} ({a['always_up_baseline_hit_count']}/{a['always_up_baseline_eval_n']})"
            days = "/".join(str(a[f"decision_weekday_{d}"]) for d in ("mon","tue","wed","thu","fri","sat","sun"))
            gap_txt = f"{a['h8_gap_crossing_count']}/{a['cohort_origins']}"
            report.append(f"| {fold} | `{filter_name}` | {a['cohort_origins']}/{a['joint_validation_origins']} | {fmt_num(a['filter_coverage'],3)} | {acc} | {rej} | {maj} | {up} | {days} | {gap_txt} |")
    report.extend([
        "",
        "`H8_only` is the single-horizon reference; its accepted rows are exactly the origins where the H8 forecast is nonzero. H4/H8, H8/H24, and unanimous rules are separate abstention-aware agreement filters; no two-of-three majority rule was evaluated.",
        "",
        "## Outputs and limits",
        "",
        "- `outer_predictions.csv` contains one row per origin with separate horizon predictions, actual labels, and timestamps.",
        "- `agreement_metrics_by_fold.csv` contains accepted and rejected counts, coverage, direction hits, zero counts, baselines, weekday distribution, and H8 gap-crossing audits for every filter, fold, and pooled descriptive cohort.",
        "- `fit_index.csv` and `model_coefficients_and_scalers.npz` retain all 12 fits and train-only scaling parameters.",
        "- `illustrative_examples.json` contains real rows from the saved outer predictions; no timestamp was invented.",
        "- Overlapping horizons/origins are dependent; no IID significance test or profitability claim is made. Results are conditional on the fixed historical folds and filters.",
        "",
    ])
    (out_dir / "results_report.md").write_text("\n".join(report), encoding="utf-8")

    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - started
    if cpu_seconds > CPU_LIMIT_SECONDS:
        raise RuntimeError(f"CPU cap exceeded: {cpu_seconds:.3f}s")
    run_summary = {
        "run_id": run_id,
        "status": "complete",
        "protocol_file": str(PROTOCOL_PATH.relative_to(REPO)),
        "protocol_sha256": protocol_hash,
        "source_run_id": source_summary["run_id"],
        "source_prefix_sha256_verified": expected_prefix_sha,
        "source_prefix_rows": n_rows,
        "test_rows_read": 0,
        "fits_completed": fit_id,
        "fits_planned": 12,
        "outer_fold_membership": fold_membership_summary,
        "core_feature_count": len(feature_indices),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "thread_count": 1,
        "device": "CPU",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "saved_prior_inner_predictions_available": True,
        "saved_prior_outer_predictions_available": False,
        "claims": {"profitability": False, "statistical_significance": False, "independent_confirmation": False, "trading_signal": False},
        "artifacts": {
            "outer_predictions": "outer_predictions.csv",
            "metrics": "agreement_metrics_by_fold.csv",
            "fit_index": "fit_index.csv",
            "parameters": "model_coefficients_and_scalers.npz",
            "examples": "illustrative_examples.json",
            "report": "results_report.md",
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))
    print("OUTPUT_DIR", out_dir)


if __name__ == "__main__":
    main()
