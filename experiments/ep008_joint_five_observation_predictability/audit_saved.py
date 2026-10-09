"""Audit the published ECB snapshot and frozen predictions without fitting models."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trading_ai.experiments.ep008_data import build_targets, load_ecb_zip  # noqa: E402
from trading_ai.experiments.ep008_metrics import score_movement, score_quantiles, score_return  # noqa: E402

RUN_ID = "ep008-9f1cb85ac383e69b9704c537"
EXPECTED_ARCHIVE_SHA256 = "e39ffa4c4e8cf3207f2b2589ce5ed5aea56253b35e33c66928d1f7ed76a9d941"
ARCHIVE = ROOT / "data/raw/ecb_eurofxref_hist/eurofxref-hist.zip"
RUN_DIR = ROOT / "experiments/ep008_joint_five_observation_predictability/runs" / RUN_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _calendar_qc(rates: pd.DataFrame) -> dict[str, object]:
    start, end = date(2000, 1, 1), date(2025, 12, 31)
    observed = set(rates.loc[rates.date.between(start, end), "date"])
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    missing = [day for day in days if day not in observed]
    weekday_gaps = [day.isoformat() for day in missing if day.weekday() < 5]
    return {
        "calendar_days": len(days),
        "calendar_dates_without_rate": len(missing),
        "weekdays_without_rate": len(weekday_gaps),
        "weekdays_without_rate_examples": weekday_gaps[:12],
        "forward_fill_applied": False,
    }


def audit() -> dict[str, object]:
    actual_hash = sha256(ARCHIVE)
    if actual_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"archive SHA-256 mismatch: {actual_hash}")
    rates = load_ecb_zip(ARCHIVE)
    period = rates.loc[rates.date.between(date(2000, 1, 1), date(2025, 12, 31))].copy()
    if period.empty or period.date.duplicated().any():
        raise ValueError("development-period rates are empty or duplicate publication dates")

    predictions_path = RUN_DIR / "cold_predictions.csv"
    ids_path = RUN_DIR / "cold_ids.csv"
    predictions = pd.read_csv(predictions_path, dtype={"sample_id": str})
    frozen_ids = pd.read_csv(ids_path, dtype={"sample_id": str})
    required = {"sample_id", "decision_date", "endpoint_date"}
    if not required.issubset(predictions.columns) or not required.issubset(frozen_ids.columns):
        raise ValueError("saved predictions or frozen IDs are missing identity columns")
    if predictions.sample_id.duplicated().any() or frozen_ids.sample_id.duplicated().any():
        raise ValueError("saved predictions or frozen IDs contain duplicate sample IDs")
    for frame in (predictions, frozen_ids):
        for column in ("decision_date", "endpoint_date"):
            frame[column] = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="raise").dt.date
    if predictions.sample_id.tolist() != frozen_ids.sample_id.tolist():
        raise ValueError("saved prediction IDs do not match the frozen ID file")

    labels = build_targets(rates, predictions.decision_date.tolist())
    if labels.sample_id.tolist() != predictions.sample_id.tolist():
        raise ValueError("reconstructed labels do not align with saved prediction IDs")
    joined = predictions.merge(labels, on=["sample_id", "decision_date", "endpoint_date"], validate="one_to_one")
    if len(joined) != 186:
        raise ValueError(f"expected 186 saved forecast rows; found {len(joined)}")

    y_return = joined.cumulative_return_bps.to_numpy(float)
    y_movement = joined.movement_target_bps2.to_numpy(float)
    scores: dict[str, object] = {"n": len(joined), "return_mae_bps": {}, "movement_qlike": {}, "quantile_mean_pinball_bps": {}}
    for name in ("zero", "train_mean", "train_median", "linear", "boosting"):
        scores["return_mae_bps"][name] = score_return(y_return, joined[f"return_{name}_bps"].to_numpy(float))["mae_bps"]
    for name in ("trailing5", "ewma", "linear", "boosting"):
        metric = score_movement(y_movement, joined[f"movement_{name}_raw_bps2"].to_numpy(float))
        scores["movement_qlike"][name] = metric["qlike"]
        if name == "linear":
            scores["movement_linear_floor_clipped_count"] = metric["prediction_floor_clipped_count"]
    for name in ("rolling_empirical", "ewma_scaled_empirical", "linear", "boosting"):
        raw = np.column_stack([
            joined[f"quantiles_{name}_raw_q05_bps"],
            joined[f"quantiles_{name}_raw_q50_bps"],
            joined[f"quantiles_{name}_raw_q95_bps"],
        ]).astype(float)
        metric = score_quantiles(y_return, raw)
        scores["quantile_mean_pinball_bps"][name] = metric["mean_pinball_bps"]

    return {
        "experiment_id": "ep008-joint-five-observation",
        "public_episode": "EP007",
        "run_id": RUN_ID,
        "audit_type": "after-the-fact audit of saved predictions; no model fitting or phase execution",
        "source": {
            "url": "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip",
            "archive_sha256": actual_hash,
            "archive_size_bytes": ARCHIVE.stat().st_size,
            "rate_definition": "USD per 1 EUR; ECB daily reference rates",
            "date_range": [str(period.date.min()), str(period.date.max())],
            "observations": len(period),
            "duplicate_dates": int(period.date.duplicated().sum()),
            "missing_usd_values": int(period.USD.isna().sum()),
            "non_numeric_or_nonfinite_usd_values": 0,
            "nonpositive_usd_values": int((period.USD <= 0).sum()),
            **_calendar_qc(rates),
        },
        "saved_run": {
            "cold_rows": len(joined),
            "decision_range": [str(joined.decision_date.min()), str(joined.decision_date.max())],
            "endpoint_range": [str(joined.endpoint_date.min()), str(joined.endpoint_date.max())],
            "metrics_recomputed_from_saved_predictions": scores,
            "overlapping_five_observation_targets": True,
        },
        "interpretation": "ECB reference rates are informational and non-executable. Results are a reproducible ML demonstration, not a trading strategy or a promise of profit.",
    }


def main() -> None:
    global ARCHIVE, RUN_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()
    ARCHIVE, RUN_DIR = args.archive, args.run_dir
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
