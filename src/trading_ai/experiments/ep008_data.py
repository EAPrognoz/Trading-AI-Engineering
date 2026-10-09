"""Data contracts and observation-aligned samples for EP008."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd


REQUIRED_CURRENCIES = ("USD", "GBP", "JPY", "CHF")
FEATURE_COLUMNS = (
    "r_usd_0",
    "r_usd_1",
    "r_usd_2",
    "c_usd_5",
    "c_usd_20",
    "rms_usd_5",
    "rms_usd_20",
    "rms_usd_60",
    "c_gbp_5",
    "rms_gbp_20",
    "c_jpy_5",
    "rms_jpy_20",
    "c_chf_5",
    "rms_chf_20",
)
FEATURE_WARMUP_RETURNS = 60
HORIZON = 5
DEVELOPMENT_START = date(2000, 1, 1)
DEVELOPMENT_END = date(2025, 12, 31)
COLD_START = date(2026, 1, 1)
COLD_END = date(2026, 9, 30)
MAX_ARCHIVE_MEMBER_BYTES = 10_000_000


@dataclass(frozen=True)
class Fold:
    name: str
    train: pd.DataFrame
    validation: pd.DataFrame
    purged_train_ids: tuple[str, ...]
    purged_validation_ids: tuple[str, ...]


def _date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def _validate_rates(rates: pd.DataFrame) -> pd.DataFrame:
    required = {"date", *REQUIRED_CURRENCIES}
    missing = required.difference(rates.columns)
    if missing:
        raise ValueError(f"rates are missing required columns: {sorted(missing)}")
    ordered = rates.loc[:, ["date", *REQUIRED_CURRENCIES]].copy()
    try:
        ordered["date"] = ordered["date"].map(_date)
    except (TypeError, ValueError) as error:
        raise ValueError("publication dates must be valid dates") from error
    if ordered["date"].duplicated().any():
        raise ValueError("publication dates must be unique")
    if not ordered["date"].is_monotonic_increasing:
        raise ValueError("publication dates must be sorted oldest first")
    for currency in REQUIRED_CURRENCIES:
        try:
            ordered[currency] = pd.to_numeric(ordered[currency], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{currency} rates must be numeric") from error
        values = ordered[currency].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{currency} rates must be positive finite values")
    return ordered.reset_index(drop=True)


def load_ecb_zip(path: str | Path) -> pd.DataFrame:
    """Read the single expected ECB CSV member without extracting archive paths."""
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    try:
        with ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != "eurofxref-hist.csv":
                raise ValueError("ECB archive must contain exactly the eurofxref-hist.csv member")
            member = members[0]
            if member.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError("ECB CSV member exceeds the permitted size")
            if archive.testzip() is not None:
                raise ValueError("ECB ZIP contains a corrupt member")
            with archive.open(member, "r") as stream:
                frame = pd.read_csv(stream, usecols=lambda name: name in {"Date", *REQUIRED_CURRENCIES})
    except BadZipFile as error:
        raise ValueError("ECB archive is not a valid ZIP file") from error
    required_columns = {"Date", *REQUIRED_CURRENCIES}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"ECB CSV is missing required columns: {sorted(missing)}")
    dates = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("ECB publication dates must be valid YYYY-MM-DD dates")
    frame = frame.rename(columns={"Date": "date"})
    frame["date"] = dates.dt.date
    frame = frame.loc[:, ["date", *REQUIRED_CURRENCIES]]
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    return _validate_rates(frame)


def build_features(rates: pd.DataFrame) -> pd.DataFrame:
    """Build exactly the frozen 14 features known by each publication date."""
    ordered = _validate_rates(rates)
    date_values = ordered["date"].tolist()
    prices = {currency: ordered[currency].to_numpy(dtype=float) for currency in REQUIRED_CURRENCIES}
    returns: dict[str, np.ndarray] = {}
    for currency, values in prices.items():
        currency_returns = np.full(len(values), np.nan, dtype=float)
        currency_returns[1:] = values[1:] / values[:-1] - 1.0
        returns[currency] = currency_returns

    records: list[dict[str, object]] = []
    for i in range(FEATURE_WARMUP_RETURNS, len(ordered)):
        row: dict[str, object] = {"decision_date": date_values[i]}
        usd_returns = returns["USD"]
        for lag in range(3):
            row[f"r_usd_{lag}"] = float(usd_returns[i - lag])
        for window in (5, 20):
            row[f"c_usd_{window}"] = float(prices["USD"][i] / prices["USD"][i - window] - 1.0)
            row[f"rms_usd_{window}"] = float(
                np.sqrt(np.mean(usd_returns[i - window + 1 : i + 1] ** 2))
            )
        row["rms_usd_60"] = float(np.sqrt(np.mean(usd_returns[i - 59 : i + 1] ** 2)))
        for currency in ("GBP", "JPY", "CHF"):
            lower = currency.lower()
            row[f"c_{lower}_5"] = float(prices[currency][i] / prices[currency][i - 5] - 1.0)
            row[f"rms_{lower}_20"] = float(
                np.sqrt(np.mean(returns[currency][i - 19 : i + 1] ** 2))
            )
        records.append(row)
    return pd.DataFrame(records, columns=["decision_date", *FEATURE_COLUMNS])


def build_targets(rates: pd.DataFrame, decision_dates: list[object]) -> pd.DataFrame:
    """Build the five-published-observation return and mean-squared-return labels."""
    ordered = _validate_rates(rates)
    dates = ordered["date"].tolist()
    positions = {value: index for index, value in enumerate(dates)}
    normalized = [_date(value) for value in decision_dates]
    if len(set(normalized)) != len(normalized):
        raise ValueError("decision dates must be unique")
    rows: list[dict[str, object]] = []
    usd = ordered["USD"].to_numpy(dtype=float)
    for decision in normalized:
        if decision not in positions:
            raise ValueError(f"decision date is absent from rates: {decision}")
        i = positions[decision]
        endpoint_index = i + HORIZON
        if endpoint_index >= len(dates):
            raise ValueError(f"five-observation endpoint is unavailable for {decision}")
        future_returns = usd[i + 1 : endpoint_index + 1] / usd[i:endpoint_index] - 1.0
        endpoint = dates[endpoint_index]
        rows.append(
            {
                "sample_id": f"{decision.isoformat()}|{endpoint.isoformat()}",
                "decision_date": decision,
                "endpoint_date": endpoint,
                "cumulative_return_bps": float((usd[endpoint_index] / usd[i] - 1.0) * 10_000.0),
                "movement_target_bps2": float(np.mean(future_returns**2) * 1e8),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "decision_date",
            "endpoint_date",
            "cumulative_return_bps",
            "movement_target_bps2",
        ],
    )


def walk_forward_splits(samples: pd.DataFrame) -> list[Fold]:
    """Build expanding 2017–2025 validation folds with endpoint purging."""
    required = {"sample_id", "decision_date", "endpoint_date"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"samples are missing split columns: {sorted(missing)}")
    rows = samples.copy()
    rows["decision_date"] = rows["decision_date"].map(_date)
    rows["endpoint_date"] = rows["endpoint_date"].map(_date)
    rows = rows.sort_values("decision_date", kind="stable").reset_index(drop=True)
    if rows["sample_id"].duplicated().any() or rows["decision_date"].duplicated().any():
        raise ValueError("sample IDs and decision dates must be unique")
    if (rows["endpoint_date"] <= rows["decision_date"]).any():
        raise ValueError("endpoint dates must follow decision dates")

    periods = (
        ("2017-2019", date(2016, 12, 31), date(2017, 1, 1), date(2019, 12, 31)),
        ("2020-2022", date(2019, 12, 31), date(2020, 1, 1), date(2022, 12, 31)),
        ("2023-2025", date(2022, 12, 31), date(2023, 1, 1), date(2025, 12, 31)),
    )
    folds: list[Fold] = []
    for name, train_end, validation_start, validation_end in periods:
        train_candidates = rows.loc[
            rows["decision_date"].between(DEVELOPMENT_START, train_end)
        ]
        validation_candidates = rows.loc[
            rows["decision_date"].between(validation_start, validation_end)
        ]
        train = train_candidates.loc[train_candidates["endpoint_date"] <= train_end].copy()
        validation = validation_candidates.loc[
            validation_candidates["endpoint_date"].between(validation_start, validation_end)
        ].copy()
        if train.empty or validation.empty:
            raise ValueError(f"{name} fold is empty after endpoint purging")
        train_ids = set(train["sample_id"].astype(str))
        validation_ids = set(validation["sample_id"].astype(str))
        folds.append(
            Fold(
                name=name,
                train=train.reset_index(drop=True),
                validation=validation.reset_index(drop=True),
                purged_train_ids=tuple(
                    train_candidates.loc[~train_candidates["sample_id"].isin(train_ids), "sample_id"].astype(str)
                ),
                purged_validation_ids=tuple(
                    validation_candidates.loc[
                        ~validation_candidates["sample_id"].isin(validation_ids), "sample_id"
                    ].astype(str)
                ),
            )
        )
    return folds


def cold_ids(rates: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Create cold decision/endpoint IDs using dates only, without computing labels."""
    ordered = _validate_rates(rates)
    if "decision_date" not in features:
        raise ValueError("features must contain decision_date")
    dates = ordered["date"].tolist()
    positions = {value: index for index, value in enumerate(dates)}
    records: list[dict[str, object]] = []
    seen: set[date] = set()
    for raw_decision in features["decision_date"]:
        decision = _date(raw_decision)
        if not COLD_START <= decision <= COLD_END:
            continue
        if decision in seen:
            raise ValueError("decision dates in features must be unique")
        seen.add(decision)
        if decision not in positions:
            raise ValueError(f"cold decision date is absent from rates: {decision}")
        endpoint_index = positions[decision] + HORIZON
        if endpoint_index >= len(dates):
            continue
        endpoint = dates[endpoint_index]
        if COLD_START <= endpoint <= COLD_END:
            records.append(
                {
                    "sample_id": f"{decision.isoformat()}|{endpoint.isoformat()}",
                    "decision_date": decision,
                    "endpoint_date": endpoint,
                }
            )
    return pd.DataFrame(records, columns=["sample_id", "decision_date", "endpoint_date"])


def ids_sha256(ids: pd.DataFrame) -> str:
    """Hash the exact ordered decision|endpoint IDs with LF line endings."""
    if "sample_id" not in ids:
        raise ValueError("ID frame must contain sample_id")
    values = ids["sample_id"].astype(str).tolist()
    if len(set(values)) != len(values):
        raise ValueError("sample IDs must be unique")
    serialized = "".join(value + "\n" for value in values).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
