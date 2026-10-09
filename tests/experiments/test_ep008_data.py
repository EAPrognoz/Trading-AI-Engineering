from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from trading_ai.experiments.ep008_data import (
    FEATURE_COLUMNS,
    build_features,
    build_targets,
    cold_ids,
    ids_sha256,
    load_ecb_zip,
    walk_forward_splits,
)


def _history(n: int = 120) -> pd.DataFrame:
    dates = pd.bdate_range("1999-10-01", periods=n)
    index = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "date": dates.date,
            "USD": 1.1 * np.cumprod(1.0 + 0.0001 + (index % 7) * 0.00001),
            "GBP": 0.85 * np.cumprod(1.0 + 0.0002 + (index % 5) * 0.00002),
            "JPY": 130.0 * np.cumprod(1.0 - 0.0001 + (index % 9) * 0.00003),
            "CHF": 0.96 * np.cumprod(1.0 + (index % 4) * 0.00001),
        }
    )


def _zip_csv(name: str, csv_text: str, path: Path) -> Path:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(name, csv_text)
    return path


def test_loads_single_ecb_member_and_sorts_rates_oldest_first(tmp_path: Path) -> None:
    path = _zip_csv(
        "eurofxref-hist.csv",
        "Date,USD,JPY,GBP,CHF\n2020-01-03,1.2,120,0.8,0.9\n2020-01-02,1.1,121,0.7,0.8\n",
        tmp_path / "rates.zip",
    )

    rates = load_ecb_zip(path)

    assert rates.columns.tolist() == ["date", "USD", "GBP", "JPY", "CHF"]
    assert rates["date"].tolist() == [date(2020, 1, 2), date(2020, 1, 3)]
    assert rates.iloc[0][["USD", "GBP", "JPY", "CHF"]].astype(float).tolist() == [1.1, 0.7, 121.0, 0.8]


@pytest.mark.parametrize("member", ["../eurofxref-hist.csv", "nested/eurofxref-hist.csv"])
def test_rejects_unsafe_or_nested_zip_member(tmp_path: Path, member: str) -> None:
    path = _zip_csv(member, "Date,USD,GBP,JPY,CHF\n2020-01-02,1,1,1,1\n", tmp_path / "unsafe.zip")

    with pytest.raises(ValueError, match="member"):
        load_ecb_zip(path)


def test_rejects_duplicate_dates_and_invalid_required_rates(tmp_path: Path) -> None:
    duplicate = _zip_csv(
        "eurofxref-hist.csv",
        "Date,USD,GBP,JPY,CHF\n2020-01-02,1,1,1,1\n2020-01-02,1.1,1,1,1\n",
        tmp_path / "duplicate.zip",
    )
    bad_rate = _zip_csv(
        "eurofxref-hist.csv",
        "Date,USD,GBP,JPY,CHF\n2020-01-02,1,1,0,1\n",
        tmp_path / "bad.zip",
    )

    with pytest.raises(ValueError, match="unique"):
        load_ecb_zip(duplicate)
    with pytest.raises(ValueError, match="positive finite"):
        load_ecb_zip(bad_rate)


def test_builds_frozen_features_and_five_publication_targets() -> None:
    rates = _history()
    features = build_features(rates)
    decision_index = 70
    decision_date = rates.iloc[decision_index]["date"]
    row = features.loc[features["decision_date"] == decision_date].iloc[0]

    assert len(FEATURE_COLUMNS) == 14
    assert features.columns.tolist() == ["decision_date", *FEATURE_COLUMNS]
    assert not any("target" in column for column in features.columns)
    assert np.isfinite(row.loc[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all()

    usd = rates["USD"].to_numpy(dtype=float)
    returns = np.full(len(usd), np.nan)
    returns[1:] = usd[1:] / usd[:-1] - 1.0
    expected = {
        "r_usd_0": returns[decision_index],
        "r_usd_1": returns[decision_index - 1],
        "r_usd_2": returns[decision_index - 2],
        "c_usd_5": usd[decision_index] / usd[decision_index - 5] - 1.0,
        "c_usd_20": usd[decision_index] / usd[decision_index - 20] - 1.0,
        "rms_usd_5": np.sqrt(np.mean(returns[decision_index - 4 : decision_index + 1] ** 2)),
        "rms_usd_20": np.sqrt(np.mean(returns[decision_index - 19 : decision_index + 1] ** 2)),
        "rms_usd_60": np.sqrt(np.mean(returns[decision_index - 59 : decision_index + 1] ** 2)),
    }
    for feature, value in expected.items():
        assert row[feature] == pytest.approx(value)
    for currency in ("GBP", "JPY", "CHF"):
        prices = rates[currency].to_numpy(dtype=float)
        returns_for_currency = np.full(len(prices), np.nan)
        returns_for_currency[1:] = prices[1:] / prices[:-1] - 1.0
        assert row[f"c_{currency.lower()}_5"] == pytest.approx(
            prices[decision_index] / prices[decision_index - 5] - 1.0
        )
        assert row[f"rms_{currency.lower()}_20"] == pytest.approx(
            np.sqrt(np.mean(returns_for_currency[decision_index - 19 : decision_index + 1] ** 2))
        )

    targets = build_targets(rates, [decision_date])
    future_returns = usd[decision_index + 1 : decision_index + 6] / usd[decision_index : decision_index + 5] - 1.0
    assert targets.loc[0, "sample_id"] == f"{decision_date}|{rates.iloc[decision_index + 5]['date']}"
    assert targets.loc[0, "endpoint_date"] == rates.iloc[decision_index + 5]["date"]
    assert targets.loc[0, "cumulative_return_bps"] == pytest.approx(
        (usd[decision_index + 5] / usd[decision_index] - 1.0) * 10_000
    )
    assert targets.loc[0, "movement_target_bps2"] == pytest.approx(np.mean(future_returns**2) * 1e8)


def test_features_at_decision_are_unchanged_by_later_rates() -> None:
    rates = _history()
    decision_date = rates.iloc[70]["date"]
    before = build_features(rates).set_index("decision_date").loc[decision_date]
    changed = rates.copy()
    changed.loc[changed.index > 70, ["USD", "GBP", "JPY", "CHF"]] *= 1.25
    after = build_features(changed).set_index("decision_date").loc[decision_date]

    assert before.loc[list(FEATURE_COLUMNS)].to_numpy(dtype=float) == pytest.approx(
        after.loc[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    )


def test_walk_forward_splits_purge_labels_that_cross_partition_boundaries() -> None:
    examples = [
        ("old", date(2015, 1, 5), date(2015, 1, 12)),
        ("f1_train_cross", date(2016, 12, 29), date(2017, 1, 6)),
        ("f1_valid", date(2017, 1, 3), date(2017, 1, 10)),
        ("f1_valid_cross", date(2019, 12, 30), date(2020, 1, 7)),
        ("f2_valid", date(2020, 1, 3), date(2020, 1, 10)),
        ("f2_valid_cross", date(2022, 12, 29), date(2023, 1, 6)),
        ("f3_valid", date(2023, 1, 3), date(2023, 1, 10)),
        ("f3_valid_end", date(2025, 12, 23), date(2025, 12, 31)),
        ("f3_end_cross", date(2025, 12, 24), date(2026, 1, 2)),
    ]
    rows = pd.DataFrame(examples, columns=["sample_id", "decision_date", "endpoint_date"])

    folds = walk_forward_splits(rows)

    assert [fold.name for fold in folds] == ["2017-2019", "2020-2022", "2023-2025"]
    assert folds[0].train["sample_id"].tolist() == ["old"]
    assert folds[0].validation["sample_id"].tolist() == ["f1_valid"]
    assert folds[0].purged_train_ids == ("f1_train_cross",)
    assert folds[0].purged_validation_ids == ("f1_valid_cross",)
    assert set(folds[1].train["sample_id"]) == {"old", "f1_train_cross", "f1_valid"}
    assert folds[1].validation["sample_id"].tolist() == ["f2_valid"]
    assert folds[1].purged_validation_ids == ("f2_valid_cross",)
    assert folds[2].validation["sample_id"].tolist() == ["f3_valid", "f3_valid_end"]
    assert folds[2].purged_validation_ids == ("f3_end_cross",)


def test_cold_ids_include_only_decisions_and_endpoints_inside_window() -> None:
    prior_dates = pd.bdate_range(end="2025-12-31", periods=100)
    cold_dates = pd.bdate_range("2026-01-02", periods=191)
    dates = prior_dates.append(cold_dates)
    rates = pd.DataFrame(
        {
            "date": dates.date,
            "USD": 1.1 + np.arange(len(dates)) * 0.0001,
            "GBP": 0.8 + np.arange(len(dates)) * 0.0001,
            "JPY": 130 + np.arange(len(dates)) * 0.01,
            "CHF": 0.9 + np.arange(len(dates)) * 0.0001,
        }
    )
    ids = cold_ids(rates, build_features(rates))

    assert len(ids) == 186
    assert ids["sample_id"].is_unique
    assert ids["decision_date"].min() >= date(2026, 1, 1)
    assert ids["endpoint_date"].max() <= date(2026, 9, 30)
    assert (ids["endpoint_date"] > ids["decision_date"]).all()


def test_id_hash_uses_exact_ordered_ids_with_lf_line_endings() -> None:
    ids = pd.DataFrame({"sample_id": ["2026-01-02|2026-01-09", "2026-01-05|2026-01-12"]})

    assert ids_sha256(ids) == hashlib.sha256(
        b"2026-01-02|2026-01-09\n2026-01-05|2026-01-12\n"
    ).hexdigest()


def test_approved_ecb_snapshot_has_complete_selected_currency_columns() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    archive = repo_root / "data/raw/ecb_eurofxref_hist/eurofxref-hist.zip"

    rates = load_ecb_zip(archive)

    assert len(rates) == 7110
    assert rates["date"].is_unique
    for currency in ("USD", "GBP", "JPY", "CHF"):
        values = rates[currency].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert (values > 0).all()
