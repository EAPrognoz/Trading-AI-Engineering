"""Chronological train/validation/test splitting with label-boundary purging."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    purged_train_rows: int
    purged_validation_rows: int


def chronological_split(
    samples: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> ChronologicalSplit:
    """Split ordered samples and purge labels crossing partition boundaries.

    A sample belongs to train only when both its decision timestamp and its
    target timestamp are strictly before the validation boundary. The same rule
    is applied between validation and test.
    """
    required = {"decision_timestamp", "target_timestamp"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"missing split columns: {sorted(missing)}")

    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must leave a test partition")

    ordered = samples.sort_values("decision_timestamp").reset_index(drop=True)
    decision = pd.to_datetime(ordered["decision_timestamp"], utc=True, errors="raise")
    target = pd.to_datetime(ordered["target_timestamp"], utc=True, errors="raise")

    if decision.duplicated().any():
        raise ValueError("decision_timestamp must be unique")
    if not decision.is_monotonic_increasing:
        raise ValueError("decision_timestamp must be ordered")
    if (target <= decision).any():
        raise ValueError("target_timestamp must be later than decision_timestamp")

    n = len(ordered)
    if n < 10:
        raise ValueError("not enough samples for a three-way chronological split")

    validation_index = int(n * train_fraction)
    test_index = int(n * (train_fraction + validation_fraction))
    if validation_index <= 0 or test_index <= validation_index or test_index >= n:
        raise ValueError("split fractions produce an empty partition")

    validation_start = decision.iloc[validation_index]
    test_start = decision.iloc[test_index]

    train_candidates = ordered[decision < validation_start]
    train_target = pd.to_datetime(train_candidates["target_timestamp"], utc=True)
    train = train_candidates[train_target < validation_start].copy()

    validation_candidates = ordered[
        (decision >= validation_start) & (decision < test_start)
    ]
    validation_target = pd.to_datetime(validation_candidates["target_timestamp"], utc=True)
    validation = validation_candidates[validation_target < test_start].copy()

    test = ordered[decision >= test_start].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("purging produced an empty partition")

    return ChronologicalSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        validation_start=validation_start,
        test_start=test_start,
        purged_train_rows=int(len(train_candidates) - len(train)),
        purged_validation_rows=int(len(validation_candidates) - len(validation)),
    )
