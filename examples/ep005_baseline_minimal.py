"""Episode 005: create the simplest benchmark first.

Usage:
    python examples/ep005_baseline_minimal.py path/to/accepted.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from trading_ai.experiments.baseline_dataset import prepare_episode005_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the minimal Episode 005 majority-class baseline."
    )
    parser.add_argument("input", help="Validated H1 CSV snapshot.")
    parser.add_argument(
        "--contract",
        default="configs/experiments/ep005_baselines.toml",
        help="Episode 005 experiment-contract TOML.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional EP002 manifest.json for provenance verification.",
    )
    args = parser.parse_args()

    prepared = prepare_episode005_split(
        args.input,
        args.contract,
        source_manifest_path=args.manifest,
    )
    y_train = prepared.split.train["target_h1_direction"]
    y_validation = prepared.split.validation["target_h1_direction"]

    majority_class = y_train.value_counts().idxmax()
    prediction = pd.Series(majority_class, index=y_validation.index)
    accuracy = (prediction == y_validation).mean()

    print(f"Training majority class: {majority_class}")
    print(f"Validation rows: {len(y_validation)}")
    print(f"B0 majority-class accuracy: {accuracy:.4f}")
    print()
    print("Now a more complex model has something concrete to beat.")


if __name__ == "__main__":
    main()
