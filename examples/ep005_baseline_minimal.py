"""Episode 005: create the simplest benchmark first.

Usage:
    python examples/ep005_baseline_minimal.py path/to/accepted.csv
    python examples/ep005_baseline_minimal.py --bundle-manifest path/to/bundle_manifest.json --contract configs/experiments/ep005_btc_mtf_baselines.toml
"""

from __future__ import annotations

import argparse

import pandas as pd

from trading_ai.experiments.baseline_dataset import prepare_episode005_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the minimal Episode 005 majority-class baseline."
    )
    parser.add_argument("input", nargs="?", help="Validated H1 CSV snapshot.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", dest="input_option", help="Validated H1 CSV snapshot.")
    source.add_argument("--bundle-manifest", help="Verified BTC H1/H4/D1 bundle manifest.")
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
    if args.input and (args.input_option or args.bundle_manifest):
        parser.error("positional input cannot be combined with another source mode")
    snapshot = args.input or args.input_option
    if (snapshot is None) == (args.bundle_manifest is None):
        parser.error("choose exactly one source mode: input or bundle manifest")
    if args.bundle_manifest and args.manifest:
        parser.error("--manifest cannot be combined with --bundle-manifest")

    prepared = prepare_episode005_split(
        snapshot,
        args.contract,
        source_manifest_path=args.manifest,
        bundle_manifest_path=args.bundle_manifest,
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
