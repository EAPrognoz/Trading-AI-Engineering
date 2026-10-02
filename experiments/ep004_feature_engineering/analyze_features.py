from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ai.features.report import analyze_feature_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Episode 004 feature contract.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Validated H1 CSV snapshot.")
    source.add_argument("--bundle-manifest", help="Validated BTC bundle manifest.")
    parser.add_argument(
        "--contract",
        help="Feature-contract TOML path.",
    )
    parser.add_argument(
        "--manifest",
        help="EP002 manifest.json for the accepted snapshot.",
    )
    parser.add_argument(
        "--experiment-contract",
        help="EP005 experiment contract defining the train diagnostic scope.",
    )
    parser.add_argument("--output", required=True, help="JSON report path.")
    args = parser.parse_args()
    if (args.input is None) != (args.manifest is None):
        parser.error("--input and --manifest must be supplied together")
    feature_contract = args.contract or (
        "configs/features/ep004_btc_mtf_features.toml"
        if args.bundle_manifest else "configs/features/ep004_baseline_features.toml"
    )
    experiment_contract = args.experiment_contract or (
        "configs/experiments/ep005_btc_mtf_baselines.toml"
        if args.bundle_manifest else "configs/experiments/ep005_baselines.toml"
    )

    report = analyze_feature_contract(
        args.input,
        feature_contract,
        source_manifest_path=args.manifest,
        bundle_manifest_path=args.bundle_manifest,
        experiment_contract_path=experiment_contract,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
