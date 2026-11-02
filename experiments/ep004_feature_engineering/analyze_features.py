from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ai.features.report import analyze_feature_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Episode 004 feature contract.")
    parser.add_argument("--input", required=True, help="Validated H1 CSV snapshot.")
    parser.add_argument(
        "--contract",
        default="configs/features/ep004_baseline_features.toml",
        help="Feature-contract TOML path.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="EP002 manifest.json for the accepted snapshot.",
    )
    parser.add_argument(
        "--experiment-contract",
        default="configs/experiments/ep005_baselines.toml",
        help="EP005 experiment contract defining the train diagnostic scope.",
    )
    parser.add_argument("--output", required=True, help="JSON report path.")
    args = parser.parse_args()

    report = analyze_feature_contract(
        args.input,
        args.contract,
        source_manifest_path=args.manifest,
        experiment_contract_path=args.experiment_contract,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
