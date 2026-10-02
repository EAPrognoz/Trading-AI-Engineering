from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ai.experiments.baseline_run import run_episode005_validation_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Episode 005 validation baselines.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Validated H1 CSV snapshot.")
    source.add_argument("--bundle-manifest", help="Verified BTC H1/H4/D1 bundle manifest.")
    parser.add_argument(
        "--contract",
        default="configs/experiments/ep005_baselines.toml",
        help="Episode 005 experiment-contract TOML.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="EP002 manifest.json for the accepted snapshot.",
    )
    parser.add_argument("--output", required=True, help="JSON report path.")
    args = parser.parse_args()
    if args.bundle_manifest and args.manifest:
        parser.error("--manifest cannot be combined with --bundle-manifest")
    if args.input and not args.manifest:
        parser.error("--manifest is required with --input")

    report = run_episode005_validation_baselines(
        args.input,
        args.contract,
        source_manifest_path=args.manifest,
        bundle_manifest_path=args.bundle_manifest,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
