from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ai.experiments.baseline_run import run_episode005_validation_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Episode 005 validation baselines.")
    parser.add_argument("--input", required=True, help="Validated H1 CSV snapshot.")
    parser.add_argument(
        "--contract",
        default="configs/experiments/ep005_baselines.toml",
        help="Episode 005 experiment-contract TOML.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="EP002 manifest.json for the accepted snapshot.",
    )
    parser.add_argument("--output", required=True, help="JSON report path.")
    args = parser.parse_args()

    report = run_episode005_validation_baselines(
        args.input,
        args.contract,
        source_manifest_path=args.manifest,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
