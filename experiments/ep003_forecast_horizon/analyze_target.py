from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ai.targets.report import analyze_h1_direction_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Episode 003 H1 direction target.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Validated H1 CSV snapshot.")
    source.add_argument("--bundle-manifest", help="Validated BTC bundle manifest.")
    parser.add_argument(
        "--manifest",
        help="EP002 manifest.json for the accepted snapshot.",
    )
    parser.add_argument("--output", required=True, help="JSON report path.")
    args = parser.parse_args()
    if (args.input is None) != (args.manifest is None):
        parser.error("--input and --manifest must be supplied together")

    report = analyze_h1_direction_target(
        args.input,
        source_manifest_path=args.manifest,
        bundle_manifest_path=args.bundle_manifest,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
