from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ai.targets.report import analyze_h1_direction_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Episode 003 H1 direction target.")
    parser.add_argument("--input", required=True, help="Validated H1 CSV snapshot.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="EP002 manifest.json for the accepted snapshot.",
    )
    parser.add_argument("--output", required=True, help="JSON report path.")
    args = parser.parse_args()

    report = analyze_h1_direction_target(
        args.input,
        source_manifest_path=args.manifest,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
