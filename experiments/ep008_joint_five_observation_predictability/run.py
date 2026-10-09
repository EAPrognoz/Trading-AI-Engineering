"""Command line entry point for the gated EP008 experiment phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ai.experiments.ep008_run import (
    freeze_phase,
    predict_cold_phase,
    run_validation_phase,
    score_cold_phase,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "freeze", "predict", "score"))
    parser.add_argument("--run-id", help="required for freeze, predict, and score")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.phase == "validate":
        if args.run_id:
            parser.error("--run-id is created by the validate phase")
        result = {"run_id": run_validation_phase(root)}
    else:
        if not args.run_id:
            parser.error(f"--run-id is required for {args.phase}")
        operation = {
            "freeze": freeze_phase,
            "predict": predict_cold_phase,
            "score": score_cold_phase,
        }[args.phase]
        result = operation(root, args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
