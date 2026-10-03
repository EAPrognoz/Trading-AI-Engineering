"""Run the frozen four-window Episode 006 BTC validation experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tomllib

from trading_ai.experiments.ep006_stability import (
    DEFAULT_CONFIG_PATH,
    run_episode006_stability,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("EP006 command paths must stay inside the project copy") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("EP006_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    source_dir = args.source_dir or (PROJECT_ROOT / config["source"])
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    output_dir = args.output_dir
    if output_dir is None:
        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = PROJECT_ROOT / ".local" / f"ep006-bitcoin-stability-{run_stamp}"
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    command_text = (
        "python experiments/ep006_baselines/run_ep006.py"
        f" --config {_repo_relative(config_path)}"
        f" --source-dir {_repo_relative(source_dir)}"
        f" --output-dir {_repo_relative(output_dir)}"
    )
    report = run_episode006_stability(
        source_dir,
        output_dir,
        config_path=config_path,
        command_text=command_text,
    )
    print(json.dumps({
        "experiment_id": report["experiment_id"],
        "output_dir": _repo_relative(output_dir),
        "pooled_validation_metrics": report["pooled_validation_metrics"],
        "locked_test": report["locked_test"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
