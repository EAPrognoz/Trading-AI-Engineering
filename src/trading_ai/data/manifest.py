"""Run manifests for reproducible Episode 002 data artifacts."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
import platform
import sys

import numpy as np
import pandas as pd

from trading_ai.data.request import MarketDataRequest


def market_contract_id(timeframe: str) -> str:
    if timeframe not in {"H1", "H4", "D1"}:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    return f"ep002-{timeframe.lower()}-market-data-v1"


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_manifest() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def build_manifest(
    *,
    request: MarketDataRequest,
    status: str,
    provenance: dict[str, Any],
    code_version: str | None,
    files: dict[str, Path],
) -> dict[str, Any]:
    contract_id = market_contract_id(request.timeframe)
    file_entries = {
        name: {"path": path.name, "sha256": sha256_file(path)}
        for name, path in files.items()
    }
    public_provenance = dict(provenance)
    for artifact in ("raw_response", "validation_report", "accepted_dataset"):
        if artifact in file_entries:
            public_provenance[f"{artifact}_sha256"] = file_entries[artifact]["sha256"]
    manifest = {
        "contract_id": contract_id,
        "status": status,
        "request": request.to_dict(),
        "provenance": public_provenance,
        "environment": environment_manifest(),
        "code_version": code_version,
        "files": file_entries,
    }
    if status == "accepted":
        accepted_sha = manifest["files"]["accepted_dataset"]["sha256"]
        manifest["dataset_id"] = f"{contract_id}:{accepted_sha}"
    return manifest
