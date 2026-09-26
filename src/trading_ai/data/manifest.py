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
    return {
        "contract_id": "ep002-h1-market-data-v1",
        "status": status,
        "request": request.to_dict(),
        "provenance": provenance,
        "environment": environment_manifest(),
        "code_version": code_version,
        "files": {
            name: {
                "path": path.name,
                "sha256": sha256_file(path),
            }
            for name, path in files.items()
        },
    }
