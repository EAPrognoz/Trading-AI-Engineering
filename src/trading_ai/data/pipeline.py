"""Observable Episode 002 market-data pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from trading_ai.data.manifest import build_manifest
from trading_ai.data.request import MarketDataRequest
from trading_ai.data.time_policy import apply_h1_time_policy
from trading_ai.data.validation import audit_h1_records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def process_h1_response(
    raw: pd.DataFrame,
    *,
    request: MarketDataRequest,
    run_dir: str | Path,
    provenance: dict[str, Any],
    code_version: str | None = None,
) -> dict[str, Any]:
    """Preserve raw evidence and produce either accepted data or rejection evidence."""
    run_path = Path(run_dir)
    if run_path.exists():
        raise FileExistsError(f"run directory already exists: {run_path}")
    run_path.mkdir(parents=True)

    request_path = run_path / "request.json"
    raw_path = run_path / "raw_response.csv"
    report_path = run_path / "validation_report.json"
    manifest_path = run_path / "manifest.json"

    _write_json(request_path, request.to_dict())
    raw.to_csv(raw_path, index=False)

    time_result = apply_h1_time_policy(raw, request)
    candidate = time_result.accepted_range
    issues = audit_h1_records(candidate)

    if not time_result.coverage["coverage_ok"]:
        issues.append(
            {
                "code": "requested_coverage_incomplete",
                "severity": "review",
                "details": time_result.coverage,
            }
        )

    accepted = not issues
    report = {
        "contract_id": "ep002-h1-market-data-v1",
        "status": "accepted" if accepted else "rejected",
        "input_rows": int(len(raw)),
        "candidate_rows": int(len(candidate)),
        "accepted_rows": int(len(candidate)) if accepted else 0,
        "exclusions": time_result.exclusions,
        "coverage": time_result.coverage,
        "issues": issues,
        "scope_note": (
            "Checks passed means only that the declared Episode 002 checks passed; "
            "it does not certify market truth, point-in-time feed reconstruction, "
            "or strategy profitability."
        ),
    }
    _write_json(report_path, report)

    files = {
        "request": request_path,
        "raw_response": raw_path,
        "validation_report": report_path,
    }

    if accepted:
        accepted_path = run_path / "accepted.csv"
        candidate.to_csv(accepted_path, index=False)
        files["accepted_dataset"] = accepted_path
        status = "accepted"
    else:
        rejection_path = run_path / "rejection_report.json"
        _write_json(
            rejection_path,
            {
                "status": "rejected",
                "issues": issues,
                "accepted_dataset_written": False,
            },
        )
        files["rejection_report"] = rejection_path
        status = "rejected"

    manifest = build_manifest(
        request=request,
        status=status,
        provenance=provenance,
        code_version=code_version,
        files=files,
    )
    _write_json(manifest_path, manifest)

    return {
        "status": status,
        "run_dir": str(run_path),
        "manifest": manifest,
        "validation_report": report,
    }
