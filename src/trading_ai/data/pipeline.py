"""Observable Episode 002 market-data pipeline."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import pandas as pd

from trading_ai.data.manifest import build_manifest, market_contract_id
from trading_ai.data.request import MarketDataRequest
from trading_ai.data.time_policy import apply_market_time_policy
from trading_ai.data.validation import audit_market_records


_PUBLIC_PROVENANCE_KEYS = frozenset({
    "source_type", "symbol", "timeframe", "terminal_build",
    "terminal_version", "metatrader5_package_version", "retrieved_at_utc",
    "request_start_utc", "request_end_utc", "cutoff_utc",
    "feature_max_lookback_bars", "feature_contract_sha256", "source_path",
})


def _public_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(provenance)
    if set(result) - _PUBLIC_PROVENANCE_KEYS:
        raise ValueError("provenance contains unsupported or private fields")
    if "source_path" in result:
        source_path = result["source_path"]
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("provenance source_path must be a filename or path")
        result["source_path"] = PureWindowsPath(source_path).name
    for value in result.values():
        if isinstance(value, str) and (
            Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
        ):
            raise ValueError("provenance contains an absolute path")
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def process_market_response(
    raw: pd.DataFrame,
    *,
    request: MarketDataRequest,
    run_dir: str | Path,
    provenance: Mapping[str, Any],
    code_version: str | None = None,
) -> dict[str, Any]:
    """Preserve raw evidence and produce either accepted data or rejection evidence."""
    safe_provenance = _public_provenance(provenance)
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

    time_result = apply_market_time_policy(raw, request)
    candidate = time_result.accepted_range
    issues = audit_market_records(candidate, request.timeframe)

    if time_result.exclusions["invalid_timestamp"]:
        issues.append(
            {
                "code": "invalid_timestamp",
                "severity": "error",
                "details": {
                    "rows": time_result.exclusions["invalid_timestamp"],
                    "scope": "raw_response",
                },
            }
        )

    if candidate.empty:
        issues.append(
            {
                "code": "empty_eligible_range",
                "severity": "error",
                "details": {},
            }
        )

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
        "contract_id": market_contract_id(request.timeframe),
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
        provenance=safe_provenance,
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


def process_h1_response(
    raw: pd.DataFrame,
    *,
    request: MarketDataRequest,
    run_dir: str | Path,
    provenance: Mapping[str, Any],
    code_version: str | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for the original Episode 002 H1 pipeline."""
    if request.timeframe != "H1":
        raise ValueError("process_h1_response requires H1")
    return process_market_response(
        raw, request=request, run_dir=run_dir,
        provenance=provenance, code_version=code_version,
    )
