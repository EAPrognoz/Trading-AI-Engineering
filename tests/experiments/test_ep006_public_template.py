from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from trading_ai.experiments.ep006_stability import _read_config

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs/experiments/ep006_btc_stability.toml"


def test_public_config_is_a_data_neutral_template() -> None:
    config = tomllib.loads(TEMPLATE.read_text(encoding="utf-8"))

    if config["source"] != ".local/ep006-user-data":
        pytest.fail("public template must use a generic local input path")
    for key in (
        "dataset_id",
        "raw_manifest_sha256",
        "coverage_reconciliation_sha256",
        "source_revision",
        "source_tree_sha256",
        "normalized_source_snapshot_sha256",
    ):
        value = config.get(key, "")
        if re.fullmatch(r"[0-9a-f]{40,64}", str(value)):
            pytest.fail("public template must not contain exact local provenance pins")
        if value not in ("", "SET_LOCALLY"):
            pytest.fail("public template pins must be blank or marked for local setup")


def test_public_template_cannot_run_until_local_pins_are_supplied() -> None:
    with pytest.raises(ValueError, match="local config"):
        _read_config(TEMPLATE)


def test_public_notebook_does_not_pin_private_run_size_or_directory() -> None:
    notebook_path = ROOT / "experiments/ep006_baselines/EP006_Bitcoin_Baseline_Stability_CodeLab.ipynb"
    notebook = __import__("json").loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    if re.search(r"(?i)\b[0-9][0-9,]*\s+eligible validation predictions\b", source):
        pytest.fail("public CodeLab must derive validation row counts from the local run")
    if re.search(r"(?i)\.local/ep006-bitcoin-stability-[0-9]{8}", source):
        pytest.fail("public CodeLab must use a generic local run directory")
    if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in notebook.get("cells", [])):
        pytest.fail("public CodeLab must not contain saved run outputs")
