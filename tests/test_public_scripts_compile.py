from __future__ import annotations

from pathlib import Path

import pytest


SCRIPT_ROOTS = (Path("examples"), Path("experiments"))
SCRIPT_PATHS = sorted(
    path
    for root in SCRIPT_ROOTS
    for path in root.rglob("*.py")
)


def test_public_scripts_are_discovered() -> None:
    assert SCRIPT_PATHS


@pytest.mark.parametrize("path", SCRIPT_PATHS, ids=str)
def test_public_script_compiles(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
