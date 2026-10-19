"""Clean-room byte-identity tests for both public parent-Gold derivations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "crn-2026-003-finite-domain-dsa"


@pytest.mark.parametrize(
    ("script", "gold_member"),
    [
        (
            "build_archived_gold.py",
            f"studies/{STUDY_ID}/evidence/gold/archived/recomputed.json",
        ),
        (
            "build_prospective_parent_gold.py",
            f"studies/{STUDY_ID}/evidence/gold/inference/parent_summary.json",
        ),
    ],
)
def test_parent_gold_derivation_is_byte_identical(
    tmp_path: Path, script: str, gold_member: str
) -> None:
    output = tmp_path / Path(gold_member).name
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name(script)),
            "--release-root",
            str(RELEASE_ROOT),
            "--output",
            str(output),
        ],
        cwd=RELEASE_ROOT,
        check=True,
    )
    assert output.read_bytes() == (RELEASE_ROOT / gold_member).read_bytes()
