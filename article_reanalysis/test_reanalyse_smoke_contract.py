"""Explicit pytest contract for the release-stage reanalysis smoke CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ANALYSIS_ROOT = Path(__file__).resolve().parent
SMOKE_CLI = ANALYSIS_ROOT / "test_reanalyse_smoke.py"
RELEASE_ROOT_VARIABLE = "CRN_RELEASE_SMOKE_ROOT"
SUMMARY_RELATIVE = Path("results/crn_ca_pr1_20261017/evaluation/summary.csv")


def test_release_reanalysis_smoke_when_assembled_input_is_supplied() -> None:
    """Run fully when opted in; otherwise record an explicit dependency skip."""

    configured = os.environ.get(RELEASE_ROOT_VARIABLE)
    if configured is None:
        pytest.skip(
            f"release-only smoke not run: set {RELEASE_ROOT_VARIABLE} to the "
            "assembled collision-study root containing the 12,672-row cube"
        )

    study_root = Path(configured).resolve()
    summary = study_root / SUMMARY_RELATIVE
    assert summary.is_file(), (
        f"{RELEASE_ROOT_VARIABLE} was supplied, but the required release input "
        f"is absent: {summary}"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SMOKE_CLI),
            "--study-root",
            str(study_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PASS: portable reanalysis CLI reproduced all smoke-test invariants" in completed.stdout
