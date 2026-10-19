#!/usr/bin/env python3
"""Release-stage smoke CLI for reanalyse.py using a supplied assembled root.

This module intentionally defines no pytest tests.  It requires the complete
12,672-row release input and must be invoked explicitly with ``--study-root``.
The lightweight workspace pytest suite exercises its execution contract through
``test_reanalyse_smoke_contract.py`` and reports a skip when that release-only
input has not been supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    """Run the standalone portable smoke test.

    Argument parsing is intentionally confined to this entry point so pytest
    can import the module during directory-wide collection.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name("reanalyse.py"),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="tccn-reanalysis-smoke-") as temp:
        out_dir = Path(temp)
        completed = subprocess.run(
            [
                sys.executable,
                str(args.script.resolve()),
                "--study-root",
                str(args.study_root.resolve()),
                "--out-dir",
                str(out_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        reported = json.loads(completed.stdout)
        output = json.loads(
            (out_dir / "reanalysis_output.json").read_text(encoding="utf-8")
        )

        assert output["row_counts"] == {"full_dqn": 3840, "total": 12672}
        assert not output["summary_identity_violations"]
        assert not output["original_seven_action_reproduction_errors"]
        original = output["original_seven_action"]
        assert original["full_pooled_eligible"] == 160_503_847
        assert original["full_pooled_abstentions"] == 140_383_782
        assert original["full_pooled_attempts"] == 20_120_065
        attempted = output["attempted_only_logical_audit"]
        assert math.isclose(
            attempted["validation_equal_checkpoint_factor_selected_agreement"],
            0.48691837704984364,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        assert len(attempted["per_checkpoint"]) == 20
        for record in attempted["per_checkpoint"]:
            assert record["calibration_mask_coverage"] == 63
            assert record["validation_mask_coverage"] == 63
            for key in ("calibration_certificate", "validation_certificate"):
                certificate = record[key]
                assert certificate["edge_count"] == 30
                assert certificate["acyclic"] is False
                assert certificate["representing_order_count"] == 0

        control = output["coordinate_positive_control"]
        assert control["case_count"] == 378
        assert control["logical_exact_agreement"] == 1.0
        assert control["logical"]["edge_count"] == 15
        assert control["logical"]["acyclic"] is True
        assert control["logical"]["representing_order_count"] == 1
        assert control["physical_masks_with_multiple_selected_actions"] == 57
        assert control["physical"]["edge_count"] == 30
        assert control["physical"]["acyclic"] is False

        with (out_dir / "attempted_only_logical_audit.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            assert len(list(csv.DictReader(handle))) == 20
        with (out_dir / "mechanism_audit_figure_data.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            assert len(list(csv.DictReader(handle))) == 80
        assert reported["output_sha256"]
        assert reported["attempted_audit_csv_sha256"]
        assert reported["figure_csv_sha256"]

    print("PASS: portable reanalysis CLI reproduced all smoke-test invariants")


if __name__ == "__main__":
    main()
