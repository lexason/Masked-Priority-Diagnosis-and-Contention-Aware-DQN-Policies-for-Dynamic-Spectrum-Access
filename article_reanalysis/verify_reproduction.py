#!/usr/bin/env python3
"""Compare clean-room article reanalysis with released Gold outputs.

CSV outputs are required to be byte-identical. JSON structure and nonnumeric
values are exact; finite floating-point values use a tight tolerance to admit
last-bit variation across NumPy/SciPy builds while rejecting scientific drift.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ABSOLUTE_TOLERANCE = 1e-12
RELATIVE_TOLERANCE = 1e-12


class VerificationError(RuntimeError):
    """Raised when clean-room output differs materially from Gold."""


def compare_values(
    gold: Any, candidate: Any, location: str, differences: list[str], maxima: dict[str, float]
) -> None:
    if isinstance(gold, bool) or isinstance(candidate, bool):
        if gold is not candidate:
            differences.append(f"{location}: {candidate!r} != {gold!r}")
        return
    if isinstance(gold, (int, float)) and isinstance(candidate, (int, float)):
        gold_float = float(gold)
        candidate_float = float(candidate)
        if not math.isfinite(gold_float) or not math.isfinite(candidate_float):
            if gold_float != candidate_float:
                differences.append(f"{location}: nonfinite values differ")
            return
        absolute = abs(gold_float - candidate_float)
        maxima["maximum_absolute_difference"] = max(maxima["maximum_absolute_difference"], absolute)
        if not math.isclose(
            gold_float,
            candidate_float,
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        ):
            differences.append(f"{location}: {candidate!r} != {gold!r}")
        return
    if isinstance(gold, dict) and isinstance(candidate, dict):
        if set(gold) != set(candidate):
            differences.append(
                f"{location}: keys differ; missing={sorted(set(gold) - set(candidate))}, "
                f"extra={sorted(set(candidate) - set(gold))}"
            )
            return
        for key in sorted(gold):
            compare_values(gold[key], candidate[key], f"{location}.{key}", differences, maxima)
        return
    if isinstance(gold, list) and isinstance(candidate, list):
        if len(gold) != len(candidate):
            differences.append(f"{location}: list lengths differ ({len(candidate)} != {len(gold)})")
            return
        for index, (gold_item, candidate_item) in enumerate(zip(gold, candidate, strict=True)):
            compare_values(
                gold_item,
                candidate_item,
                f"{location}[{index}]",
                differences,
                maxima,
            )
        return
    if type(gold) is not type(candidate) or gold != candidate:
        differences.append(f"{location}: {candidate!r} != {gold!r}")


def verify(gold_root: Path, candidate_root: Path) -> dict[str, Any]:
    exact_files = (
        "reanalysis/attempted_only_logical_audit.csv",
        "reanalysis/mechanism_audit_figure_data.csv",
    )
    json_files = (
        "reanalysis/reanalysis_output.json",
        "inference/recomputed.json",
    )
    exact_results = []
    for relative in exact_files:
        gold = (gold_root / relative).read_bytes()
        candidate = (candidate_root / relative).read_bytes()
        if gold != candidate:
            raise VerificationError(f"byte mismatch: {relative}")
        exact_results.append(relative)

    maxima = {"maximum_absolute_difference": 0.0}
    json_results = []
    for relative in json_files:
        gold = json.loads((gold_root / relative).read_text(encoding="utf-8"))
        candidate = json.loads((candidate_root / relative).read_text(encoding="utf-8"))
        differences: list[str] = []
        compare_values(gold, candidate, "$", differences, maxima)
        if differences:
            preview = "; ".join(differences[:10])
            raise VerificationError(f"JSON mismatch in {relative}: {preview}")
        json_results.append(relative)
    return {
        "valid": True,
        "byte_identical_csv": exact_results,
        "tolerance_validated_json": json_results,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        **maxima,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.gold_root, args.candidate_root)
    except (OSError, ValueError, json.JSONDecodeError, VerificationError) as exc:
        print(f"reproduction verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
