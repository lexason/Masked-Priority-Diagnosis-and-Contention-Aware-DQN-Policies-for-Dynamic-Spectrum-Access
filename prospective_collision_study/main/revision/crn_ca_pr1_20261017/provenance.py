"""Fail-closed source freeze and result sealing for CRN-CA-PR1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from main.revision.crn_ca_pr1_20261017.config import (
    ALL_POLICIES,
    DEFAULT_RESULTS_ROOT,
    DEPENDENCY_FILENAMES,
    FACTORS,
    FREEZE_PATH,
    LEARNED_POLICIES,
    PACKAGE_DIR,
    PROTOCOL_VERSION,
    REPOSITORY_ROOT,
    SOURCE_FILENAMES,
    STUDY_ID,
    TRAINING_SEEDS,
    protocol_payload,
    protocol_sha256,
)

_RUNTIME_DISTRIBUTIONS = ("matplotlib", "numpy", "pandas", "scipy", "torch")
_REQUIRED_ANALYSIS_FILES = (
    "analysis/summary.json",
    "analysis/comparisons.csv",
    "analysis/cell_effects.csv",
    "analysis/checkpoint_effects.csv",
    "analysis/factorial_means.csv",
    "analysis/mechanism_certificates.csv",
    "analysis/figure_collision_aware_data.csv",
    "analysis/fig_collision_aware_results.pdf",
    "analysis/FIGURE_MANIFEST.json",
)
_EXPECTED_ANALYSIS_ARTIFACTS = {
    "summary": "analysis/summary.json",
    "comparisons": "analysis/comparisons.csv",
    "cell_effects": "analysis/cell_effects.csv",
    "checkpoint_effects": "analysis/checkpoint_effects.csv",
    "factorial_means": "analysis/factorial_means.csv",
    "mechanism": "analysis/mechanism_certificates.csv",
}
_EXPECTED_FIGURE_ARTIFACTS = {
    "figure_data": "analysis/figure_collision_aware_data.csv",
    "figure": "analysis/fig_collision_aware_results.pdf",
}
_EXPECTED_FIGURE_RECORD_COUNTS = {
    ("a", "cell_effect"): len(FACTORS),
    ("a", "checkpoint_effect"): len(TRAINING_SEEDS),
    ("a", "equal_cell_macro_effect"): 1,
    ("b", "cell_effect"): len(FACTORS),
    ("b", "checkpoint_effect"): len(TRAINING_SEEDS),
    ("b", "equal_cell_macro_effect"): 1,
    ("c", "factorial_cell_mean"): len(ALL_POLICIES) * len(FACTORS),
    ("d", "checkpoint_mechanism"): len(TRAINING_SEEDS),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes() -> dict[str, str]:
    missing = [name for name in SOURCE_FILENAMES if not (PACKAGE_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze incomplete source set: {missing}")
    return {name: sha256_file(PACKAGE_DIR / name) for name in sorted(SOURCE_FILENAMES)}


def _dependency_hashes() -> dict[str, str]:
    missing = [
        name for name in DEPENDENCY_FILENAMES if not (REPOSITORY_ROOT / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"cannot freeze incomplete dependency set: {missing}")
    return {
        name: sha256_file(REPOSITORY_ROOT / name)
        for name in sorted(DEPENDENCY_FILENAMES)
    }


def _runtime_versions() -> dict[str, Any]:
    distributions: dict[str, str] = {}
    for name in _RUNTIME_DISTRIBUTIONS:
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required runtime distribution is missing: {name}") from error
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "system": platform.system(),
        "machine": platform.machine(),
        "distributions": distributions,
    }


def initialise_freeze(path: Path = FREEZE_PATH) -> dict[str, Any]:
    """Create the immutable pre-outcome manifest; never overwrite one."""
    if path.exists():
        raise FileExistsError(f"protocol freeze already exists: {path}")
    sources = _source_hashes()
    dependencies = _dependency_hashes()
    runtime = _runtime_versions()
    payload: dict[str, Any] = {
        "schema_version": 2,
        "study_id": STUDY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha256(),
        "protocol": protocol_payload(),
        "source_sha256": sources,
        "aggregate_source_sha256": canonical_sha256(sources),
        "dependency_sha256": dependencies,
        "aggregate_dependency_sha256": canonical_sha256(dependencies),
        "runtime_versions": runtime,
        "aggregate_frozen_inputs_sha256": canonical_sha256(
            {
                "dependencies": dependencies,
                "protocol": protocol_payload(),
                "runtime": runtime,
                "sources": sources,
            }
        ),
    }
    _atomic_json_new(path, payload)
    verify_freeze(path)
    return payload


def verify_freeze(path: Path = FREEZE_PATH) -> dict[str, Any]:
    """Fail unless protocol constants and every inventoried source are unchanged."""
    if not path.is_file():
        raise FileNotFoundError(
            f"missing protocol freeze {path}; production execution is prohibited"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported protocol-freeze schema")
    if payload.get("study_id") != STUDY_ID:
        raise ValueError("study ID mismatch in protocol freeze")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("protocol version mismatch in protocol freeze")
    if payload.get("protocol_sha256") != protocol_sha256():
        raise ValueError("frozen protocol constants no longer match config.py")
    if payload.get("protocol") != protocol_payload():
        raise ValueError("frozen protocol payload no longer matches config.py")
    current = _source_hashes()
    if payload.get("source_sha256") != current:
        changed = sorted(
            name
            for name in set(current) | set(payload.get("source_sha256", {}))
            if current.get(name) != payload.get("source_sha256", {}).get(name)
        )
        raise ValueError(f"source freeze violation: {changed}")
    if payload.get("aggregate_source_sha256") != canonical_sha256(current):
        raise ValueError("aggregate source digest mismatch")
    dependencies = _dependency_hashes()
    if payload.get("dependency_sha256") != dependencies:
        changed = sorted(
            name
            for name in set(dependencies) | set(payload.get("dependency_sha256", {}))
            if dependencies.get(name) != payload.get("dependency_sha256", {}).get(name)
        )
        raise ValueError(f"dependency freeze violation: {changed}")
    if payload.get("aggregate_dependency_sha256") != canonical_sha256(dependencies):
        raise ValueError("aggregate dependency digest mismatch")
    runtime = _runtime_versions()
    if payload.get("runtime_versions") != runtime:
        raise ValueError("frozen runtime versions no longer match the active environment")
    expected_inputs_digest = canonical_sha256(
        {
            "dependencies": dependencies,
            "protocol": protocol_payload(),
            "runtime": runtime,
            "sources": current,
        }
    )
    if payload.get("aggregate_frozen_inputs_sha256") != expected_inputs_digest:
        raise ValueError("aggregate frozen-input digest mismatch")
    return payload


def _required_result_paths(results_root: Path) -> tuple[Path, ...]:
    required = [
        results_root / "evaluation" / "summary.csv",
        results_root / "VALIDATION_REPORT.json",
        *(results_root / relative for relative in _REQUIRED_ANALYSIS_FILES),
    ]
    for policy in LEARNED_POLICIES:
        for seed in TRAINING_SEEDS:
            run_dir = results_root / "training" / policy / f"seed_{seed}"
            required.extend(
                (run_dir / "final.pt", run_dir / "metadata.json", run_dir / "episodes.csv")
            )
    return tuple(required)


def _validate_required_results(
    results_root: Path, *, frozen_inputs_sha256: str | None = None
) -> None:
    required_paths = _required_result_paths(results_root)
    missing = [
        path.relative_to(results_root).as_posix()
        for path in required_paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"cannot seal incomplete results: {missing[:5]}")
    empty = [
        path.relative_to(results_root).as_posix()
        for path in required_paths
        if path.stat().st_size == 0
    ]
    if empty:
        raise ValueError(f"cannot seal empty required artifacts: {empty[:5]}")

    analysis_summary_path = results_root / "analysis" / "summary.json"
    analysis_summary = json.loads(analysis_summary_path.read_text(encoding="utf-8"))
    if analysis_summary.get("study_id") != STUDY_ID:
        raise ValueError("analysis summary study ID mismatch")
    if analysis_summary.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("analysis summary protocol version mismatch")
    if analysis_summary.get("analysis_status") != "COMPLETE":
        raise ValueError("analysis summary is not structurally complete")
    if analysis_summary.get("artifacts") != _EXPECTED_ANALYSIS_ARTIFACTS:
        raise ValueError("analysis summary artifact inventory mismatch")

    evaluation_path = results_root / "evaluation" / "summary.csv"
    figure_manifest_path = results_root / "analysis" / "FIGURE_MANIFEST.json"
    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    if figure_manifest.get("schema_version") != 2:
        raise ValueError("unsupported figure-manifest schema")
    if figure_manifest.get("study_id") != STUDY_ID:
        raise ValueError("figure manifest study ID mismatch")
    if figure_manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("figure manifest protocol version mismatch")
    if (
        frozen_inputs_sha256 is not None
        and figure_manifest.get("frozen_inputs_sha256") != frozen_inputs_sha256
    ):
        raise ValueError("figure manifest frozen-input identity mismatch")
    if int(figure_manifest.get("panels", 0)) < 4:
        raise ValueError("figure manifest contains fewer than four scientific panels")
    if figure_manifest.get("source_summary") != "evaluation/summary.csv":
        raise ValueError("figure manifest points to an unexpected evaluation summary")
    if figure_manifest.get("source_summary_sha256") != sha256_file(evaluation_path):
        raise ValueError("figure manifest evaluation digest mismatch")
    if figure_manifest.get("analysis_summary") != "analysis/summary.json":
        raise ValueError("figure manifest points to an unexpected analysis summary")
    if figure_manifest.get("analysis_summary_sha256") != sha256_file(
        analysis_summary_path
    ):
        raise ValueError("figure manifest analysis-summary digest mismatch")
    expected_sources = {
        "evaluation/summary.csv": sha256_file(evaluation_path),
        **{
            relative: sha256_file(results_root / relative)
            for relative in _EXPECTED_ANALYSIS_ARTIFACTS.values()
        },
    }
    if figure_manifest.get("source_sha256") != dict(sorted(expected_sources.items())):
        raise ValueError("figure manifest source inventory mismatch")
    for path_key, digest_key in (
        ("figure_data", "figure_data_sha256"),
        ("figure", "figure_sha256"),
    ):
        relative = figure_manifest.get(path_key)
        if relative != _EXPECTED_FIGURE_ARTIFACTS[path_key]:
            raise ValueError(f"figure manifest has unexpected {path_key}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"figure manifest {path_key} escapes results root")
        artifact = results_root / relative_path
        try:
            artifact.relative_to(results_root)
        except ValueError as error:
            raise ValueError(f"figure manifest {path_key} escapes results root") from error
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if figure_manifest.get(digest_key) != sha256_file(artifact):
            raise ValueError(f"figure manifest digest mismatch for {relative}")

    figure_data_path = results_root / _EXPECTED_FIGURE_ARTIFACTS["figure_data"]
    with figure_data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        figure_rows = list(reader)
    if not figure_rows or not {"panel", "record_type"} <= set(reader.fieldnames or ()):
        raise ValueError("figure data lacks panel provenance")
    observed_counts: dict[tuple[str, str], int] = {}
    for row in figure_rows:
        key = (row["panel"], row["record_type"])
        observed_counts[key] = observed_counts.get(key, 0) + 1
    if observed_counts != _EXPECTED_FIGURE_RECORD_COUNTS:
        raise ValueError("figure data does not contain the four prespecified panels")
    figure_path = results_root / _EXPECTED_FIGURE_ARTIFACTS["figure"]
    if not figure_path.read_bytes().startswith(b"%PDF-"):
        raise ValueError("figure artifact is not a PDF")


def _production_hashes(results_root: Path, seal_path: Path) -> dict[str, str]:
    files = sorted(
        path
        for path in results_root.rglob("*")
        if path.is_file() and path != seal_path and "smoke" not in path.parts
    )
    if not files:
        raise FileNotFoundError("no production results to seal")
    symlinks = [path for path in files if path.is_symlink()]
    if symlinks:
        raise ValueError(f"result inventory contains symbolic links: {symlinks[:3]}")
    return {
        path.relative_to(results_root).as_posix(): sha256_file(path) for path in files
    }


def seal_results(results_root: Path = DEFAULT_RESULTS_ROOT) -> dict[str, Any]:
    """Validate and inventory the complete results; never overwrite a seal."""
    freeze = verify_freeze()
    seal_path = results_root / "RESULTS_SEAL.json"
    if seal_path.exists():
        raise FileExistsError(f"results already sealed: {seal_path}")
    from main.revision.crn_ca_pr1_20261017.validation import validate

    report = validate(results_root, smoke=False)
    if report.get("status") != "PASS":
        raise ValueError("production validation did not pass")
    _validate_required_results(
        results_root,
        frozen_inputs_sha256=freeze["aggregate_frozen_inputs_sha256"],
    )
    hashes = _production_hashes(results_root, seal_path)
    payload = {
        "schema_version": 2,
        "study_id": STUDY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(),
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "protocol_freeze_sha256": sha256_file(FREEZE_PATH),
        "frozen_inputs_sha256": freeze["aggregate_frozen_inputs_sha256"],
        "validation_report_sha256": sha256_file(
            results_root / "VALIDATION_REPORT.json"
        ),
        "file_sha256": hashes,
        "aggregate_results_sha256": canonical_sha256(hashes),
    }
    _atomic_json_new(seal_path, payload)
    return verify_results_seal(results_root)


def verify_results_seal(results_root: Path = DEFAULT_RESULTS_ROOT) -> dict[str, Any]:
    """Fail unless the seal and every sealed production artifact remain exact."""
    freeze = verify_freeze()
    seal_path = results_root / "RESULTS_SEAL.json"
    if not seal_path.is_file():
        raise FileNotFoundError(f"missing result seal: {seal_path}")
    payload = json.loads(seal_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported result-seal schema")
    if payload.get("study_id") != STUDY_ID:
        raise ValueError("result-seal study ID mismatch")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("result-seal protocol version mismatch")
    if payload.get("protocol_sha256") != protocol_sha256():
        raise ValueError("result-seal protocol digest mismatch")
    if payload.get("protocol_freeze_sha256") != sha256_file(FREEZE_PATH):
        raise ValueError("result seal refers to a different protocol freeze")
    if payload.get("frozen_inputs_sha256") != freeze["aggregate_frozen_inputs_sha256"]:
        raise ValueError("result seal refers to different frozen inputs")

    from main.revision.crn_ca_pr1_20261017.validation import validate

    report = validate(results_root, smoke=False)
    if report.get("status") != "PASS":
        raise ValueError("sealed production validation did not pass")
    _validate_required_results(
        results_root,
        frozen_inputs_sha256=freeze["aggregate_frozen_inputs_sha256"],
    )
    validation_digest = sha256_file(results_root / "VALIDATION_REPORT.json")
    if payload.get("validation_report_sha256") != validation_digest:
        raise ValueError("sealed validation-report digest mismatch")
    current = _production_hashes(results_root, seal_path)
    if payload.get("file_sha256") != current:
        changed = sorted(
            name
            for name in set(current) | set(payload.get("file_sha256", {}))
            if current.get(name) != payload.get("file_sha256", {}).get(name)
        )
        raise ValueError(f"sealed result inventory changed: {changed[:5]}")
    if payload.get("aggregate_results_sha256") != canonical_sha256(current):
        raise ValueError("aggregate sealed-result digest mismatch")
    return payload


def _atomic_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite existing output: {path}") from None
    except BaseException:
        raise
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    subparsers.add_parser("verify")
    seal_parser = subparsers.add_parser("seal-results")
    seal_parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    verify_seal_parser = subparsers.add_parser("verify-seal")
    verify_seal_parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT
    )
    args = parser.parse_args()
    if args.command == "init":
        payload = initialise_freeze()
    elif args.command == "verify":
        payload = verify_freeze()
    elif args.command == "seal-results":
        payload = seal_results(args.results_root)
    else:
        payload = verify_results_seal(args.results_root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
