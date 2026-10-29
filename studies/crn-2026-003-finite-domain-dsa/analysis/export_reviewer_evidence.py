#!/usr/bin/env python3
# ruff: noqa: E501
"""Build compact, deterministic reviewer evidence from sealed study inputs.

This is a read-only derivation.  It does not train, simulate, refit, or change
any scientific result.  The exporter intentionally fails closed if a pinned
input, checkpoint, inventory, row count, join, or arithmetic invariant drifts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import struct
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

SCRIPT = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT.parents[3]
DEFAULT_STUDY_ROOT = SCRIPT.parents[1]
DEFAULT_GITHUB_RECEIPT = Path(
    "studies/crn-2026-003-finite-domain-dsa/release/"
    "github_current_commit_v1.3.0_receipt.json"
)

# Workspace lookup paths and public-deposit locators are deliberately separate.
# The former are used only to read sealed inputs; only the latter may be emitted.
_LOCAL_ARCHIVE_ROOT = "main_" + "archive"
_RAW_GIT_STATE_KEY = "status_" + "porcelain"
ARCHIVED_PRIORITY_AUDIT_SOURCE = Path(_LOCAL_ARCHIVE_ROOT).joinpath(
    "revision", "v65_48h", "audit_rerun_certificate",
    "priority_collapse_audit.json",
).as_posix()
ARCHIVED_PRIORITY_SCRIPT_SOURCE = Path(_LOCAL_ARCHIVE_ROOT).joinpath(
    "revision", "audit_priority_collapse.py",
).as_posix()
ARCHIVED_BOUNDARY_AUDIT_SOURCE = Path(_LOCAL_ARCHIVE_ROOT).joinpath(
    "revision", "v65_48h", "audit_rerun", "feature_adversarial_audit.json",
).as_posix()
ARCHIVED_BOUNDARY_SCRIPT_SOURCE = Path(_LOCAL_ARCHIVE_ROOT).joinpath(
    "revision", "audit_feature_sensitivity.py",
).as_posix()
REPLICATION_PRIORITY_SCRIPT_SOURCE = Path(_LOCAL_ARCHIVE_ROOT).joinpath(
    "revision", "crn_mp_pr1_20261017", "audit_priority_collapse.py",
).as_posix()
PROSPECTIVE_FREEZE_SOURCE = Path(_LOCAL_ARCHIVE_ROOT).joinpath(
    "revision", "crn_ca_pr1_20261017", "PROTOCOL_FREEZE.json",
).as_posix()

PUBLIC_COMPACT_REPLICATION = (
    "independent_replications/independent_masked_priority/"
    "CRN_MP_PR1_20261017_Compact_Replication.zip"
)
PUBLIC_PROSPECTIVE_ROOT = "prospective_collision_study"

PUBLIC_OUTPUT_NAMES = (
    "analysis_input_documentation.json",
    "analysis_input_index.csv",
    "boundary_witnesses.json",
    "checkpoint_registry.json",
    "claim_to_artifact_manifest.json",
    "paired_effects.json",
    "reviewer_evidence_manifest.json",
    "statistical_recipe_and_results.json",
    "study_a_certificates.json",
    "study_b_priority_audits.json",
)
_LOCAL_ARCHIVE_PATTERN = re.escape("main_" + "archive") + r"/"
_EDITORIAL_PATTERN = "Editorial" + "_Review"
_WORKSPACE_USER_PATTERN = "sdr" + "-research"
_CREDENTIAL_FILE_PATTERN = re.escape("." + "env" + "." + "zenodo")
_ZENODO_CREDENTIAL_NAME = "ZENODO" + "_ACCESS_TOKEN"
_GITHUB_CREDENTIAL_NAME = "GITHUB" + "_TOKEN"
_PRIVATE_KEY_MARKER = "-" * 5 + "BEGIN " + "(?:RSA |EC |OPENSSH )?" + "PRIVATE KEY" + "-" * 5

PUBLIC_FORBIDDEN_PATTERNS = (
    (
        "absolute filesystem path",
        re.compile(
            r"(?<![A-Za-z0-9:/])/(?!/)(?:[A-Za-z0-9._~-]+/)"
            r"+[A-Za-z0-9._~-]+"
        ),
    ),
    (
        "Windows user path",
        re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]{1,2}Users[\\/]{1,2}"),
    ),
    ("temporary filesystem path", re.compile(r"(?i)(?<![A-Za-z0-9])/(?:tmp|private/tmp)(?:/|\b)")),
    ("workspace username", re.compile(rf"(?i)\b{re.escape(_WORKSPACE_USER_PATTERN)}\b")),
    ("editorial workspace locator", re.compile(_EDITORIAL_PATTERN, re.IGNORECASE)),
    (
        "coauthor-feedback filename",
        re.compile("Final Feedback_" + "From Co-authors", re.IGNORECASE),
    ),
    (
        "editorial-blueprint filename",
        re.compile("TCCN_Author_" + "Revision_Blueprint", re.IGNORECASE),
    ),
    (
        "private workspace protocol",
        re.compile("SIX_SIGMA_" + "RESEARCH_PROTOCOL", re.IGNORECASE),
    ),
    ("raw Git workspace status", re.compile(_RAW_GIT_STATE_KEY, re.IGNORECASE)),
    ("private local archive path", re.compile(_LOCAL_ARCHIVE_PATTERN, re.IGNORECASE)),
    ("credential environment file", re.compile(_CREDENTIAL_FILE_PATTERN, re.IGNORECASE)),
    (
        "credential name",
        re.compile(
            rf"(?i)\b(?:{re.escape(_ZENODO_CREDENTIAL_NAME)}|"
            rf"{re.escape(_GITHUB_CREDENTIAL_NAME)}|"
            r"access[_-]?token|api[_-]?key|client[_-]?secret|pass"
            r"word|pass"
            r"wd)\b"
        ),
    ),
    ("GitHub credential", re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Bearer credential", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("private key", re.compile(_PRIVATE_KEY_MARKER)),
)

ARCHIVED_SEEDS = (42, 7, 1007, 2007, 101, 202, 303, 404)
REPLICATION_SEEDS = (10853, 64172, 34184, 69087, 14660, 63964, 87425, 63575)
PROSPECTIVE_SEEDS = (
    71101, 71209, 71317, 71423, 71531, 71639, 71747, 71851, 71959, 72067,
    72173, 72281, 72389, 72497, 72601, 72709, 72817, 72923, 73031, 73139,
)
PROSPECTIVE_VARIANTS = (
    "collision_aware_dqn",
    "history_free_dqn",
    "coordination_blind_dqn",
)
FACTORS = (
    "iid_o30_a10", "iid_o30_a25", "iid_o60_a10", "iid_o60_a25",
    "markov_o30_a10", "markov_o30_a25", "markov_o60_a10",
    "markov_o60_a25",
)
ENVIRONMENT_SEEDS = (
    52010101, 52010211, 52010307, 52010419, 52010503, 52010613,
    52010709, 52010821, 52010907, 52011019, 52011103, 52011217,
    52011301, 52011409, 52011503, 52011619, 52011707, 52011811,
    52011919, 52012027, 52012119, 52012233, 52012307, 52012419,
)
PROSPECTIVE_POLICIES = (
    *PROSPECTIVE_VARIANTS,
    "phase_orthogonal",
    "quality_greedy_persistent",
    "round_robin",
    "uniform_legal",
    "static_priority",
    "centralized_quality_matching",
)
COMPARATORS = (
    "phase_orthogonal",
    "history_free_dqn",
    "coordination_blind_dqn",
    "quality_greedy_persistent",
    "round_robin",
    "uniform_legal",
    "static_priority",
    "centralized_quality_matching",
)

# These hashes are release identities, not values discovered at run time.
PINNED_INPUT_SHA256 = {
    "results/camera_ready_20261014_corrected/training/training_manifest.json":
        "9adc922102899041e43526ef99c0a4d08edcc68eba4780e1763fecead1f67220",
    "results/camera_ready_20261014_corrected/provenance/source_manifest.json":
        "9e9e3162048480ea141390a9cf35e57ff4c56974f1541b16a63aaaed1b83bab2",
    "results/camera_ready_20261014_corrected/core/core_manifest.json":
        "6d6b95613f1d2b1ab100874d5247c66a7e183d9035fb781e391925234d61b339",
    "results/crn_mp_pr1_20261017/training/training_manifest.json":
        "795bedb865c3f3dc461a85bb3ba89668d02613b13e88d979692ea9b5dfbdcb2e",
    "results/crn_mp_pr1_20261017/provenance/source_manifest.json":
        "a15b343da9a07220877e080663b13d0165cbf2be3ef9977a2a0891bac65132cb",
    "results/crn_mp_pr1_20261017/core/core_manifest.json":
        "320ff3b2783408b6f4b50c34682bb2b9a4d6498cacf1a2524d4ac05a54705ee5",
    "results/crn_mp_pr1_20261017/REPLICATION_RESULTS_SEAL.json":
        "823953171434a565a70695528efcaa35887cb44573fd828b1322a9d774be7f9f",
    ARCHIVED_PRIORITY_AUDIT_SOURCE:
        "5f9e6b90feacf59d67f06e1e3bd5fe1278b7f8c111da4025cdf57ea7015a19d6",
    "results/crn_mp_pr1_20261017/audits/priority_collapse/priority_collapse_audit.json":
        "f687847da8e41ac3f4896fe31a421babf91b375a7a88a34ea08130312768830f",
    ARCHIVED_BOUNDARY_AUDIT_SOURCE:
        "ebfb66657c2944f393598bcd24b441d99442295e5ae166232f2b188c3a325dab",
    PROSPECTIVE_FREEZE_SOURCE:
        "2877300949e33a3ea63441501dd31621427874760accb0826d135708ac05afc0",
    "results/crn_ca_pr1_20261017/RESULTS_SEAL.json":
        "d5e9841d6e3bdb243b784c4499a44b23532e237e7364db9d9952710e8713f2f8",
    "results/crn_ca_pr1_20261017/evaluation/summary.csv":
        "89fd4e331634213d197f5d8a7c126dfb6cbc48b9c3a7b99f1adef5d9c09187ae",
    "results/crn_ca_pr1_20261017/analysis/checkpoint_effects.csv":
        "e8ad6ce8a9d7e10d79b8865d07aa228e4daa65a161fd73137a2bbecdc57817ff",
    "results/crn_ca_pr1_20261017/analysis/mechanism_certificates.csv":
        "126f2b70b26fe376f04c8586393e1d529611a0ec8682d1eaa84fc943d0e96ce2",
    "studies/crn-2026-003-finite-domain-dsa/evidence/gold/reanalysis/reanalysis_output.json":
        "ab6ccafc932ddca28a2aec40d6e8f46091ab17992955aed988a764649affe2e0",
    "studies/crn-2026-003-finite-domain-dsa/evidence/gold/inference/recomputed.json":
        "5199f52a44d119010207c2e0358217d2a14ed238f57bb9a743f76f7c189ccca3",
    "studies/crn-2026-003-finite-domain-dsa/evidence/gold/inference/parent_summary.json":
        "c3fc74b020a801baee303e1d6741256af89c8c53d540f4e18de6da1ddcc7d812",
}

SUMMARY_FIELDS = (
    "study_id", "protocol_version", "protocol_sha256", "run_id", "smoke",
    "policy", "training_seed", "checkpoint_sha256", "factor_id",
    "occupancy_process", "occupancy_probability", "arrival_probability",
    "replication_index", "environment_seed", "logical_to_physical",
    "burn_in_steps", "measurement_steps", "trace_sha256", "starting_backlog",
    "arrivals", "deliveries", "ending_backlog", "network_goodput",
    "finite_window_service_fraction", "capacity_ceiling_per_slot",
    "opportunity_utilization", "attempted", "singleton_attempts",
    "collision_attempts", "collision_attempt_rate", "link_failures",
    "conditional_link_failure_rate", "voluntary_deferrals", "forced_deferrals",
    "mean_network_backlog", "delivery_jain_fairness", "zero_delivery_run",
    "priority_eligible_decisions", "priority_matches", "static_priority_agreement",
    "logical_action_counts", "physical_action_counts", "mask_action_counts_shape",
    "mask_action_counts",
)


class EvidenceError(RuntimeError):
    """A fail-closed evidence or join invariant failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def assert_public_text_safe(label: str, data: bytes | str) -> None:
    """Reject workspace-local paths, user identities, and credential material."""
    if isinstance(data, bytes):
        try:
            value = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvidenceError(f"public reviewer output is not UTF-8: {label}") from exc
    else:
        value = data
    for description, pattern in PUBLIC_FORBIDDEN_PATTERNS:
        require(
            pattern.search(value) is None,
            f"public reviewer output contains {description}: {label}",
        )


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False,
    ) + "\n"
    assert_public_text_safe(path.name, text)
    path.write_text(text, encoding="utf-8")


def rel(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _package_member(locator: str) -> dict[str, str]:
    return {"mode": "package_member", "locator": locator}


def _nested_package_member(inner_member: str) -> dict[str, str]:
    return {
        "mode": "nested_archive_member",
        "locator": f"{PUBLIC_COMPACT_REPLICATION}::{inner_member}",
        "outer_member": PUBLIC_COMPACT_REPLICATION,
        "inner_member": inner_member,
    }


def _hash_only(reason: str) -> dict[str, str]:
    return {"mode": "hash_identity_only", "reason": reason}


def _public_repository(relative: str) -> dict[str, str]:
    return {
        "mode": "public_repository",
        "repository_path": relative,
    }


def public_access_for_source(relative: str) -> dict[str, str]:
    """Map a private read locator to a truthful, public-stable access route.

    No private workspace locator is returned. Inputs absent from the compact
    deposit are explicitly represented by their already-exported hash identity.
    """
    fixed_members = {
        ARCHIVED_PRIORITY_AUDIT_SOURCE: "outputs/priority_collapse_audit.json",
        ARCHIVED_PRIORITY_SCRIPT_SOURCE: "scripts/audits/audit_priority_collapse.py",
        ARCHIVED_BOUNDARY_AUDIT_SOURCE: "outputs/feature_adversarial_audit.json",
        ARCHIVED_BOUNDARY_SCRIPT_SOURCE: "scripts/audits/audit_feature_sensitivity.py",
        PROSPECTIVE_FREEZE_SOURCE: (
            f"{PUBLIC_PROSPECTIVE_ROOT}/main/revision/"
            "crn_ca_pr1_20261017/PROTOCOL_FREEZE.json"
        ),
    }
    if relative in fixed_members:
        return _package_member(fixed_members[relative])
    if relative == REPLICATION_PRIORITY_SCRIPT_SOURCE:
        return _nested_package_member(
            "source/main/revision/crn_mp_pr1_20261017/audit_priority_collapse.py"
        )

    archived_prefix = "results/camera_ready_20261014_corrected/"
    if relative.startswith(archived_prefix):
        suffix = relative.removeprefix(archived_prefix)
        match = re.fullmatch(
            r"training/seed_(\d+)/dqn_seed\1_final\.pt", suffix,
        )
        if match:
            seed = match.group(1)
            return _package_member(
                f"checkpoints/seed_{seed}/dqn_seed{seed}_final.pt"
            )
        return _hash_only(
            "sealed source input is omitted from the compact deposit; its "
            "SHA-256 identity and derived reviewer record are retained"
        )

    replication_prefix = "results/crn_mp_pr1_20261017/"
    if relative == (
        replication_prefix + "provenance/crn_mp_pr1_20261017_protocol.md"
    ):
        return _nested_package_member(
            "source/main/revision/crn_mp_pr1_20261017_protocol.md"
        )
    if relative.startswith(replication_prefix):
        suffix = relative.removeprefix(replication_prefix)
        if suffix in {"REPLICATION_RESULTS_SEAL.json", "REPLICATION_INVENTORY.json"}:
            return _nested_package_member(suffix)
        return _nested_package_member(f"results/{suffix}")

    prospective_prefix = "results/crn_ca_pr1_20261017/"
    if relative.startswith(prospective_prefix):
        suffix = relative.removeprefix(prospective_prefix)
        return _package_member(
            f"{PUBLIC_PROSPECTIVE_ROOT}/results/crn_ca_pr1_20261017/{suffix}"
        )

    if relative.startswith("scenarios/"):
        return _package_member(relative)
    if relative.startswith("main/manuscript/figures/"):
        return _package_member(relative)
    if relative.startswith("main/manuscript/"):
        return _hash_only(
            "active manuscript source is intentionally excluded from public "
            "code and data packages; the published article is available via DOI"
        )
    if relative.startswith("studies/crn-2026-003-finite-domain-dsa/release/"):
        return _hash_only(
            "release-control receipt is intentionally excluded from the public "
            "code tree and reproducibility payload to avoid circular self-attestation"
        )
    if relative.startswith("studies/crn-2026-003-finite-domain-dsa/"):
        return _package_member(relative)
    raise EvidenceError(f"no public access classification for sealed source identity: {relative}")


def public_locator_for_source(relative: str) -> str:
    access = public_access_for_source(relative)
    require(
        access["mode"] in {"package_member", "nested_archive_member"},
        f"source has no public package locator: {relative}",
    )
    return access["locator"]


def verify_pins(project_root: Path) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for relative, expected in PINNED_INPUT_SHA256.items():
        path = project_root / relative
        require(path.is_file(), f"missing pinned input: {relative}")
        actual = sha256(path)
        require(actual == expected, f"pinned input drift: {relative}: {actual} != {expected}")
        verified[relative] = {"sha256": actual, "bytes": path.stat().st_size}
    return verified


def verify_file(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    actual = sha256(path)
    require(actual == expected, f"{label} hash mismatch: {actual} != {expected}: {path}")


def load_summary(project_root: Path) -> list[dict[str, str]]:
    path = project_root / "results/crn_ca_pr1_20261017/evaluation/summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == SUMMARY_FIELDS, "summary.csv schema drift")
        rows = list(reader)
    require(len(rows) == 12_672, f"expected 12,672 aggregate rows, got {len(rows)}")
    require(len({row["run_id"] for row in rows}) == 12_672, "run_id is not unique")
    require({row["smoke"].lower() for row in rows} == {"false"}, "smoke row entered evidence cube")
    require({row["policy"] for row in rows} == set(PROSPECTIVE_POLICIES), "policy set drift")
    require({row["factor_id"] for row in rows} == set(FACTORS), "factor set drift")
    require({int(row["environment_seed"]) for row in rows} == set(ENVIRONMENT_SEEDS), "environment seed drift")
    require({row["protocol_sha256"] for row in rows} == {
        "a2cfac7ac2e3c58e313f7d92af3ddfc8c0878471b0d449a38a168949b1842fcd"
    }, "protocol identity drift")
    for policy in PROSPECTIVE_VARIANTS:
        selected = [row for row in rows if row["policy"] == policy]
        require(len(selected) == 20 * 8 * 24, f"incomplete learned cube: {policy}")
        require({int(row["training_seed"]) for row in selected} == set(PROSPECTIVE_SEEDS), f"seed drift: {policy}")
    for policy in set(PROSPECTIVE_POLICIES) - set(PROSPECTIVE_VARIANTS):
        selected = [row for row in rows if row["policy"] == policy]
        require(len(selected) == 8 * 24, f"incomplete baseline cube: {policy}")
        require(all(not row["training_seed"] and not row["checkpoint_sha256"] for row in selected), f"baseline carries checkpoint: {policy}")
    return rows


def _source_version(project_root: Path, manifest_relative: str) -> dict[str, Any]:
    path = project_root / manifest_relative
    payload = read_json(path)
    source = payload.get("source", {})
    raw_git = payload.get("git")
    require(isinstance(raw_git, dict), f"source manifest lacks Git provenance: {manifest_relative}")
    branch = raw_git.get("branch")
    head = raw_git.get("head")
    tracked_diff_sha256 = raw_git.get("tracked_diff_sha256")
    raw_state = raw_git.get(_RAW_GIT_STATE_KEY)
    require(isinstance(branch, str) and branch, f"source branch is absent: {manifest_relative}")
    require(
        isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40}", head) is not None,
        f"source commit identity is malformed: {manifest_relative}",
    )
    require(
        isinstance(tracked_diff_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", tracked_diff_sha256) is not None,
        f"tracked-diff identity is malformed: {manifest_relative}",
    )
    require(isinstance(raw_state, str), f"source workspace state is absent: {manifest_relative}")
    aggregate_source_sha256 = source.get("aggregate_sha256") or payload.get("source_sha256")
    require(
        isinstance(aggregate_source_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", aggregate_source_sha256) is not None,
        f"aggregate source identity is malformed: {manifest_relative}",
    )
    public_git = {
        "branch": branch,
        "head": head,
        "tracked_diff_sha256": tracked_diff_sha256,
        "workspace_state_sha256": hashlib.sha256(raw_state.encode("utf-8")).hexdigest(),
        "workspace_state_entry_count": len(raw_state.splitlines()) if raw_state else 0,
        "working_tree_clean_at_capture": raw_state == "",
        "privacy_note": (
            "machine-local working-tree filenames are intentionally omitted; "
            "the state digest and sealed source-manifest digest preserve identity"
        ),
    }
    return {
        "manifest_access": public_access_for_source(manifest_relative),
        "manifest_sha256": sha256(path),
        "aggregate_source_sha256": aggregate_source_sha256,
        "git": public_git,
    }


def _study_a_run_records(
    project_root: Path,
    cohort_root: Path,
    seed: int,
    checkpoint_sha: str,
    sealed_summaries: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted((cohort_root / "core/runs").glob(f"*/dqn_trainseed_{seed}/*.summary.json"))
    require(len(paths) == 100, f"expected 100 Study-A core summaries for {cohort_root.name}/{seed}, got {len(paths)}")
    exported: list[dict[str, Any]] = []
    internal: list[dict[str, Any]] = []
    for path in paths:
        inventory_key = path.relative_to(cohort_root).as_posix()
        require(inventory_key in sealed_summaries, f"run summary absent from pinned core inventory: {inventory_key}")
        inventory_item = sealed_summaries[inventory_key]
        verify_file(path, inventory_item["summary_sha256"], "Study-A run summary")
        record = read_json(path)
        spec = record.get("spec", {})
        counts = record.get("counts", {})
        invariants = record.get("invariants", {})
        require(record.get("status") == "valid", f"invalid run summary: {path}")
        require(spec.get("checkpoint_training_seeds") == [seed], f"run/checkpoint seed join failed: {path}")
        require(record.get("checkpoint_sha256") == [checkpoint_sha], f"run/checkpoint hash join failed: {path}")
        require(record.get("spec_sha256") == inventory_item["spec_sha256"], f"run/inventory spec hash mismatch: {path}")
        require(record.get("artifact_sha256") == inventory_item["artifact_sha256"], f"run/inventory artifact hash mismatch: {path}")
        require(all(invariants.get(key) is True for key in (
            "acceptance_identity", "action_legality", "one_winner_le_one_minus_excess", "row_count_identity"
        )), f"run invariant failed: {path}")
        require(counts.get("rows") == counts.get("attempts") + counts.get("forced_abstentions"), f"all-occupied accounting failed: {path}")
        source_relative = rel(project_root, path)
        item = {
            "record_id": f"{spec['scenario_label']}/{spec['deployment_id']}/env_{spec['environment_seed']}",
            "summary_access": public_access_for_source(source_relative),
            "summary_sha256": sha256(path),
            "spec_sha256": record["spec_sha256"],
            "trace_sha256": record["trace_sha256"],
            "artifact_sha256": record["artifact_sha256"],
        }
        exported.append(item)
        internal.append({"path": path, "record": record})
    require(len({item["record_id"] for item in exported}) == 100, "duplicate Study-A record identifier")
    return exported, internal


def _study_a_registry(
    project_root: Path,
    cohort_name: str,
    result_dir: str,
    seeds: Sequence[int],
    audit_relative: str,
    audit_script_relative: str,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    cohort_root = project_root / result_dir
    manifest_path = cohort_root / "training/training_manifest.json"
    manifest = read_json(manifest_path)
    items = manifest.get("completion", {}).get("items", [])
    require(manifest.get("complete") is True, f"training manifest is not complete: {result_dir}")
    require(tuple(manifest.get("planned_seeds", ())) == tuple(seeds), f"training seed order drift: {result_dir}")
    require(len(items) == 8, f"expected eight selected checkpoints: {result_dir}")
    by_seed = {int(item["seed"]): item for item in items}
    require(set(by_seed) == set(seeds), f"selected checkpoint seed set drift: {result_dir}")
    audit_path = project_root / audit_relative
    script_path = project_root / audit_script_relative
    require(script_path.is_file(), f"missing retained audit implementation: {audit_script_relative}")
    source_relative = f"{result_dir}/provenance/source_manifest.json"
    source_version = _source_version(project_root, source_relative)
    core_manifest = read_json(cohort_root / "core/core_manifest.json")
    core_items = core_manifest.get("completion", {}).get("items", [])
    require(len(core_items) == 1_000, f"expected 1,000 sealed core items: {result_dir}")
    sealed_summaries = {item["summary"]: item for item in core_items}
    require(len(sealed_summaries) == 1_000, f"duplicate core inventory path: {result_dir}")
    rows: list[dict[str, Any]] = []
    summaries: dict[int, list[dict[str, Any]]] = {}
    for seed in seeds:
        item = by_seed[seed]
        checkpoint_path = cohort_root / item["checkpoint"]
        metadata_path = cohort_root / item["metadata"]
        verify_file(checkpoint_path, item["checkpoint_sha256"], "selected checkpoint")
        verify_file(metadata_path, item["metadata_sha256"], "checkpoint metadata")
        metadata = read_json(metadata_path)
        require(metadata["seed"] == seed, f"metadata seed mismatch: {metadata_path}")
        require(metadata["checkpoint_sha256"] == item["checkpoint_sha256"], f"metadata checkpoint mismatch: {metadata_path}")
        evaluation_records, internal = _study_a_run_records(
            project_root, cohort_root, seed, item["checkpoint_sha256"], sealed_summaries
        )
        summaries[seed] = internal
        checkpoint_source = rel(project_root, checkpoint_path)
        metadata_source = rel(project_root, metadata_path)
        rows.append({
            "checkpoint_id": f"{cohort_name}:dqn:{seed}",
            "cohort": cohort_name,
            "policy_variant": "masked_dqn_final",
            "training_seed": seed,
            "checkpoint": public_locator_for_source(checkpoint_source),
            "checkpoint_access": public_access_for_source(checkpoint_source),
            "checkpoint_sha256": item["checkpoint_sha256"],
            "metadata_access": public_access_for_source(metadata_source),
            "metadata_sha256": item["metadata_sha256"],
            "selection_rule": metadata["selection_mode"],
            "training_configuration": {
                key: metadata[key] for key in (
                    "episodes", "selected_episode", "steps_per_episode",
                    "environment_decision_count", "transition_count",
                    "gradient_update_count", "forced_abstention_count", "node_count",
                    "scenario_name", "scenario_path", "scenario_sha256",
                    "reward_weights", "epsilon_schedule", "action_schema", "latency_model",
                )
            },
            "source_version": source_version,
            "audit_version": {
                "audit_output": public_locator_for_source(audit_relative),
                "audit_output_access": public_access_for_source(audit_relative),
                "audit_output_sha256": sha256(audit_path),
                "implementation": public_locator_for_source(audit_script_relative),
                "implementation_access": public_access_for_source(audit_script_relative),
                "implementation_sha256": sha256(script_path),
            },
            "dataset_membership": {
                "study": "Study A",
                "cohort": cohort_name,
                "scientific_checkpoint": True,
                "smoke_fixture": False,
                "core_evaluation_record_count": 100,
            },
            "evaluation_records": evaluation_records,
        })
    return rows, summaries


def _prospective_registry(
    project_root: Path, summary_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    results_root = project_root / "results/crn_ca_pr1_20261017"
    seal = read_json(results_root / "RESULTS_SEAL.json")
    seal_files = seal["file_sha256"]
    freeze_relative = PROSPECTIVE_FREEZE_SOURCE
    freeze = read_json(project_root / freeze_relative)
    require(freeze["protocol_sha256"] == seal["protocol_sha256"], "freeze/seal protocol mismatch")
    rows: list[dict[str, Any]] = []
    for variant in PROSPECTIVE_VARIANTS:
        for seed in PROSPECTIVE_SEEDS:
            base_relative = f"training/{variant}/seed_{seed}"
            checkpoint_relative = f"{base_relative}/final.pt"
            metadata_relative = f"{base_relative}/metadata.json"
            require(checkpoint_relative in seal_files and metadata_relative in seal_files, f"unsealed prospective checkpoint: {variant}/{seed}")
            checkpoint_path = results_root / checkpoint_relative
            metadata_path = results_root / metadata_relative
            verify_file(checkpoint_path, seal_files[checkpoint_relative], "prospective checkpoint")
            verify_file(metadata_path, seal_files[metadata_relative], "prospective metadata")
            metadata = read_json(metadata_path)
            require(metadata["model_seed"] == seed and metadata["policy_variant"] == variant, f"prospective metadata join failed: {variant}/{seed}")
            require(metadata["checkpoint_sha256"] == seal_files[checkpoint_relative], f"prospective checkpoint metadata mismatch: {variant}/{seed}")
            selected = [
                row for row in summary_rows
                if row["policy"] == variant and int(row["training_seed"]) == seed
            ]
            require(len(selected) == 192, f"expected 192 evaluation rows: {variant}/{seed}")
            require({row["checkpoint_sha256"] for row in selected} == {metadata["checkpoint_sha256"]}, f"evaluation/checkpoint join failed: {variant}/{seed}")
            evaluation_records = [
                {
                    "run_id": row["run_id"],
                    "factor_id": row["factor_id"],
                    "replication_index": int(row["replication_index"]),
                    "environment_seed": int(row["environment_seed"]),
                    "trace_sha256": row["trace_sha256"],
                }
                for row in sorted(selected, key=lambda r: (FACTORS.index(r["factor_id"]), int(r["replication_index"])))
            ]
            checkpoint_source = rel(project_root, checkpoint_path)
            metadata_source = rel(project_root, metadata_path)
            seal_source = "results/crn_ca_pr1_20261017/RESULTS_SEAL.json"
            rows.append({
                "checkpoint_id": f"prospective_collision:{variant}:{seed}",
                "cohort": "prospective_collision",
                "policy_variant": variant,
                "training_seed": seed,
                "checkpoint": public_locator_for_source(checkpoint_source),
                "checkpoint_access": public_access_for_source(checkpoint_source),
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "metadata_access": public_access_for_source(metadata_source),
                "metadata_sha256": seal_files[metadata_relative],
                "selection_rule": metadata["selection_rule"],
                "training_configuration": {
                    key: metadata[key] for key in (
                        "study_id", "protocol_version", "protocol_sha256",
                        "episodes", "steps_per_episode", "global_environment_steps",
                        "gradient_steps", "replay_total_insertions",
                        "replay_node_insertions", "seed_index", "selection_rule",
                    )
                },
                "source_version": {
                    "protocol_freeze": public_locator_for_source(freeze_relative),
                    "protocol_freeze_access": public_access_for_source(freeze_relative),
                    "protocol_freeze_sha256": sha256(project_root / freeze_relative),
                    "protocol_sha256": freeze["protocol_sha256"],
                    "aggregate_source_sha256": freeze["aggregate_source_sha256"],
                    "results_seal": public_locator_for_source(seal_source),
                    "results_seal_access": public_access_for_source(seal_source),
                    "results_seal_sha256": sha256(results_root / "RESULTS_SEAL.json"),
                },
                "audit_version": {
                    "attempted_priority_audit": "studies/crn-2026-003-finite-domain-dsa/evidence/gold/reanalysis/reanalysis_output.json",
                    "attempted_priority_audit_sha256": PINNED_INPUT_SHA256["studies/crn-2026-003-finite-domain-dsa/evidence/gold/reanalysis/reanalysis_output.json"],
                    "independent_reanalysis_implementation": "article_reanalysis/reanalyse.py",
                    "independent_reanalysis_implementation_sha256": sha256(project_root / "studies/crn-2026-003-finite-domain-dsa/analysis/reanalyse.py"),
                },
                "dataset_membership": {
                    "study": "Study B",
                    "cohort": "prospective_collision",
                    "scientific_checkpoint": True,
                    "smoke_fixture": False,
                    "evaluation_record_count": 192,
                },
                "evaluation_records": evaluation_records,
            })
    return rows


def build_checkpoint_registry(
    project_root: Path, summary_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, dict[int, list[dict[str, Any]]]]]:
    archived, archived_summaries = _study_a_registry(
        project_root,
        "archived_discovery",
        "results/camera_ready_20261014_corrected",
        ARCHIVED_SEEDS,
        ARCHIVED_PRIORITY_AUDIT_SOURCE,
        ARCHIVED_PRIORITY_SCRIPT_SOURCE,
    )
    replication, replication_summaries = _study_a_registry(
        project_root,
        "independent_replication",
        "results/crn_mp_pr1_20261017",
        REPLICATION_SEEDS,
        "results/crn_mp_pr1_20261017/audits/priority_collapse/priority_collapse_audit.json",
        REPLICATION_PRIORITY_SCRIPT_SOURCE,
    )
    prospective = _prospective_registry(project_root, summary_rows)
    records = archived + replication + prospective
    require(len(records) == 76, f"checkpoint registry is not 76 rows: {len(records)}")
    require(len({record["checkpoint_id"] for record in records}) == 76, "duplicate checkpoint_id")
    require(len({record["checkpoint_sha256"] for record in records}) == 76, "checkpoint hashes are not unique")
    return ({
        "schema_version": 1,
        "derivation": "deterministic joins over sealed/package inputs; no training or simulation",
        "counts": {
            "total_scientific_checkpoints": 76,
            "archived_discovery": 8,
            "independent_replication": 8,
            "prospective_collision": 60,
            "excluded_smoke_fixtures": 1,
        },
        "records": records,
    }, {
        "archived_discovery": archived_summaries,
        "independent_replication": replication_summaries,
    })


def _accounting(
    summaries: list[dict[str, Any]], scenarios: set[str], environments: set[int],
) -> dict[str, Any]:
    selected = [
        item["record"] for item in summaries
        if item["record"]["spec"]["scenario_label"] in scenarios
        and int(item["record"]["spec"]["environment_seed"]) in environments
    ]
    expected = len(scenarios) * len(environments)
    require(len(selected) == expected, f"Study-A split identity incomplete: {len(selected)} != {expected}")
    rows = sum(int(item["counts"]["rows"]) for item in selected)
    attempts = sum(int(item["counts"]["attempts"]) for item in selected)
    forced = sum(int(item["counts"]["forced_abstentions"]) for item in selected)
    require(rows == attempts + forced, "Study-A split row accounting failed")
    return {
        "run_count": len(selected),
        "total_node_decisions": rows,
        "attempted_channel_decisions": attempts,
        "all_occupied_empty_legal_set_decisions": forced,
        "identity": "total_node_decisions = attempted_channel_decisions + all_occupied_empty_legal_set_decisions",
        "identity_pass": True,
        "agreement_denominator_excludes_empty_legal_set": True,
        "evidence": "retained valid run summaries; each also carries action_legality and row_count_identity=true",
    }


def _audit_certificate_id(
    cohort: str, seed: int, checkpoint_sha: str, audit_sha: str,
) -> str:
    return "study-a-" + canonical_sha256({
        "cohort": cohort,
        "training_seed": seed,
        "checkpoint_sha256": checkpoint_sha,
        "audit_output_sha256": audit_sha,
    })[:24]


def build_study_a_certificates(
    project_root: Path,
    checkpoint_registry: dict[str, Any],
    summary_lookup: dict[str, dict[int, list[dict[str, Any]]]],
) -> dict[str, Any]:
    registry = {
        (record["cohort"], int(record["training_seed"])): record
        for record in checkpoint_registry["records"]
        if record["cohort"] in {"archived_discovery", "independent_replication"}
    }
    inputs = (
        (
            "archived_discovery",
            ARCHIVED_PRIORITY_AUDIT_SOURCE,
            ARCHIVED_PRIORITY_SCRIPT_SOURCE,
        ),
        (
            "independent_replication",
            "results/crn_mp_pr1_20261017/audits/priority_collapse/priority_collapse_audit.json",
            REPLICATION_PRIORITY_SCRIPT_SOURCE,
        ),
    )
    certificates: list[dict[str, Any]] = []
    for cohort, audit_relative, script_relative in inputs:
        audit_path = project_root / audit_relative
        script_path = project_root / script_relative
        audit = read_json(audit_path)
        definition = audit["audit_definition"]
        audit_sha = sha256(audit_path)
        audit_records = audit["per_checkpoint"]
        expected_seeds = ARCHIVED_SEEDS if cohort == "archived_discovery" else REPLICATION_SEEDS
        require(len(audit_records) == 8, f"Study-A audit is not eight records: {audit_relative}")
        require({int(r["training_seed"]) for r in audit_records} == set(expected_seeds), f"Study-A audit seed drift: {audit_relative}")
        for record in sorted(audit_records, key=lambda item: expected_seeds.index(int(item["training_seed"]))):
            seed = int(record["training_seed"])
            checkpoint = registry[(cohort, seed)]
            summaries = summary_lookup[cohort][seed]
            calibration = _accounting(
                summaries,
                set(definition["calibration_scenarios"]),
                {int(value) for value in definition["calibration_environment_seeds"]},
            )
            validation = _accounting(
                summaries,
                set(definition["heldout_scenarios"]),
                {int(value) for value in definition["heldout_environment_seeds"]},
            )
            full_core = _accounting(
                summaries,
                set(definition["full_core_scenarios"]),
                {int(value) for value in definition["full_core_environment_seeds"]},
            )
            require(calibration["attempted_channel_decisions"] == record["calibration_attempts"], f"calibration attempts mismatch: {cohort}/{seed}")
            require(validation["attempted_channel_decisions"] == record["heldout_attempts"], f"validation attempts mismatch: {cohort}/{seed}")
            require(full_core["attempted_channel_decisions"] == record["full_core_attempts"], f"full-core attempts mismatch: {cohort}/{seed}")
            require(record["tied_best_orders"] == 1, f"nonunique selected priority order: {cohort}/{seed}")
            require(record["calibration_matches"] == record["calibration_attempts"], f"nonexact calibration: {cohort}/{seed}")
            require(record["heldout_matches"] == record["heldout_attempts"], f"nonexact validation: {cohort}/{seed}")
            require(record["full_core_matches"] == record["full_core_attempts"], f"nonexact full core: {cohort}/{seed}")
            certificates.append({
                "certificate_id": _audit_certificate_id(cohort, seed, checkpoint["checkpoint_sha256"], audit_sha),
                "cohort": cohort,
                "training_seed": seed,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "fitted_priority_high_to_low": [int(x) for x in record["fitted_priority_high_to_low"].split(">")],
                "selection_and_ties": {
                    "candidate_orders": math.factorial(6),
                    "selected_order_count": record["tied_best_orders"],
                    "unique_selected_order": record["tied_best_orders"] == 1,
                    "criterion": "maximize exact physical-channel agreement on calibration attempted decisions",
                },
                "split_identity": {
                    "calibration_scenarios": definition["calibration_scenarios"],
                    "calibration_environment_seeds": definition["calibration_environment_seeds"],
                    "validation_scenarios": definition["heldout_scenarios"],
                    "validation_environment_seeds": definition["heldout_environment_seeds"],
                    "full_core_scenarios": definition["full_core_scenarios"],
                    "full_core_environment_seeds": definition["full_core_environment_seeds"],
                },
                "calibration": {
                    "attempts": record["calibration_attempts"],
                    "matches": record["calibration_matches"],
                    "agreement": record["calibration_agreement"],
                    "distinct_nonempty_masks": record["calibration_distinct_nonempty_masks"],
                    "all_occupied_accounting": calibration,
                },
                "validation_without_refit": {
                    "attempts": record["heldout_attempts"],
                    "matches": record["heldout_matches"],
                    "agreement": record["heldout_agreement"],
                    "distinct_nonempty_masks": record["heldout_distinct_nonempty_masks"],
                    "all_occupied_accounting": validation,
                },
                "complete_core": {
                    "attempts": record["full_core_attempts"],
                    "matches": record["full_core_matches"],
                    "agreement": record["full_core_agreement"],
                    "distinct_nonempty_masks": record["full_core_distinct_nonempty_masks"],
                    "all_occupied_accounting": full_core,
                },
                "counterfactual_assay": {
                    key: value for key, value in record.items()
                    if key.startswith("counterfactual_")
                },
                "precedence_certificate": {
                    key: value for key, value in record.items()
                    if "precedence" in key or key.endswith("certificate_sha256")
                },
                "audit_provenance": {
                    "output": public_locator_for_source(audit_relative),
                    "output_access": public_access_for_source(audit_relative),
                    "output_sha256": audit_sha,
                    "implementation": public_locator_for_source(script_relative),
                    "implementation_access": public_access_for_source(script_relative),
                    "implementation_sha256": sha256(script_path),
                    "audit_schema_version": audit.get("schema_version", "legacy-retained-output"),
                    "source_sha256": audit.get("source_sha256"),
                },
                "interpretation_boundary": audit.get("interpretation_boundary"),
            })
    require(len(certificates) == 16, f"Study-A certificate count is not 16: {len(certificates)}")
    require(len({item["certificate_id"] for item in certificates}) == 16, "duplicate Study-A certificate ID")
    return {
        "schema_version": 1,
        "record_count": 16,
        "method": "exhaustive 6! masked-priority fit; frozen calibration order evaluated without refitting",
        "records": certificates,
    }


def parse_mask_action_counts(value: str) -> list[list[int]]:
    values = [int(item) for item in value.split("-")]
    require(len(values) == 64 * 7, f"mask-action count length is {len(values)}, expected 448")
    require(all(value >= 0 for value in values), "negative mask-action count")
    return [values[index * 7:(index + 1) * 7] for index in range(64)]


def add_counts(target: list[list[int]], source: list[list[int]]) -> None:
    for mask in range(64):
        for action in range(7):
            target[mask][action] += source[mask][action]


def count_sha256(counts: list[list[int]]) -> str:
    payload = b"".join(struct.pack("<q", value) for row in counts for value in row[:6])
    return hashlib.sha256(payload).hexdigest()


def graph_from_counts(counts: list[list[int]]) -> tuple[list[list[int]], list[list[bool]]]:
    support = [[0] * 6 for _ in range(6)]
    for mask in range(1, 64):
        legal = [channel for channel in range(6) if mask & (1 << channel)]
        for chosen in legal:
            selected = counts[mask][chosen]
            if selected:
                for other in legal:
                    if other != chosen:
                        support[chosen][other] += selected
    adjacency = [[support[i][j] > 0 for j in range(6)] for i in range(6)]
    return support, adjacency


def shortest_cycle(counts: list[list[int]]) -> dict[str, Any]:
    support, adjacency = graph_from_counts(counts)
    selected_cycle: tuple[int, ...] | None = None
    for length in range(2, 7):
        candidates: list[tuple[int, ...]] = []
        for cycle in itertools.permutations(range(6), length):
            if cycle[0] != min(cycle):
                continue  # canonical rotation
            if all(adjacency[cycle[index]][cycle[(index + 1) % length]] for index in range(length)):
                candidates.append(cycle)
        if candidates:
            selected_cycle = min(candidates)
            break
    require(selected_cycle is not None, "cyclic certificate lacks a cycle witness")
    edges = []
    for index, source in enumerate(selected_cycle):
        target = selected_cycle[(index + 1) % len(selected_cycle)]
        mask_witness = next(
            mask for mask in range(1, 64)
            if (mask & (1 << source)) and (mask & (1 << target)) and counts[mask][source] > 0
        )
        edges.append({
            "preferred_channel": source,
            "over_channel": target,
            "aggregate_support_count": support[source][target],
            "smallest_supporting_legal_mask_bits": mask_witness,
            "selected_action_count_at_supporting_mask": counts[mask_witness][source],
        })
    return {
        "length": len(selected_cycle),
        "closed_channel_path": [*selected_cycle, selected_cycle[0]],
        "canonicalization": "shortest length, then lexicographically smallest directed simple cycle after minimum-node rotation",
        "edge_witnesses": edges,
    }


def build_study_b_priority_records(
    project_root: Path,
    summary_rows: list[dict[str, str]],
    checkpoint_registry: dict[str, Any],
) -> dict[str, Any]:
    gold_path = project_root / "studies/crn-2026-003-finite-domain-dsa/evidence/gold/reanalysis/reanalysis_output.json"
    gold = read_json(gold_path)["attempted_only_logical_audit"]
    records_by_seed = {int(item["training_seed"]): item for item in gold["per_checkpoint"]}
    require(set(records_by_seed) == set(PROSPECTIVE_SEEDS), "Study-B attempted audit seed drift")
    registry = {
        int(record["training_seed"]): record
        for record in checkpoint_registry["records"]
        if record["cohort"] == "prospective_collision" and record["policy_variant"] == "collision_aware_dqn"
    }
    per_seed_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        if row["policy"] == "collision_aware_dqn":
            per_seed_rows[int(row["training_seed"])].append(row)
    exported: list[dict[str, Any]] = []
    for seed in PROSPECTIVE_SEEDS:
        groups = per_seed_rows[seed]
        require(len(groups) == 192, f"Study-B full-policy rows incomplete: {seed}")
        partitions = {
            "calibration": [row for row in groups if int(row["replication_index"]) < 12],
            "validation": [row for row in groups if int(row["replication_index"]) >= 12],
            "full_support": groups,
        }
        counts: dict[str, list[list[int]]] = {}
        for name, rows in partitions.items():
            matrix = [[0] * 7 for _ in range(64)]
            for row in rows:
                require(row["mask_action_counts_shape"] == "64x7", f"mask count shape drift: {row['run_id']}")
                add_counts(matrix, parse_mask_action_counts(row["mask_action_counts"]))
            counts[name] = matrix
        source = records_by_seed[seed]
        mapping = {
            "calibration": source["calibration_certificate"],
            "validation": source["validation_certificate"],
            "full_support": source["full_support_certificate"],
        }
        certificates: dict[str, Any] = {}
        for name in ("calibration", "validation", "full_support"):
            matrix = counts[name]
            cert = mapping[name]
            support, adjacency = graph_from_counts(matrix)
            edge_count = sum(int(value) for row in adjacency for value in row)
            attempted = sum(sum(row[:6]) for row in matrix)
            require(count_sha256(matrix) == cert["counts_64x6_le_i8_sha256"], f"Study-B count hash mismatch: {seed}/{name}")
            require(edge_count == cert["edge_count"], f"Study-B edge count mismatch: {seed}/{name}")
            require(attempted == cert["attempts"], f"Study-B attempt count mismatch: {seed}/{name}")
            require(cert["acyclic"] is False and cert["representing_order_count"] == 0, f"Study-B certificate unexpectedly representable: {seed}/{name}")
            certificates[name] = {
                **cert,
                "shortest_cycle_witness": shortest_cycle(matrix),
            }
        checkpoint = registry[seed]
        exported.append({
            "record_id": f"study-b-full-policy-{seed}",
            "training_seed": seed,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "evaluation_run_ids": [row["run_id"] for row in sorted(groups, key=lambda r: (FACTORS.index(r["factor_id"]), int(r["replication_index"])))],
            "calibration_indices": list(range(12)),
            "validation_indices": list(range(12, 24)),
            "selection_and_ties": {
                "candidate_orders": math.factorial(6),
                "score": "equal-factor/equal-environment-block attempted-action agreement",
                "numerical_tie_rule": "numpy.isclose(rtol=0, atol=1e-15) at the maximum score",
                "calibration_tie_count": source["calibration_tie_count"],
                "calibration_tied_orders": source["calibration_tied_orders"],
                "selected_order_index": source["selected_order_index"],
                "selected_order": source["selected_order_original_validation_tie_rule"],
                "validation_tie_break": "among calibration ties, maximize validation score; remaining tie resolves by ascending permutation index",
                "validation_optimal_tie_count": source["validation_optimal_tie_count"],
                "validation_optimal_orders": source["validation_optimal_orders"],
            },
            "agreement": {
                key: value for key, value in source.items()
                if "agreement" in key or key.endswith("_count") or key.endswith("_coverage")
            },
            "certificates": certificates,
            "provenance": {
                "aggregate_input": public_locator_for_source(
                    "results/crn_ca_pr1_20261017/evaluation/summary.csv"
                ),
                "aggregate_input_access": public_access_for_source(
                    "results/crn_ca_pr1_20261017/evaluation/summary.csv"
                ),
                "aggregate_input_sha256": PINNED_INPUT_SHA256["results/crn_ca_pr1_20261017/evaluation/summary.csv"],
                "gold_audit": rel(project_root, gold_path),
                "gold_audit_sha256": sha256(gold_path),
                "implementation": "article_reanalysis/reanalyse.py",
                "implementation_sha256": sha256(project_root / "studies/crn-2026-003-finite-domain-dsa/analysis/reanalyse.py"),
            },
        })
    require(len(exported) == 20, f"Study-B full-policy record count is not 20: {len(exported)}")
    return {
        "schema_version": 1,
        "record_count": 20,
        "coordinate_system": "logical channels; attempted actions only; abstentions excluded",
        "records": exported,
    }


def build_boundary_witnesses(
    project_root: Path, checkpoint_registry: dict[str, Any],
) -> dict[str, Any]:
    audit_relative = ARCHIVED_BOUNDARY_AUDIT_SOURCE
    audit_path = project_root / audit_relative
    audit = read_json(audit_path)
    script_relative = ARCHIVED_BOUNDARY_SCRIPT_SOURCE
    script_path = project_root / script_relative
    registry = {
        int(record["training_seed"]): record
        for record in checkpoint_registry["records"]
        if record["cohort"] == "archived_discovery"
    }
    witnesses: list[dict[str, Any]] = []
    for checkpoint_record in audit["per_checkpoint"]:
        seed = int(checkpoint_record["training_seed"])
        checkpoint = registry[seed]
        for ordinal, source in enumerate(checkpoint_record["crossing_witnesses"], start=1):
            mask = int(source["legal_mask_bits"])
            dynamic = list(source["dynamic_features"])
            require(len(dynamic) == 24, f"boundary witness dynamic dimension drift: {seed}/{ordinal}")
            occupancy = [0.0 if mask & (1 << channel) else 1.0 for channel in range(6)]
            input_vector = occupancy + dynamic + [0.0] * 98
            require(len(input_vector) == 128, "boundary witness input is not 128-dimensional")
            require(set(source["legal_channels"]) == {c for c in range(6) if mask & (1 << c)}, f"boundary witness mask mismatch: {seed}/{ordinal}")
            require(source["priority_action"] in source["legal_channels"], f"illegal priority action: {seed}/{ordinal}")
            require(source["target_alternative"] in source["legal_channels"], f"illegal target alternative: {seed}/{ordinal}")
            require(source["executed_action"] in source["legal_channels"], f"illegal executed action: {seed}/{ordinal}")
            witness_id = f"study-a-boundary-{seed}-mask{mask:02d}-alt{source['target_alternative']}"
            witnesses.append({
                "witness_id": witness_id,
                "training_seed": seed,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "checkpoint": checkpoint["checkpoint"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "input": {
                    "dtype": "float32",
                    "shape": [128],
                    "values": input_vector,
                    "coordinate_blocks": {
                        "0:6": "occupancy indicator; 0=legal/idle and 1=unavailable/occupied",
                        "6:30": "optimized constructed dynamic features",
                        "30:128": "fixed zero compatibility coordinates",
                    },
                    "legal_mask_bits": mask,
                    "legal_channels": source["legal_channels"],
                },
                "q_comparison": {
                    "dtype": "float32",
                    "priority_action": source["priority_action"],
                    "target_alternative": source["target_alternative"],
                    "executed_action": source["executed_action"],
                    "pairwise_margin": source["target_pairwise_q_margin"],
                    "margin_definition": "Q(priority_action) - Q(target_alternative)",
                    "nonpositive_margin_found": source["target_pairwise_q_margin"] <= 0.0,
                    "full_q_vector_retained": False,
                },
                "source_location": {
                    "audit_record_training_seed": seed,
                    "crossing_witness_ordinal_one_based": ordinal,
                },
            })
    witnesses.sort(key=lambda item: (item["training_seed"], item["input"]["legal_mask_bits"], item["q_comparison"]["target_alternative"]))
    require(len(witnesses) == 44, f"expected 44 boundary witnesses, got {len(witnesses)}")
    require(len({item["witness_id"] for item in witnesses}) == 44, "boundary witness IDs are not unique")
    require(all(item["q_comparison"]["nonpositive_margin_found"] for item in witnesses), "positive-margin row entered boundary witnesses")
    return {
        "schema_version": 1,
        "record_count": 44,
        "scope": audit["scope"],
        "method": audit["method"],
        "provenance": {
            "audit_output": public_locator_for_source(audit_relative),
            "audit_output_access": public_access_for_source(audit_relative),
            "audit_output_sha256": sha256(audit_path),
            "implementation": public_locator_for_source(script_relative),
            "implementation_access": public_access_for_source(script_relative),
            "implementation_sha256": sha256(script_path),
        },
        "limitation": (
            "The retained audit stores the decisive float32 pairwise margin but not all six raw Q values. "
            "Raw Q vectors are therefore not invented here; recovering them requires a source-verified "
            "checkpoint forward pass. These are constructed falsification witnesses, not claimed generator-manifold states."
        ),
        "records": witnesses,
    }


def _effect_role(comparator: str) -> str:
    return {
        "phase_orthogonal": "primary_efficacy",
        "history_free_dqn": "primary_ablation",
        "coordination_blind_dqn": "diagnostic",
        "quality_greedy_persistent": "secondary_holm",
        "round_robin": "secondary_holm",
        "uniform_legal": "secondary_holm",
        "static_priority": "secondary_holm",
        "centralized_quality_matching": "descriptive_oracle",
    }[comparator]


def build_paired_effects(
    project_root: Path,
    summary_rows: list[dict[str, str]],
    checkpoint_registry: dict[str, Any],
) -> dict[str, Any]:
    effects_path = project_root / "results/crn_ca_pr1_20261017/analysis/checkpoint_effects.csv"
    with effects_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    goodput_rows = [row for row in source_rows if row["metric"] == "network_goodput"]
    require(len(goodput_rows) == 20 * 8, f"expected 160 stored goodput effects, got {len(goodput_rows)}")
    source_effect = {
        (int(row["training_seed"]), row["comparator"]): float(row["paired_macro_difference"])
        for row in goodput_rows
    }
    require(len(source_effect) == 160, "duplicate checkpoint-effect row")
    by_key: dict[tuple[str, int | None, str, int], dict[str, str]] = {}
    for row in summary_rows:
        seed = int(row["training_seed"]) if row["training_seed"] else None
        key = (row["policy"], seed, row["factor_id"], int(row["replication_index"]))
        require(key not in by_key, f"duplicate aggregate cube cell: {key}")
        by_key[key] = row
    registry = {
        (record["policy_variant"], int(record["training_seed"])): record
        for record in checkpoint_registry["records"]
        if record["cohort"] == "prospective_collision"
    }
    records: list[dict[str, Any]] = []
    for seed in PROSPECTIVE_SEEDS:
        treatment_checkpoint = registry[("collision_aware_dqn", seed)]
        comparisons: list[dict[str, Any]] = []
        for comparator in COMPARATORS:
            values: list[float] = []
            for factor in FACTORS:
                for replication in range(24):
                    treatment = float(by_key[("collision_aware_dqn", seed, factor, replication)]["network_goodput"])
                    comparator_seed = seed if comparator in PROSPECTIVE_VARIANTS else None
                    control = float(by_key[(comparator, comparator_seed, factor, replication)]["network_goodput"])
                    values.append(treatment - control)
            require(len(values) == 192, "paired macro does not contain 192 equal-weight cells")
            recomputed = fmean(values)
            stored = source_effect[(seed, comparator)]
            require(math.isclose(recomputed, stored, rel_tol=0.0, abs_tol=2e-15), f"checkpoint effect mismatch: {seed}/{comparator}: {recomputed} != {stored}")
            if comparator in PROSPECTIVE_VARIANTS:
                comparator_checkpoint = registry[(comparator, seed)]
                comparator_identity: dict[str, Any] = {
                    "checkpoint_id": comparator_checkpoint["checkpoint_id"],
                    "checkpoint_sha256": comparator_checkpoint["checkpoint_sha256"],
                }
            else:
                comparator_identity = {
                    "identity": "deterministic trace-keyed nonlearning policy",
                    "checkpoint_id": None,
                    "checkpoint_sha256": None,
                }
            comparisons.append({
                "comparator": comparator,
                "role": _effect_role(comparator),
                "paired_macro_difference": stored,
                "stored_decimal": next(
                    row["paired_macro_difference"] for row in goodput_rows
                    if int(row["training_seed"]) == seed and row["comparator"] == comparator
                ),
                "recomputed_from_192_cells": recomputed,
                "comparator_identity": comparator_identity,
            })
        records.append({
            "record_id": f"study-b-paired-effects-{seed}",
            "training_seed": seed,
            "treatment_checkpoint_id": treatment_checkpoint["checkpoint_id"],
            "treatment_checkpoint_sha256": treatment_checkpoint["checkpoint_sha256"],
            "comparisons": comparisons,
        })
    require(len(records) == 20, "paired-effect export is not 20 records")
    return {
        "schema_version": 1,
        "record_count": 20,
        "metric": "network_goodput",
        "effect_direction": "collision_aware_dqn minus comparator",
        "full_precision_policy": "JSON binary64 value plus the exact source CSV decimal; no display rounding",
        "equal_cell_weights": {
            "factor_weight": "1/8",
            "label_shift_weight": "1/6",
            "environment_block_within_shift_weight": "1/4",
            "individual_factor_environment_cell_weight": "1/192",
            "training_unit_weight_in_population_point_estimate": "1/20",
            "note": "nonlearning comparator cubes are broadcast across training units; no fictitious comparator seed is created",
        },
        "provenance": {
            "stored_effects": public_locator_for_source(rel(project_root, effects_path)),
            "stored_effects_access": public_access_for_source(rel(project_root, effects_path)),
            "stored_effects_sha256": sha256(effects_path),
            "aggregate_input": public_locator_for_source(
                "results/crn_ca_pr1_20261017/evaluation/summary.csv"
            ),
            "aggregate_input_access": public_access_for_source(
                "results/crn_ca_pr1_20261017/evaluation/summary.csv"
            ),
            "aggregate_input_sha256": PINNED_INPUT_SHA256["results/crn_ca_pr1_20261017/evaluation/summary.csv"],
        },
        "records": records,
    }


def write_analysis_input_index(
    output: Path, summary_rows: list[dict[str, str]],
) -> dict[str, Any]:
    fields = (
        "input_row_number", "run_id", "policy", "training_seed",
        "checkpoint_sha256", "factor_id", "replication_index", "environment_seed",
        "logical_to_physical", "trace_sha256",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for number, row in enumerate(summary_rows, start=1):
            writer.writerow({
                "input_row_number": number,
                "run_id": row["run_id"],
                "policy": row["policy"],
                "training_seed": row["training_seed"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "factor_id": row["factor_id"],
                "replication_index": row["replication_index"],
                "environment_seed": row["environment_seed"],
                "logical_to_physical": row["logical_to_physical"],
                "trace_sha256": row["trace_sha256"],
            })
    run_ids = sorted(row["run_id"] for row in summary_rows)
    trace_ids = sorted(row["trace_sha256"] for row in summary_rows)
    return {
        "path": output.name,
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "rows": len(summary_rows),
        "unique_run_ids": len(set(run_ids)),
        "unique_trace_identities": len(set(trace_ids)),
        "sorted_run_id_set_sha256": hashlib.sha256(("\n".join(run_ids) + "\n").encode()).hexdigest(),
        "sorted_trace_identity_multiset_sha256": hashlib.sha256(("\n".join(trace_ids) + "\n").encode()).hexdigest(),
    }


def _field(
    name: str, dtype: str, units: str | None, description: str,
    nullable: bool = False, undefined_when: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "serialized_type": dtype,
        "units": units,
        "description": description,
        "nullable": nullable,
        "undefined_when": undefined_when,
    }


def data_dictionary() -> list[dict[str, Any]]:
    fields = [
        _field("study_id", "string", None, "Frozen parent study identifier."),
        _field("protocol_version", "string", None, "Frozen protocol semantic version."),
        _field("protocol_sha256", "sha256 hex", None, "Identity of the frozen protocol."),
        _field("run_id", "string", None, "Unique aggregate evaluation-record key."),
        _field("smoke", "boolean string", None, "Whether the row is a non-evidence smoke run; all released rows are false."),
        _field("policy", "categorical string", None, "Evaluated learned policy or deterministic baseline."),
        _field("training_seed", "nullable integer", None, "Training-unit identifier for learned policies.", True, "nonlearning comparator"),
        _field("checkpoint_sha256", "nullable sha256 hex", None, "Executable checkpoint identity for learned policies.", True, "nonlearning comparator"),
        _field("factor_id", "categorical string", None, "One of eight fixed occupancy-by-arrival factorial cells."),
        _field("occupancy_process", "categorical string", None, "IID or Markov primary-occupancy process."),
        _field("occupancy_probability", "binary64 decimal", "probability", "Nominal/stationary primary-occupancy probability."),
        _field("arrival_probability", "binary64 decimal", "probability per node-slot", "Bernoulli packet-arrival probability."),
        _field("replication_index", "integer", None, "Environment-block index 0 through 23; modulo six is the label shift."),
        _field("environment_seed", "integer", None, "Frozen exogenous trace seed."),
        _field("logical_to_physical", "hyphen-delimited 6-integer permutation", None, "Logical-to-physical channel-label map."),
        _field("burn_in_steps", "integer", "global slots", "Unmeasured burn-in duration."),
        _field("measurement_steps", "integer", "global slots", "Measured evaluation duration."),
        _field("trace_sha256", "sha256 hex", None, "Identity of the excluded slot-level exogenous trace."),
        _field("starting_backlog", "integer", "packets", "Total queue occupancy at measurement-window start."),
        _field("arrivals", "integer", "packets", "Arrivals during the measurement window."),
        _field("deliveries", "integer", "packets", "Successful packet deliveries during the measurement window."),
        _field("ending_backlog", "integer", "packets", "Total queue occupancy at measurement-window end."),
        _field("network_goodput", "binary64 decimal", "packets per global slot", "Deliveries divided by measurement steps."),
        _field("finite_window_service_fraction", "binary64 decimal", "fraction", "Deliveries divided by starting backlog plus arrivals.", True, "starting_backlog + arrivals = 0"),
        _field("capacity_ceiling_per_slot", "binary64 decimal", "packets per global slot", "Trace-conditioned optimistic service ceiling."),
        _field("opportunity_utilization", "binary64 decimal", "fraction", "Goodput divided by the trace-conditioned capacity ceiling.", True, "capacity ceiling = 0"),
        _field("attempted", "integer", "node attempts", "Non-abstaining attempted transmissions."),
        _field("singleton_attempts", "integer", "node attempts", "Attempts without a same-channel contender."),
        _field("collision_attempts", "integer", "node attempts", "Attempts involved in destructive contention."),
        _field("collision_attempt_rate", "binary64 decimal", "fraction of attempts", "Collision attempts divided by attempts.", True, "attempted = 0"),
        _field("link_failures", "integer", "node attempts", "Singleton attempts that failed the stochastic link."),
        _field("conditional_link_failure_rate", "nullable binary64 decimal", "fraction of singleton attempts", "Link failures divided by singleton attempts.", True, "singleton_attempts = 0"),
        _field("voluntary_deferrals", "integer", "node decisions", "Eligible decisions selecting abstention."),
        _field("forced_deferrals", "integer", "node decisions", "Deferrals forced by an empty legal set."),
        _field("mean_network_backlog", "binary64 decimal", "packets", "Time-average total network queue occupancy."),
        _field("delivery_jain_fairness", "nullable binary64 decimal", "index", "Jain fairness over all 15 per-node delivery totals.", True, "total deliveries = 0"),
        _field("zero_delivery_run", "boolean string", None, "Whether no packets were delivered."),
        _field("priority_eligible_decisions", "integer", "node decisions", "Decisions included in the seven-action priority assay."),
        _field("priority_matches", "integer", "node decisions", "Decisions matching the fitted reference order."),
        _field("static_priority_agreement", "binary64 decimal", "fraction", "Priority matches divided by eligible decisions.", True, "priority_eligible_decisions = 0"),
        _field("logical_action_counts", "hyphen-delimited 7 integers", "decisions", "Logical actions 0--5 plus abstain."),
        _field("physical_action_counts", "hyphen-delimited 7 integers", "decisions", "Physical actions 0--5 plus abstain."),
        _field("mask_action_counts_shape", "string", None, "Declared flattened contingency-table shape; fixed at 64x7."),
        _field("mask_action_counts", "hyphen-delimited 448 integers", "decisions", "Row-major logical legal-mask by logical-action contingency table."),
    ]
    require(tuple(item["name"] for item in fields) == SUMMARY_FIELDS, "data dictionary does not exactly cover input schema")
    return fields


def denominator_audit(summary_rows: list[dict[str, str]]) -> dict[str, Any]:
    definitions = {
        "collision_attempt_rate": lambda row: int(row["attempted"]) == 0,
        "conditional_link_failure_rate": lambda row: int(row["singleton_attempts"]) == 0,
        "finite_window_service_fraction": lambda row: int(row["starting_backlog"]) + int(row["arrivals"]) == 0,
        "opportunity_utilization": lambda row: float(row["capacity_ceiling_per_slot"]) == 0.0,
        "delivery_jain_fairness": lambda row: int(row["deliveries"]) == 0,
        "static_priority_agreement": lambda row: int(row["priority_eligible_decisions"]) == 0,
    }
    serialized_missing = {
        "conditional_link_failure_rate": True,
        "delivery_jain_fairness": True,
    }
    metrics: dict[str, Any] = {}
    for field, predicate in definitions.items():
        undefined = [row for row in summary_rows if predicate(row)]
        expected_blank = serialized_missing.get(field, False)
        if undefined:
            if expected_blank:
                require(all(row[field] == "" for row in undefined), f"undefined {field} is not blank")
            else:
                require(all(row[field] != "" for row in undefined), f"unexpected blank convention for {field}")
        metrics[field] = {
            "undefined_row_count": len(undefined),
            "serialized_as_blank_when_undefined": expected_blank,
            "undefined_by_policy": {
                policy: sum(1 for row in undefined if row["policy"] == policy)
                for policy in PROSPECTIVE_POLICIES
            },
        }
    require(metrics["conditional_link_failure_rate"]["undefined_row_count"] == 384, "conditional link undefined count drift")
    require(metrics["delivery_jain_fairness"]["undefined_row_count"] == 384, "fairness undefined count drift")
    return metrics


def build_input_documentation(
    project_root: Path,
    summary_rows: list[dict[str, str]],
    index_identity: dict[str, Any],
) -> dict[str, Any]:
    summary_source = "results/crn_ca_pr1_20261017/evaluation/summary.csv"
    summary_path = project_root / summary_source
    summary_package_path = public_locator_for_source(summary_source)
    return {
        "schema_version": 1,
        "analysis_input": {
            "path": summary_package_path,
            "access": public_access_for_source(summary_source),
            "sha256": sha256(summary_path),
            "bytes": summary_path.stat().st_size,
            "rows_excluding_header": len(summary_rows),
            "columns": len(SUMMARY_FIELDS),
        },
        "compact_identity_index": index_identity,
        "run_key": {
            "unique_field": "run_id",
            "cell_key_for_learned_policies": ["policy", "training_seed", "factor_id", "replication_index"],
            "cell_key_for_nonlearning_policies": ["policy", "factor_id", "replication_index"],
        },
        "trace_identity": {
            "field": "trace_sha256",
            "interpretation": "cryptographic identity of the excluded slot-level trace; identity is retained even though the 9.31-GB traces are outside the compact package",
        },
        "analysis_entry_points": [
            {
                "script": "article_reanalysis/recompute_inference.py",
                "sha256": sha256(project_root / "studies/crn-2026-003-finite-domain-dsa/analysis/recompute_inference.py"),
                "command": (
                    "uv run --frozen --no-sync python "
                    "article_reanalysis/recompute_inference.py "
                    f"{summary_package_path} --output recomputed_inference.json"
                ),
            },
            {
                "script": "article_reanalysis/reanalyse.py",
                "sha256": sha256(project_root / "studies/crn-2026-003-finite-domain-dsa/analysis/reanalyse.py"),
                "command": (
                    "uv run --frozen --no-sync python article_reanalysis/reanalyse.py "
                    f"--study-root {PUBLIC_PROSPECTIVE_ROOT} "
                    "--out-dir reproduced_reanalysis"
                ),
            },
        ],
        "data_dictionary": data_dictionary(),
        "undefined_denominator_audit": denominator_audit(summary_rows),
        "known_reconstruction_boundary": "Per-node delivery vectors are absent; stored run-level Jain fairness cannot be independently reconstructed from this aggregate input.",
    }


def build_statistical_recipe(
    project_root: Path,
    study_b_records: dict[str, Any],
    paired_effects: dict[str, Any],
) -> dict[str, Any]:
    inference_path = project_root / "studies/crn-2026-003-finite-domain-dsa/evidence/gold/inference/recomputed.json"
    parent_path = project_root / "studies/crn-2026-003-finite-domain-dsa/evidence/gold/inference/parent_summary.json"
    inference = read_json(inference_path)
    parent = read_json(parent_path)
    comparisons = inference["inference"]["comparisons"]
    for comparator in COMPARATORS:
        values = [
            next(item["paired_macro_difference"] for item in record["comparisons"] if item["comparator"] == comparator)
            for record in paired_effects["records"]
        ]
        require(math.isclose(fmean(values), comparisons[comparator]["point_effect"], rel_tol=0.0, abs_tol=2e-15), f"sealed point effect mismatch: {comparator}")
    tie_counts = [record["selection_and_ties"]["calibration_tie_count"] for record in study_b_records["records"]]
    validation_ties = [record["selection_and_ties"]["validation_optimal_tie_count"] for record in study_b_records["records"]]
    require(len(tie_counts) == 20 and all(value >= 1 for value in tie_counts), "invalid calibration tie counts")
    return {
        "schema_version": 1,
        "estimand_equation_9": {
            "definition": "For training unit t, Delta_t(c) = (1/8) sum_f [(1/6) sum_s {(1/4) sum_{b in shift s} (Y_full(t,f,b)-Y_c(t or shared,f,b))}]",
            "implementation_equivalence": "The sealed cube is balanced: mean over the 8 factor cells and 24 environment blocks gives weights 1/8, 1/6, and 1/4 exactly.",
            "checkpoint_effect_count": 20,
            "population_point_estimate": "arithmetic mean of the 20 checkpoint effects",
            "nonlearning_comparator_handling": "broadcast the single trace-keyed comparator cube across training units; do not invent comparator seeds",
        },
        "priority_order_selection": {
            "calibration_indices": list(range(12)),
            "validation_indices": list(range(12, 24)),
            "shift_definition": "replication_index modulo 6",
            "candidate_orders": 720,
            "ties": "scores within absolute tolerance 1e-15 and rtol 0 of the maximum are tied",
            "tie_break": "maximize validation agreement within calibration ties, then ascending permutation index",
            "calibration_tie_counts_by_training_seed": dict(
                zip(PROSPECTIVE_SEEDS, tie_counts, strict=True)
            ),
            "validation_optimal_tie_counts_by_training_seed": dict(
                zip(PROSPECTIVE_SEEDS, validation_ties, strict=True)
            ),
            "note": "The attempted-only reanalysis discloses validation tie sensitivity; it is not refitted into the confirmatory estimand.",
        },
        "crossed_bootstrap": {
            "rng": "numpy.random.Generator(PCG64)",
            "seed": 20261017,
            "replicates": 10_000,
            "plan_sha256": "3b26f415c4992bb579afe0615649eaa08233f81f8ae7b3f48ce87c8b4fbd3245",
            "training_resampling": "draw 20 training-unit indices with replacement; one shared draw across all policies and cells",
            "environment_resampling": "within each of six fixed label-shift strata, draw four environment blocks with replacement; one joint draw across policies, seeds, and factorial cells",
            "factorial_cells": "fixed and equally weighted",
            "quantile_algorithm": "numpy.quantile default linear interpolation",
            "quantiles": [0.05, 0.95],
            "interval_scope": "pointwise, not simultaneous",
            "one_sided_table_endpoint": "5th percentile for a lower bound (or 95th percentile for an upper bound)",
        },
        "exact_sign_flip": {
            "unit": "20 paired checkpoint effects",
            "centered_value": "checkpoint effect minus the null margin",
            "statistic": "sum of centered effects; equivalent to the mean for the fixed n=20 tail comparison",
            "enumeration": "all 2^20 independent plus/minus assignments",
            "tail": "one-sided inclusive P(T* >= T_observed - tolerance)",
            "floating_tolerance": "float64 epsilon * 64 * max(1, abs(observed sum), sum(abs(centered effects)))",
            "p_value_denominator": 2 ** 20,
            "margins": {comparator: (0.05 if comparator == "phase_orthogonal" else 0.0) for comparator in COMPARATORS if comparator != "centralized_quality_matching"},
            "assumption_boundary": "exact conditional on joint sign-invariance/exchangeability of the 20 centered paired effects; it is not an unconditional randomization test",
        },
        "holm_family": {
            "members": ["quality_greedy_persistent", "round_robin", "uniform_legal", "static_priority"],
            "ordering": "ascending (raw p-value, comparator name); adjusted values use cumulative max of (remaining hypotheses * p)",
            "raw_p_values": parent["secondary_holm_family"]["raw_p_values"],
            "adjusted_p_values": parent["secondary_holm_family"]["adjusted_p_values"],
            "alpha": parent["secondary_holm_family"]["alpha"],
        },
        "undefined_denominators": {
            "policy": "macro means omit undefined ratios rather than replacing them by zero",
            "source_serialization": "conditional link failure and Jain fairness are blank when their denominators are zero; the exported input documentation reports exact counts",
            "fairness_boundary": "per-node deliveries are absent, so the stored Jain scalar is auditable by identity and missingness but not independently recomputable without replay",
        },
        "sealed_results": {
            "comparisons": comparisons,
            "method_metadata": inference["inference"]["method"],
            "confirmatory_gates": parent["confirmatory_gates"],
            "secondary_holm_family": parent["secondary_holm_family"],
        },
        "provenance": {
            "independent_inference_output": rel(project_root, inference_path),
            "independent_inference_output_sha256": sha256(inference_path),
            "parent_summary": rel(project_root, parent_path),
            "parent_summary_sha256": sha256(parent_path),
            "implementation": "article_reanalysis/recompute_inference.py",
            "implementation_sha256": sha256(project_root / "studies/crn-2026-003-finite-domain-dsa/analysis/recompute_inference.py"),
        },
    }


def _project_path(project_root: Path, supplied: Path) -> tuple[Path, str]:
    """Resolve a CLI-supplied project artifact and reject paths outside the tree."""
    candidate = supplied if supplied.is_absolute() else project_root / supplied
    try:
        relative = candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise EvidenceError(
            f"routed artifact is outside the project root: {supplied}"
        ) from exc
    return candidate, relative


def build_claim_routes(
    project_root: Path,
    output_dir: Path,
    github_receipt_path: Path = DEFAULT_GITHUB_RECEIPT,
) -> dict[str, Any]:
    claims_path = project_root / "studies/crn-2026-003-finite-domain-dsa/manuscript/claims.yaml"
    freeze_relative = PROSPECTIVE_FREEZE_SOURCE
    seal_relative = "results/crn_ca_pr1_20261017/RESULTS_SEAL.json"
    freeze_path = project_root / freeze_relative
    seal_path = project_root / seal_relative
    freeze = read_json(freeze_path)
    seal = read_json(seal_path)
    require(
        freeze["aggregate_source_sha256"]
        == "43166a4456347424bc04577b03d793206ea37086211e3423bcb0013d2ee9ccc7",
        "evaluated aggregate source identity drift",
    )
    require(
        freeze["protocol_sha256"]
        == "a2cfac7ac2e3c58e313f7d92af3ddfc8c0878471b0d449a38a168949b1842fcd",
        "evaluated protocol identity drift",
    )
    require(
        seal["protocol_freeze_sha256"] == sha256(freeze_path),
        "results seal does not bind the protocol freeze",
    )
    require(
        seal["frozen_inputs_sha256"] == freeze["aggregate_frozen_inputs_sha256"],
        "results seal does not bind the frozen inputs",
    )
    require(
        freeze["frozen_at_utc"] < seal["sealed_at_utc"],
        "protocol/result chronology is not monotone",
    )

    def source_ref(relative: str) -> dict[str, Any]:
        path = project_root / relative
        require(path.is_file(), f"missing routed source artifact: {relative}")
        access = public_access_for_source(relative)
        result = {
            "access": access,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        if "locator" in access:
            result["path"] = access["locator"]
        elif "repository_path" in access:
            result["path"] = access["repository_path"]
        return result

    def reviewer_ref(name: str) -> dict[str, Any]:
        path = output_dir / name
        require(path.is_file(), f"missing routed reviewer artifact: {name}")
        return {
            "artifact": name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    figure_manifest_relative = "main/manuscript/figures/publication_figures.manifest.json"
    figure_manifest = read_json(project_root / figure_manifest_relative)
    for name, identity in figure_manifest["figures"].items():
        figure = project_root / "main/manuscript/figures" / name
        verify_file(figure, identity["sha256"], f"numbered publication figure {name}")

    release_config = read_json(
        project_root
        / "studies/crn-2026-003-finite-domain-dsa/release/release_config.json"
    )
    release = release_config["release"]
    release_version = release["version"]
    zenodo_receipt_relative = (
        "studies/crn-2026-003-finite-domain-dsa/release/"
        f"zenodo_new_version_v{release_version}_draft_receipt.json"
    )
    github_receipt_file, github_receipt_relative = _project_path(
        project_root, github_receipt_path,
    )
    zenodo_receipt = read_json(project_root / zenodo_receipt_relative)
    github_receipt = read_json(github_receipt_file)
    new_doi = zenodo_receipt["new_draft_identity"]["reserved_doi"]
    repository = github_receipt["repository"]
    require(new_doi == release["doi"], "new-version DOI drift")
    require(
        repository
        == "https://github.com/lexason/Masked-Priority-Diagnosis-and-Contention-Aware-DQN-Policies-for-Dynamic-Spectrum-Access",
        "public repository route drift",
    )
    zenodo_receipt_reference = source_ref(zenodo_receipt_relative)
    zenodo_receipt_reference["receipt_scope"] = (
        f"v{release_version}_version_reservation"
    )
    github_receipt_reference = source_ref(github_receipt_relative)
    github_receipt_reference["receipt_scope"] = (
        f"v{release_version}_final_public_commit_observation"
        if github_receipt_relative == DEFAULT_GITHUB_RECEIPT.as_posix()
        else "pre_push_staging_observation"
    )
    routes = {
        "A_ARCHIVE_PRIORITY": ["checkpoint_registry.json", "study_a_certificates.json"],
        "A_DIRECT_REPLICATION": ["checkpoint_registry.json", "study_a_certificates.json"],
        "A_OFF_SUPPORT": ["boundary_witnesses.json"],
        "B_PRIMARY_PHASE": ["paired_effects.json", "statistical_recipe_and_results.json", "analysis_input_documentation.json"],
        "B_PHASE_NEGATIVE_SENSITIVITY": ["paired_effects.json", "statistical_recipe_and_results.json"],
        "B_OUTCOME_HISTORY": ["paired_effects.json", "statistical_recipe_and_results.json"],
        "B_ATTEMPTED_LOGICAL": ["study_b_priority_audits.json", "analysis_input_documentation.json"],
        "B_ATTEMPTED_FIT_SENSITIVITY": ["study_b_priority_audits.json", "analysis_input_documentation.json"],
        "B_QUALITY_GREEDY": ["paired_effects.json", "statistical_recipe_and_results.json"],
        "B_SECONDARY_COMPARATOR_FAMILY": ["paired_effects.json", "statistical_recipe_and_results.json"],
        "B_COORDINATION_REMOVAL": ["paired_effects.json", "statistical_recipe_and_results.json"],
        "B_COLLISION_PATHWAY": ["analysis_input_documentation.json", "statistical_recipe_and_results.json"],
        "B_FAIRNESS": ["analysis_input_documentation.json", "statistical_recipe_and_results.json"],
        "B_OPERATIONAL_METRICS": ["analysis_input_documentation.json", "statistical_recipe_and_results.json"],
    }
    return {
        "schema_version": 1,
        "claim_registry": rel(project_root, claims_path),
        "claim_registry_access": public_access_for_source(rel(project_root, claims_path)),
        "claim_registry_sha256": sha256(claims_path),
        "evaluated_source_identity": {
            "parent_study_id": freeze["study_id"],
            "aggregate_source_sha256": freeze["aggregate_source_sha256"],
            "protocol_and_config_identity": {
                "protocol_version": freeze["protocol_version"],
                "protocol_sha256": freeze["protocol_sha256"],
                "config_py_sha256": freeze["source_sha256"]["config.py"],
                "environment_py_sha256": freeze["source_sha256"]["environment.py"],
                "model_py_sha256": freeze["source_sha256"]["model.py"],
                "policies_py_sha256": freeze["source_sha256"]["policies.py"],
                "train_py_sha256": freeze["source_sha256"]["train.py"],
                "evaluate_py_sha256": freeze["source_sha256"]["evaluate.py"],
                "analyze_py_sha256": freeze["source_sha256"]["analyze.py"],
                "aggregate_frozen_inputs_sha256": freeze["aggregate_frozen_inputs_sha256"],
                "aggregate_dependency_sha256": freeze["aggregate_dependency_sha256"],
            },
            "frozen_chronology": {
                "events": [
                    {
                        "sequence": 1,
                        "event": "protocol_source_inputs_dependencies_frozen",
                        "at_utc": freeze["frozen_at_utc"],
                        "record": source_ref(freeze_relative),
                    },
                    {
                        "sequence": 2,
                        "event": "prospective_results_validated_and_sealed",
                        "at_utc": seal["sealed_at_utc"],
                        "record": source_ref(seal_relative),
                        "aggregate_results_sha256": seal["aggregate_results_sha256"],
                        "validation_report_sha256": seal["validation_report_sha256"],
                    },
                ],
                "freeze_precedes_results_seal": True,
                "results_seal_binds_protocol_freeze": True,
                "results_seal_binds_frozen_inputs": True,
            },
        },
        "public_access_routes": {
            "target_zenodo_version": {
                "doi": new_doi,
                "doi_url": f"https://doi.org/{new_doi}",
                "record_url": f"https://zenodo.org/records/{zenodo_receipt['new_draft_identity']['reserved_record_id']}",
                "concept_doi": zenodo_receipt["new_draft_identity"]["concept_doi"],
                "target_version": zenodo_receipt["target_version"],
                "receipt_state": (
                    "submitted_public" if zenodo_receipt["new_draft_submitted"]
                    else "reserved_new_version_draft_not_submitted_at_receipt"
                ),
                "receipt": zenodo_receipt_reference,
            },
            "github": {
                "repository": repository,
                "commit": github_receipt["commit"],
                "visibility_at_receipt": github_receipt["repository_visibility_at_observation"],
                "public_visibility_verified": github_receipt["public_visibility_verified"],
                "receipt": github_receipt_reference,
            },
        },
        "routes": routes,
        "numbered_manuscript_elements": {
            "tables": [
                {
                    "number": "Table I",
                    "label": "tab:related-work-matrix",
                    "subject": "bounded comparison with five selected DSA studies",
                    "manuscript_source": {
                        **source_ref("main/manuscript/main.tex"),
                        "locator": "tab:related-work-matrix",
                        "status": "hash-bound non-public manuscript source; not sealed scientific evidence",
                    },
                    "evidence_route": [
                        {
                            **source_ref("main/manuscript/references.bib"),
                            "status": "hash-bound non-public manuscript bibliography",
                        },
                    ],
                    "claim_ids": [],
                    "route_scope": "literature synthesis; no experimental result is asserted by this table",
                },
                {
                    "number": "Table II",
                    "label": "tab:study-a-scenarios",
                    "subject": "Study A scenario definitions",
                    "manuscript_source": {
                        **source_ref("main/manuscript/main.tex"),
                        "locator": "tab:study-a-scenarios",
                        "status": "hash-bound non-public manuscript source; not sealed scientific evidence",
                    },
                    "evidence_route": [
                        source_ref("results/camera_ready_20261014_corrected/provenance/camera_ready_protocol_20261014.md"),
                        source_ref("results/crn_mp_pr1_20261017/provenance/crn_mp_pr1_20261017_protocol.md"),
                        reviewer_ref("checkpoint_registry.json"),
                    ],
                    "claim_ids": ["A_ARCHIVE_PRIORITY", "A_DIRECT_REPLICATION"],
                },
                {
                    "number": "Table III",
                    "label": "tab:prospective-results",
                    "subject": "Study B decision results",
                    "manuscript_source": {
                        **source_ref("main/manuscript/main.tex"),
                        "locator": "tab:prospective-results",
                        "status": "hash-bound non-public manuscript source; not sealed scientific evidence",
                    },
                    "evidence_route": [
                        source_ref("studies/crn-2026-003-finite-domain-dsa/evidence/gold/manuscript/result_macros.manifest.json"),
                        reviewer_ref("paired_effects.json"),
                        reviewer_ref("study_b_priority_audits.json"),
                        reviewer_ref("statistical_recipe_and_results.json"),
                        reviewer_ref("analysis_input_documentation.json"),
                    ],
                    "claim_ids": [
                        "B_PRIMARY_PHASE", "B_OUTCOME_HISTORY", "B_SEVEN_ACTION_GATE",
                        "B_QUALITY_GREEDY", "B_COORDINATION_REMOVAL",
                    ],
                },
            ],
            "figures": [
                {
                    "number": "Fig. 1",
                    "label": "fig:architecture",
                    "subject": "evidence architecture",
                    "manuscript_artifact": source_ref("main/manuscript/figures/system_architecture.pdf"),
                    "evidence_route": [
                        source_ref("studies/crn-2026-003-finite-domain-dsa/analysis/generate_architecture_figure.py"),
                        source_ref("studies/crn-2026-003-finite-domain-dsa/manuscript/claims.yaml"),
                        reviewer_ref("checkpoint_registry.json"),
                        {"artifact": "claim_to_artifact_manifest.json", "self_reference": True},
                    ],
                    "claim_ids": [],
                },
                {
                    "number": "Fig. 2",
                    "label": "fig:archived-certificate",
                    "subject": "finite-domain archived evidence",
                    "manuscript_artifact": source_ref("main/manuscript/figures/fig_archived_evidence.pdf"),
                    "evidence_route": [
                        source_ref(figure_manifest_relative),
                        source_ref("studies/crn-2026-003-finite-domain-dsa/evidence/gold/figure_inputs/source_manifest.json"),
                        reviewer_ref("study_a_certificates.json"),
                        reviewer_ref("boundary_witnesses.json"),
                    ],
                    "claim_ids": [
                        "A_ARCHIVE_PRIORITY", "A_DIRECT_REPLICATION", "A_OFF_SUPPORT",
                        "A_ANALYTICAL_RECONSTRUCTION", "A_DISPERSION_PATH",
                    ],
                },
                {
                    "number": "Fig. 3",
                    "label": "fig:collision-effects",
                    "subject": "prospective goodput and sensitivity",
                    "manuscript_artifact": source_ref("main/manuscript/figures/fig_collision_effects.pdf"),
                    "evidence_route": [
                        source_ref(figure_manifest_relative),
                        source_ref("studies/crn-2026-003-finite-domain-dsa/evidence/gold/figure_inputs/source_manifest.json"),
                        reviewer_ref("paired_effects.json"),
                        reviewer_ref("statistical_recipe_and_results.json"),
                    ],
                    "claim_ids": [
                        "B_PRIMARY_PHASE", "B_PHASE_NEGATIVE_SENSITIVITY",
                        "B_OUTCOME_HISTORY", "B_OUTCOME_HISTORY_SENSITIVITY",
                        "B_SECONDARY_COMPARATOR_FAMILY", "B_COORDINATION_REMOVAL",
                    ],
                },
                {
                    "number": "Fig. 4",
                    "label": "fig:collision-mechanisms",
                    "subject": "policy mechanism and operating diagnostics",
                    "manuscript_artifact": source_ref("main/manuscript/figures/fig_collision_mechanisms.pdf"),
                    "evidence_route": [
                        source_ref(figure_manifest_relative),
                        source_ref("studies/crn-2026-003-finite-domain-dsa/evidence/gold/figure_inputs/source_manifest.json"),
                        reviewer_ref("study_b_priority_audits.json"),
                        reviewer_ref("analysis_input_documentation.json"),
                        reviewer_ref("statistical_recipe_and_results.json"),
                    ],
                    "claim_ids": [
                        "B_SEVEN_ACTION_GATE", "B_ATTEMPTED_LOGICAL",
                        "B_ATTEMPTED_FIT_SENSITIVITY", "B_COORDINATE_CONTROL",
                        "B_COLLISION_PATHWAY", "B_LOAD_BOUND", "B_FAIRNESS",
                        "B_OPERATIONAL_METRICS",
                    ],
                },
            ],
        },
        "scope_note": "Only claims materially served by the new reviewer exports are routed; other claims retain their existing claim-registry evidence.",
    }


def _write_and_record(output_dir: Path, name: str, value: Any) -> dict[str, Any]:
    path = output_dir / name
    write_json(path, value)
    return {"file": name, "sha256": sha256(path), "bytes": path.stat().st_size}


def generate(
    project_root: Path,
    study_root: Path,
    output_dir: Path,
    github_receipt_path: Path = DEFAULT_GITHUB_RECEIPT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    study_root = study_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    verified_inputs = verify_pins(project_root)
    summary_rows = load_summary(project_root)

    checkpoint_registry, study_a_summaries = build_checkpoint_registry(project_root, summary_rows)
    study_a = build_study_a_certificates(project_root, checkpoint_registry, study_a_summaries)
    study_b = build_study_b_priority_records(project_root, summary_rows, checkpoint_registry)
    boundaries = build_boundary_witnesses(project_root, checkpoint_registry)
    effects = build_paired_effects(project_root, summary_rows, checkpoint_registry)

    artifacts: list[dict[str, Any]] = []
    artifacts.append(_write_and_record(output_dir, "checkpoint_registry.json", checkpoint_registry))
    artifacts.append(_write_and_record(output_dir, "study_a_certificates.json", study_a))
    artifacts.append(_write_and_record(output_dir, "study_b_priority_audits.json", study_b))
    artifacts.append(_write_and_record(output_dir, "boundary_witnesses.json", boundaries))
    artifacts.append(_write_and_record(output_dir, "paired_effects.json", effects))

    index_path = output_dir / "analysis_input_index.csv"
    index_identity = write_analysis_input_index(index_path, summary_rows)
    assert_public_text_safe(index_path.name, index_path.read_bytes())
    artifacts.append({"file": index_path.name, "sha256": sha256(index_path), "bytes": index_path.stat().st_size})
    input_documentation = build_input_documentation(project_root, summary_rows, index_identity)
    artifacts.append(_write_and_record(output_dir, "analysis_input_documentation.json", input_documentation))
    statistics = build_statistical_recipe(project_root, study_b, effects)
    artifacts.append(_write_and_record(output_dir, "statistical_recipe_and_results.json", statistics))
    claim_routes = build_claim_routes(
        project_root, output_dir, github_receipt_path=github_receipt_path,
    )
    artifacts.append(_write_and_record(output_dir, "claim_to_artifact_manifest.json", claim_routes))

    manifest = {
        "schema_version": 1,
        "study_id": "crn-2026-003-finite-domain-dsa",
        "generator": rel(project_root, SCRIPT),
        "generator_sha256": sha256(SCRIPT),
        "derivation_contract": {
            "read_only_inputs": True,
            "new_training": False,
            "new_simulation": False,
            "new_experiment": False,
            "display_rounding": False,
            "fail_closed": True,
        },
        "expected_record_counts": {
            "checkpoint_registry": 76,
            "study_a_certificates": 16,
            "study_b_full_policy_records": 20,
            "boundary_witnesses": 44,
            "paired_effect_records": 20,
            "analysis_input_rows": 12_672,
        },
        "artifacts": artifacts,
        "verified_source_inputs": [
            {
                "source_id": "sealed-input-" + canonical_sha256({"relative": relative})[:16],
                "sha256": identity["sha256"],
                "bytes": identity["bytes"],
                "access": public_access_for_source(relative),
            }
            for relative, identity in sorted(verified_inputs.items())
        ],
        "truthful_nonreconstruction": [
            "The 44 retained boundary witnesses include the decisive float32 pairwise Q margin, but not a retained six-value raw Q vector; the exporter does not fabricate it.",
            "Per-node delivery vectors are not present in the aggregate cube, so Jain fairness scalars cannot be independently reconstructed without replay.",
            "The compact index retains slot-trace identities, not the excluded 9.31-GB slot-level trace payload.",
        ],
    }
    manifest_path = output_dir / "reviewer_evidence_manifest.json"
    write_json(manifest_path, manifest)
    actual_names = tuple(sorted(path.name for path in output_dir.iterdir() if path.is_file()))
    require(
        actual_names == PUBLIC_OUTPUT_NAMES,
        f"reviewer output inventory mismatch: {actual_names}",
    )
    for name in actual_names:
        assert_public_text_safe(name, (output_dir / name).read_bytes())
    return {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "artifact_count_excluding_manifest": len(artifacts),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    result.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_STUDY_ROOT / "evidence/reviewer",
    )
    result.add_argument(
        "--github-receipt",
        type=Path,
        default=DEFAULT_GITHUB_RECEIPT,
        help=(
            "project-relative or absolute public GitHub verification receipt; "
            "defaults to the final current-version receipt. A legacy receipt may be "
            "supplied explicitly only for pre-push staging."
        ),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    result = generate(
        args.project_root,
        args.study_root,
        args.output_dir,
        github_receipt_path=args.github_receipt,
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
