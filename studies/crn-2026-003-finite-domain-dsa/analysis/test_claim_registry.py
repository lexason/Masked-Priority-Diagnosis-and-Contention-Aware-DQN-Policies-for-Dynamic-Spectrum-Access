"""Public-package checks for claim lineage without manuscript source files."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

ANALYSIS_ROOT = Path(__file__).resolve().parent
STUDY_ROOT = ANALYSIS_ROOT.parent
RELEASE_ROOT = STUDY_ROOT.parents[1]
REGISTRY_PATH = STUDY_ROOT / "manuscript/claims.yaml"
MACRO_PATH = STUDY_ROOT / "evidence/gold/manuscript/result_macros.tex"
MACRO_MANIFEST_PATH = STUDY_ROOT / "evidence/gold/manuscript/result_macros.manifest.json"
SANITIZATION_PATH = RELEASE_ROOT / "PUBLIC_SANITIZATION.json"
PUBLIC_PROVENANCE_PATH = STUDY_ROOT / "PUBLIC_PROVENANCE.json"
COMPACT_MEMBER = (
    "independent_replications/independent_masked_priority/"
    "CRN_MP_PR1_20261017_Compact_Replication.zip"
)
EXPECTED_CLAIM_IDS = {
    "A_ANALYTICAL_RECONSTRUCTION",
    "A_ARCHIVE_PRIORITY",
    "A_ASSAY_DISCRIMINATION",
    "A_DIRECT_REPLICATION",
    "A_DISPERSION_PATH",
    "A_OFF_SUPPORT",
    "A_SHARED_MASK_SENSITIVITY",
    "A_TONE_RANK_CHECK",
    "B_ATTEMPTED_FIT_SENSITIVITY",
    "B_ATTEMPTED_LOGICAL",
    "B_CENTRALIZED_REFERENCE",
    "B_COLLISION_PATHWAY",
    "B_COORDINATE_CONTROL",
    "B_COORDINATION_REMOVAL",
    "B_DESIGN_AND_AUDIT_COUNTS",
    "B_ELIGIBLE_PARTITION",
    "B_FAIRNESS",
    "B_LOAD_BOUND",
    "B_OPERATIONAL_METRICS",
    "B_OUTCOME_HISTORY",
    "B_OUTCOME_HISTORY_SENSITIVITY",
    "B_PHASE_NEGATIVE_SENSITIVITY",
    "B_PRIMARY_PHASE",
    "B_QUALITY_GREEDY",
    "B_SECONDARY_COMPARATOR_FAMILY",
    "B_SEVEN_ACTION_GATE",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


def resolve_release_member(value: str) -> Path:
    if value.startswith("reproducibility/"):
        return RELEASE_ROOT / value.removeprefix("reproducibility/")
    if value.startswith("gold/"):
        return STUDY_ROOT / "evidence" / value
    return RELEASE_ROOT / value


def test_registry_identity_review_boundary_and_required_claims() -> None:
    registry = load_registry()
    provenance = json.loads(PUBLIC_PROVENANCE_PATH.read_text(encoding="utf-8"))
    review = provenance["control_approvals"]["claim_registry_review"]
    scientific_digest = provenance["scientific_payload_manifest"]["sha256"]
    assert registry["schema_version"] == 2
    assert registry["study_id"] == "crn-2026-003-finite-domain-dsa"
    assert registry["registry_status"] == review["status"]
    assert registry["reviewer"] == review["reviewed_by"]
    if review["status"] == "approved":
        assert isinstance(review["reviewed_by"], str)
        assert review["reviewed_by"].strip()
        assert isinstance(review["reviewed_at"], str)
        assert review["reviewed_at"].strip()
        assert review["scientific_payload_manifest_sha256"] == scientific_digest
    else:
        assert review == {
            "status": "pending_independent_claim_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "scientific_payload_manifest_sha256": None,
        }
    ids = [claim["id"] for claim in registry["claims"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == EXPECTED_CLAIM_IDS


def test_every_claim_member_and_evidence_identity_resolves() -> None:
    sanitization = json.loads(SANITIZATION_PATH.read_text(encoding="utf-8"))
    for claim in load_registry()["claims"]:
        release_member = resolve_release_member(claim["release_member"])
        assert release_member.is_file(), (claim["id"], release_member)
        for evidence in claim["evidence"]:
            assert len({"local", "parent"} & set(evidence)) == 1, claim["id"]
            assert re.fullmatch(r"[0-9a-f]{64}", evidence["sha256"])
            if "local" in evidence:
                path = STUDY_ROOT / evidence["local"]
                assert path.is_file(), (claim["id"], path)
                assert sha256(path) == evidence["sha256"], (claim["id"], path)
            else:
                path = RELEASE_ROOT / evidence["parent"]
                assert path.is_file(), (claim["id"], path)
                if evidence["parent"] == COMPACT_MEMBER:
                    assert evidence["sha256"] == sanitization["original_archive_sha256"]
                    assert sha256(path) == sanitization["sanitized_archive_sha256"]
                else:
                    assert sha256(path) == evidence["sha256"], (claim["id"], path)


def test_generated_macro_interface_and_claim_coverage_are_hash_bound() -> None:
    registry = load_registry()
    interface = registry["generated_result_interface"]
    assert sha256(MACRO_PATH) == interface["sha256"]
    assert sha256(MACRO_MANIFEST_PATH) == interface["manifest_sha256"]
    manifest = json.loads(MACRO_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["artifact_sha256"] == sha256(MACRO_PATH)
    definitions = {
        item.decode("ascii")
        for item in re.findall(rb"\\newcommand\{(\\Result[A-Za-z]+)\}", MACRO_PATH.read_bytes())
    }
    registered = {
        "\\" + macro for claim in registry["claims"] for macro in claim.get("value_macros", [])
    }
    assert definitions
    assert len(definitions) == 157
    assert definitions == registered


def test_public_claim_controls_have_no_private_references() -> None:
    prohibited = (
        "Editorial" + "_Review",
        "SIX" + "_SIGMA_RESEARCH_PROTOCOL",
        "/" + "home/",
        "/" + "tmp/",
        "main" + "_archive/",
        "manuscript" + ".pdf",
    )
    for path in (REGISTRY_PATH, MACRO_PATH, MACRO_MANIFEST_PATH):
        text = path.read_text(encoding="utf-8")
        assert not [token for token in prohibited if token in text]
