"""Focused contract tests for the compact reviewer-evidence export."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("export_reviewer_evidence.py")
SPEC = importlib.util.spec_from_file_location("export_reviewer_evidence", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)

PROJECT_ROOT = MODULE_PATH.parents[3]
STUDY_ROOT = MODULE_PATH.parents[1]
REVIEWER = STUDY_ROOT / "evidence/reviewer"
LEGACY_GITHUB_RECEIPT = Path(
    "studies/crn-2026-003-finite-domain-dsa/release/"
    "github_current_commit_receipt.json"
)
BASE_ARCHIVE = PROJECT_ROOT / ("main_" + "archive") / (
    "submission_collision_aware_dsa_20261017/"
    "Collision_Aware_DSA_Reproducibility_20261017.zip"
)
PUBLIC_PACKAGE = (PROJECT_ROOT / "PUBLIC_SANITIZATION.json").is_file()
PUBLIC_COMPACT_ARCHIVE = PROJECT_ROOT / EXPORT.PUBLIC_COMPACT_REPLICATION


def _public_member_context() -> tuple[set[str], bytes]:
    """Return public members and compact-archive bytes in either layout."""
    if PUBLIC_PACKAGE:
        members = {
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in PROJECT_ROOT.rglob("*")
            if path.is_file() and ".venv" not in path.parts
        }
        return members, PUBLIC_COMPACT_ARCHIVE.read_bytes()
    with zipfile.ZipFile(BASE_ARCHIVE) as outer:
        return set(outer.namelist()), outer.read(EXPORT.PUBLIC_COMPACT_REPLICATION)


def _json(name: str) -> dict:
    return json.loads((REVIEWER / name).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_record_contract_and_manifest_hashes() -> None:
    assert len(_json("checkpoint_registry.json")["records"]) == 76
    assert len(_json("study_a_certificates.json")["records"]) == 16
    assert len(_json("study_b_priority_audits.json")["records"]) == 20
    assert len(_json("boundary_witnesses.json")["records"]) == 44
    assert len(_json("paired_effects.json")["records"]) == 20
    with (REVIEWER / "analysis_input_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12_672
    assert len({row["run_id"] for row in rows}) == 12_672

    manifest = _json("reviewer_evidence_manifest.json")
    assert manifest["derivation_contract"] == {
        "display_rounding": False,
        "fail_closed": True,
        "new_experiment": False,
        "new_simulation": False,
        "new_training": False,
        "read_only_inputs": True,
    }
    for item in manifest["artifacts"]:
        path = REVIEWER / item["file"]
        assert path.stat().st_size == item["bytes"]
        assert _sha(path) == item["sha256"]


def _walk(value: object) -> Iterator[object]:
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(key)
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def test_all_ten_public_exports_and_packaged_sources_are_privacy_safe() -> None:
    files = sorted(path for path in REVIEWER.iterdir() if path.is_file())
    assert tuple(path.name for path in files) == EXPORT.PUBLIC_OUTPUT_NAMES
    for path in files:
        data = path.read_bytes()
        EXPORT.assert_public_text_safe(path.name, data)
        if path.suffix == ".json":
            for value in _walk(json.loads(data)):
                if isinstance(value, str):
                    EXPORT.assert_public_text_safe(path.name, value)
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    for value in _walk(row):
                        if isinstance(value, str):
                            EXPORT.assert_public_text_safe(path.name, value)

    # These two Python files are also public release members.
    for path in (MODULE_PATH, Path(__file__)):
        text = path.read_text(encoding="utf-8")
        if text.startswith("#!"):
            text = text.split("\n", 1)[1]
        for description, pattern in EXPORT.PUBLIC_FORBIDDEN_PATTERNS:
            if description == "absolute filesystem path":
                # Public Python source legitimately contains package-relative
                # slash fragments; reject only machine-local absolute roots.
                roots = tuple(
                    "/" + part + "/"
                    for part in ("ho" + "me", "tm" + "p", "Us" + "ers", "ro" + "ot")
                )
                assert all(root not in text for root in roots), path.name
            else:
                assert pattern.search(text) is None, (path.name, description)


@pytest.mark.parametrize(
    "payload",
    [
        "/" + "home" + "/analyst/private.json",
        "/" + "tmp" + "/reviewer-output.json",
        "C:" + "\\" + "Users" + "\\analyst\\private.json",
        "sdr" + "-research",
        "Editorial" + "_Review/private.docx",
        "main_" + "archive/private.json",
        "." + "env" + "." + "zenodo",
        "ZENODO" + "_ACCESS_TOKEN=value",
        "ghp_" + "A" * 32,
        "Bearer " + "A" * 32,
        "-" * 5 + "BEGIN PRIVATE KEY" + "-" * 5,
    ],
)
def test_privacy_guard_rejects_nested_sensitive_values(payload: str) -> None:
    nested = {"outer": [{"inner": payload}]}
    with pytest.raises(EXPORT.EvidenceError):
        EXPORT.assert_public_text_safe("nested.json", json.dumps(nested))


def test_registry_cohorts_and_study_a_accounting() -> None:
    registry = _json("checkpoint_registry.json")
    cohorts: dict[str, int] = {}
    for record in registry["records"]:
        cohorts[record["cohort"]] = cohorts.get(record["cohort"], 0) + 1
        assert len(record["checkpoint_sha256"]) == 64
        assert record["dataset_membership"]["smoke_fixture"] is False
    assert cohorts == {
        "archived_discovery": 8,
        "independent_replication": 8,
        "prospective_collision": 60,
    }

    certificates = _json("study_a_certificates.json")["records"]
    for certificate in certificates:
        assert certificate["selection_and_ties"]["unique_selected_order"] is True
        for key in ("calibration", "validation_without_refit", "complete_core"):
            block = certificate[key]
            assert block["agreement"] == 1.0
            accounting = block["all_occupied_accounting"]
            assert accounting["identity_pass"] is True
            assert accounting["total_node_decisions"] == (
                accounting["attempted_channel_decisions"]
                + accounting["all_occupied_empty_legal_set_decisions"]
            )


def _access_records(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        if value.get("mode") in {
            "package_member", "nested_archive_member",
            "public_repository", "hash_identity_only",
        }:
            yield value
        for item in value.values():
            yield from _access_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _access_records(item)


def test_all_declared_public_access_routes_resolve_or_are_hash_only() -> None:
    outer_names, compact = _public_member_context()
    with zipfile.ZipFile(io.BytesIO(compact)) as inner:
        inner_names = set(inner.namelist())

    access_records = []
    for path in sorted(REVIEWER.glob("*.json")):
        access_records.extend(_access_records(json.loads(path.read_text(encoding="utf-8"))))
    assert len(access_records) >= 1_700
    for access in access_records:
        mode = access["mode"]
        if mode == "package_member":
            locator = access["locator"]
            assert not locator.startswith("/")
            assert ".." not in Path(locator).parts
            assert locator in outer_names or (PROJECT_ROOT / locator).is_file(), locator
        elif mode == "nested_archive_member":
            assert access["outer_member"] == EXPORT.PUBLIC_COMPACT_REPLICATION
            assert access["outer_member"] in outer_names
            assert access["inner_member"] in inner_names
            assert access["locator"] == (
                f"{access['outer_member']}::{access['inner_member']}"
            )
        elif mode == "public_repository":
            repository_path = access["repository_path"]
            assert not repository_path.startswith("/")
            assert ".." not in Path(repository_path).parts
            assert (PROJECT_ROOT / repository_path).is_file(), repository_path
        else:
            assert mode == "hash_identity_only"
            assert access.get("reason")
            assert "locator" not in access
            assert "repository_path" not in access


def test_public_repository_routes_follow_github_allowlist_and_exclude_receipts() -> None:
    release_dir = STUDY_ROOT / "release"
    if not (release_dir / "build_github_stage.py").is_file():
        pytest.skip(
            "workspace-only GitHub staging tool is intentionally omitted from the public package"
        )
    sys.path.insert(0, str(release_dir))
    try:
        github_builder = importlib.import_module("build_github_stage")
    finally:
        sys.path.pop(0)

    repository_routes = []
    for path in sorted(REVIEWER.glob("*.json")):
        for access in _access_records(json.loads(path.read_text(encoding="utf-8"))):
            if access["mode"] == "public_repository":
                repository_routes.append(access["repository_path"])
    for repository_path in repository_routes:
        assert github_builder.source_allowed(repository_path)
        assert "/release/" not in repository_path
        assert not repository_path.endswith("_receipt.json")

    routing = _json("claim_to_artifact_manifest.json")["public_access_routes"]
    for receipt in (
        routing["github"]["receipt"],
        routing["target_zenodo_version"]["receipt"],
    ):
        assert receipt["access"]["mode"] == "hash_identity_only"
        assert len(receipt["sha256"]) == 64
        assert "path" not in receipt
        assert "repository_path" not in receipt["access"]


def test_checkpoint_routes_and_sanitized_source_versions() -> None:
    registry = _json("checkpoint_registry.json")["records"]
    for record in registry:
        assert record["checkpoint"] == record["checkpoint_access"]["locator"]
        assert record["checkpoint_access"]["mode"] in {
            "package_member", "nested_archive_member",
        }
        assert record["metadata_access"]["mode"] in {
            "package_member", "nested_archive_member", "hash_identity_only",
        }
        source = record["source_version"]
        if record["cohort"] in {"archived_discovery", "independent_replication"}:
            git = source["git"]
            assert set(git) == {
                "branch", "head", "tracked_diff_sha256",
                "workspace_state_sha256", "workspace_state_entry_count",
                "working_tree_clean_at_capture", "privacy_note",
            }
            assert len(git["head"]) == 40
            assert len(git["tracked_diff_sha256"]) == 64
            assert len(git["workspace_state_sha256"]) == 64
            assert source["manifest_access"]["mode"] in {
                "nested_archive_member", "hash_identity_only",
            }
            for evaluation in record["evaluation_records"]:
                assert evaluation["summary_access"]["mode"] in {
                    "nested_archive_member", "hash_identity_only",
                }
        else:
            assert source["protocol_freeze_access"]["mode"] == "package_member"
            assert source["results_seal_access"]["mode"] == "package_member"


def test_documented_entry_points_use_extracted_package_layout() -> None:
    document = _json("analysis_input_documentation.json")
    summary = (
        "prospective_collision_study/results/crn_ca_pr1_20261017/"
        "evaluation/summary.csv"
    )
    assert document["analysis_input"]["path"] == summary
    assert document["analysis_input"]["access"] == {
        "mode": "package_member", "locator": summary,
    }
    by_script = {item["script"]: item for item in document["analysis_entry_points"]}
    assert set(by_script) == {
        "article_reanalysis/recompute_inference.py",
        "article_reanalysis/reanalyse.py",
    }
    recompute = by_script["article_reanalysis/recompute_inference.py"]["command"]
    assert summary in recompute
    assert " --input " not in recompute
    assert recompute.endswith("--output recomputed_inference.json")
    reanalyse = by_script["article_reanalysis/reanalyse.py"]["command"]
    assert "--study-root prospective_collision_study" in reanalyse
    assert "--out-dir reproduced_reanalysis" in reanalyse

    if PUBLIC_PACKAGE:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "analysis/recompute_inference.py" in readme
        assert "article_reanalysis/recompute_inference.py" in readme
        assert (PROJECT_ROOT / summary).is_file()
    else:
        config = json.loads(
            (STUDY_ROOT / "release/release_config.json").read_text(encoding="utf-8")
        )
        assert "analysis/recompute_inference.py" in config["article_reanalysis"]
        assert "analysis/reanalyse.py" in config["article_reanalysis"]
        with zipfile.ZipFile(BASE_ARCHIVE) as outer:
            assert summary in outer.namelist()


def test_cycles_boundary_inputs_and_paired_effect_precision() -> None:
    priority = _json("study_b_priority_audits.json")["records"]
    for record in priority:
        for split in ("calibration", "validation", "full_support"):
            certificate = record["certificates"][split]
            assert certificate["acyclic"] is False
            assert certificate["representing_order_count"] == 0
            cycle = certificate["shortest_cycle_witness"]
            assert cycle["length"] >= 2
            assert cycle["closed_channel_path"][0] == cycle["closed_channel_path"][-1]
            assert len(cycle["edge_witnesses"]) == cycle["length"]
            assert all(edge["aggregate_support_count"] > 0 for edge in cycle["edge_witnesses"])

    boundaries = _json("boundary_witnesses.json")["records"]
    assert len({record["witness_id"] for record in boundaries}) == 44
    for record in boundaries:
        assert record["input"]["dtype"] == "float32"
        assert record["input"]["shape"] == [128]
        assert len(record["input"]["values"]) == 128
        assert record["q_comparison"]["dtype"] == "float32"
        assert record["q_comparison"]["pairwise_margin"] <= 0.0

    effects = _json("paired_effects.json")
    assert effects["equal_cell_weights"]["individual_factor_environment_cell_weight"] == "1/192"
    for record in effects["records"]:
        assert len(record["comparisons"]) == 8
        for comparison in record["comparisons"]:
            assert abs(
                comparison["paired_macro_difference"]
                - comparison["recomputed_from_192_cells"]
            ) <= 2e-15


def test_r4_source_access_and_numbered_element_routes() -> None:
    routing = _json("claim_to_artifact_manifest.json")
    source = routing["evaluated_source_identity"]
    assert source["aggregate_source_sha256"] == (
        "43166a4456347424bc04577b03d793206ea37086211e3423bcb0013d2ee9ccc7"
    )
    identity = source["protocol_and_config_identity"]
    assert identity["protocol_version"] == "2.0.0"
    assert identity["protocol_sha256"] == (
        "a2cfac7ac2e3c58e313f7d92af3ddfc8c0878471b0d449a38a168949b1842fcd"
    )
    assert identity["config_py_sha256"] == (
        "e8829bc08d99aa7d3a1d09739a23148d5097a4d78f97f858babe6cdf85883c2d"
    )
    chronology = source["frozen_chronology"]
    assert chronology["freeze_precedes_results_seal"] is True
    assert chronology["results_seal_binds_protocol_freeze"] is True
    assert chronology["results_seal_binds_frozen_inputs"] is True
    assert [event["sequence"] for event in chronology["events"]] == [1, 2]
    assert chronology["events"][0]["at_utc"] < chronology["events"][1]["at_utc"]

    access = routing["public_access_routes"]
    assert access["target_zenodo_version"]["doi"] == "10.5281/zenodo.23527228"
    assert access["target_zenodo_version"]["doi_url"] == (
        "https://doi.org/10.5281/zenodo.23527228"
    )
    assert access["github"]["repository"] == (
        "https://github.com/lexason/"
        "Masked-Priority-Diagnosis-and-Contention-Aware-DQN-Policies-for-Dynamic-Spectrum-Access"
    )
    assert access["github"]["public_visibility_verified"] is True

    numbered = routing["numbered_manuscript_elements"]
    assert [(item["number"], item["label"]) for item in numbered["tables"]] == [
        ("Table I", "tab:related-work-matrix"),
        ("Table II", "tab:study-a-scenarios"),
        ("Table III", "tab:prospective-results"),
    ]
    assert [(item["number"], item["label"]) for item in numbered["figures"]] == [
        ("Fig. 1", "fig:architecture"),
        ("Fig. 2", "fig:archived-certificate"),
        ("Fig. 3", "fig:collision-effects"),
        ("Fig. 4", "fig:collision-mechanisms"),
    ]
    assert all(item["evidence_route"] for item in numbered["tables"])
    assert all(item["evidence_route"] for item in numbered["figures"])
    for item in numbered["tables"]:
        source = item["manuscript_source"]
        assert source["access"]["mode"] == "hash_identity_only"
        assert len(source["sha256"]) == 64
        assert "path" not in source
    bibliography = numbered["tables"][0]["evidence_route"][0]
    assert bibliography["access"]["mode"] == "hash_identity_only"
    assert len(bibliography["sha256"]) == 64
    assert "path" not in bibliography


def test_github_receipt_default_and_explicit_staging_override() -> None:
    if PUBLIC_PACKAGE:
        pytest.skip(
            "receipt-staging regression test requires intentionally omitted workspace provenance inputs"
        )
    args = EXPORT.parser().parse_args([])
    assert args.github_receipt == EXPORT.DEFAULT_GITHUB_RECEIPT
    assert args.github_receipt.as_posix().endswith(
        "release/github_current_commit_v1.1.0_receipt.json"
    )

    staged = EXPORT.build_claim_routes(
        PROJECT_ROOT,
        REVIEWER,
        github_receipt_path=LEGACY_GITHUB_RECEIPT,
    )
    staged_receipt = staged["public_access_routes"]["github"]["receipt"]
    assert staged_receipt["access"]["mode"] == "hash_identity_only"
    assert staged_receipt["receipt_scope"] == "pre_push_staging_observation"
    assert staged_receipt["sha256"] == _sha(PROJECT_ROOT / LEGACY_GITHUB_RECEIPT)
    assert "path" not in staged_receipt

    final_receipt = PROJECT_ROOT / EXPORT.DEFAULT_GITHUB_RECEIPT
    if final_receipt.is_file():
        final = EXPORT.build_claim_routes(
            PROJECT_ROOT,
            REVIEWER,
            github_receipt_path=EXPORT.DEFAULT_GITHUB_RECEIPT,
        )
        final_receipt_record = final["public_access_routes"]["github"]["receipt"]
        assert final_receipt_record["access"]["mode"] == "hash_identity_only"
        assert final_receipt_record["receipt_scope"] == (
            "v1.1_final_public_commit_observation"
        )
        assert final_receipt_record["sha256"] == _sha(final_receipt)
        assert "path" not in final_receipt_record
    else:
        with pytest.raises(EXPORT.EvidenceError, match="cannot read JSON"):
            EXPORT.build_claim_routes(
                PROJECT_ROOT,
                REVIEWER,
                github_receipt_path=EXPORT.DEFAULT_GITHUB_RECEIPT,
            )


def test_regeneration_is_byte_deterministic(tmp_path: Path) -> None:
    if PUBLIC_PACKAGE:
        pytest.skip(
            "full reviewer-export regeneration requires intentionally omitted sealed workspace inputs"
        )
    rebuilt = tmp_path / "reviewer"
    receipt_scope = (
        _json("claim_to_artifact_manifest.json")["public_access_routes"]
        ["github"]["receipt"]["receipt_scope"]
    )
    recorded_receipt = (
        EXPORT.DEFAULT_GITHUB_RECEIPT
        if receipt_scope == "v1.1_final_public_commit_observation"
        else LEGACY_GITHUB_RECEIPT
    )
    EXPORT.generate(
        PROJECT_ROOT,
        STUDY_ROOT,
        rebuilt,
        github_receipt_path=recorded_receipt,
    )
    expected_files = sorted(path.name for path in REVIEWER.iterdir() if path.is_file())
    assert sorted(path.name for path in rebuilt.iterdir() if path.is_file()) == expected_files
    for name in expected_files:
        if name == "reviewer_evidence_manifest.json":
            expected = _json(name)
            actual = json.loads((rebuilt / name).read_text(encoding="utf-8"))
            # During the pre-push bootstrap window the checked-in outputs
            # intentionally retain the prior generator hash. All scientific
            # content and routed evidence must nevertheless remain identical.
            expected["generator_sha256"] = actual["generator_sha256"]
            assert actual == expected
        else:
            assert (rebuilt / name).read_bytes() == (REVIEWER / name).read_bytes(), name
