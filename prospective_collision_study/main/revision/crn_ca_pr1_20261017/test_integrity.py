"""Adversarial regression tests for the v2 validation and provenance gates."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from main.revision.crn_ca_pr1_20261017 import provenance
from main.revision.crn_ca_pr1_20261017 import validation as validation_module
from main.revision.crn_ca_pr1_20261017.config import (
    DEPENDENCY_FILENAMES,
    GAMMA,
    LEARNED_POLICIES,
    N_ACTIONS,
    N_CHANNELS,
    SMOKE_MODEL_SEED,
    STATE_DIM,
)
from main.revision.crn_ca_pr1_20261017.evaluate import build_tasks, evaluate_task
from main.revision.crn_ca_pr1_20261017.model import CollisionAwareDQN, ReplayBatch
from main.revision.crn_ca_pr1_20261017.provenance import (
    initialise_freeze,
    sha256_file,
    verify_freeze,
)
from main.revision.crn_ca_pr1_20261017.train import train_one
from main.revision.crn_ca_pr1_20261017.validation import (
    validate,
    validate_evaluation,
    validate_training,
)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def valid_smoke_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("integrity-smoke")
    for policy in LEARNED_POLICIES:
        train_one(
            seed_index=0,
            output_root=root,
            policy_variant=policy,
            smoke=True,
        )
    rows = [evaluate_task(task) for task in build_tasks(output_root=root, smoke=True)]
    evaluation_path = root / "smoke" / "evaluation" / "summary.csv"
    evaluation_path.parent.mkdir(parents=True)
    _write_rows(evaluation_path, rows)
    assert validate(root, smoke=True)["status"] == "PASS"
    return root


def _copy_smoke(source: Path, destination_parent: Path) -> Path:
    destination = destination_parent / "smoke-copy"
    shutil.copytree(source, destination)
    return destination


def test_validation_report_is_repeatable_but_not_overwritable(
    valid_smoke_root: Path, tmp_path: Path
) -> None:
    first = validate(valid_smoke_root, smoke=True)
    assert validate(valid_smoke_root, smoke=True) == first

    destination = tmp_path / "report.json"
    validation_module._write_json_once_or_identical(destination, {"value": 1})
    validation_module._write_json_once_or_identical(destination, {"value": 1})
    with pytest.raises(FileExistsError, match="non-identical"):
        validation_module._write_json_once_or_identical(destination, {"value": 2})


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("run_id", "tampered-run", "noncanonical run ID"),
        ("checkpoint_sha256", "0" * 64, "checkpoint identity"),
        ("replication_index", "1", "replication index"),
        ("logical_to_physical", "1-0-2-3-4-5", "channel permutation"),
        ("measurement_steps", "51", "measurement duration"),
        ("occupancy_process", "markov", "occupancy process"),
        ("trace_sha256", "0" * 64, "complete common traces"),
    ],
)
def test_evaluation_identity_tampering_is_rejected(
    valid_smoke_root: Path,
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    root = _copy_smoke(valid_smoke_root, tmp_path)
    path = root / "smoke" / "evaluation" / "summary.csv"
    rows = _read_rows(path)
    learned = next(row for row in rows if row["policy"] in LEARNED_POLICIES)
    learned[field] = replacement
    _write_rows(path, rows)
    with pytest.raises(ValueError, match=message):
        validate_evaluation(root, smoke=True)


def test_evaluation_cartesian_membership_is_exact(
    valid_smoke_root: Path, tmp_path: Path
) -> None:
    root = _copy_smoke(valid_smoke_root, tmp_path)
    path = root / "smoke" / "evaluation" / "summary.csv"
    rows = _read_rows(path)
    _write_rows(path, rows[:-1])
    with pytest.raises(ValueError, match="expected 9 evaluation rows"):
        validate_evaluation(root, smoke=True)


def test_evaluation_csv_schema_is_exact(valid_smoke_root: Path, tmp_path: Path) -> None:
    root = _copy_smoke(valid_smoke_root, tmp_path)
    path = root / "smoke" / "evaluation" / "summary.csv"
    rows = _read_rows(path)
    for row in rows:
        row["unregistered_metric"] = "1"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="CSV schema mismatch"):
        validate_evaluation(root, smoke=True)


def test_mask_table_illegal_action_is_rejected(
    valid_smoke_root: Path,
) -> None:
    path = valid_smoke_root / "smoke" / "evaluation" / "summary.csv"
    rows = _read_rows(path)
    row = next(item for item in rows if int(item["priority_eligible_decisions"]) > 0)
    counts = [int(value) for value in row["mask_action_counts"].split("-")]
    source = next(index for index, value in enumerate(counts) if value > 0)
    illegal = N_ACTIONS + 1  # mask 1 exposes channel 0, never channel 1.
    counts[source] -= 1
    counts[illegal] += 1
    row["mask_action_counts"] = "-".join(str(value) for value in counts)
    with pytest.raises(ValueError, match="illegal action"):
        validation_module._validate_evaluation_metrics(
            row, measurement_steps=int(row["measurement_steps"])
        )


def test_action_count_and_priority_count_identities_are_recomputed(
    valid_smoke_root: Path,
) -> None:
    path = valid_smoke_root / "smoke" / "evaluation" / "summary.csv"
    rows = _read_rows(path)
    attempted_row = next(item for item in rows if int(item["attempted"]) > 0)
    logical = [int(value) for value in attempted_row["logical_action_counts"].split("-")]
    source = next(index for index in range(N_CHANNELS) if logical[index] > 0)
    logical[source] -= 1
    logical[-1] += 1
    attempted_row["logical_action_counts"] = "-".join(str(value) for value in logical)
    with pytest.raises(ValueError, match="logical attempted-action"):
        validation_module._validate_evaluation_metrics(
            attempted_row,
            measurement_steps=int(attempted_row["measurement_steps"]),
        )

    rows = _read_rows(path)
    priority_row = next(
        item
        for item in rows
        if int(item["priority_eligible_decisions"]) > 0
        and int(item["priority_matches"]) > 0
    )
    matches = int(priority_row["priority_matches"]) - 1
    eligible = int(priority_row["priority_eligible_decisions"])
    priority_row["priority_matches"] = str(matches)
    priority_row["static_priority_agreement"] = str(matches / eligible)
    with pytest.raises(ValueError, match="mask/action table"):
        validation_module._validate_evaluation_metrics(
            priority_row,
            measurement_steps=int(priority_row["measurement_steps"]),
        )


def test_training_schedule_and_common_trace_tampering_is_rejected(
    valid_smoke_root: Path, tmp_path: Path
) -> None:
    metadata_root = _copy_smoke(valid_smoke_root, tmp_path / "metadata")
    metadata_path = (
        metadata_root
        / "smoke"
        / "training"
        / LEARNED_POLICIES[0]
        / "seed_smoke"
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["global_environment_steps"] += 1
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="global_environment_steps"):
        validate_training(metadata_root, smoke=True)

    trace_root = _copy_smoke(valid_smoke_root, tmp_path / "trace")
    episodes_path = (
        trace_root
        / "smoke"
        / "training"
        / LEARNED_POLICIES[0]
        / "seed_smoke"
        / "episodes.csv"
    )
    rows = _read_rows(episodes_path)
    rows[0]["trace_sha256"] = "0" * 64
    _write_rows(episodes_path, rows)
    with pytest.raises(ValueError, match="common training traces"):
        validate_training(trace_root, smoke=True)


def test_checkpoint_top_level_identity_is_revalidated(
    valid_smoke_root: Path, tmp_path: Path
) -> None:
    root = _copy_smoke(valid_smoke_root, tmp_path)
    run_dir = (
        root
        / "smoke"
        / "training"
        / LEARNED_POLICIES[0]
        / "seed_smoke"
    )
    checkpoint_path = run_dir / "final.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["seed"] = SMOKE_MODEL_SEED + 1
    torch.save(payload, checkpoint_path)
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_sha256"] = sha256_file(checkpoint_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="top-level seed"):
        validate_training(root, smoke=True)


class _OneBatchReplay:
    def __init__(self, batch: ReplayBatch) -> None:
        self._batch = batch

    def ready(self) -> bool:
        return True

    def sample(self) -> ReplayBatch:
        return self._batch


def _controlled_update_loss(*, terminal: bool) -> float:
    agent = CollisionAwareDQN(seed=6_711, training=True)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        for parameter in agent.target.parameters():
            parameter.zero_()
        agent.online.net[-1].bias[1] = 100.0
        agent.online.net[-1].bias[2] = 1.0
        agent.target.net[-1].bias[1] = 1_000.0
        agent.target.net[-1].bias[2] = 2.0
    next_masks = torch.zeros((1, N_ACTIONS), dtype=torch.bool)
    next_masks[:, 2] = True
    batch = ReplayBatch(
        states=torch.zeros((1, STATE_DIM)),
        actions=torch.zeros(1, dtype=torch.int64),
        rewards=torch.zeros(1),
        next_states=torch.zeros((1, STATE_DIM)),
        next_masks=next_masks,
        dones=torch.tensor([float(terminal)]),
    )
    loss = agent.update(_OneBatchReplay(batch))  # type: ignore[arg-type]
    assert loss is not None
    return loss


def test_double_dqn_masks_targets_and_terminal_transitions_do_not_bootstrap() -> None:
    assert _controlled_update_loss(terminal=False) == pytest.approx(GAMMA * 2.0 - 0.5)
    assert _controlled_update_loss(terminal=True) == pytest.approx(0.0)


def test_freeze_binds_sources_dependencies_and_runtime(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    payload = initialise_freeze(path)
    assert set(payload["dependency_sha256"]) == set(DEPENDENCY_FILENAMES)
    assert payload["runtime_versions"]["distributions"]["torch"]
    assert verify_freeze(path) == payload

    payload["runtime_versions"]["python"] = "0.0.0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime versions"):
        verify_freeze(path)


def test_seal_contract_lists_every_analysis_and_figure_artifact(tmp_path: Path) -> None:
    required = {
        path.relative_to(tmp_path).as_posix()
        for path in provenance._required_result_paths(tmp_path)
    }
    expected = {
        "analysis/summary.json",
        "analysis/comparisons.csv",
        "analysis/cell_effects.csv",
        "analysis/checkpoint_effects.csv",
        "analysis/factorial_means.csv",
        "analysis/mechanism_certificates.csv",
        "analysis/figure_collision_aware_data.csv",
        "analysis/fig_collision_aware_results.pdf",
        "analysis/FIGURE_MANIFEST.json",
    }
    assert expected <= required


def test_seal_invokes_validation_before_rejecting_incomplete_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, bool]] = []

    def fake_validate(root: Path, *, smoke: bool) -> dict[str, str]:
        calls.append((root, smoke))
        return {"status": "PASS"}

    monkeypatch.setattr(
        provenance,
        "verify_freeze",
        lambda: {"aggregate_frozen_inputs_sha256": "0" * 64},
    )
    monkeypatch.setattr(validation_module, "validate", fake_validate)
    with pytest.raises(FileNotFoundError, match="incomplete results"):
        provenance.seal_results(tmp_path)
    assert calls == [(tmp_path, False)]
    assert not (tmp_path / "RESULTS_SEAL.json").exists()


def test_atomic_manifest_creation_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    provenance._atomic_json_new(path, {"value": 1})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        provenance._atomic_json_new(path, {"value": 1})


def test_figure_manifest_rejects_unregistered_artifact_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = tmp_path / "evaluation" / "summary.csv"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text("header\n", encoding="utf-8")
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    expected_analysis = {"summary": "analysis/summary.json"}
    monkeypatch.setattr(
        provenance, "_EXPECTED_ANALYSIS_ARTIFACTS", expected_analysis
    )
    summary = analysis / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "study_id": provenance.STUDY_ID,
                "protocol_version": provenance.PROTOCOL_VERSION,
                "analysis_status": "COMPLETE",
                "artifacts": expected_analysis,
            }
        ),
        encoding="utf-8",
    )
    manifest = analysis / "FIGURE_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "study_id": provenance.STUDY_ID,
                "protocol_version": provenance.PROTOCOL_VERSION,
                "panels": 4,
                "source_summary": "evaluation/summary.csv",
                "source_summary_sha256": sha256_file(evaluation),
                "analysis_summary": "analysis/summary.json",
                "analysis_summary_sha256": sha256_file(summary),
                "source_sha256": {
                    "analysis/summary.json": sha256_file(summary),
                    "evaluation/summary.csv": sha256_file(evaluation),
                },
                "figure_data": "../outside.csv",
                "figure_data_sha256": "0" * 64,
                "figure": "analysis/fig_collision_aware_results.pdf",
                "figure_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        provenance,
        "_required_result_paths",
        lambda _root: (evaluation, summary, manifest),
    )
    with pytest.raises(ValueError, match="unexpected figure_data"):
        provenance._validate_required_results(tmp_path)


def test_figure_manifest_cannot_seal_stale_analysis_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = tmp_path / "evaluation" / "summary.csv"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text("evaluation\n", encoding="utf-8")
    validation_report = tmp_path / "VALIDATION_REPORT.json"
    validation_report.write_text("{}\n", encoding="utf-8")
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    expected_analysis = {
        "summary": "analysis/summary.json",
        "comparisons": "analysis/comparisons.csv",
    }
    summary = analysis / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "study_id": provenance.STUDY_ID,
                "protocol_version": provenance.PROTOCOL_VERSION,
                "analysis_status": "COMPLETE",
                "artifacts": expected_analysis,
            }
        ),
        encoding="utf-8",
    )
    comparisons = analysis / "comparisons.csv"
    comparisons.write_text("effect\n1\n", encoding="utf-8")
    figure_data = analysis / "figure_collision_aware_data.csv"
    figure_data.write_text(
        "panel,record_type\n"
        "a,cell_effect\n"
        "b,cell_effect\n"
        "c,factorial_cell_mean\n"
        "d,checkpoint_mechanism\n",
        encoding="utf-8",
    )
    figure = analysis / "fig_collision_aware_results.pdf"
    figure.write_bytes(b"%PDF-1.4\nsynthetic\n")
    manifest = analysis / "FIGURE_MANIFEST.json"
    frozen_inputs = "f" * 64
    sources = {
        "analysis/comparisons.csv": sha256_file(comparisons),
        "analysis/summary.json": sha256_file(summary),
        "evaluation/summary.csv": sha256_file(evaluation),
    }
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "study_id": provenance.STUDY_ID,
                "protocol_version": provenance.PROTOCOL_VERSION,
                "frozen_inputs_sha256": frozen_inputs,
                "panels": 4,
                "source_summary": "evaluation/summary.csv",
                "source_summary_sha256": sha256_file(evaluation),
                "analysis_summary": "analysis/summary.json",
                "analysis_summary_sha256": sha256_file(summary),
                "source_sha256": sources,
                "figure_data": "analysis/figure_collision_aware_data.csv",
                "figure_data_sha256": sha256_file(figure_data),
                "figure": "analysis/fig_collision_aware_results.pdf",
                "figure_sha256": sha256_file(figure),
            }
        ),
        encoding="utf-8",
    )
    required = (
        evaluation,
        validation_report,
        summary,
        comparisons,
        figure_data,
        figure,
        manifest,
    )
    monkeypatch.setattr(provenance, "_required_result_paths", lambda _root: required)
    monkeypatch.setattr(
        provenance, "_EXPECTED_ANALYSIS_ARTIFACTS", expected_analysis
    )
    monkeypatch.setattr(
        provenance,
        "_EXPECTED_FIGURE_RECORD_COUNTS",
        {
            ("a", "cell_effect"): 1,
            ("b", "cell_effect"): 1,
            ("c", "factorial_cell_mean"): 1,
            ("d", "checkpoint_mechanism"): 1,
        },
    )
    provenance._validate_required_results(
        tmp_path, frozen_inputs_sha256=frozen_inputs
    )

    comparisons.write_text("effect\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source inventory"):
        provenance._validate_required_results(
            tmp_path, frozen_inputs_sha256=frozen_inputs
        )
