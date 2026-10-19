"""Focused invariants for the v2 policy and evaluation design."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.revision.crn_ca_pr1_20261017 import evaluate as evaluate_module
from main.revision.crn_ca_pr1_20261017.config import (
    ABSTAIN_ACTION,
    ALL_POLICIES,
    BASELINES,
    EVALUATION_SEEDS,
    FACTORS,
    LEARNED_POLICIES,
    LEARNED_POLICY_NAME,
    N_ACTIONS,
    N_CHANNELS,
    N_NODES,
    PROTOCOL_VERSION,
    SMOKE_EPISODES,
    SMOKE_MODEL_SEED,
    SMOKE_STEPS_PER_EPISODE,
    STATE_DIM,
    STUDY_ID,
    TRAINING_SEEDS,
    protocol_sha256,
)
from main.revision.crn_ca_pr1_20261017.evaluate import (
    EvaluationTask,
    build_tasks,
    evaluate_task,
)
from main.revision.crn_ca_pr1_20261017.model import CollisionAwareDQN
from main.revision.crn_ca_pr1_20261017.policies import (
    CentralizedQualityMatchingPolicy,
    PhaseOrthogonalPolicy,
    QualityGreedyPersistentPolicy,
)
from main.revision.crn_ca_pr1_20261017.provenance import sha256_file


def _states_and_masks(
    *, idle_channels: tuple[int, ...], phase: int, queued_nodes: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    states = np.zeros((N_NODES, STATE_DIM), dtype=np.float32)
    states[:, idle_channels] = 1.0
    states[:, STATE_DIM - N_NODES + phase] = 1.0
    masks = np.zeros((N_NODES, N_ACTIONS), dtype=bool)
    masks[:, ABSTAIN_ACTION] = True
    masks[np.ix_(queued_nodes, idle_channels)] = True
    return states, masks


def test_phase_orthogonal_mapping_is_exact_and_does_not_backfill() -> None:
    states, masks = _states_and_masks(
        idle_channels=(0, 2, 4), phase=4, queued_nodes=(4, 6)
    )
    actions = PhaseOrthogonalPolicy().choose(states, masks)
    assert actions[4] == 0
    assert actions[5] == ABSTAIN_ACTION
    assert actions[6] == 4
    assert np.sum(actions < N_CHANNELS) == 2
    assert len(set(actions[actions < N_CHANNELS])) == 2


def test_centralized_oracle_finds_maximum_quality_one_to_one_matching() -> None:
    states, masks = _states_and_masks(
        idle_channels=(0, 1), phase=0, queued_nodes=(0, 1, 2)
    )
    quality_start = N_CHANNELS
    states[0, quality_start : quality_start + 2] = (0.90, 0.10)
    states[1, quality_start : quality_start + 2] = (0.80, 0.70)
    states[2, quality_start : quality_start + 2] = (0.10, 0.95)
    actions = CentralizedQualityMatchingPolicy().choose(states, masks)
    assert actions[0] == 0
    assert actions[2] == 1
    assert actions[1] == ABSTAIN_ACTION


def test_quality_persistent_access_uses_idle_channels_over_population() -> None:
    states, masks = _states_and_masks(
        idle_channels=tuple(range(N_CHANNELS)), phase=0, queued_nodes=(0, 1)
    )
    quality_start = N_CHANNELS
    states[0, quality_start + 4] = 0.9
    states[1, quality_start + 2] = 0.8

    class StubRng:
        def __init__(self) -> None:
            self.draws = iter((0.4, np.nextafter(0.4, 1.0)))

        def random(self) -> float:
            return float(next(self.draws))

        def choice(self, values: np.ndarray) -> np.integer:
            return values[0]

    policy = QualityGreedyPersistentPolicy(seed=1)
    policy._rng = StubRng()  # type: ignore[assignment]  # noqa: SLF001
    actions = policy.choose(states, masks)
    assert actions[0] == 4
    assert actions[1] == ABSTAIN_ACTION


def test_v2_task_grid_is_complete_unique_and_uses_variant_paths() -> None:
    root = Path("/").joinpath("tmp", "crn_ca_v2_plan_only")
    tasks = build_tasks(output_root=root, smoke=False)
    expected = len(FACTORS) * len(EVALUATION_SEEDS) * (
        len(LEARNED_POLICIES) * len(TRAINING_SEEDS) + len(BASELINES)
    )
    assert len(tasks) == expected == 12_672
    assert len({task.run_id for task in tasks}) == expected
    assert sum(task.policy in LEARNED_POLICIES for task in tasks) == 11_520
    first = next(task for task in tasks if task.policy == LEARNED_POLICY_NAME)
    assert first.checkpoint_path == (
        root
        / "training"
        / LEARNED_POLICY_NAME
        / f"seed_{TRAINING_SEEDS[0]}"
        / "final.pt"
    )


def test_aggregate_mask_action_table_and_metric_identities() -> None:
    task = EvaluationTask(
        policy="phase_orthogonal",
        factor=FACTORS[0],
        replication_index=0,
        environment_seed=98_765_431,
        training_seed=None,
        checkpoint_path=None,
        burn_in_steps=2,
        measurement_steps=20,
        smoke=True,
    )
    row = evaluate_task(task)
    table = np.asarray(
        [int(value) for value in row["mask_action_counts"].split("-")],
        dtype=np.int64,
    ).reshape(64, 7)
    assert row["mask_action_counts_shape"] == "64x7"
    assert table.shape == (64, 7)
    assert int(np.sum(table)) == row["priority_eligible_decisions"]
    assert row["singleton_attempts"] == row["deliveries"] + row["link_failures"]
    assert row["network_goodput"] == row["deliveries"] / task.measurement_steps
    if row["zero_delivery_run"]:
        assert row["delivery_jain_fairness"] == ""


def test_checkpoint_loader_enforces_embedded_and_sidecar_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final.pt"
    total_steps = SMOKE_EPISODES * SMOKE_STEPS_PER_EPISODE
    metadata: dict[str, object] = {
        "study_id": STUDY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(),
        "seed_index": 0,
        "model_seed": SMOKE_MODEL_SEED,
        "episodes": SMOKE_EPISODES,
        "steps_per_episode": SMOKE_STEPS_PER_EPISODE,
        "global_environment_steps": total_steps,
        "gradient_steps": 72,
        "replay_total_insertions": total_steps * N_NODES,
        "replay_node_insertions": [total_steps] * N_NODES,
        "smoke": True,
        "policy_variant": LEARNED_POLICY_NAME,
        "selection_rule": "final_iterate_only",
    }
    agent = CollisionAwareDQN(seed=SMOKE_MODEL_SEED, training=False)
    agent.gradient_steps = 72
    agent.save(checkpoint, metadata=metadata)
    sidecar = {
        **metadata,
        "checkpoint_sha256": sha256_file(checkpoint),
        "wall_seconds": 0.0,
    }
    sidecar_path = checkpoint.with_name("metadata.json")
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True) + "\n", encoding="utf-8"
    )
    task = EvaluationTask(
        policy=LEARNED_POLICY_NAME,
        factor=FACTORS[0],
        replication_index=0,
        environment_seed=123,
        training_seed=SMOKE_MODEL_SEED,
        checkpoint_path=checkpoint,
        burn_in_steps=1,
        measurement_steps=1,
        smoke=True,
    )
    policy, digest = evaluate_module._load_policy(task)  # noqa: SLF001
    assert policy is not None
    assert digest == sidecar["checkpoint_sha256"]

    sidecar["policy_variant"] = LEARNED_POLICIES[1]
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="embedded metadata"):
        evaluate_module._load_policy(task)  # noqa: SLF001


def test_smoke_plan_contains_every_registered_policy_once() -> None:
    tasks = build_tasks(output_root=Path("/").joinpath("tmp", "crn_ca_v2_smoke_plan"), smoke=True)
    assert [task.policy for task in tasks] == list(ALL_POLICIES)
    assert len({task.run_id for task in tasks}) == len(ALL_POLICIES)
