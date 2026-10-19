"""Invariant tests for the isolated prospective collision experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from main.revision.crn_ca_pr1_20261017.config import (
    ABSTAIN_ACTION,
    EVALUATION_SEEDS,
    FACTORS,
    LEARNED_POLICIES,
    N_ACTIONS,
    N_CHANNELS,
    N_NODES,
    STATE_DIM,
    label_permutation,
    protocol_payload,
    protocol_sha256,
)
from main.revision.crn_ca_pr1_20261017.environment import CollisionEnvironment
from main.revision.crn_ca_pr1_20261017.evaluate import build_tasks, evaluate_task
from main.revision.crn_ca_pr1_20261017.model import (
    ArrayReplay,
    CollisionAwareDQN,
    observation_for_policy,
)
from main.revision.crn_ca_pr1_20261017.provenance import initialise_freeze, verify_freeze
from main.revision.crn_ca_pr1_20261017.train import train_one
from main.revision.crn_ca_pr1_20261017.validation import validate


def _environment(seed: int = 123) -> CollisionEnvironment:
    return CollisionEnvironment(
        factor=FACTORS[0],
        seed=seed,
        logical_to_physical=tuple(range(N_CHANNELS)),
    )


def _force_all_idle_reliable(env: CollisionEnvironment) -> tuple[np.ndarray, np.ndarray]:
    env._occupied[:] = False  # noqa: SLF001
    env._current_quality[:] = 1.0  # noqa: SLF001
    env._current_link_uniform[:] = 0.0  # noqa: SLF001
    env.queues[:] = 1
    return env._build_states_and_masks()  # noqa: SLF001


def test_protocol_digest_is_canonical() -> None:
    assert protocol_sha256() == protocol_sha256()
    assert len(protocol_payload()["factors"]) == 8


def test_evaluation_label_permutations_are_exactly_balanced() -> None:
    permutations = [label_permutation(index) for index in range(len(EVALUATION_SEEDS))]
    assert len(set(permutations)) == N_CHANNELS
    for permutation in set(permutations):
        assert permutations.count(permutation) == 4
        assert sorted(permutation) == list(range(N_CHANNELS))


def test_source_freeze_round_trip_to_temporary_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "freeze.json"
    created = initialise_freeze(manifest)
    verified = verify_freeze(manifest)
    assert created == verified
    assert verified["protocol_sha256"] == protocol_sha256()


def test_destructive_collision_and_packet_conservation() -> None:
    env = _environment()
    env.begin_slot()
    _, masks = _force_all_idle_reliable(env)
    actions = np.full(N_NODES, ABSTAIN_ACTION, dtype=np.int64)
    actions[0] = 0
    actions[1] = 0
    before = int(np.sum(env.queues))
    result = env.resolve(actions)
    assert result.collisions[:2].tolist() == [True, True]
    assert not np.any(result.delivered[:2])
    assert int(np.sum(result.queue_after)) == before
    assert masks[0, 0]
    assert masks[1, 0]


def test_singleton_delivery_depends_on_selected_action() -> None:
    env = _environment(seed=124)
    env.begin_slot()
    _force_all_idle_reliable(env)
    actions = np.full(N_NODES, ABSTAIN_ACTION, dtype=np.int64)
    actions[0] = 3
    before = int(np.sum(env.queues))
    result = env.resolve(actions)
    assert result.delivered[0]
    assert result.physical_actions[0] == 3
    assert int(np.sum(result.queue_after)) == before - 1


def test_trace_is_policy_independent() -> None:
    left = _environment(seed=125)
    right = _environment(seed=125)
    for _ in range(20):
        _, left_masks = left.begin_slot()
        _, right_masks = right.begin_slot()
        left_actions = np.full(N_NODES, ABSTAIN_ACTION, dtype=np.int64)
        right_actions = np.full(N_NODES, ABSTAIN_ACTION, dtype=np.int64)
        for node_id in range(N_NODES):
            legal = np.flatnonzero(right_masks[node_id, :N_CHANNELS])
            if legal.size:
                right_actions[node_id] = legal[node_id % legal.size]
        left.resolve(left_actions)
        right.resolve(right_actions)
        assert left.trace_sha256 == right.trace_sha256
        assert left_masks.shape == right_masks.shape


def test_replay_inserts_all_nodes_equally() -> None:
    replay = ArrayReplay(capacity=2 * N_NODES, seed=4)
    states = np.zeros((N_NODES, STATE_DIM), dtype=np.float32)
    actions = np.full(N_NODES, ABSTAIN_ACTION, dtype=np.int64)
    rewards = np.zeros(N_NODES, dtype=np.float32)
    masks = np.zeros((N_NODES, N_ACTIONS), dtype=bool)
    masks[:, ABSTAIN_ACTION] = True
    replay.add_all_nodes(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=states,
        next_masks=masks,
        done=False,
    )
    assert replay.total_inserted == N_NODES
    np.testing.assert_array_equal(replay.node_insertions, np.ones(N_NODES))


def test_dqn_never_emits_masked_action() -> None:
    agent = CollisionAwareDQN(seed=5, training=False)
    states = np.zeros((N_NODES, STATE_DIM), dtype=np.float32)
    masks = np.zeros((N_NODES, N_ACTIONS), dtype=bool)
    masks[:, ABSTAIN_ACTION] = True
    masks[0, 2] = True
    actions = agent.choose(states, masks, epsilon=0.0)
    assert np.all(masks[np.arange(N_NODES), actions])


def test_observation_interventions_zero_only_preregistered_blocks() -> None:
    states = np.ones((N_NODES, STATE_DIM), dtype=np.float32)
    full = observation_for_policy(states, LEARNED_POLICIES[0])
    history_free = observation_for_policy(states, LEARNED_POLICIES[1])
    coordination_blind = observation_for_policy(states, LEARNED_POLICIES[2])

    np.testing.assert_array_equal(full, states)
    assert np.all(history_free[:, 2 * N_CHANNELS : 4 * N_CHANNELS] == 0.0)
    outcome_start = 4 * N_CHANNELS + N_ACTIONS
    assert np.all(history_free[:, outcome_start : outcome_start + 5] == 0.0)
    assert np.all(history_free[:, : 2 * N_CHANNELS] == 1.0)
    assert np.all(coordination_blind[:, -2 * N_NODES :] == 0.0)
    assert np.all(coordination_blind[:, : -2 * N_NODES] == 1.0)


def test_end_to_end_smoke(tmp_path: Path) -> None:
    for policy_variant in LEARNED_POLICIES:
        metadata = train_one(
            seed_index=0,
            output_root=tmp_path,
            policy_variant=policy_variant,
            smoke=True,
        )
        assert metadata["selection_rule"] == "final_iterate_only"
    tasks = build_tasks(output_root=tmp_path, smoke=True)
    rows = [evaluate_task(task) for task in tasks]
    destination = tmp_path / "smoke" / "evaluation" / "summary.csv"
    destination.parent.mkdir(parents=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = validate(tmp_path, smoke=True)
    assert report["status"] == "PASS"
    saved = json.loads((tmp_path / "smoke" / "VALIDATION_REPORT.json").read_text())
    assert saved == report


def test_four_panel_figure_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from main.revision.crn_ca_pr1_20261017 import plot
    from main.revision.crn_ca_pr1_20261017.config import (
        ALL_POLICIES,
        HISTORY_FREE_POLICY_NAME,
        LEARNED_POLICY_NAME,
        PRIMARY_BASELINE_NAME,
        PROTOCOL_VERSION,
        STUDY_ID,
        TRAINING_SEEDS,
    )

    monkeypatch.setattr(
        plot,
        "verify_freeze",
        lambda: {"aggregate_frozen_inputs_sha256": "0" * 64},
    )
    monkeypatch.setattr(plot, "validate", lambda _root, smoke: {"status": "PASS"})
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    (evaluation_dir / "summary.csv").write_text("fixture\n", encoding="utf-8")
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "summary.json").write_text(
        json.dumps(
            {
                "study_id": STUDY_ID,
                "protocol_version": PROTOCOL_VERSION,
                "analysis_status": "COMPLETE",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def write_rows(name: str, rows: list[dict[str, object]]) -> None:
        path = analysis_dir / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    comparisons = [
        {
            "treatment": LEARNED_POLICY_NAME,
            "comparator": comparator,
            "metric": "network_goodput",
            "difference": difference,
            "one_sided_95_lcb": difference - 0.01,
            "one_sided_95_ucb": difference + 0.01,
            "exact_sign_flip_p_raw": 0.01,
        }
        for comparator, difference in (
            (PRIMARY_BASELINE_NAME, 0.08),
            (HISTORY_FREE_POLICY_NAME, 0.03),
        )
    ]
    write_rows("comparisons.csv", comparisons)

    cell_effects: list[dict[str, object]] = []
    for comparator, base_difference in (
        (PRIMARY_BASELINE_NAME, 0.08),
        (HISTORY_FREE_POLICY_NAME, 0.03),
    ):
        for factor_index, factor in enumerate(FACTORS):
            difference = base_difference + 0.001 * factor_index
            cell_effects.append(
                {
                    "treatment": LEARNED_POLICY_NAME,
                    "comparator": comparator,
                    "metric": "network_goodput",
                    "factor_id": factor.factor_id,
                    "difference": difference,
                    "one_sided_95_lcb": difference - 0.01,
                    "one_sided_95_ucb": difference + 0.01,
                }
            )
    write_rows("cell_effects.csv", cell_effects)

    checkpoint_effects = [
        {
            "metric": "network_goodput",
            "comparator": comparator,
            "role": "fixture",
            "training_seed": seed,
            "paired_macro_difference": base_difference + 0.0001 * seed_index,
        }
        for comparator, base_difference in (
            (PRIMARY_BASELINE_NAME, 0.08),
            (HISTORY_FREE_POLICY_NAME, 0.03),
        )
        for seed_index, seed in enumerate(TRAINING_SEEDS)
    ]
    write_rows("checkpoint_effects.csv", checkpoint_effects)

    factorial_means: list[dict[str, object]] = []
    for factor_index, factor in enumerate(FACTORS):
        for policy_index, policy in enumerate(ALL_POLICIES):
            factorial_means.append(
                {
                    "factor_id": factor.factor_id,
                    "policy": policy,
                    "metric": "network_goodput",
                    "mean": 0.4 + 0.03 * policy_index + 0.005 * factor_index,
                }
            )
    write_rows("factorial_means.csv", factorial_means)

    mechanisms = [
        {
            "training_seed": seed,
            "validation_best_equal_cell_agreement": 0.70 + 0.005 * seed_index,
            "exact_priority_representable": False,
        }
        for seed_index, seed in enumerate(TRAINING_SEEDS)
    ]
    write_rows("mechanism_certificates.csv", mechanisms)

    manifest = plot.generate(tmp_path)
    assert manifest["panels"] == 4
    assert (tmp_path / manifest["figure"]).is_file()
    assert len(manifest["source_sha256"]) == 7
    with pytest.raises(FileExistsError):
        plot.generate(tmp_path)
