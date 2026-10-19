"""Paired common-trace evaluation for the prospective collision study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from main.revision.crn_ca_pr1_20261017.config import (
    ALL_POLICIES,
    BASELINES,
    BATCH_SIZE,
    DEFAULT_RESULTS_ROOT,
    EVAL_BURN_IN_STEPS,
    EVAL_MEASUREMENT_STEPS,
    EVALUATION_SEEDS,
    FACTORS,
    LEARNED_POLICIES,
    N_CHANNELS,
    N_NODES,
    PROTOCOL_VERSION,
    SMOKE_ENVIRONMENT_SEED,
    SMOKE_EPISODES,
    SMOKE_EVAL_BURN_IN_STEPS,
    SMOKE_EVAL_MEASUREMENT_STEPS,
    SMOKE_MODEL_SEED,
    SMOKE_STEPS_PER_EPISODE,
    STUDY_ID,
    TRAIN_EPISODES,
    TRAIN_STEPS_PER_EPISODE,
    TRAINING_SEEDS,
    Factor,
    label_permutation,
    protocol_sha256,
)
from main.revision.crn_ca_pr1_20261017.environment import CollisionEnvironment
from main.revision.crn_ca_pr1_20261017.model import (
    CollisionAwareDQN,
    observation_for_policy,
)
from main.revision.crn_ca_pr1_20261017.policies import Policy, build_baseline
from main.revision.crn_ca_pr1_20261017.provenance import sha256_file, verify_freeze


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    policy: str
    factor: Factor
    replication_index: int
    environment_seed: int
    training_seed: int | None
    checkpoint_path: Path | None
    burn_in_steps: int
    measurement_steps: int
    smoke: bool

    @property
    def run_id(self) -> str:
        training = "baseline" if self.training_seed is None else str(self.training_seed)
        return (
            f"{self.policy}__train-{training}__{self.factor.factor_id}"
            f"__rep-{self.replication_index:02d}__env-{self.environment_seed}"
        )


class _LearnedPolicyAdapter:
    def __init__(self, agent: CollisionAwareDQN, policy_name: str) -> None:
        self._agent = agent
        self._policy_name = policy_name

    def choose(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        observations = observation_for_policy(states, self._policy_name)
        return self._agent.choose(observations, masks, epsilon=0.0)


def _jain(counts: np.ndarray) -> float | None:
    values = counts.astype(np.float64)
    denominator = len(values) * float(np.sum(values**2))
    if denominator == 0.0:
        return None
    return float(np.sum(values) ** 2 / denominator)


def _policy_seed(task: EvaluationTask) -> int:
    encoded = f"{STUDY_ID}|{task.policy}|{task.factor.factor_id}|{task.environment_seed}"
    return int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:8], "big") % (2**32)


def _load_policy(task: EvaluationTask) -> tuple[Policy, str]:
    if task.policy in LEARNED_POLICIES:
        if task.checkpoint_path is None or task.training_seed is None:
            raise ValueError("learned-policy task requires checkpoint and training seed")
        agent, metadata = CollisionAwareDQN.load(task.checkpoint_path)
        if task.smoke:
            seed_index = 0
            episodes = SMOKE_EPISODES
            steps_per_episode = SMOKE_STEPS_PER_EPISODE
        else:
            try:
                seed_index = TRAINING_SEEDS.index(task.training_seed)
            except ValueError as error:
                raise ValueError("unregistered checkpoint training seed") from error
            episodes = TRAIN_EPISODES
            steps_per_episode = TRAIN_STEPS_PER_EPISODE
        total_steps = episodes * steps_per_episode
        first_update_slot = (BATCH_SIZE + N_NODES - 1) // N_NODES
        expected_gradient_steps = max(0, total_steps - first_update_slot + 1)
        expected_metadata: dict[str, object] = {
            "study_id": STUDY_ID,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": protocol_sha256(),
            "seed_index": seed_index,
            "model_seed": task.training_seed,
            "smoke": task.smoke,
            "policy_variant": task.policy,
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "global_environment_steps": total_steps,
            "gradient_steps": expected_gradient_steps,
            "replay_total_insertions": total_steps * N_NODES,
            "replay_node_insertions": [total_steps] * N_NODES,
            "selection_rule": "final_iterate_only",
        }
        mismatches = {
            key: {"expected": expected, "observed": metadata.get(key)}
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"checkpoint metadata mismatch: {mismatches}")
        if agent.seed != task.training_seed:
            raise ValueError("checkpoint payload seed mismatch")

        checkpoint_sha256 = sha256_file(task.checkpoint_path)
        sidecar_path = task.checkpoint_path.with_name("metadata.json")
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"missing checkpoint metadata sidecar: {sidecar_path}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("checkpoint digest differs from training sidecar")
        embedded_sidecar = {
            key: value
            for key, value in sidecar.items()
            if key not in {"checkpoint_sha256", "wall_seconds"}
        }
        if embedded_sidecar != metadata:
            raise ValueError("checkpoint embedded metadata differs from training sidecar")
        return _LearnedPolicyAdapter(agent, task.policy), checkpoint_sha256
    return (
        build_baseline(task.policy, seed=_policy_seed(task), factor=task.factor),
        "",
    )


def evaluate_task(task: EvaluationTask) -> dict[str, Any]:
    """Evaluate one policy on one factor/trace without writing row-level data."""
    if not task.smoke:
        verify_freeze()
    torch.set_num_threads(1)
    policy, checkpoint_sha256 = _load_policy(task)
    if task.smoke:
        permutation = tuple(range(N_CHANNELS))
    else:
        permutation = label_permutation(task.replication_index)
    env = CollisionEnvironment(
        factor=task.factor,
        seed=task.environment_seed,
        logical_to_physical=permutation,
    )

    delivered_by_node = np.zeros(N_NODES, dtype=np.int64)
    logical_action_counts = np.zeros(N_CHANNELS + 1, dtype=np.int64)
    physical_action_counts = np.zeros(N_CHANNELS, dtype=np.int64)
    mask_action_counts = np.zeros((1 << N_CHANNELS, N_CHANNELS + 1), dtype=np.int64)
    starting_backlog = 0
    arrivals = 0
    deliveries = 0
    collision_attempts = 0
    link_failures = 0
    attempted = 0
    voluntary_deferrals = 0
    forced_deferrals = 0
    priority_eligible = 0
    priority_matches = 0
    backlog_slot_sum = 0
    capacity_ceiling_sum = 0
    total_steps = task.burn_in_steps + task.measurement_steps

    for step in range(total_steps):
        states, masks = env.begin_slot()
        actions = policy.choose(states, masks)
        if actions.shape != (N_NODES,) or np.any(~masks[np.arange(N_NODES), actions]):
            raise ValueError(f"{task.policy} emitted an illegal action")
        result = env.resolve(actions)
        if step == task.burn_in_steps - 1:
            starting_backlog = int(np.sum(result.queue_after))
        if step < task.burn_in_steps:
            continue

        arrivals += int(np.sum(result.arrivals))
        deliveries += int(np.sum(result.delivered))
        collision_attempts += int(np.sum(result.collisions))
        link_failures += int(np.sum(result.link_failures))
        attempted += int(np.sum(result.attempted))
        voluntary_deferrals += int(np.sum(result.voluntary_deferrals))
        forced_deferrals += int(np.sum(result.forced_deferrals))
        delivered_by_node += result.delivered.astype(np.int64)
        backlog_slot_sum += int(np.sum(result.queue_after))
        capacity_ceiling_sum += min(
            int(np.sum(result.queue_before > 0)), result.idle_channels
        )

        logical_action_counts += np.bincount(actions, minlength=N_CHANNELS + 1)
        chosen_physical = result.physical_actions[result.attempted]
        if chosen_physical.size:
            physical_action_counts += np.bincount(
                chosen_physical, minlength=N_CHANNELS
            )
        eligible_nodes = (result.queue_before > 0) & np.any(
            masks[:, :N_CHANNELS], axis=1
        )
        if np.any(eligible_nodes):
            eligible_indices = np.flatnonzero(eligible_nodes)
            bit_weights = 1 << np.arange(N_CHANNELS, dtype=np.int64)
            mask_ids = (
                masks[eligible_indices, :N_CHANNELS].astype(np.int64) @ bit_weights
            )
            np.add.at(
                mask_action_counts,
                (mask_ids, actions[eligible_indices]),
                1,
            )
            static_actions = np.argmax(masks[:, :N_CHANNELS], axis=1)
            priority_eligible += int(np.sum(eligible_nodes))
            priority_matches += int(
                np.sum(actions[eligible_nodes] == static_actions[eligible_nodes])
            )

    ending_backlog = int(np.sum(env.queues))
    packet_denominator = starting_backlog + arrivals
    if packet_denominator != deliveries + ending_backlog:
        raise AssertionError(
            "packet conservation failure: starting backlog + arrivals "
            "!= deliveries + ending backlog"
        )
    if task.measurement_steps <= 0:
        raise ValueError("measurement_steps must be positive")
    singleton_attempts = attempted - collision_attempts
    if singleton_attempts != deliveries + link_failures:
        raise AssertionError("singleton-attempt accounting failure")
    fairness = _jain(delivered_by_node)
    encoded_mask_actions = "-".join(
        str(int(value)) for value in mask_action_counts.ravel()
    )
    return {
        "study_id": STUDY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(),
        "run_id": task.run_id,
        "smoke": task.smoke,
        "policy": task.policy,
        "training_seed": "" if task.training_seed is None else task.training_seed,
        "checkpoint_sha256": checkpoint_sha256,
        "factor_id": task.factor.factor_id,
        "occupancy_process": task.factor.occupancy_process,
        "occupancy_probability": task.factor.occupancy_probability,
        "arrival_probability": task.factor.arrival_probability,
        "replication_index": task.replication_index,
        "environment_seed": task.environment_seed,
        "logical_to_physical": "-".join(str(value) for value in permutation),
        "burn_in_steps": task.burn_in_steps,
        "measurement_steps": task.measurement_steps,
        "trace_sha256": env.trace_sha256,
        "starting_backlog": starting_backlog,
        "arrivals": arrivals,
        "deliveries": deliveries,
        "ending_backlog": ending_backlog,
        "network_goodput": deliveries / task.measurement_steps,
        "finite_window_service_fraction": (
            deliveries / packet_denominator if packet_denominator else ""
        ),
        "capacity_ceiling_per_slot": capacity_ceiling_sum / task.measurement_steps,
        "opportunity_utilization": (
            deliveries / capacity_ceiling_sum if capacity_ceiling_sum else ""
        ),
        "attempted": attempted,
        "singleton_attempts": singleton_attempts,
        "collision_attempts": collision_attempts,
        "collision_attempt_rate": collision_attempts / attempted if attempted else 0.0,
        "link_failures": link_failures,
        "conditional_link_failure_rate": (
            link_failures / singleton_attempts if singleton_attempts else ""
        ),
        "voluntary_deferrals": voluntary_deferrals,
        "forced_deferrals": forced_deferrals,
        "mean_network_backlog": backlog_slot_sum / task.measurement_steps,
        "delivery_jain_fairness": "" if fairness is None else fairness,
        "zero_delivery_run": int(deliveries == 0),
        "priority_eligible_decisions": priority_eligible,
        "priority_matches": priority_matches,
        "static_priority_agreement": (
            priority_matches / priority_eligible if priority_eligible else ""
        ),
        "logical_action_counts": "-".join(str(value) for value in logical_action_counts),
        "physical_action_counts": "-".join(str(value) for value in physical_action_counts),
        "mask_action_counts_shape": f"{1 << N_CHANNELS}x{N_CHANNELS + 1}",
        "mask_action_counts": encoded_mask_actions,
    }


def build_tasks(*, output_root: Path, smoke: bool) -> list[EvaluationTask]:
    """Construct the complete preregistered Cartesian evaluation plan."""
    if smoke:
        factor = FACTORS[0]
        tasks = []
        for policy in ALL_POLICIES:
            learned = policy in LEARNED_POLICIES
            checkpoint = (
                output_root
                / "smoke"
                / "training"
                / policy
                / "seed_smoke"
                / "final.pt"
                if learned
                else None
            )
            tasks.append(
                EvaluationTask(
                    policy=policy,
                    factor=factor,
                    replication_index=0,
                    environment_seed=SMOKE_ENVIRONMENT_SEED + 10_000,
                    training_seed=SMOKE_MODEL_SEED if learned else None,
                    checkpoint_path=checkpoint,
                    burn_in_steps=SMOKE_EVAL_BURN_IN_STEPS,
                    measurement_steps=SMOKE_EVAL_MEASUREMENT_STEPS,
                    smoke=True,
                )
            )
        return tasks

    tasks: list[EvaluationTask] = []
    for factor in FACTORS:
        for replication_index, environment_seed in enumerate(EVALUATION_SEEDS):
            for policy in LEARNED_POLICIES:
                for training_seed in TRAINING_SEEDS:
                    checkpoint = (
                        output_root
                        / "training"
                        / policy
                        / f"seed_{training_seed}"
                        / "final.pt"
                    )
                    tasks.append(
                        EvaluationTask(
                            policy=policy,
                            factor=factor,
                            replication_index=replication_index,
                            environment_seed=environment_seed,
                            training_seed=training_seed,
                            checkpoint_path=checkpoint,
                            burn_in_steps=EVAL_BURN_IN_STEPS,
                            measurement_steps=EVAL_MEASUREMENT_STEPS,
                            smoke=False,
                        )
                    )
            for baseline in BASELINES:
                tasks.append(
                    EvaluationTask(
                        policy=baseline,
                        factor=factor,
                        replication_index=replication_index,
                        environment_seed=environment_seed,
                        training_seed=None,
                        checkpoint_path=None,
                        burn_in_steps=EVAL_BURN_IN_STEPS,
                        measurement_steps=EVAL_MEASUREMENT_STEPS,
                        smoke=False,
                    )
                )
    run_ids = [task.run_id for task in tasks]
    if len(run_ids) != len(set(run_ids)):
        raise AssertionError("evaluation plan contains duplicate run IDs")
    return tasks


def _worker(task: EvaluationTask) -> dict[str, Any]:
    return evaluate_task(task)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    tasks = build_tasks(output_root=args.output_root, smoke=args.smoke)
    if args.plan:
        learned = sum(task.policy in LEARNED_POLICIES for task in tasks)
        payload = {
            "study_id": STUDY_ID,
            "total_runs": len(tasks),
            "learned_runs": learned,
            "baseline_runs": len(tasks) - learned,
            "factor_cells": 1 if args.smoke else len(FACTORS),
            "environment_seeds_per_cell": 1 if args.smoke else len(EVALUATION_SEEDS),
            "balanced_label_cycles_per_cell": (
                0 if args.smoke else len(EVALUATION_SEEDS) // N_CHANNELS
            ),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if not args.smoke:
        verify_freeze()
    missing = [
        str(task.checkpoint_path)
        for task in tasks
        if task.checkpoint_path and not task.checkpoint_path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing learned checkpoints: {missing[:3]}")
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        rows = list(executor.map(_worker, tasks, chunksize=1))
    rows.sort(key=lambda row: str(row["run_id"]))
    destination = (
        args.output_root / "smoke" / "evaluation" / "summary.csv"
        if args.smoke
        else args.output_root / "evaluation" / "summary.csv"
    )
    _write_csv(destination, rows)
    print(json.dumps({"rows": len(rows), "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
