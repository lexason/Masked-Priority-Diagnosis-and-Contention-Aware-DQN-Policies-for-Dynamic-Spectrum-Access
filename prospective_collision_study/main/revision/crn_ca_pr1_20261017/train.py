"""Fixed-final-iterate training for the collision-aware shared DQN."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from main.revision.crn_ca_pr1_20261017.config import (
    ABSTAIN_ACTION,
    DEFAULT_RESULTS_ROOT,
    EPSILON_END,
    EPSILON_START,
    FACTORS,
    LEARNED_POLICIES,
    LEARNED_POLICY_NAME,
    N_ACTIONS,
    N_CHANNELS,
    N_NODES,
    PROTOCOL_VERSION,
    SMOKE_ENVIRONMENT_SEED,
    SMOKE_EPISODES,
    SMOKE_MODEL_SEED,
    SMOKE_STEPS_PER_EPISODE,
    STATE_DIM,
    STUDY_ID,
    TRAIN_EPISODES,
    TRAIN_STEPS_PER_EPISODE,
    TRAINING_SEEDS,
    protocol_sha256,
    training_environment_seed,
)
from main.revision.crn_ca_pr1_20261017.environment import CollisionEnvironment
from main.revision.crn_ca_pr1_20261017.model import (
    ArrayReplay,
    CollisionAwareDQN,
    observation_for_policy,
)
from main.revision.crn_ca_pr1_20261017.provenance import sha256_file, verify_freeze


def _epsilon(step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return EPSILON_END
    fraction = step / (total_steps - 1)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def _logical_permutation(shift: int) -> tuple[int, ...]:
    return tuple((logical + shift) % N_CHANNELS for logical in range(N_CHANNELS))


def _file_sha256(path: Path) -> str:
    return sha256_file(path)


def train_one(
    *,
    seed_index: int,
    output_root: Path,
    policy_variant: str = LEARNED_POLICY_NAME,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train one prespecified final iterate and write auditable artifacts."""
    if not smoke:
        verify_freeze()
    if policy_variant not in LEARNED_POLICIES:
        raise ValueError(f"unknown learned policy variant: {policy_variant}")
    if smoke:
        if seed_index != 0:
            raise ValueError("smoke mode has exactly one seed index (0)")
        model_seed = SMOKE_MODEL_SEED
        episodes = SMOKE_EPISODES
        steps_per_episode = SMOKE_STEPS_PER_EPISODE
        run_dir = (
            output_root / "smoke" / "training" / policy_variant / "seed_smoke"
        )
    else:
        if not 0 <= seed_index < len(TRAINING_SEEDS):
            raise ValueError("seed_index outside preregistered training seeds")
        model_seed = TRAINING_SEEDS[seed_index]
        episodes = TRAIN_EPISODES
        steps_per_episode = TRAIN_STEPS_PER_EPISODE
        run_dir = output_root / "training" / policy_variant / f"seed_{model_seed}"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite training run: {run_dir}")
    run_dir.mkdir(parents=True)

    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True)
    agent = CollisionAwareDQN(seed=model_seed, training=True)
    replay = ArrayReplay(seed=model_seed + 2)
    total_steps = episodes * steps_per_episode
    global_step = 0
    episode_rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()

    for episode in range(episodes):
        factor = FACTORS[(episode + seed_index) % len(FACTORS)]
        # Advance the label shift once per complete factorial block. This
        # crosses every factor with every shift instead of parity-confounding
        # an 8-cell factor cycle with a 6-shift label cycle.
        label_shift = (episode // len(FACTORS) + seed_index) % N_CHANNELS
        permutation = _logical_permutation(label_shift)
        environment_seed = (
            SMOKE_ENVIRONMENT_SEED + episode
            if smoke
            else training_environment_seed(seed_index, episode)
        )
        env = CollisionEnvironment(
            factor=factor,
            seed=environment_seed,
            logical_to_physical=permutation,
        )
        raw_states, masks = env.begin_slot()
        states = observation_for_policy(raw_states, policy_variant)
        reward_sum = 0.0
        delivered = 0
        collisions = 0
        losses: list[float] = []
        for step_in_episode in range(steps_per_episode):
            epsilon = _epsilon(global_step, total_steps)
            actions = agent.choose(states, masks, epsilon=epsilon)
            result = env.resolve(actions)
            reward_sum += float(np.sum(result.rewards))
            delivered += int(np.sum(result.delivered))
            collisions += int(np.sum(result.collisions))
            terminal = step_in_episode == steps_per_episode - 1
            if terminal:
                next_states = np.zeros((N_NODES, STATE_DIM), dtype=np.float32)
                next_masks = np.zeros((N_NODES, N_ACTIONS), dtype=bool)
                next_masks[:, ABSTAIN_ACTION] = True
            else:
                raw_next_states, next_masks = env.begin_slot()
                next_states = observation_for_policy(raw_next_states, policy_variant)
            replay.add_all_nodes(
                states=states,
                actions=actions,
                rewards=result.rewards,
                next_states=next_states,
                next_masks=next_masks,
                done=terminal,
            )
            loss = agent.update(replay)
            if loss is not None:
                losses.append(loss)
            states, masks = next_states, next_masks
            global_step += 1
        episode_rows.append(
            {
                "episode": episode,
                "factor_id": factor.factor_id,
                "environment_seed": environment_seed,
                "label_shift": label_shift,
                "epsilon_final": _epsilon(global_step - 1, total_steps),
                "mean_node_reward": reward_sum / (steps_per_episode * N_NODES),
                "deliveries": delivered,
                "collision_attempts": collisions,
                "mean_loss": float(np.mean(losses)) if losses else "",
                "trace_sha256": env.trace_sha256,
            }
        )

    expected_insertions = total_steps * N_NODES
    if replay.total_inserted != expected_insertions:
        raise AssertionError("all-node replay insertion count mismatch")
    if not np.all(replay.node_insertions == total_steps):
        raise AssertionError("replay contains unequal node representation")
    checkpoint_path = run_dir / "final.pt"
    checkpoint_metadata: dict[str, Any] = {
        "study_id": STUDY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(),
        "smoke": smoke,
        "policy_variant": policy_variant,
        "seed_index": seed_index,
        "model_seed": model_seed,
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "global_environment_steps": total_steps,
        "gradient_steps": agent.gradient_steps,
        "replay_total_insertions": replay.total_inserted,
        "replay_node_insertions": replay.node_insertions.tolist(),
        "selection_rule": "final_iterate_only",
    }
    agent.save(checkpoint_path, metadata=checkpoint_metadata)
    metadata = {
        **checkpoint_metadata,
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        # Runtime is operational provenance, not embedded in the scientific
        # checkpoint, so identical reruns can remain byte-reproducible.
        "wall_seconds": time.perf_counter() - wall_start,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_csv(run_dir / "episodes.csv", episode_rows)
    return metadata


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty training table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _worker(arguments: tuple[int, str, str, bool]) -> dict[str, Any]:
    seed_index, policy_variant, root, smoke = arguments
    return train_one(
        seed_index=seed_index,
        policy_variant=policy_variant,
        output_root=Path(root),
        smoke=smoke,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed-index", type=int)
    group.add_argument("--all", action="store_true")
    group.add_argument("--plan", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--policy",
        choices=LEARNED_POLICIES,
        default=LEARNED_POLICY_NAME,
        help="learned variant for a single --seed-index run",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()
    if args.plan:
        plan = {
            "study_id": STUDY_ID,
            "learned_policy_variants": list(LEARNED_POLICIES),
            "training_seeds": 1 if args.smoke else len(TRAINING_SEEDS),
            "total_final_checkpoints": (
                len(LEARNED_POLICIES)
                if args.smoke
                else len(LEARNED_POLICIES) * len(TRAINING_SEEDS)
            ),
            "episodes_per_seed": SMOKE_EPISODES if args.smoke else TRAIN_EPISODES,
            "steps_per_episode": (
                SMOKE_STEPS_PER_EPISODE if args.smoke else TRAIN_STEPS_PER_EPISODE
            ),
            "all_node_transitions_per_seed": (
                (SMOKE_EPISODES if args.smoke else TRAIN_EPISODES)
                * (SMOKE_STEPS_PER_EPISODE if args.smoke else TRAIN_STEPS_PER_EPISODE)
                * N_NODES
            ),
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.all:
        indices = [0] if args.smoke else list(range(len(TRAINING_SEEDS)))
        tasks = [
            (index, policy_variant, str(args.output_root), args.smoke)
            for policy_variant in LEARNED_POLICIES
            for index in indices
        ]
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(tasks)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            outputs = list(executor.map(_worker, tasks))
    else:
        assert args.seed_index is not None
        outputs = [
            train_one(
                seed_index=args.seed_index,
                output_root=args.output_root,
                policy_variant=args.policy,
                smoke=args.smoke,
            )
        ]
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
