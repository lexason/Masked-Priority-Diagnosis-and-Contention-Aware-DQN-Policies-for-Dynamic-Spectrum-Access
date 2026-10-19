"""Fail-closed structural and numerical validation for CRN-CA-PR1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from main.revision.crn_ca_pr1_20261017.config import (
    ALL_POLICIES,
    BASELINES,
    BATCH_SIZE,
    DEFAULT_RESULTS_ROOT,
    EPSILON_END,
    EPSILON_START,
    EVAL_BURN_IN_STEPS,
    EVAL_MEASUREMENT_STEPS,
    EVALUATION_SEEDS,
    FACTORS,
    LEARNED_POLICIES,
    N_ACTIONS,
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
    label_permutation,
    protocol_sha256,
    training_environment_seed,
)
from main.revision.crn_ca_pr1_20261017.model import CollisionAwareDQN
from main.revision.crn_ca_pr1_20261017.provenance import (
    canonical_sha256,
    sha256_file,
    verify_freeze,
)

_HEX = frozenset("0123456789abcdef")
_TRAINING_FIELDS = (
    "episode",
    "factor_id",
    "environment_seed",
    "label_shift",
    "epsilon_final",
    "mean_node_reward",
    "deliveries",
    "collision_attempts",
    "mean_loss",
    "trace_sha256",
)
_EVALUATION_FIELDS = (
    "study_id",
    "protocol_version",
    "protocol_sha256",
    "run_id",
    "smoke",
    "policy",
    "training_seed",
    "checkpoint_sha256",
    "factor_id",
    "occupancy_process",
    "occupancy_probability",
    "arrival_probability",
    "replication_index",
    "environment_seed",
    "logical_to_physical",
    "burn_in_steps",
    "measurement_steps",
    "trace_sha256",
    "starting_backlog",
    "arrivals",
    "deliveries",
    "ending_backlog",
    "network_goodput",
    "finite_window_service_fraction",
    "capacity_ceiling_per_slot",
    "opportunity_utilization",
    "attempted",
    "singleton_attempts",
    "collision_attempts",
    "collision_attempt_rate",
    "link_failures",
    "conditional_link_failure_rate",
    "voluntary_deferrals",
    "forced_deferrals",
    "mean_network_backlog",
    "delivery_jain_fairness",
    "zero_delivery_run",
    "priority_eligible_decisions",
    "priority_matches",
    "static_priority_agreement",
    "logical_action_counts",
    "physical_action_counts",
    "mask_action_counts_shape",
    "mask_action_counts",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _require_csv_schema(
    rows: list[dict[str, str]], *, expected: tuple[str, ...], path: Path
) -> None:
    if tuple(rows[0]) != expected:
        raise ValueError(
            f"CSV schema mismatch in {path}: expected {expected}; got {tuple(rows[0])}"
        )
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"ragged CSV row in {path}")


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise ValueError(f"invalid SHA-256 for {label}")


def _parse_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalised = str(value).strip().lower()
    if normalised == "true":
        return True
    if normalised == "false":
        return False
    raise ValueError(f"invalid Boolean for {label}: {value!r}")


def _integer(row: dict[str, str], key: str) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid integer {key} in {row.get('run_id', '<row>')}"
        ) from error
    if value < 0:
        raise ValueError(f"negative {key} in {row.get('run_id', '<row>')}")
    return value


def _finite(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid float {key} in {row.get('run_id', '<row>')}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in {row.get('run_id', '<row>')}")
    return value


def _assert_close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: expected {expected!r}; got {actual!r}")


def _expected_gradient_steps(total_steps: int) -> int:
    first_ready_slot = math.ceil(BATCH_SIZE / N_NODES)
    return max(0, total_steps - first_ready_slot + 1)


def _training_run_dir(
    results_root: Path, *, policy: str, seed: int, smoke: bool
) -> Path:
    if smoke:
        return results_root / "smoke" / "training" / policy / "seed_smoke"
    return results_root / "training" / policy / f"seed_{seed}"


def _expected_epsilon(step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return EPSILON_END
    fraction = step / (total_steps - 1)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def validate_training(results_root: Path, *, smoke: bool) -> dict[str, Any]:
    """Validate every checkpoint and its exact preregistered training schedule."""
    seeds = (SMOKE_MODEL_SEED,) if smoke else TRAINING_SEEDS
    episodes = SMOKE_EPISODES if smoke else TRAIN_EPISODES
    steps_per_episode = SMOKE_STEPS_PER_EPISODE if smoke else TRAIN_STEPS_PER_EPISODE
    total_steps = episodes * steps_per_episode
    gradient_steps = _expected_gradient_steps(total_steps)
    artifact_hashes: dict[str, str] = {}
    trace_sets: dict[tuple[int, int], set[str]] = defaultdict(set)

    for policy in LEARNED_POLICIES:
        for seed_index, seed in enumerate(seeds):
            effective_seed_index = 0 if smoke else seed_index
            run_dir = _training_run_dir(
                results_root, policy=policy, seed=seed, smoke=smoke
            )
            checkpoint = run_dir / "final.pt"
            metadata_path = run_dir / "metadata.json"
            episodes_path = run_dir / "episodes.csv"
            rows = _read_csv(episodes_path)
            _require_csv_schema(rows, expected=_TRAINING_FIELDS, path=episodes_path)
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            expected_metadata: dict[str, Any] = {
                "study_id": STUDY_ID,
                "protocol_version": PROTOCOL_VERSION,
                "protocol_sha256": protocol_sha256(),
                "smoke": smoke,
                "policy_variant": policy,
                "seed_index": effective_seed_index,
                "model_seed": seed,
                "episodes": episodes,
                "steps_per_episode": steps_per_episode,
                "global_environment_steps": total_steps,
                "gradient_steps": gradient_steps,
                "replay_total_insertions": total_steps * N_NODES,
                "replay_node_insertions": [total_steps] * N_NODES,
                "selection_rule": "final_iterate_only",
            }
            for key, expected in expected_metadata.items():
                if metadata.get(key) != expected:
                    raise ValueError(
                        f"training metadata {key} mismatch for {policy}/{seed}: "
                        f"expected {expected!r}; got {metadata.get(key)!r}"
                    )
            expected_metadata_keys = set(expected_metadata) | {
                "checkpoint_sha256",
                "wall_seconds",
            }
            if set(metadata) != expected_metadata_keys:
                raise ValueError(
                    f"unexpected training metadata fields for {policy}/{seed}: "
                    f"{sorted(set(metadata) ^ expected_metadata_keys)}"
                )
            wall_seconds = float(metadata["wall_seconds"])
            if not math.isfinite(wall_seconds) or wall_seconds < 0.0:
                raise ValueError(f"invalid wall time for {policy}/{seed}")
            checkpoint_digest = sha256_file(checkpoint)
            _require_sha256(checkpoint_digest, label=f"checkpoint {policy}/{seed}")
            if metadata["checkpoint_sha256"] != checkpoint_digest:
                raise ValueError(f"checkpoint digest mismatch for {policy}/{seed}")

            agent, checkpoint_metadata = CollisionAwareDQN.load(checkpoint)
            if agent.seed != seed:
                raise ValueError(f"checkpoint top-level seed mismatch for {policy}/{seed}")
            if agent.gradient_steps != gradient_steps:
                raise ValueError(
                    f"checkpoint top-level gradient count mismatch for {policy}/{seed}"
                )
            if checkpoint_metadata != expected_metadata:
                raise ValueError(
                    f"checkpoint embedded metadata mismatch for {policy}/{seed}"
                )

            if len(rows) != episodes:
                raise ValueError(
                    f"training episode count mismatch for {policy}/{seed}: "
                    f"expected {episodes}; got {len(rows)}"
                )
            for episode, row in enumerate(rows):
                expected_factor = FACTORS[
                    (episode + effective_seed_index) % len(FACTORS)
                ]
                expected_shift = (
                    episode // len(FACTORS) + effective_seed_index
                ) % N_CHANNELS
                expected_environment_seed = (
                    SMOKE_ENVIRONMENT_SEED + episode
                    if smoke
                    else training_environment_seed(effective_seed_index, episode)
                )
                if int(row["episode"]) != episode:
                    raise ValueError(f"noncanonical episode order for {policy}/{seed}")
                if row["factor_id"] != expected_factor.factor_id:
                    raise ValueError(
                        f"factor schedule mismatch for {policy}/{seed}/episode-{episode}"
                    )
                if int(row["label_shift"]) != expected_shift:
                    raise ValueError(
                        f"label schedule mismatch for {policy}/{seed}/episode-{episode}"
                    )
                if int(row["environment_seed"]) != expected_environment_seed:
                    raise ValueError(
                        f"environment schedule mismatch for {policy}/{seed}/episode-{episode}"
                    )
                expected_epsilon = _expected_epsilon(
                    (episode + 1) * steps_per_episode - 1, total_steps
                )
                _assert_close(
                    float(row["epsilon_final"]),
                    expected_epsilon,
                    label=f"epsilon {policy}/{seed}/episode-{episode}",
                )
                _finite(row, "mean_node_reward")
                for field in ("deliveries", "collision_attempts"):
                    _integer(row, field)
                if row["mean_loss"]:
                    _finite(row, "mean_loss")
                trace = row["trace_sha256"]
                _require_sha256(
                    trace, label=f"training trace {policy}/{seed}/episode-{episode}"
                )
                trace_sets[(seed, episode)].add(trace)

            for artifact in (checkpoint, metadata_path, episodes_path):
                relative = artifact.relative_to(results_root).as_posix()
                artifact_hashes[relative] = sha256_file(artifact)

    mismatched_traces = [key for key, values in trace_sets.items() if len(values) != 1]
    if mismatched_traces:
        raise ValueError(
            "learned variants did not receive common training traces: "
            f"{mismatched_traces[:3]}"
        )
    expected_trace_groups = len(seeds) * episodes
    if len(trace_sets) != expected_trace_groups:
        raise ValueError(
            f"expected {expected_trace_groups} training trace groups; got {len(trace_sets)}"
        )
    return {
        "training_variants_validated": len(LEARNED_POLICIES),
        "training_seeds_validated": len(seeds),
        "training_trace_groups_validated": len(trace_sets),
        "training_artifact_aggregate_sha256": canonical_sha256(artifact_hashes),
    }


def _evaluation_expectations(
    results_root: Path, *, smoke: bool
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    factors = (FACTORS[0],) if smoke else FACTORS
    environments = (
        ((0, SMOKE_ENVIRONMENT_SEED + 10_000),)
        if smoke
        else tuple(enumerate(EVALUATION_SEEDS))
    )
    training_seeds = (SMOKE_MODEL_SEED,) if smoke else TRAINING_SEEDS
    burn_in = SMOKE_EVAL_BURN_IN_STEPS if smoke else EVAL_BURN_IN_STEPS
    measurement = SMOKE_EVAL_MEASUREMENT_STEPS if smoke else EVAL_MEASUREMENT_STEPS
    expectations: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    checkpoint_digests: dict[tuple[str, int], str] = {}
    for policy in LEARNED_POLICIES:
        for seed in training_seeds:
            checkpoint = _training_run_dir(
                results_root, policy=policy, seed=seed, smoke=smoke
            ) / "final.pt"
            checkpoint_digests[(policy, seed)] = sha256_file(checkpoint)

    for factor in factors:
        for replication_index, environment_seed in environments:
            permutation = (
                tuple(range(N_CHANNELS))
                if smoke
                else label_permutation(replication_index)
            )
            common = {
                "factor": factor,
                "replication_index": replication_index,
                "environment_seed": environment_seed,
                "logical_to_physical": "-".join(str(value) for value in permutation),
                "burn_in_steps": burn_in,
                "measurement_steps": measurement,
            }
            for policy in LEARNED_POLICIES:
                for seed in training_seeds:
                    key = (policy, str(seed), factor.factor_id, environment_seed)
                    expectations[key] = {
                        **common,
                        "training_seed": str(seed),
                        "checkpoint_sha256": checkpoint_digests[(policy, seed)],
                        "run_id": (
                            f"{policy}__train-{seed}__{factor.factor_id}"
                            f"__rep-{replication_index:02d}__env-{environment_seed}"
                        ),
                    }
            for policy in BASELINES:
                key = (policy, "", factor.factor_id, environment_seed)
                expectations[key] = {
                    **common,
                    "training_seed": "",
                    "checkpoint_sha256": "",
                    "run_id": (
                        f"{policy}__train-baseline__{factor.factor_id}"
                        f"__rep-{replication_index:02d}__env-{environment_seed}"
                    ),
                }
    return expectations


def _validate_evaluation_metrics(row: dict[str, str], *, measurement_steps: int) -> None:
    run_id = row["run_id"]
    counts = {
        key: _integer(row, key)
        for key in (
            "starting_backlog",
            "arrivals",
            "deliveries",
            "ending_backlog",
            "attempted",
            "singleton_attempts",
            "collision_attempts",
            "link_failures",
            "voluntary_deferrals",
            "forced_deferrals",
            "priority_eligible_decisions",
            "priority_matches",
        )
    }
    if counts["starting_backlog"] + counts["arrivals"] != (
        counts["deliveries"] + counts["ending_backlog"]
    ):
        raise ValueError(f"packet conservation failed in {run_id}")
    if counts["attempted"] != (
        counts["singleton_attempts"] + counts["collision_attempts"]
    ):
        raise ValueError(f"attempt outcome partition failed in {run_id}")
    if counts["singleton_attempts"] != (
        counts["deliveries"] + counts["link_failures"]
    ):
        raise ValueError(f"singleton outcome partition failed in {run_id}")
    if counts["priority_matches"] > counts["priority_eligible_decisions"]:
        raise ValueError(f"priority matches exceed eligible decisions in {run_id}")

    denominator = counts["starting_backlog"] + counts["arrivals"]
    _assert_close(
        _finite(row, "network_goodput"),
        counts["deliveries"] / measurement_steps,
        label=f"network goodput in {run_id}",
    )
    service_fraction = row["finite_window_service_fraction"].strip()
    if denominator:
        _assert_close(
            float(service_fraction),
            counts["deliveries"] / denominator,
            label=f"finite-window service fraction in {run_id}",
        )
    elif service_fraction:
        raise ValueError(f"service fraction must be empty without packets in {run_id}")
    expected_collision_rate = (
        counts["collision_attempts"] / counts["attempted"]
        if counts["attempted"]
        else 0.0
    )
    _assert_close(
        _finite(row, "collision_attempt_rate"),
        expected_collision_rate,
        label=f"collision rate in {run_id}",
    )
    conditional_failure = row["conditional_link_failure_rate"].strip()
    if counts["singleton_attempts"]:
        _assert_close(
            float(conditional_failure),
            counts["link_failures"] / counts["singleton_attempts"],
            label=f"conditional link-failure rate in {run_id}",
        )
    elif conditional_failure:
        raise ValueError(
            f"conditional link-failure rate must be empty without singletons in {run_id}"
        )

    capacity_per_slot = _finite(row, "capacity_ceiling_per_slot")
    capacity_sum_float = capacity_per_slot * measurement_steps
    capacity_sum = round(capacity_sum_float)
    _assert_close(
        capacity_sum_float,
        float(capacity_sum),
        label=f"capacity-ceiling count in {run_id}",
    )
    if not counts["deliveries"] <= capacity_sum <= N_CHANNELS * measurement_steps:
        raise ValueError(f"invalid capacity ceiling in {run_id}")
    utilization = row["opportunity_utilization"].strip()
    if capacity_sum:
        _assert_close(
            float(utilization),
            counts["deliveries"] / capacity_sum,
            label=f"opportunity utilization in {run_id}",
        )
    elif utilization:
        raise ValueError(f"utilization must be empty without capacity in {run_id}")

    zero_delivery = _integer(row, "zero_delivery_run")
    if zero_delivery not in {0, 1} or zero_delivery != int(counts["deliveries"] == 0):
        raise ValueError(f"zero-delivery flag mismatch in {run_id}")
    fairness_text = row["delivery_jain_fairness"].strip()
    if counts["deliveries"]:
        fairness = float(fairness_text)
        if not math.isfinite(fairness) or not 0.0 <= fairness <= 1.0:
            raise ValueError(f"invalid delivery fairness in {run_id}")
    elif fairness_text:
        raise ValueError(f"fairness must be empty for a zero-delivery run in {run_id}")
    if _finite(row, "mean_network_backlog") < 0.0:
        raise ValueError(f"negative mean backlog in {run_id}")

    agreement_text = row["static_priority_agreement"].strip()
    eligible = counts["priority_eligible_decisions"]
    if eligible:
        agreement = float(agreement_text)
        expected_agreement = counts["priority_matches"] / eligible
        if not math.isfinite(agreement):
            raise ValueError(f"non-finite priority agreement in {run_id}")
        _assert_close(
            agreement,
            expected_agreement,
            label=f"priority agreement in {run_id}",
        )
    elif agreement_text:
        raise ValueError(
            f"priority agreement must be empty with no eligible decisions in {run_id}"
        )
    if (
        row["policy"] == "static_priority"
        and eligible
        and float(agreement_text) != 1.0
    ):
        raise ValueError("static-priority reference disagrees with itself")

    try:
        logical_counts = [int(value) for value in row["logical_action_counts"].split("-")]
        physical_counts = [int(value) for value in row["physical_action_counts"].split("-")]
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid action-count encoding in {run_id}") from error
    if len(logical_counts) != N_ACTIONS or any(value < 0 for value in logical_counts):
        raise ValueError(f"invalid logical action counts in {run_id}")
    if len(physical_counts) != N_CHANNELS or any(value < 0 for value in physical_counts):
        raise ValueError(f"invalid physical action counts in {run_id}")
    if sum(logical_counts) != N_NODES * measurement_steps:
        raise ValueError(f"logical action count total mismatch in {run_id}")
    if sum(logical_counts[:N_CHANNELS]) != counts["attempted"]:
        raise ValueError(f"logical attempted-action total mismatch in {run_id}")
    if sum(physical_counts) != counts["attempted"]:
        raise ValueError(f"physical action count total mismatch in {run_id}")
    try:
        permutation = [int(value) for value in row["logical_to_physical"].split("-")]
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid channel permutation in {run_id}") from error
    if len(permutation) != N_CHANNELS or sorted(permutation) != list(range(N_CHANNELS)):
        raise ValueError(f"invalid channel permutation in {run_id}")
    expected_physical_counts = [0] * N_CHANNELS
    for logical_channel, physical_channel in enumerate(permutation):
        expected_physical_counts[physical_channel] = logical_counts[logical_channel]
    if physical_counts != expected_physical_counts:
        raise ValueError(f"logical/physical action counts disagree in {run_id}")
    if counts["voluntary_deferrals"] + counts["forced_deferrals"] > logical_counts[-1]:
        raise ValueError(f"deferred-packet count exceeds abstentions in {run_id}")
    if row["mask_action_counts_shape"] != f"{1 << N_CHANNELS}x{N_ACTIONS}":
        raise ValueError(f"mask/action count shape mismatch in {run_id}")
    try:
        mask_action_counts = [int(value) for value in row["mask_action_counts"].split("-")]
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid mask/action count encoding in {run_id}") from error
    if len(mask_action_counts) != (1 << N_CHANNELS) * N_ACTIONS or any(
        value < 0 for value in mask_action_counts
    ):
        raise ValueError(f"invalid mask/action counts in {run_id}")
    if sum(mask_action_counts) != eligible:
        raise ValueError(f"mask/action count total mismatch in {run_id}")
    if any(mask_action_counts[:N_ACTIONS]):
        raise ValueError(f"mask/action table includes an ineligible empty mask in {run_id}")
    for mask_id in range(1 << N_CHANNELS):
        row_offset = mask_id * N_ACTIONS
        for action in range(N_CHANNELS):
            if not mask_id & (1 << action) and mask_action_counts[row_offset + action]:
                raise ValueError(
                    f"mask/action table records an illegal action in {run_id}"
                )
    recomputed_priority_matches = sum(
        mask_action_counts[mask_id * N_ACTIONS + (mask_id & -mask_id).bit_length() - 1]
        for mask_id in range(1, 1 << N_CHANNELS)
    )
    if recomputed_priority_matches != counts["priority_matches"]:
        raise ValueError(f"priority-match count disagrees with mask/action table in {run_id}")


def validate_evaluation(results_root: Path, *, smoke: bool) -> dict[str, Any]:
    """Validate the exact Cartesian evaluation design and all row identities."""
    path = (
        results_root / "smoke" / "evaluation" / "summary.csv"
        if smoke
        else results_root / "evaluation" / "summary.csv"
    )
    rows = _read_csv(path)
    _require_csv_schema(rows, expected=_EVALUATION_FIELDS, path=path)
    expectations = _evaluation_expectations(results_root, smoke=smoke)
    if len(rows) != len(expectations):
        raise ValueError(f"expected {len(expectations)} evaluation rows; got {len(rows)}")

    actual: dict[tuple[str, str, str, int], dict[str, str]] = {}
    run_ids: set[str] = set()
    trace_groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    trace_members: dict[tuple[str, int], int] = defaultdict(int)
    factor_lookup = {factor.factor_id: factor for factor in FACTORS}
    for row in rows:
        try:
            environment_seed = int(row["environment_seed"])
            key = (
                row["policy"],
                row["training_seed"].strip(),
                row["factor_id"],
                environment_seed,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed evaluation identity fields") from error
        if key in actual:
            raise ValueError(f"duplicate evaluation design key: {key}")
        actual[key] = row
        if row["run_id"] in run_ids:
            raise ValueError(f"duplicate evaluation run ID: {row['run_id']}")
        run_ids.add(row["run_id"])

    expected_keys = set(expectations)
    actual_keys = set(actual)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:3]
        unexpected = sorted(actual_keys - expected_keys)[:3]
        raise ValueError(
            f"evaluation Cartesian design mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )

    for key, expected in expectations.items():
        row = actual[key]
        run_id = expected["run_id"]
        if row["run_id"] != run_id:
            raise ValueError(f"noncanonical run ID for {key}: {row['run_id']}")
        if row["study_id"] != STUDY_ID:
            raise ValueError(f"study ID mismatch in {run_id}")
        if row["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError(f"protocol version mismatch in {run_id}")
        if row["protocol_sha256"] != protocol_sha256():
            raise ValueError(f"protocol digest mismatch in {run_id}")
        if _parse_bool(row["smoke"], label=f"smoke in {run_id}") is not smoke:
            raise ValueError(f"smoke flag mismatch in {run_id}")
        if row["policy"] not in ALL_POLICIES:
            raise ValueError(f"unknown policy in {run_id}")
        if row["training_seed"].strip() != expected["training_seed"]:
            raise ValueError(f"training seed mismatch in {run_id}")
        if row["checkpoint_sha256"].strip() != expected["checkpoint_sha256"]:
            raise ValueError(f"checkpoint identity mismatch in {run_id}")

        factor = factor_lookup[row["factor_id"]]
        if row["occupancy_process"] != factor.occupancy_process:
            raise ValueError(f"occupancy process mismatch in {run_id}")
        _assert_close(
            float(row["occupancy_probability"]),
            factor.occupancy_probability,
            label=f"occupancy probability in {run_id}",
        )
        _assert_close(
            float(row["arrival_probability"]),
            factor.arrival_probability,
            label=f"arrival probability in {run_id}",
        )
        if int(row["replication_index"]) != expected["replication_index"]:
            raise ValueError(f"replication index mismatch in {run_id}")
        if row["logical_to_physical"] != expected["logical_to_physical"]:
            raise ValueError(f"channel permutation mismatch in {run_id}")
        if int(row["burn_in_steps"]) != expected["burn_in_steps"]:
            raise ValueError(f"burn-in duration mismatch in {run_id}")
        if int(row["measurement_steps"]) != expected["measurement_steps"]:
            raise ValueError(f"measurement duration mismatch in {run_id}")

        trace = row["trace_sha256"]
        _require_sha256(trace, label=f"evaluation trace {run_id}")
        group = (row["factor_id"], int(row["environment_seed"]))
        trace_groups[group].add(trace)
        trace_members[group] += 1
        _validate_evaluation_metrics(
            row, measurement_steps=int(expected["measurement_steps"])
        )

    expected_groups = 1 if smoke else len(FACTORS) * len(EVALUATION_SEEDS)
    expected_members = len(LEARNED_POLICIES) * (
        1 if smoke else len(TRAINING_SEEDS)
    ) + len(BASELINES)
    if len(trace_groups) != expected_groups:
        raise ValueError(
            f"expected {expected_groups} common-trace groups; got {len(trace_groups)}"
        )
    bad_groups = [
        group
        for group in trace_groups
        if len(trace_groups[group]) != 1 or trace_members[group] != expected_members
    ]
    if bad_groups:
        raise ValueError(
            "policies did not receive complete common traces: "
            f"{bad_groups[:3]}"
        )
    return {
        "evaluation_rows_validated": len(rows),
        "common_trace_groups_validated": len(trace_groups),
        "evaluation_summary_sha256": sha256_file(path),
    }


def _write_json_once_or_identical(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to overwrite non-identical output: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise FileExistsError(
                    f"refusing to overwrite concurrently created output: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def validate(results_root: Path, *, smoke: bool) -> dict[str, Any]:
    """Run every validation gate and write an immutable, input-bound report."""
    if not smoke:
        verify_freeze()
    report: dict[str, Any] = {
        "study_id": STUDY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(),
        "smoke": smoke,
        "status": "PASS",
    }
    report.update(validate_training(results_root, smoke=smoke))
    report.update(validate_evaluation(results_root, smoke=smoke))
    destination = (
        results_root / "smoke" / "VALIDATION_REPORT.json"
        if smoke
        else results_root / "VALIDATION_REPORT.json"
    )
    _write_json_once_or_identical(destination, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.results_root, smoke=args.smoke), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
