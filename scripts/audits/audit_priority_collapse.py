"""Audit frozen DQN checkpoints for masked static-priority collapse.

The audit deliberately uses only information available in the archived action
logs: the legal-action mask and the executed physical-channel action.  For each
training seed, a channel order is fitted on three S1/S2 environment
realizations, evaluated without refitting on two disjoint S1/S2 realizations,
and then checked over the complete 20-environment S1--S5 core archive.  Six
channels permit exhaustive enumeration of all 6! candidate orders.

The script also computes exact reference values for fixed-priority and
uniform-legal selectors under independent Bernoulli masks, and exact
shared-mask collision-domain consequences by enumerating all 2^6 masks.  Those
calculations are analytical diagnostics; they do not retroactively turn the
evaluated simulator into a packet-level experiment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve()


def _discover_project_root() -> Path:
    """Return the nearest checkout containing the DQN implementation."""

    candidates = (Path.cwd().resolve(), *SCRIPT.parents)
    for candidate in candidates:
        if (candidate / "ai/agents/dqn_agent.py").is_file():
            return candidate
    return Path.cwd().resolve()


DEFAULT_PROJECT_ROOT = _discover_project_root()
DEFAULT_TRACE_ROOT = Path("traces/core/runs")
DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("reproduced")
TRAINING_SEEDS = (42, 7, 1007, 2007, 101, 202, 303, 404)
CALIBRATION_ENVS = (2610000, 2620000, 2630000)
VALIDATION_ENVS = (2640000, 2650000)
S4_ENVS = (*CALIBRATION_ENVS, *VALIDATION_ENVS)
ALL_CORE_ENVS = tuple(2_610_000 + 10_000 * index for index in range(20))
ALL_CORE_SCENARIOS = ("S1", "S2", "S3", "S4", "S5")
N_CHANNELS = 6
N_NODES = 15
TONE_CHANNEL = 1
COUNTERFACTUAL_SAMPLES_PER_MASK = 128
S4_STEP_SECONDS = 0.1
S4_MEAN_ON_SECONDS = 0.5
S4_MEAN_OFF_SECONDS = 0.167


@dataclass
class Aggregate:
    """Sufficient statistics for fitting and checking a priority order."""

    counts: dict[tuple[int, int], int]
    attempted: int = 0
    abstentions: int = 0
    accepted: int = 0
    tone_selected: int = 0
    tone_accepted: int = 0

    @classmethod
    def empty(cls) -> Aggregate:
        return cls(counts=defaultdict(int))


def _mask_bits(raw: str) -> int:
    values = json.loads(raw)
    if len(values) != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} mask entries, found {len(values)}")
    bits = 0
    for channel, legal in enumerate(values):
        if int(legal):
            bits |= 1 << channel
    return bits


def _load_dqn_agent(project_root: Path) -> type:
    source = project_root / "ai/agents/dqn_agent.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"DQN project source not found at {source}. "
            "Pass --project-root pointing to the ChainRadioNet checkout."
        )
    project = str(project_root)
    if project not in sys.path:
        sys.path.insert(0, project)
    try:
        from ai.agents.dqn_agent import DQNAgent
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f"Could not import DQNAgent from {source}. Install the project's Python "
            "dependencies and pass the correct --project-root."
        ) from exc
    return DQNAgent


def _trace_path(trace_root: Path, scenario: str, seed: int, env_seed: int) -> Path:
    return trace_root / scenario / f"dqn_trainseed_{seed}" / f"env_{env_seed}.csv.gz"


def _accumulate(paths: Iterable[Path]) -> Aggregate:
    result = Aggregate.empty()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required raw trace is missing: {path}. Check --trace-root and "
                "obtain the complete row-level trace archive."
            )
        with gzip.open(path, mode="rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "legal_action_mask",
                "channel_selected",
                "attempted",
                "accepted",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Missing required fields in {path}")
            for row in reader:
                if int(row["attempted"]) == 0:
                    result.abstentions += 1
                    continue
                mask = _mask_bits(row["legal_action_mask"])
                action = int(row["channel_selected"])
                if not (mask & (1 << action)):
                    raise ValueError(f"Illegal archived action in {path}: mask={mask}, a={action}")
                result.counts[(mask, action)] += 1
                result.attempted += 1
                accepted = int(row["accepted"])
                result.accepted += accepted
                if action == TONE_CHANNEL:
                    result.tone_selected += 1
                    result.tone_accepted += accepted
    return result


def _predict(order: tuple[int, ...], mask: int) -> int:
    for channel in order:
        if mask & (1 << channel):
            return channel
    raise ValueError("Priority prediction requested for an empty legal set")


def _agreement(order: tuple[int, ...], aggregate: Aggregate) -> tuple[int, int, float]:
    matches = sum(
        count
        for (mask, action), count in aggregate.counts.items()
        if _predict(order, mask) == action
    )
    total = aggregate.attempted
    return matches, total, matches / total if total else math.nan


def _distinct_nonempty_masks(aggregate: Aggregate) -> int:
    return len({mask for mask, _action in aggregate.counts})


def _precedence_certificate(aggregate: Aggregate) -> dict[str, object]:
    """Construct an exact graph certificate for masked-priority equivalence.

    Every attempted record with legal mask ``M`` and selected action ``a``
    requires ``a`` to precede each other member of ``M``.  The resulting graph
    is acyclic exactly when all records admit a static priority order; its
    topological ordering is unique exactly when the representing order is
    unique.  Counts are retained as auditable edge support but do not alter the
    graph logic.
    """

    support: Counter[tuple[int, int]] = Counter()
    for (mask, action), count in aggregate.counts.items():
        for channel in range(N_CHANNELS):
            if channel != action and mask & (1 << channel):
                support[(action, channel)] += count

    adjacency = [set() for _ in range(N_CHANNELS)]
    indegree = [0] * N_CHANNELS
    for higher, lower in support:
        adjacency[higher].add(lower)
        indegree[lower] += 1

    available = sorted(channel for channel, degree in enumerate(indegree) if degree == 0)
    topological_order: list[int] = []
    unique = True
    while available:
        if len(available) != 1:
            unique = False
        channel = available.pop(0)
        topological_order.append(channel)
        for lower in sorted(adjacency[channel]):
            indegree[lower] -= 1
            if indegree[lower] == 0:
                available.append(lower)
                available.sort()

    acyclic = len(topological_order) == N_CHANNELS
    if not acyclic:
        unique = False
    payload: dict[str, object] = {
        "attempted_records": aggregate.attempted,
        "distinct_mask_action_pairs": len(aggregate.counts),
        "edge_count": len(support),
        "edges": [
            {"higher": higher, "lower": lower, "support_count": support[(higher, lower)]}
            for higher, lower in sorted(support)
        ],
        "acyclic": acyclic,
        "unique_topological_order": unique,
        "topological_order_high_to_low": topological_order if acyclic else None,
    }
    payload["certificate_sha256"] = hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    return payload


def _fit_order(aggregate: Aggregate) -> tuple[tuple[int, ...], int, int]:
    scored: list[tuple[int, tuple[int, ...]]] = []
    for order in itertools.permutations(range(N_CHANNELS)):
        matches, _, _ = _agreement(order, aggregate)
        scored.append((matches, order))
    best_matches = max(score for score, _ in scored)
    best_orders = [order for score, order in scored if score == best_matches]
    return min(best_orders), best_matches, len(best_orders)


def _masked_actions(network: torch.nn.Module, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        q_values = network(torch.from_numpy(states).to(dtype=torch.float32))[:, :N_CHANNELS]
    legal = torch.from_numpy(masks.astype(np.bool_))
    q_values = q_values.masked_fill(~legal, float("-inf"))
    return q_values.argmax(dim=1).cpu().numpy()


def _counterfactual_audit(
    seed: int,
    order: tuple[int, ...],
    checkpoint_root: Path,
    dqn_agent_class: type,
) -> dict[str, float | int]:
    """Stress dynamic feature blocks while holding each legal mask fixed.

    These synthetic [0,1] perturbations are deliberately broader than the
    empirical generator support.  They test network responsiveness; the
    held-out trace audit remains the evidentiary basis for behavior on the
    evaluated scenarios.
    """

    checkpoint = checkpoint_root / f"seed_{seed}" / f"dqn_seed{seed}_final.pt"
    agent = dqn_agent_class(
        node_id=f"audit_seed_{seed}", n_channels=N_CHANNELS, seed=seed
    )
    agent.load_checkpoint(checkpoint)
    network = agent._online  # Read-only audit of the archived executable network.
    network.eval()

    rng = np.random.default_rng(20_261_016 + seed)
    masks_list: list[np.ndarray] = []
    occupancy_list: list[np.ndarray] = []
    expected_list: list[int] = []
    for mask_bits in range(1, 1 << N_CHANNELS):
        mask = np.asarray(
            [bool(mask_bits & (1 << channel)) for channel in range(N_CHANNELS)],
            dtype=np.bool_,
        )
        occupancy = (~mask).astype(np.float32)
        expected = _predict(order, mask_bits)
        for _ in range(COUNTERFACTUAL_SAMPLES_PER_MASK):
            masks_list.append(mask)
            occupancy_list.append(occupancy)
            expected_list.append(expected)
    masks = np.stack(masks_list)
    occupancy = np.stack(occupancy_list)
    expected = np.asarray(expected_list, dtype=np.int64)
    n_states = len(expected)

    base = np.zeros((n_states, 128), dtype=np.float32)
    base[:, 0:N_CHANNELS] = occupancy
    base[:, N_CHANNELS : 5 * N_CHANNELS] = 0.5
    base_actions = _masked_actions(network, base, masks)

    full = base.copy()
    full[:, N_CHANNELS : 5 * N_CHANNELS] = rng.random((n_states, 4 * N_CHANNELS), dtype=np.float32)
    full_actions = _masked_actions(network, full, masks)
    result: dict[str, float | int] = {
        "counterfactual_states": n_states,
        "counterfactual_full_priority_agreement": float(np.mean(full_actions == expected)),
        "counterfactual_full_argmax_flip_rate": float(np.mean(full_actions != base_actions)),
    }

    blocks = {
        "interference": (N_CHANNELS, 2 * N_CHANNELS),
        "power": (2 * N_CHANNELS, 3 * N_CHANNELS),
        "occupancy_history": (3 * N_CHANNELS, 4 * N_CHANNELS),
        "acceptance_history": (4 * N_CHANNELS, 5 * N_CHANNELS),
    }
    for name, (start, stop) in blocks.items():
        perturbed = base.copy()
        perturbed[:, start:stop] = rng.random((n_states, stop - start), dtype=np.float32)
        actions = _masked_actions(network, perturbed, masks)
        result[f"counterfactual_{name}_argmax_flip_rate"] = float(np.mean(actions != base_actions))
        result[f"counterfactual_{name}_priority_agreement"] = float(np.mean(actions == expected))
    return result


def _fixed_priority_unique_channels(occupancy: float) -> float:
    free = 1.0 - occupancy
    return sum(1.0 - (1.0 - free * occupancy**rank) ** N_NODES for rank in range(N_CHANNELS))


def _uniform_legal_unique_channels(occupancy: float) -> float:
    per_channel_selection = (1.0 - occupancy**N_CHANNELS) / N_CHANNELS
    return N_CHANNELS * (1.0 - (1.0 - per_channel_selection) ** N_NODES)


def _s4_stationary_occupancy() -> float:
    p_off = 1.0 - math.exp(-S4_STEP_SECONDS / S4_MEAN_ON_SECONDS)
    p_on = 1.0 - math.exp(-S4_STEP_SECONDS / S4_MEAN_OFF_SECONDS)
    return p_on / (p_on + p_off)


def _s4_acceptance_for_tone_rank(rank: int, occupancy: float) -> float:
    """Acceptance conditional on an attempt when tone selections always fail."""

    if not 1 <= rank <= N_CHANNELS:
        raise ValueError(f"Tone rank must be in [1, {N_CHANNELS}], found {rank}")
    tone_selection = (1.0 - occupancy) * occupancy ** (rank - 1)
    attempt_probability = 1.0 - occupancy**N_CHANNELS
    return 1.0 - tone_selection / attempt_probability


def _independent_mask_population_unique_channels(
    orders: list[tuple[int, ...]], occupancy: float
) -> float:
    if len(orders) != N_NODES:
        raise ValueError(f"Expected {N_NODES} node orders, found {len(orders)}")
    ranks = [{channel: rank for rank, channel in enumerate(order)} for order in orders]
    free = 1.0 - occupancy
    expected = 0.0
    for channel in range(N_CHANNELS):
        probability_unused = 1.0
        for rank in ranks:
            probability_selected = free * occupancy ** rank[channel]
            probability_unused *= 1.0 - probability_selected
        expected += 1.0 - probability_unused
    return expected


def _shared_mask_metrics(
    orders: list[tuple[int, ...]], occupancy: float, offered_load: float = 1.0
) -> dict[str, float]:
    """Exact single-collision-domain metrics over all shared masks.

    Each node is independently active with probability ``offered_load`` and a
    transmission succeeds only if exactly one active node selects the channel.
    The returned attempted count is an expectation, not packets delivered by
    the original emulator.
    """

    totals = Counter()
    for mask in range(1 << N_CHANNELS):
        occupied_count = N_CHANNELS - mask.bit_count()
        free_count = mask.bit_count()
        probability = occupancy**occupied_count * (1.0 - occupancy) ** free_count
        if mask == 0:
            totals["prob_all_occupied"] += probability
            continue
        actions = [_predict(order, mask) for order in orders]
        multiplicities = Counter(actions)
        totals["attempts"] += probability * len(actions) * offered_load
        totals["unique_channels"] += probability * sum(
            1.0 - (1.0 - offered_load) ** count for count in multiplicities.values()
        )
        totals["singleton_successes"] += probability * sum(
            count * offered_load * (1.0 - offered_load) ** (count - 1)
            for count in multiplicities.values()
        )
        totals["colliding_attempts"] += probability * sum(
            count * offered_load * (1.0 - (1.0 - offered_load) ** (count - 1))
            for count in multiplicities.values()
        )
    attempts = totals["attempts"]
    return {
        "expected_attempts_per_cycle": totals["attempts"],
        "expected_unique_channels_per_cycle": totals["unique_channels"],
        "expected_singleton_successes_per_cycle": totals["singleton_successes"],
        "destructive_collision_delivery_fraction": (
            totals["singleton_successes"] / attempts if attempts else math.nan
        ),
        "colliding_attempt_fraction": (
            totals["colliding_attempts"] / attempts if attempts else math.nan
        ),
        "probability_all_channels_occupied": totals["prob_all_occupied"],
        "offered_load_probability": offered_load,
    }


def _shared_mask_uniform_random_metrics(
    occupancy: float, offered_load: float = 1.0
) -> dict[str, float]:
    totals = Counter()
    for free_count in range(N_CHANNELS + 1):
        probability = (
            math.comb(N_CHANNELS, free_count)
            * (1.0 - occupancy) ** free_count
            * occupancy ** (N_CHANNELS - free_count)
        )
        if free_count == 0:
            totals["prob_all_occupied"] += probability
            continue
        attempts = N_NODES * offered_load
        per_node_channel_probability = offered_load / free_count
        unique = free_count * (1.0 - (1.0 - per_node_channel_probability) ** N_NODES)
        singleton = N_NODES * offered_load * (1.0 - per_node_channel_probability) ** (N_NODES - 1)
        totals["attempts"] += probability * attempts
        totals["unique_channels"] += probability * unique
        totals["singleton_successes"] += probability * singleton
    attempts = totals["attempts"]
    return {
        "expected_attempts_per_cycle": attempts,
        "expected_unique_channels_per_cycle": totals["unique_channels"],
        "expected_singleton_successes_per_cycle": totals["singleton_successes"],
        "destructive_collision_delivery_fraction": (
            totals["singleton_successes"] / attempts if attempts else math.nan
        ),
        "colliding_attempt_fraction": (
            1.0 - totals["singleton_successes"] / attempts if attempts else math.nan
        ),
        "probability_all_channels_occupied": totals["prob_all_occupied"],
        "offered_load_probability": offered_load,
    }


def _cyclic_channel_orders() -> list[tuple[int, ...]]:
    return [
        tuple((channel + node) % N_CHANNELS for channel in range(N_CHANNELS))
        for node in range(N_NODES)
    ]


def _write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    trace_root: Path = DEFAULT_TRACE_ROOT,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    if not trace_root.is_dir():
        raise FileNotFoundError(
            f"Core raw-trace root not found: {trace_root}. "
            "Pass --trace-root pointing to the directory containing S1--S5."
        )
    missing_checkpoints = [
        checkpoint_root / f"seed_{seed}" / f"dqn_seed{seed}_final.pt"
        for seed in TRAINING_SEEDS
        if not (
            checkpoint_root / f"seed_{seed}" / f"dqn_seed{seed}_final.pt"
        ).is_file()
    ]
    if missing_checkpoints:
        preview = ", ".join(str(path) for path in missing_checkpoints[:2])
        suffix = ", and others" if len(missing_checkpoints) > 2 else ""
        raise FileNotFoundError(
            f"Required DQN checkpoint(s) missing: {preview}{suffix}. "
            "Pass --checkpoint-root pointing to the directory containing seed_* folders."
        )
    dqn_agent_class = _load_dqn_agent(project_root)
    rows: list[dict[str, object]] = []
    fitted: dict[int, tuple[int, ...]] = {}
    aggregates: dict[int, dict[str, Aggregate]] = {}
    precedence_certificates: dict[str, dict[str, dict[str, object]]] = {}

    for seed in TRAINING_SEEDS:
        calibration = _accumulate(
            _trace_path(trace_root, scenario, seed, env)
            for scenario in ("S1", "S2")
            for env in CALIBRATION_ENVS
        )
        validation = _accumulate(
            _trace_path(trace_root, scenario, seed, env)
            for scenario in ("S1", "S2")
            for env in VALIDATION_ENVS
        )
        s4 = _accumulate(
            _trace_path(trace_root, "S4", seed, env) for env in ALL_CORE_ENVS
        )
        full_core = _accumulate(
            _trace_path(trace_root, scenario, seed, env)
            for scenario in ALL_CORE_SCENARIOS
            for env in ALL_CORE_ENVS
        )
        order, fit_matches, tied_orders = _fit_order(calibration)
        cal_matches, cal_total, cal_rate = _agreement(order, calibration)
        val_matches, val_total, val_rate = _agreement(order, validation)
        s4_matches, s4_total, s4_rate = _agreement(order, s4)
        core_matches, core_total, core_rate = _agreement(order, full_core)
        if fit_matches != cal_matches:
            raise AssertionError("Fitted and recomputed calibration matches disagree")
        seed_certificates = {
            "calibration": _precedence_certificate(calibration),
            "heldout": _precedence_certificate(validation),
            "full_core": _precedence_certificate(full_core),
        }
        for domain, certificate in seed_certificates.items():
            if not bool(certificate["acyclic"]):
                raise AssertionError(f"Seed {seed} {domain} precedence graph is cyclic")
            if not bool(certificate["unique_topological_order"]):
                raise AssertionError(
                    f"Seed {seed} {domain} precedence graph lacks a unique ordering"
                )
            if certificate["topological_order_high_to_low"] != list(order):
                raise AssertionError(
                    f"Seed {seed} {domain} certificate disagrees with exhaustive fit"
                )
        precedence_certificates[str(seed)] = seed_certificates
        fitted[seed] = order
        aggregates[seed] = {
            "calibration": calibration,
            "validation": validation,
            "s4": s4,
            "full_core": full_core,
        }
        rows.append(
            {
                "training_seed": seed,
                "fitted_priority_high_to_low": ">".join(map(str, order)),
                "tied_best_orders": tied_orders,
                "calibration_attempts": cal_total,
                "calibration_matches": cal_matches,
                "calibration_agreement": cal_rate,
                "calibration_distinct_nonempty_masks": _distinct_nonempty_masks(calibration),
                "heldout_attempts": val_total,
                "heldout_matches": val_matches,
                "heldout_agreement": val_rate,
                "heldout_distinct_nonempty_masks": _distinct_nonempty_masks(validation),
                "s4_attempts": s4_total,
                "s4_matches": s4_matches,
                "s4_agreement": s4_rate,
                "s4_distinct_nonempty_masks": _distinct_nonempty_masks(s4),
                "full_core_attempts": core_total,
                "full_core_matches": core_matches,
                "full_core_agreement": core_rate,
                "full_core_distinct_nonempty_masks": _distinct_nonempty_masks(full_core),
                "calibration_precedence_edge_count": seed_certificates["calibration"][
                    "edge_count"
                ],
                "calibration_precedence_acyclic": seed_certificates["calibration"][
                    "acyclic"
                ],
                "calibration_precedence_unique": seed_certificates["calibration"][
                    "unique_topological_order"
                ],
                "calibration_certificate_sha256": seed_certificates["calibration"][
                    "certificate_sha256"
                ],
                "full_core_precedence_edge_count": seed_certificates["full_core"][
                    "edge_count"
                ],
                "full_core_precedence_acyclic": seed_certificates["full_core"]["acyclic"],
                "full_core_precedence_unique": seed_certificates["full_core"][
                    "unique_topological_order"
                ],
                "full_core_certificate_sha256": seed_certificates["full_core"][
                    "certificate_sha256"
                ],
                "tone_channel_rank": order.index(TONE_CHANNEL) + 1,
                "s4_tone_selections": s4.tone_selected,
                "s4_tone_selection_fraction": s4.tone_selected / s4.attempted,
                "s4_acceptance": s4.accepted / s4.attempted,
                "s4_tone_acceptance": (
                    s4.tone_accepted / s4.tone_selected if s4.tone_selected else math.nan
                ),
            }
            | _counterfactual_audit(
                seed,
                order,
                checkpoint_root,
                dqn_agent_class,
            )
        )

    source_orders = [fitted[seed] for seed in TRAINING_SEEDS]
    rotations: list[list[tuple[int, ...]]] = []
    for rotation in range(len(source_orders)):
        rotations.append(
            [source_orders[(node + rotation) % len(source_orders)] for node in range(N_NODES)]
        )

    total_calibration = sum(row["calibration_attempts"] for row in rows)
    total_heldout = sum(row["heldout_attempts"] for row in rows)
    total_s4 = sum(row["s4_attempts"] for row in rows)
    total_cal_matches = sum(row["calibration_matches"] for row in rows)
    total_val_matches = sum(row["heldout_matches"] for row in rows)
    total_s4_matches = sum(row["s4_matches"] for row in rows)
    total_core = sum(row["full_core_attempts"] for row in rows)
    total_core_matches = sum(row["full_core_matches"] for row in rows)

    analytical: dict[str, object] = {}
    for label, occupancy in (("S1", 0.30),):
        analytical[label] = {
            "occupancy_probability": occupancy,
            "independent_masks": {
                "fixed_priority_expected_unique_channels": _fixed_priority_unique_channels(
                    occupancy
                ),
                "uniform_legal_expected_unique_channels": _uniform_legal_unique_channels(occupancy),
                "fitted_heterogeneous_rotation_expected_unique_channels": sum(
                    _independent_mask_population_unique_channels(rotation, occupancy)
                    for rotation in rotations
                )
                / len(rotations),
            },
            "shared_mask_destructive_collision": {
                "homogeneous_fitted_priority": _shared_mask_metrics(
                    [source_orders[0]] * N_NODES, occupancy
                ),
                "fitted_heterogeneous_rotation_mean": {
                    key: sum(
                        _shared_mask_metrics(rotation, occupancy)[key] for rotation in rotations
                    )
                    / len(rotations)
                    for key in _shared_mask_metrics(rotations[0], occupancy)
                },
                "cyclic_node_priority": _shared_mask_metrics(_cyclic_channel_orders(), occupancy),
                "uniform_random_legal": _shared_mask_uniform_random_metrics(occupancy),
            },
        }
    s4_occupancy = _s4_stationary_occupancy()
    analytical["S4"] = {
        "configured_continuous_time_target_occupancy": (
            S4_MEAN_ON_SECONDS / (S4_MEAN_ON_SECONDS + S4_MEAN_OFF_SECONDS)
        ),
        "implemented_discrete_time_stationary_occupancy": s4_occupancy,
        "step_seconds": S4_STEP_SECONDS,
        "mean_on_seconds": S4_MEAN_ON_SECONDS,
        "mean_off_seconds": S4_MEAN_OFF_SECONDS,
        "predicted_acceptance_by_tone_rank": {
            str(rank): _s4_acceptance_for_tone_rank(rank, s4_occupancy)
            for rank in range(1, N_CHANNELS + 1)
        },
    }
    s1_occupancy = 0.30
    load_sweep: list[dict[str, object]] = []
    for offered_load in (index / 20.0 for index in range(1, 21)):
        load_sweep.append(
            {
                "offered_load_probability": offered_load,
                "homogeneous_fitted_priority": _shared_mask_metrics(
                    [source_orders[0]] * N_NODES, s1_occupancy, offered_load
                ),
                "fitted_heterogeneous_rotation_mean": {
                    key: sum(
                        _shared_mask_metrics(rotation, s1_occupancy, offered_load)[key]
                        for rotation in rotations
                    )
                    / len(rotations)
                    for key in (
                        "expected_attempts_per_cycle",
                        "expected_unique_channels_per_cycle",
                        "expected_singleton_successes_per_cycle",
                        "destructive_collision_delivery_fraction",
                        "colliding_attempt_fraction",
                        "probability_all_channels_occupied",
                    )
                }
                | {"offered_load_probability": offered_load},
                "uniform_random_legal": _shared_mask_uniform_random_metrics(
                    s1_occupancy, offered_load
                ),
            }
        )
    analytical["S1_shared_mask_offered_load_sweep"] = load_sweep

    result: dict[str, object] = {
        "audit_definition": {
            "calibration_scenarios": ["S1", "S2"],
            "calibration_environment_seeds": list(CALIBRATION_ENVS),
            "heldout_scenarios": ["S1", "S2"],
            "heldout_environment_seeds": list(VALIDATION_ENVS),
            "cross_scenario_check": "S4",
            "cross_scenario_environment_seeds": list(ALL_CORE_ENVS),
            "full_core_scenarios": list(ALL_CORE_SCENARIOS),
            "full_core_environment_seeds": list(ALL_CORE_ENVS),
            "fit": "exhaustive 6! priority-order agreement on legal masks and actions",
            "exact_certificate": (
                "precedence DAG over channel labels; acyclicity establishes existence and "
                "a unique topological order establishes unique representability"
            ),
            "abstentions": "excluded from physical-channel agreement; forced by empty legal set",
        },
        "aggregate": {
            "calibration_attempts": total_calibration,
            "calibration_matches": total_cal_matches,
            "calibration_agreement": total_cal_matches / total_calibration,
            "heldout_attempts": total_heldout,
            "heldout_matches": total_val_matches,
            "heldout_agreement": total_val_matches / total_heldout,
            "s4_attempts": total_s4,
            "s4_matches": total_s4_matches,
            "s4_agreement": total_s4_matches / total_s4,
            "full_core_attempts": total_core,
            "full_core_matches": total_core_matches,
            "full_core_agreement": total_core_matches / total_core,
        },
        "per_checkpoint": rows,
        "precedence_certificates": precedence_certificates,
        "analytical_references": analytical,
        "interpretation_boundary": (
            "Agreement establishes masked static-priority equivalence on the audited action-log "
            "support. Shared-mask collision calculations are exact consequences of the fitted "
            "rules under the stated counterfactual model, not packet measurements from the "
            "evaluated simulator."
        ),
    }

    _write_csv(rows, output_dir / "priority_collapse_audit.csv")
    json_path = output_dir / "priority_collapse_audit.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="ChainRadioNet checkout containing ai/agents/dqn_agent.py",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help="core raw-trace directory containing scenario folders S1--S5",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="checkpoint directory containing seed_<seed> folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for priority_collapse_audit.csv and .json",
    )
    args = parser.parse_args()
    try:
        result = run(
            project_root=args.project_root.expanduser().resolve(),
            trace_root=args.trace_root.expanduser().resolve(),
            checkpoint_root=args.checkpoint_root.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"Wrote {args.output_dir / 'priority_collapse_audit.csv'}")
    print(f"Wrote {args.output_dir / 'priority_collapse_audit.json'}")


if __name__ == "__main__":
    main()
