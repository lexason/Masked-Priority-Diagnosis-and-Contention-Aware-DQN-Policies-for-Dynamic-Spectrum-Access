"""Frozen design constants for the prospective collision-aware study.

This namespace is intentionally independent of the historical simulator and of
every prior/parallel prospective study.  Production commands must verify the
source manifest written by :mod:`provenance` before creating outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

STUDY_ID: Final = "CRN-CA-PR1-20261017"
PROTOCOL_VERSION: Final = "2.0.0"
PACKAGE_DIR: Final = Path(__file__).resolve().parent
REPOSITORY_ROOT: Final = PACKAGE_DIR.parents[2]
DEFAULT_RESULTS_ROOT: Final = REPOSITORY_ROOT / "results" / "crn_ca_pr1_20261017"
FREEZE_PATH: Final = PACKAGE_DIR / "PROTOCOL_FREEZE.json"

N_NODES: Final = 15
N_CHANNELS: Final = 6
ABSTAIN_ACTION: Final = N_CHANNELS
N_ACTIONS: Final = N_CHANNELS + 1

# Twenty independent optimisation initialisations. Environment streams are in
# separate 41M/52M namespaces and can never equal these model seeds.
TRAINING_SEEDS: Final = (
    71_101,
    71_209,
    71_317,
    71_423,
    71_531,
    71_639,
    71_747,
    71_851,
    71_959,
    72_067,
    72_173,
    72_281,
    72_389,
    72_497,
    72_601,
    72_709,
    72_817,
    72_923,
    73_031,
    73_139,
)
TRAIN_ENVIRONMENT_BASE: Final = 41_000_000
EVALUATION_SEEDS: Final = (
    52_010_101,
    52_010_211,
    52_010_307,
    52_010_419,
    52_010_503,
    52_010_613,
    52_010_709,
    52_010_821,
    52_010_907,
    52_011_019,
    52_011_103,
    52_011_217,
    52_011_301,
    52_011_409,
    52_011_503,
    52_011_619,
    52_011_707,
    52_011_811,
    52_011_919,
    52_012_027,
    52_012_119,
    52_012_233,
    52_012_307,
    52_012_419,
)
SMOKE_MODEL_SEED: Final = 91_900_001
SMOKE_ENVIRONMENT_SEED: Final = 91_900_101
ANALYSIS_BOOTSTRAP_SEED: Final = 20_261_017

TRAIN_EPISODES: Final = 400
TRAIN_STEPS_PER_EPISODE: Final = 250
EVAL_BURN_IN_STEPS: Final = 300
EVAL_MEASUREMENT_STEPS: Final = 3_000
SMOKE_EPISODES: Final = 2
SMOKE_STEPS_PER_EPISODE: Final = 40
SMOKE_EVAL_BURN_IN_STEPS: Final = 10
SMOKE_EVAL_MEASUREMENT_STEPS: Final = 50

REPLAY_CAPACITY: Final = 100_000
BATCH_SIZE: Final = 128
LEARNING_RATE: Final = 5e-4
GAMMA: Final = 0.97
TARGET_UPDATE_STEPS: Final = 500
EPSILON_START: Final = 1.0
EPSILON_END: Final = 0.05
HIDDEN_DIM: Final = 128

REWARD_DELIVERED: Final = 1.0
REWARD_COLLISION: Final = -0.50
REWARD_LINK_FAILURE: Final = -0.50
REWARD_VOLUNTARY_DEFER: Final = -0.02
REWARD_FORCED_OR_NO_PACKET: Final = 0.0
HISTORY_ALPHA: Final = 0.10
MARKOV_DEPARTURE_PROBABILITY: Final = 0.05

LEARNED_POLICY_NAME: Final = "collision_aware_dqn"
HISTORY_FREE_POLICY_NAME: Final = "history_free_dqn"
COORDINATION_BLIND_POLICY_NAME: Final = "coordination_blind_dqn"
LEARNED_POLICIES: Final = (
    LEARNED_POLICY_NAME,
    HISTORY_FREE_POLICY_NAME,
    COORDINATION_BLIND_POLICY_NAME,
)
PRIMARY_BASELINE_NAME: Final = "phase_orthogonal"
ORACLE_POLICY_NAME: Final = "centralized_quality_matching"
DECENTRALIZED_BASELINES: Final = (
    PRIMARY_BASELINE_NAME,
    "quality_greedy_persistent",
    "round_robin",
    "uniform_legal",
    "static_priority",
)
BASELINES: Final = (*DECENTRALIZED_BASELINES, ORACLE_POLICY_NAME)
ALL_POLICIES: Final = (*LEARNED_POLICIES, *BASELINES)

# Inference and finite-support mechanism-audit constants.  The 24 environment
# seeds form four complete six-shift cycles; each half therefore retains two
# complete cycles for calibration/validation of the priority-order audit.
BOOTSTRAP_REPLICATES: Final = 10_000
BOOTSTRAP_LOWER_QUANTILE: Final = 0.05
BOOTSTRAP_UPPER_QUANTILE: Final = 0.95
PRIMARY_GOODPUT_MARGIN: Final = 0.05
HISTORY_ABLATION_MARGIN: Final = 0.0
PRIORITY_AGREEMENT_THRESHOLD: Final = 0.95
PRIORITY_CALIBRATION_INDICES: Final = tuple(range(12))
PRIORITY_VALIDATION_INDICES: Final = tuple(range(12, 24))


@dataclass(frozen=True, slots=True)
class Factor:
    """One prespecified evaluation cell."""

    factor_id: str
    occupancy_process: str
    occupancy_probability: float
    arrival_probability: float


FACTORS: Final = tuple(
    Factor(
        factor_id=f"{process}_o{int(occupancy * 100)}_a{int(arrival * 100)}",
        occupancy_process=process,
        occupancy_probability=occupancy,
        arrival_probability=arrival,
    )
    for process in ("iid", "markov")
    for occupancy in (0.30, 0.60)
    for arrival in (0.10, 0.25)
)

# Observation layout: availability K, quality K, collision EWMA K,
# delivery EWMA K, previous action K+1, previous outcome 5, queue 1,
# node one-hot N, common slot phase one-hot N.
OUTCOME_CLASSES: Final = 5
STATE_DIM: Final = 4 * N_CHANNELS + N_ACTIONS + OUTCOME_CLASSES + 1 + 2 * N_NODES

SOURCE_FILENAMES: Final = (
    "PROTOCOL.md",
    "READINESS_REPORT.md",
    "README.md",
    "__init__.py",
    "analyze.py",
    "config.py",
    "environment.py",
    "evaluate.py",
    "model.py",
    "plot.py",
    "policies.py",
    "provenance.py",
    "test_evaluation_v2.py",
    "test_integrity.py",
    "test_inference.py",
    "test_pipeline.py",
    "train.py",
    "validation.py",
)

DEPENDENCY_FILENAMES: Final = ("pyproject.toml", "uv.lock")


def factor_by_id(factor_id: str) -> Factor:
    """Return a frozen factor by ID, failing on an unregistered cell."""
    matches = [factor for factor in FACTORS if factor.factor_id == factor_id]
    if len(matches) != 1:
        raise ValueError(f"unknown factor_id: {factor_id}")
    return matches[0]


def label_permutation(replication_index: int) -> tuple[int, ...]:
    """Map logical actions to physical channels using balanced cyclic shifts."""
    if not 0 <= replication_index < len(EVALUATION_SEEDS):
        raise ValueError("replication_index outside prespecified evaluation seeds")
    shift = replication_index % N_CHANNELS
    return tuple((logical + shift) % N_CHANNELS for logical in range(N_CHANNELS))


def training_environment_seed(training_seed_index: int, episode: int) -> int:
    """Return an environment seed in the disjoint training namespace."""
    if not 0 <= training_seed_index < len(TRAINING_SEEDS):
        raise ValueError("training_seed_index outside prespecified seed set")
    if not 0 <= episode < TRAIN_EPISODES:
        raise ValueError("episode outside prespecified production schedule")
    return TRAIN_ENVIRONMENT_BASE + training_seed_index * 100_000 + episode


def protocol_payload() -> dict[str, object]:
    """Return the machine-comparable preregistered design payload."""
    return {
        "study_id": STUDY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "nodes": N_NODES,
        "channels": N_CHANNELS,
        "action_count": N_ACTIONS,
        "state_dim": STATE_DIM,
        "training_seeds": list(TRAINING_SEEDS),
        "train_environment_base": TRAIN_ENVIRONMENT_BASE,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "train_episodes": TRAIN_EPISODES,
        "train_steps_per_episode": TRAIN_STEPS_PER_EPISODE,
        "eval_burn_in_steps": EVAL_BURN_IN_STEPS,
        "eval_measurement_steps": EVAL_MEASUREMENT_STEPS,
        "factors": [asdict(factor) for factor in FACTORS],
        "baselines": list(BASELINES),
        "decentralized_baselines": list(DECENTRALIZED_BASELINES),
        "oracle_policy": ORACLE_POLICY_NAME,
        "learned_policies": list(LEARNED_POLICIES),
        "balanced_label_permutations": [
            list(label_permutation(index)) for index in range(len(EVALUATION_SEEDS))
        ],
        "optimisation": {
            "replay_capacity": REPLAY_CAPACITY,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "target_update_steps": TARGET_UPDATE_STEPS,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "hidden_dim": HIDDEN_DIM,
            "checkpoint_selection": "final_iterate_only",
            "replay": "all_15_nodes_each_slot_uniform",
        },
        "reward": {
            "delivered": REWARD_DELIVERED,
            "collision": REWARD_COLLISION,
            "link_failure": REWARD_LINK_FAILURE,
            "voluntary_defer": REWARD_VOLUNTARY_DEFER,
            "forced_or_no_packet": REWARD_FORCED_OR_NO_PACKET,
        },
        "dynamics": {
            "history_alpha": HISTORY_ALPHA,
            "markov_departure_probability": MARKOV_DEPARTURE_PROBABILITY,
        },
        "observation_ablations": {
            HISTORY_FREE_POLICY_NAME: (
                "zero collision EWMA, delivery EWMA, and previous outcome; "
                "retain previous action, node identity, and common phase"
            ),
            COORDINATION_BLIND_POLICY_NAME: (
                "zero node identity and common phase; retain collision and delivery history"
            ),
        },
        "inference": {
            "estimand": (
                "equal-cell macro mean over paired training seeds, fixed factorial cells, "
                "and crossed environment-seed blocks"
            ),
            "bootstrap": "stratified_crossed_two_factor",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "one_sided_lower_quantile": BOOTSTRAP_LOWER_QUANTILE,
            "one_sided_upper_quantile": BOOTSTRAP_UPPER_QUANTILE,
            "primary_goodput_comparator": PRIMARY_BASELINE_NAME,
            "primary_goodput_margin_packets_per_slot": PRIMARY_GOODPUT_MARGIN,
            "history_ablation_margin_packets_per_slot": HISTORY_ABLATION_MARGIN,
            "priority_agreement_threshold": PRIORITY_AGREEMENT_THRESHOLD,
            "priority_calibration_indices": list(PRIORITY_CALIBRATION_INDICES),
            "priority_validation_indices": list(PRIORITY_VALIDATION_INDICES),
            "seed_level_test": "exact_one_sided_sign_flip",
        },
        "bootstrap_seed": ANALYSIS_BOOTSTRAP_SEED,
    }


def protocol_sha256() -> str:
    """Return a canonical digest of all outcome-independent design constants."""
    encoded = json.dumps(protocol_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
