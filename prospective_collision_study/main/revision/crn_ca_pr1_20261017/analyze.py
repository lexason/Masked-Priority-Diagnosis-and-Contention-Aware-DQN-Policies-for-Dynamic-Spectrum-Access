"""Preregistered crossed-block analysis for the collision-aware study.

The inferential units are independently trained final checkpoints and paired
environment-seed blocks. Factorial cells and channel-label shifts are fixed
design strata; slot and node observations are never analysed as replicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from main.revision.crn_ca_pr1_20261017.config import (
    ABSTAIN_ACTION,
    ALL_POLICIES,
    ANALYSIS_BOOTSTRAP_SEED,
    BASELINES,
    BOOTSTRAP_LOWER_QUANTILE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_UPPER_QUANTILE,
    COORDINATION_BLIND_POLICY_NAME,
    DECENTRALIZED_BASELINES,
    DEFAULT_RESULTS_ROOT,
    EVALUATION_SEEDS,
    FACTORS,
    HISTORY_ABLATION_MARGIN,
    HISTORY_FREE_POLICY_NAME,
    LEARNED_POLICIES,
    LEARNED_POLICY_NAME,
    N_ACTIONS,
    N_CHANNELS,
    ORACLE_POLICY_NAME,
    PRIMARY_BASELINE_NAME,
    PRIMARY_GOODPUT_MARGIN,
    PRIORITY_AGREEMENT_THRESHOLD,
    PRIORITY_CALIBRATION_INDICES,
    PRIORITY_VALIDATION_INDICES,
    PROTOCOL_VERSION,
    STUDY_ID,
    TRAINING_SEEDS,
)
from main.revision.crn_ca_pr1_20261017.provenance import verify_freeze
from main.revision.crn_ca_pr1_20261017.validation import validate

PRIMARY_METRIC = "network_goodput"
SECONDARY_METRIC = "finite_window_service_fraction"
ANALYSED_METRICS = (PRIMARY_METRIC, SECONDARY_METRIC)
SIGN_FLIP_ALPHA = 0.05

_ORDERS = np.asarray(
    list(itertools.permutations(range(N_CHANNELS))), dtype=np.int8
)
_MASKS = np.arange(1 << N_CHANNELS, dtype=np.int64)
_AVAILABLE = ((_MASKS[:, None] >> np.arange(N_CHANNELS)) & 1).astype(bool)
_ORDER_PREDICTIONS = np.full(
    (len(_ORDERS), len(_MASKS)), ABSTAIN_ACTION, dtype=np.int8
)
for _order_index, _order in enumerate(_ORDERS):
    for _mask in range(1, 1 << N_CHANNELS):
        _ORDER_PREDICTIONS[_order_index, _mask] = next(
            int(channel) for channel in _order if _mask & (1 << int(channel))
        )


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    """One reusable crossed, channel-shift-stratified resampling plan."""

    training_indices: np.ndarray
    environment_indices: np.ndarray

    @property
    def replicates(self) -> int:
        return int(self.training_indices.shape[0])


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty evaluation summary: {path}")
    return rows


def _row_indices(row: dict[str, str]) -> tuple[int, int]:
    factor_lookup = {factor.factor_id: index for index, factor in enumerate(FACTORS)}
    environment_lookup = {
        int(seed): index for index, seed in enumerate(EVALUATION_SEEDS)
    }
    try:
        factor_index = factor_lookup[row["factor_id"]]
        environment_index = environment_lookup[int(row["environment_seed"])]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("row contains an unregistered factor/environment key") from error
    if (
        "replication_index" in row
        and row["replication_index"] != ""
        and int(row["replication_index"]) != environment_index
    ):
        raise ValueError("replication index does not match evaluation-seed order")
    return factor_index, environment_index


def _metric_cubes(
    rows: list[dict[str, str]], metric: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Construct exact learned and baseline cubes, rejecting any key drift."""
    if not rows or metric not in rows[0]:
        raise ValueError(f"evaluation rows do not expose required metric: {metric}")
    shape = (len(TRAINING_SEEDS), len(FACTORS), len(EVALUATION_SEEDS))
    learned = {
        policy: np.full(shape, np.nan, dtype=np.float64)
        for policy in LEARNED_POLICIES
    }
    baselines = {
        policy: np.full((len(FACTORS), len(EVALUATION_SEEDS)), np.nan)
        for policy in BASELINES
    }
    training_lookup = {int(seed): index for index, seed in enumerate(TRAINING_SEEDS)}

    for row in rows:
        policy = row.get("policy", "")
        if policy not in ALL_POLICIES:
            raise ValueError(f"unregistered policy in evaluation rows: {policy!r}")
        factor_index, environment_index = _row_indices(row)
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid {metric} in {row.get('run_id', '<unknown>')}"
            ) from error
        if not np.isfinite(value):
            raise ValueError(f"non-finite {metric} in {row.get('run_id', '<unknown>')}")

        training_raw = str(row.get("training_seed", "")).strip()
        if policy in LEARNED_POLICIES:
            try:
                training_index = training_lookup[int(training_raw)]
            except (KeyError, ValueError) as error:
                raise ValueError(f"invalid training seed for learned policy {policy}") from error
            destination = learned[policy]
            if np.isfinite(destination[training_index, factor_index, environment_index]):
                raise ValueError(
                    f"duplicate learned cube cell: {policy}, {training_raw}, "
                    f"{row['factor_id']}, {row['environment_seed']}"
                )
            destination[training_index, factor_index, environment_index] = value
        else:
            if training_raw:
                raise ValueError(f"baseline {policy} unexpectedly has a training seed")
            destination = baselines[policy]
            if np.isfinite(destination[factor_index, environment_index]):
                raise ValueError(
                    f"duplicate baseline cube cell: {policy}, {row['factor_id']}, "
                    f"{row['environment_seed']}"
                )
            destination[factor_index, environment_index] = value

    incomplete = [
        name
        for name, values in (*learned.items(), *baselines.items())
        if np.any(~np.isfinite(values))
    ]
    if incomplete:
        raise ValueError(f"incomplete {metric} cubes: {incomplete}")
    return learned, baselines


def _make_bootstrap_plan(
    *,
    rng: np.random.Generator,
    replicates: int,
    allowed_environment_indices: Iterable[int] | None = None,
) -> BootstrapPlan:
    """Resample training seeds and crossed environment blocks within shifts."""
    if replicates <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    if allowed_environment_indices is None:
        allowed = np.arange(len(EVALUATION_SEEDS), dtype=np.int64)
    else:
        allowed = np.asarray(tuple(allowed_environment_indices), dtype=np.int64)
    if allowed.ndim != 1 or allowed.size == 0 or len(np.unique(allowed)) != allowed.size:
        raise ValueError("environment bootstrap indices must be unique and nonempty")
    if np.any((allowed < 0) | (allowed >= len(EVALUATION_SEEDS))):
        raise ValueError("environment bootstrap index outside the registered seed set")

    strata = [allowed[allowed % N_CHANNELS == shift] for shift in range(N_CHANNELS)]
    if any(stratum.size == 0 for stratum in strata):
        raise ValueError("every channel-label shift needs an environment stratum")
    if len({int(stratum.size) for stratum in strata}) != 1:
        raise ValueError("channel-label shift strata must be exactly balanced")

    training_indices = rng.integers(
        0,
        len(TRAINING_SEEDS),
        size=(replicates, len(TRAINING_SEEDS)),
        dtype=np.int64,
    )
    environment_indices = np.concatenate(
        [
            rng.choice(stratum, size=(replicates, stratum.size), replace=True)
            for stratum in strata
        ],
        axis=1,
    ).astype(np.int64, copy=False)
    return BootstrapPlan(training_indices, environment_indices)


def _effect_draws(
    treatment: np.ndarray,
    comparator: np.ndarray,
    plan: BootstrapPlan,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell points and paired crossed-bootstrap draws for an effect."""
    expected = (len(TRAINING_SEEDS), len(FACTORS), len(EVALUATION_SEEDS))
    if treatment.shape != expected:
        raise ValueError(f"unexpected treatment cube shape: {treatment.shape}")
    if comparator.shape == expected:
        difference = treatment - comparator
    elif comparator.shape == expected[1:]:
        difference = treatment - comparator[None, :, :]
    else:
        raise ValueError(f"unexpected comparator cube shape: {comparator.shape}")
    cell_points = np.mean(difference, axis=(0, 2))
    draws = np.empty((plan.replicates, len(FACTORS)), dtype=np.float64)
    for draw_index in range(plan.replicates):
        selected = difference[plan.training_indices[draw_index]]
        selected = selected[:, :, plan.environment_indices[draw_index]]
        draws[draw_index] = np.mean(selected, axis=(0, 2))
    return cell_points, draws


def _bounds(draws: np.ndarray, *, axis: int | None = None) -> tuple[Any, Any]:
    lower = np.quantile(draws, BOOTSTRAP_LOWER_QUANTILE, axis=axis)
    upper = np.quantile(draws, BOOTSTRAP_UPPER_QUANTILE, axis=axis)
    return lower, upper


def _exact_sign_flip_p(values: np.ndarray, *, margin: float = 0.0) -> float:
    """Exact one-sided sign-flip p-value for a positive mean over a margin."""
    centered = np.asarray(values, dtype=np.float64) - float(margin)
    if centered.ndim != 1 or centered.size == 0 or np.any(~np.isfinite(centered)):
        raise ValueError("sign-flip inputs must be a finite nonempty vector")
    if centered.size > 24:
        raise ValueError("exact sign-flip enumeration is limited to 24 paired seeds")
    distribution = np.asarray([0.0], dtype=np.float64)
    for value in centered:
        distribution = np.concatenate((distribution + value, distribution - value))
    observed = float(np.sum(centered))
    tolerance = np.finfo(np.float64).eps * 64.0 * max(
        1.0, abs(observed), float(np.sum(np.abs(centered)))
    )
    return float(np.count_nonzero(distribution >= observed - tolerance) / distribution.size)


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p-values for one declared family."""
    if not p_values:
        return {}
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("Holm family contains an invalid p-value")
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def _parse_mask_action_counts(raw: str) -> np.ndarray:
    """Parse the frozen 64-mask x 7-action row-major integer encoding."""
    tokens = [token.strip() for token in str(raw).split("-")]
    expected = (1 << N_CHANNELS) * N_ACTIONS
    if len(tokens) != expected or any(not token.isdigit() for token in tokens):
        raise ValueError(f"mask_action_counts must contain exactly {expected} integers")
    try:
        values = np.asarray([int(token) for token in tokens], dtype=np.int64)
    except (OverflowError, ValueError) as error:
        raise ValueError("mask_action_counts contains an invalid integer") from error
    if np.any(values < 0):
        raise ValueError("mask_action_counts cannot be negative")
    counts = values.reshape(1 << N_CHANNELS, N_ACTIONS)
    if np.any(counts[0] != 0):
        raise ValueError("eligible-decision counts cannot use the empty mask")
    if np.any(counts[:, :N_CHANNELS][~_AVAILABLE]):
        raise ValueError("mask_action_counts records an unavailable channel action")
    return counts


def _mask_count_cube(rows: list[dict[str, str]]) -> np.ndarray:
    """Construct the exact full-DQN mask/action count cube."""
    if not rows or "mask_action_counts" not in rows[0]:
        raise ValueError("evaluation rows do not expose mask_action_counts")
    cube = np.full(
        (
            len(TRAINING_SEEDS),
            len(FACTORS),
            len(EVALUATION_SEEDS),
            1 << N_CHANNELS,
            N_ACTIONS,
        ),
        -1,
        dtype=np.int64,
    )
    training_lookup = {int(seed): index for index, seed in enumerate(TRAINING_SEEDS)}
    for row in rows:
        if row.get("policy") != LEARNED_POLICY_NAME:
            continue
        factor_index, environment_index = _row_indices(row)
        try:
            training_index = training_lookup[int(row["training_seed"])]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid training seed in mechanism rows") from error
        if np.any(cube[training_index, factor_index, environment_index] >= 0):
            raise ValueError("duplicate full-DQN mechanism cube cell")
        counts = _parse_mask_action_counts(row["mask_action_counts"])
        eligible_raw = str(row.get("priority_eligible_decisions", "")).strip()
        if eligible_raw and int(eligible_raw) != int(np.sum(counts)):
            raise ValueError("mask/action total differs from eligible-decision count")
        cube[training_index, factor_index, environment_index] = counts
    if np.any(cube < 0):
        raise ValueError("incomplete full-DQN mechanism cube")
    return cube


def _order_scores(group_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score all 720 orders, giving every supplied group equal weight."""
    counts = np.asarray(group_counts, dtype=np.int64)
    if counts.ndim == 2:
        counts = counts[None, :, :]
    if counts.ndim != 3 or counts.shape[1:] != (1 << N_CHANNELS, N_ACTIONS):
        raise ValueError("priority counts need shape (groups, 64, 7)")
    totals = np.sum(counts, axis=(1, 2))
    if np.any(totals <= 0):
        raise ValueError("every priority-agreement group needs eligible decisions")
    matches = np.empty((len(_ORDERS), counts.shape[0]), dtype=np.int64)
    masks = np.arange(1 << N_CHANNELS)[None, :]
    for group_index, group in enumerate(counts):
        matches[:, group_index] = np.sum(
            group[masks, _ORDER_PREDICTIONS], axis=1, dtype=np.int64
        )
    scores = np.mean(matches / totals[None, :], axis=1)
    return scores, np.sum(matches, axis=1, dtype=np.int64)


def _best_order(group_counts: np.ndarray) -> dict[str, Any]:
    scores, pooled_matches = _order_scores(group_counts)
    best_score = float(np.max(scores))
    winners = np.flatnonzero(np.isclose(scores, best_score, rtol=0.0, atol=1e-15))
    selected = int(winners[0])
    return {
        "order_index": selected,
        "order": tuple(int(value) for value in _ORDERS[selected]),
        "winner_indices": tuple(int(value) for value in winners),
        "equal_group_agreement": best_score,
        "pooled_matches": int(pooled_matches[selected]),
        "tie_count": int(winners.size),
    }


def _precedence_certificate(counts: np.ndarray) -> dict[str, Any]:
    """Certify finite-support representability by any masked channel order."""
    if counts.shape != (1 << N_CHANNELS, N_ACTIONS):
        raise ValueError("precedence certificate requires pooled (64, 7) counts")
    edge_support = np.zeros((N_CHANNELS, N_CHANNELS), dtype=np.int64)
    eligible_abstentions = int(np.sum(counts[1:, ABSTAIN_ACTION]))
    for mask in range(1, 1 << N_CHANNELS):
        available = np.flatnonzero(_AVAILABLE[mask])
        for selected in available:
            support = int(counts[mask, selected])
            if support == 0:
                continue
            for alternative in available:
                if alternative != selected:
                    edge_support[selected, alternative] += support
    adjacency = edge_support > 0
    indegree = np.sum(adjacency, axis=0).astype(np.int64)
    frontier = sorted(int(node) for node in np.flatnonzero(indegree == 0))
    topological_order: list[int] = []
    unique = True
    while frontier:
        if len(frontier) != 1:
            unique = False
        node = frontier.pop(0)
        topological_order.append(node)
        for target in np.flatnonzero(adjacency[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(int(target))
                frontier.sort()
    acyclic = len(topological_order) == N_CHANNELS
    if not acyclic:
        unique = False
        topological_order = []
    edges = [
        {
            "preferred": source,
            "over": target,
            "support": int(edge_support[source, target]),
        }
        for source in range(N_CHANNELS)
        for target in range(N_CHANNELS)
        if edge_support[source, target] > 0
    ]
    return {
        "eligible_abstentions": eligible_abstentions,
        "edge_count": len(edges),
        "edges": edges,
        "acyclic": acyclic,
        "unique_topological_order": bool(acyclic and unique),
        "topological_order": topological_order,
        "exact_priority_representable": bool(acyclic and eligible_abstentions == 0),
        "counts_sha256": hashlib.sha256(
            counts.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }


def _mechanism_analysis(
    mask_cube: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration = np.asarray(PRIORITY_CALIBRATION_INDICES, dtype=np.int64)
    validation_indices = np.asarray(PRIORITY_VALIDATION_INDICES, dtype=np.int64)
    if set(calibration) & set(validation_indices):
        raise ValueError("priority calibration and validation indices overlap")
    if set(calibration) | set(validation_indices) != set(range(len(EVALUATION_SEEDS))):
        raise ValueError("priority split must partition the evaluation seeds")
    for subset in (calibration, validation_indices):
        counts = np.bincount(subset % N_CHANNELS, minlength=N_CHANNELS)
        if len({int(value) for value in counts}) != 1:
            raise ValueError("priority split is not channel-shift balanced")

    selected_matches = np.zeros(mask_cube.shape[:3], dtype=np.int64)
    eligible_totals = np.sum(mask_cube, axis=(-1, -2), dtype=np.int64)
    certificate_rows: list[dict[str, Any]] = []
    per_checkpoint_scores: list[float] = []
    validation_best_scores: list[float] = []
    structural_departures: list[bool] = []

    for training_index, training_seed in enumerate(TRAINING_SEEDS):
        by_factor_calibration = np.sum(
            np.take(mask_cube[training_index], calibration, axis=1),
            axis=1,
            dtype=np.int64,
        )
        by_factor_validation = np.sum(
            np.take(mask_cube[training_index], validation_indices, axis=1),
            axis=1,
            dtype=np.int64,
        )
        by_factor_full = np.sum(mask_cube[training_index], axis=1, dtype=np.int64)
        calibration_fit = _best_order(by_factor_calibration)
        _, calibration_pooled_matches = _order_scores(by_factor_calibration)
        validation_scores, validation_pooled_matches = _order_scores(
            by_factor_validation
        )
        calibration_winners = np.asarray(
            calibration_fit["winner_indices"], dtype=np.int64
        )
        conservative_score = float(np.max(validation_scores[calibration_winners]))
        conservative_candidates = calibration_winners[
            np.isclose(
                validation_scores[calibration_winners],
                conservative_score,
                rtol=0.0,
                atol=1e-15,
            )
        ]
        selected_order_index = int(conservative_candidates[0])
        selected_order = tuple(int(value) for value in _ORDERS[selected_order_index])
        validation_score = {
            "equal_group_agreement": conservative_score,
            "pooled_matches": int(validation_pooled_matches[selected_order_index]),
        }
        validation_best = _best_order(by_factor_validation)
        full_best = _best_order(by_factor_full)
        pooled_full = np.sum(by_factor_full, axis=0, dtype=np.int64)
        certificate = _precedence_certificate(pooled_full)

        prediction = _ORDER_PREDICTIONS[selected_order_index]
        selected = np.take_along_axis(
            mask_cube[training_index],
            prediction.reshape(1, 1, -1, 1),
            axis=3,
        ).squeeze(axis=3)
        selected_matches[training_index] = np.sum(selected, axis=2, dtype=np.int64)

        score = float(validation_score["equal_group_agreement"])
        validation_best_score = float(validation_best["equal_group_agreement"])
        structural_departure = not bool(certificate["exact_priority_representable"])
        per_checkpoint_scores.append(score)
        validation_best_scores.append(validation_best_score)
        structural_departures.append(structural_departure)
        checkpoint_pass = (
            structural_departure
            and validation_best_score < PRIORITY_AGREEMENT_THRESHOLD
        )
        certificate_rows.append(
            {
                "training_seed": int(training_seed),
                "calibration_best_order": "-".join(
                    str(value) for value in selected_order
                ),
                "calibration_best_equal_cell_agreement": calibration_fit[
                    "equal_group_agreement"
                ],
                "calibration_best_pooled_matches": int(
                    calibration_pooled_matches[selected_order_index]
                ),
                "calibration_eligible_decisions": int(np.sum(by_factor_calibration)),
                "calibration_order_tie_count": calibration_fit["tie_count"],
                "calibration_tied_order_indices_json": json.dumps(
                    calibration_fit["winner_indices"], separators=(",", ":")
                ),
                "validation_agreement_for_calibration_order": score,
                "validation_pooled_matches_for_calibration_order": validation_score[
                    "pooled_matches"
                ],
                "validation_eligible_decisions": int(np.sum(by_factor_validation)),
                "validation_best_order": "-".join(
                    str(value) for value in validation_best["order"]
                ),
                "validation_best_equal_cell_agreement": validation_best_score,
                "validation_best_order_tie_count": validation_best["tie_count"],
                "full_best_order": "-".join(
                    str(value) for value in full_best["order"]
                ),
                "full_best_equal_cell_agreement": full_best[
                    "equal_group_agreement"
                ],
                "full_eligible_decisions": int(np.sum(by_factor_full)),
                "eligible_abstentions": certificate["eligible_abstentions"],
                "precedence_edge_count": certificate["edge_count"],
                "precedence_edges_json": json.dumps(
                    certificate["edges"], sort_keys=True, separators=(",", ":")
                ),
                "precedence_acyclic": certificate["acyclic"],
                "precedence_unique_topological_order": certificate[
                    "unique_topological_order"
                ],
                "precedence_topological_order": "-".join(
                    str(value) for value in certificate["topological_order"]
                ),
                "exact_priority_representable": certificate[
                    "exact_priority_representable"
                ],
                "mask_action_counts_sha256": certificate["counts_sha256"],
                "checkpoint_mechanism_pass": checkpoint_pass,
            }
        )

    plan = _make_bootstrap_plan(
        rng=rng,
        replicates=BOOTSTRAP_REPLICATES,
        allowed_environment_indices=validation_indices,
    )
    mechanism_draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for draw_index in range(BOOTSTRAP_REPLICATES):
        training = plan.training_indices[draw_index]
        environments = plan.environment_indices[draw_index]
        matches = selected_matches[training][:, :, environments]
        totals = eligible_totals[training][:, :, environments]
        cell_totals = np.sum(totals, axis=2)
        if np.any(cell_totals <= 0):
            raise ValueError("mechanism bootstrap encountered an empty cell")
        cell_agreement = np.sum(matches, axis=2) / cell_totals
        mechanism_draws[draw_index] = float(np.mean(cell_agreement))
    mechanism_lcb, mechanism_ucb = _bounds(mechanism_draws)
    point = float(np.mean(per_checkpoint_scores))
    every_checkpoint_below = all(
        value < PRIORITY_AGREEMENT_THRESHOLD for value in validation_best_scores
    )
    every_checkpoint_nonrepresentable = all(structural_departures)
    mechanism_pass = bool(
        every_checkpoint_below
        and every_checkpoint_nonrepresentable
        and mechanism_ucb < PRIORITY_AGREEMENT_THRESHOLD
    )
    summary = {
        "estimand": (
            "equal-training-seed/equal-factor validation agreement of each "
            "calibration-fitted masked-priority order"
        ),
        "calibration_indices": calibration.tolist(),
        "validation_indices": validation_indices.tolist(),
        "agreement_point": point,
        "one_sided_95_lcb": float(mechanism_lcb),
        "one_sided_95_ucb": float(mechanism_ucb),
        "threshold": PRIORITY_AGREEMENT_THRESHOLD,
        "all_validation_best_order_agreements_below_threshold": every_checkpoint_below,
        "all_full_support_certificates_nonrepresentable": (
            every_checkpoint_nonrepresentable
        ),
        "pass": mechanism_pass,
    }
    return certificate_rows, summary


def _comparison_specifications() -> list[dict[str, Any]]:
    specifications: list[dict[str, Any]] = [
        {
            "comparator": PRIMARY_BASELINE_NAME,
            "kind": "baseline",
            "role": "primary_efficacy",
            "margin": PRIMARY_GOODPUT_MARGIN,
            "holm_family": False,
        },
        {
            "comparator": HISTORY_FREE_POLICY_NAME,
            "kind": "learned",
            "role": "primary_local_outcome_history_attribution",
            "margin": HISTORY_ABLATION_MARGIN,
            "holm_family": False,
        },
        {
            "comparator": COORDINATION_BLIND_POLICY_NAME,
            "kind": "learned",
            "role": "secondary_coordination_diagnostic",
            "margin": 0.0,
            "holm_family": False,
        },
    ]
    specifications.extend(
        {
            "comparator": baseline,
            "kind": "baseline",
            "role": "secondary_decentralized_comparator",
            "margin": 0.0,
            "holm_family": True,
        }
        for baseline in DECENTRALIZED_BASELINES
        if baseline != PRIMARY_BASELINE_NAME
    )
    specifications.append(
        {
            "comparator": ORACLE_POLICY_NAME,
            "kind": "baseline",
            "role": "descriptive_centralized_oracle",
            "margin": None,
            "holm_family": False,
        }
    )
    comparators = [str(item["comparator"]) for item in specifications]
    if len(comparators) != len(set(comparators)):
        raise AssertionError("comparison specification contains a duplicate comparator")
    return specifications


def _write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def analyze(results_root: Path) -> dict[str, Any]:
    """Run the preregistered analysis after all fail-closed validation gates."""
    verify_freeze()
    validate(results_root, smoke=False)
    rows = _read_rows(results_root / "evaluation" / "summary.csv")
    analysis_dir = results_root / "analysis"
    output_paths = {
        "summary": analysis_dir / "summary.json",
        "comparisons": analysis_dir / "comparisons.csv",
        "cell_effects": analysis_dir / "cell_effects.csv",
        "checkpoint_effects": analysis_dir / "checkpoint_effects.csv",
        "factorial_means": analysis_dir / "factorial_means.csv",
        "mechanism": analysis_dir / "mechanism_certificates.csv",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite analysis artifacts: {existing}")

    metric_cubes = {
        metric: _metric_cubes(rows, metric) for metric in ANALYSED_METRICS
    }
    rng = np.random.default_rng(ANALYSIS_BOOTSTRAP_SEED)
    plan = _make_bootstrap_plan(rng=rng, replicates=BOOTSTRAP_REPLICATES)
    specifications = _comparison_specifications()
    comparison_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    checkpoint_effect_rows: list[dict[str, Any]] = []
    goodput_by_comparator: dict[str, dict[str, Any]] = {}

    for metric in ANALYSED_METRICS:
        learned, baselines = metric_cubes[metric]
        treatment = learned[LEARNED_POLICY_NAME]
        for specification in specifications:
            comparator_name = str(specification["comparator"])
            comparator = (
                learned[comparator_name]
                if specification["kind"] == "learned"
                else baselines[comparator_name]
            )
            cell_points, cell_draws = _effect_draws(treatment, comparator, plan)
            macro_draws = np.mean(cell_draws, axis=1)
            macro_point = float(np.mean(cell_points))
            treatment_mean = float(np.mean(treatment))
            comparator_mean = float(np.mean(comparator))
            relative_difference: float | str = (
                "" if comparator_mean == 0.0 else macro_point / comparator_mean
            )
            macro_lcb, macro_ucb = _bounds(macro_draws)
            seed_effects = np.mean(
                treatment
                - (comparator if comparator.ndim == 3 else comparator[None, :, :]),
                axis=(1, 2),
            )
            checkpoint_effect_rows.extend(
                {
                    "metric": metric,
                    "comparator": comparator_name,
                    "role": specification["role"],
                    "training_seed": int(training_seed),
                    "paired_macro_difference": float(seed_effects[training_index]),
                }
                for training_index, training_seed in enumerate(TRAINING_SEEDS)
            )
            margin = specification["margin"] if metric == PRIMARY_METRIC else None
            raw_p: float | None = None
            if (
                metric == PRIMARY_METRIC
                and specification["role"] != "descriptive_centralized_oracle"
            ):
                raw_p = _exact_sign_flip_p(
                    seed_effects,
                    margin=float(margin or 0.0),
                )
            comparison_row = {
                "treatment": LEARNED_POLICY_NAME,
                "comparator": comparator_name,
                "comparator_kind": specification["kind"],
                "inferential_role": specification["role"],
                "metric": metric,
                "treatment_mean": treatment_mean,
                "comparator_mean": comparator_mean,
                "difference": macro_point,
                "relative_difference": relative_difference,
                "one_sided_95_lcb": float(macro_lcb),
                "one_sided_95_ucb": float(macro_ucb),
                "superiority_margin": "" if margin is None else float(margin),
                "exact_sign_flip_p_raw": "" if raw_p is None else raw_p,
                "exact_sign_flip_p_holm": "",
                "holm_family_member": bool(
                    metric == PRIMARY_METRIC and specification["holm_family"]
                ),
                "claim_pass": "",
            }
            comparison_rows.append(comparison_row)
            if metric == PRIMARY_METRIC:
                goodput_by_comparator[comparator_name] = {
                    "row": comparison_row,
                    "seed_effects": seed_effects,
                    "raw_p": raw_p,
                }

            cell_lcb, cell_ucb = _bounds(cell_draws, axis=0)
            treatment_cell_means = np.mean(treatment, axis=(0, 2))
            comparator_cell_means = (
                np.mean(comparator, axis=(0, 2))
                if comparator.ndim == 3
                else np.mean(comparator, axis=1)
            )
            for factor_index, factor in enumerate(FACTORS):
                cell_comparator_mean = float(comparator_cell_means[factor_index])
                cell_difference = float(cell_points[factor_index])
                cell_rows.append(
                    {
                        "treatment": LEARNED_POLICY_NAME,
                        "comparator": comparator_name,
                        "inferential_role": specification["role"],
                        "metric": metric,
                        "factor_id": factor.factor_id,
                        "occupancy_process": factor.occupancy_process,
                        "occupancy_probability": factor.occupancy_probability,
                        "arrival_probability": factor.arrival_probability,
                        "treatment_mean": float(treatment_cell_means[factor_index]),
                        "comparator_mean": cell_comparator_mean,
                        "difference": cell_difference,
                        "relative_difference": (
                            ""
                            if cell_comparator_mean == 0.0
                            else cell_difference / cell_comparator_mean
                        ),
                        "one_sided_95_lcb": float(cell_lcb[factor_index]),
                        "one_sided_95_ucb": float(cell_ucb[factor_index]),
                    }
                )

    holm_raw = {
        comparator: float(payload["raw_p"])
        for comparator, payload in goodput_by_comparator.items()
        if bool(payload["row"]["holm_family_member"])
    }
    holm_adjusted = _holm_adjust(holm_raw)
    for comparator, adjusted in holm_adjusted.items():
        row = goodput_by_comparator[comparator]["row"]
        row["exact_sign_flip_p_holm"] = adjusted
        row["claim_pass"] = bool(
            float(row["one_sided_95_lcb"]) > 0.0 and adjusted < SIGN_FLIP_ALPHA
        )

    phase_row = goodput_by_comparator[PRIMARY_BASELINE_NAME]["row"]
    history_row = goodput_by_comparator[HISTORY_FREE_POLICY_NAME]["row"]
    phase_pass = bool(
        float(phase_row["one_sided_95_lcb"]) > PRIMARY_GOODPUT_MARGIN
        and float(phase_row["exact_sign_flip_p_raw"]) < SIGN_FLIP_ALPHA
    )
    history_pass = bool(
        float(history_row["one_sided_95_lcb"]) > HISTORY_ABLATION_MARGIN
        and float(history_row["exact_sign_flip_p_raw"]) < SIGN_FLIP_ALPHA
    )
    phase_row["claim_pass"] = phase_pass
    history_row["claim_pass"] = history_pass

    mask_cube = _mask_count_cube(rows)
    mechanism_rows, mechanism_summary = _mechanism_analysis(mask_cube, rng=rng)

    factorial_rows: list[dict[str, Any]] = []
    for metric in ANALYSED_METRICS:
        learned, baselines = metric_cubes[metric]
        for policy in ALL_POLICIES:
            values = learned[policy] if policy in LEARNED_POLICIES else baselines[policy]
            cell_means = (
                np.mean(values, axis=(0, 2))
                if values.ndim == 3
                else np.mean(values, axis=1)
            )
            for factor_index, factor in enumerate(FACTORS):
                factorial_rows.append(
                    {
                        "factor_id": factor.factor_id,
                        "occupancy_process": factor.occupancy_process,
                        "occupancy_probability": factor.occupancy_probability,
                        "arrival_probability": factor.arrival_probability,
                        "policy": policy,
                        "metric": metric,
                        "mean": float(cell_means[factor_index]),
                    }
                )

    joint_pass = bool(phase_pass and history_pass and mechanism_summary["pass"])
    summary: dict[str, Any] = {
        "study_id": STUDY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "COMPLETE",
        "outcome_reporting_required_regardless_of_gates": True,
        "estimand": (
            "equal-cell macro mean paired difference over independently trained "
            "final checkpoints and crossed environment-seed blocks"
        ),
        "bootstrap": {
            "name": "stratified_crossed_two_factor_percentile",
            "replicates": BOOTSTRAP_REPLICATES,
            "training_seed_resampling": "one shared draw across every factorial cell",
            "environment_resampling": (
                "one joint draw across every factorial cell and policy, stratified "
                "within the six fixed channel-label shifts"
            ),
            "factorial_cells": "fixed and equally weighted",
            "lower_quantile": BOOTSTRAP_LOWER_QUANTILE,
            "upper_quantile": BOOTSTRAP_UPPER_QUANTILE,
        },
        "confirmatory_gates": {
            "goodput_vs_phase_orthogonal": {
                "difference": phase_row["difference"],
                "one_sided_95_lcb": phase_row["one_sided_95_lcb"],
                "one_sided_95_ucb": phase_row["one_sided_95_ucb"],
                "margin_packets_per_slot": PRIMARY_GOODPUT_MARGIN,
                "exact_sign_flip_p": phase_row["exact_sign_flip_p_raw"],
                "pass": phase_pass,
            },
            "goodput_vs_history_free_dqn": {
                "difference": history_row["difference"],
                "one_sided_95_lcb": history_row["one_sided_95_lcb"],
                "one_sided_95_ucb": history_row["one_sided_95_ucb"],
                "margin_packets_per_slot": HISTORY_ABLATION_MARGIN,
                "exact_sign_flip_p": history_row["exact_sign_flip_p_raw"],
                "pass": history_pass,
            },
            "finite_support_mechanism_departure": mechanism_summary,
            "joint_pass": joint_pass,
        },
        "secondary_holm_family": {
            "metric": PRIMARY_METRIC,
            "comparators": list(holm_raw),
            "excludes": [
                PRIMARY_BASELINE_NAME,
                HISTORY_FREE_POLICY_NAME,
                COORDINATION_BLIND_POLICY_NAME,
                ORACLE_POLICY_NAME,
            ],
            "raw_p_values": holm_raw,
            "adjusted_p_values": holm_adjusted,
            "alpha": SIGN_FLIP_ALPHA,
        },
        "centralized_oracle_role": (
            "descriptive headroom only; excluded from the Holm family and all "
            "superiority gates"
        ),
        "interpretation_gate": (
            "A collision-aware efficacy claim is permitted only if every confirmatory "
            "gate passes. Null, negative, and mixed outcomes remain reportable."
        ),
        "artifacts": {
            key: path.relative_to(results_root).as_posix()
            for key, path in output_paths.items()
        },
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_exclusive(output_paths["comparisons"], comparison_rows)
    _write_csv_exclusive(output_paths["cell_effects"], cell_rows)
    _write_csv_exclusive(output_paths["checkpoint_effects"], checkpoint_effect_rows)
    _write_csv_exclusive(output_paths["factorial_means"], factorial_rows)
    _write_csv_exclusive(output_paths["mechanism"], mechanism_rows)
    _write_json_exclusive(output_paths["summary"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.results_root), indent=2, sort_keys=True, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
