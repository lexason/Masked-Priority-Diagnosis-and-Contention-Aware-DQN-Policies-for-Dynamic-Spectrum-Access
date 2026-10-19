#!/usr/bin/env python3
"""Independent, read-only recomputation from the sealed aggregate cube.

This script deliberately does not import the study analysis implementation.
It reads the sealed CSV, reconstructs the crossed design, and writes the
machine-readable result to the explicit path supplied by ``--output``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

TRAINING_SEEDS = (
    71101, 71209, 71317, 71423, 71531, 71639, 71747, 71851, 71959, 72067,
    72173, 72281, 72389, 72497, 72601, 72709, 72817, 72923, 73031, 73139,
)
ENVIRONMENT_SEEDS = (
    52010101, 52010211, 52010307, 52010419, 52010503, 52010613,
    52010709, 52010821, 52010907, 52011019, 52011103, 52011217,
    52011301, 52011409, 52011503, 52011619, 52011707, 52011811,
    52011919, 52012027, 52012119, 52012233, 52012307, 52012419,
)
FACTORS = (
    "iid_o30_a10", "iid_o30_a25", "iid_o60_a10", "iid_o60_a25",
    "markov_o30_a10", "markov_o30_a25", "markov_o60_a10", "markov_o60_a25",
)
LEARNED = (
    "collision_aware_dqn", "history_free_dqn", "coordination_blind_dqn",
)
BASELINES = (
    "phase_orthogonal", "quality_greedy_persistent", "round_robin",
    "uniform_legal", "static_priority", "centralized_quality_matching",
)
POLICIES = (*LEARNED, *BASELINES)
FULL = "collision_aware_dqn"
BOOTSTRAP_SEED = 20261017
BOOTSTRAP_REPLICATES = 10_000
N_NODES = 15
MEASUREMENT_STEPS = 3_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(text: str, name: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"invalid {name}: {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {name}: {text!r}")
    return value


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12_672:
        raise ValueError(f"expected 12672 rows, found {len(rows)}")
    return rows


def build_cubes(
    rows: list[dict[str, str]], metric: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ti = {seed: index for index, seed in enumerate(TRAINING_SEEDS)}
    fi = {factor: index for index, factor in enumerate(FACTORS)}
    ei = {seed: index for index, seed in enumerate(ENVIRONMENT_SEEDS)}
    learned = {
        p: np.full((20, 8, 24), np.nan, dtype=np.float64) for p in LEARNED
    }
    baselines = {
        p: np.full((8, 24), np.nan, dtype=np.float64) for p in BASELINES
    }
    for row in rows:
        p = row["policy"]
        f = fi[row["factor_id"]]
        e = ei[int(row["environment_seed"])]
        value = finite(row[metric], metric)
        if p in learned:
            t = ti[int(row["training_seed"])]
            if np.isfinite(learned[p][t, f, e]):
                raise ValueError(f"duplicate learned cell {p, t, f, e}")
            learned[p][t, f, e] = value
        else:
            if row["training_seed"]:
                raise ValueError(f"baseline {p} has training seed")
            if np.isfinite(baselines[p][f, e]):
                raise ValueError(f"duplicate baseline cell {p, f, e}")
            baselines[p][f, e] = value
    for name, values in (*learned.items(), *baselines.items()):
        if np.any(~np.isfinite(values)):
            raise ValueError(f"incomplete cube: {name}")
    return learned, baselines


def bootstrap_plan() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    training_indices = rng.integers(
        0, 20, size=(BOOTSTRAP_REPLICATES, 20), dtype=np.int64
    )
    strata = [np.arange(shift, 24, 6, dtype=np.int64) for shift in range(6)]
    environment_indices = np.concatenate(
        [rng.choice(s, size=(BOOTSTRAP_REPLICATES, 4), replace=True) for s in strata],
        axis=1,
    )
    training_counts = (
        training_indices[:, :, None] == np.arange(20, dtype=np.int64)[None, None, :]
    ).sum(axis=1)
    environment_counts = (
        environment_indices[:, :, None] == np.arange(24, dtype=np.int64)[None, None, :]
    ).sum(axis=1)
    return training_indices, environment_indices, training_counts, environment_counts


def difference_cube(treatment: np.ndarray, comparator: np.ndarray) -> np.ndarray:
    if comparator.ndim == 2:
        return treatment - comparator[None, :, :]
    return treatment - comparator


def bootstrap_draws(
    difference: np.ndarray, training_counts: np.ndarray, environment_counts: np.ndarray
) -> np.ndarray:
    # Factors are fixed at 1/8. Selected training units receive 1/20, and the
    # six balanced 4-block shift strata together produce environment weight 1/24.
    training_environment = difference.mean(axis=1)
    return (
        ((training_counts @ training_environment) * environment_counts).sum(axis=1)
        / (20.0 * 24.0)
    )


def exact_sign_flip(values: np.ndarray, margin: float) -> dict[str, float | int]:
    centered = np.asarray(values, dtype=np.float64) - margin
    distribution = np.asarray([0.0], dtype=np.float64)
    for value in centered:
        distribution = np.concatenate((distribution + value, distribution - value))
    observed = float(centered.sum())
    tolerance = np.finfo(np.float64).eps * 64.0 * max(
        1.0, abs(observed), float(np.abs(centered).sum())
    )
    numerator = int(np.count_nonzero(distribution >= observed - tolerance))
    return {
        "margin": margin,
        "statistic_sum_boundary_centered": observed,
        "inclusive_tail_numerator": numerator,
        "enumeration_denominator": int(distribution.size),
        "p_value": numerator / distribution.size,
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    keys = sorted(p_values, key=lambda key: (p_values[key], key))
    result: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - rank) * p_values[key]))
        result[key] = running
    return result


def sensitivities(difference: np.ndarray, point: float) -> dict[str, Any]:
    seed_effects = difference.mean(axis=(1, 2))
    loto = np.asarray(
        [(seed_effects.sum() - seed_effects[index]) / 19.0 for index in range(20)]
    )
    loto_change = np.abs(loto - point)
    loto_max = int(np.argmax(loto_change))

    by_environment = difference.mean(axis=(0, 1))
    loe = np.empty(24, dtype=np.float64)
    for omitted in range(24):
        shift_means = []
        for shift in range(6):
            members = [index for index in range(24) if index % 6 == shift and index != omitted]
            shift_means.append(float(by_environment[members].mean()))
        loe[omitted] = fmean(shift_means)
    loe_change = np.abs(loe - point)
    loe_max = int(np.argmax(loe_change))
    return {
        "checkpoint_effects": [
            {"training_seed": seed, "effect": float(seed_effects[index])}
            for index, seed in enumerate(TRAINING_SEEDS)
        ],
        "positive_checkpoint_effect_count": int(np.count_nonzero(seed_effects > 0.0)),
        "checkpoint_effect_range": [float(seed_effects.min()), float(seed_effects.max())],
        "leave_one_training_unit": {
            "range": [float(loto.min()), float(loto.max())],
            "largest_absolute_change": float(loto_change[loto_max]),
            "most_influential_omitted_training_seed": TRAINING_SEEDS[loto_max],
            "estimate_when_omitted": float(loto[loto_max]),
            "estimates": [
                {"omitted_training_seed": seed, "effect": float(loto[index])}
                for index, seed in enumerate(TRAINING_SEEDS)
            ],
        },
        "leave_one_environment_block_shift_preserving": {
            "range": [float(loe.min()), float(loe.max())],
            "largest_absolute_change": float(loe_change[loe_max]),
            "most_influential_omitted_environment_seed": ENVIRONMENT_SEEDS[loe_max],
            "replication_index": loe_max,
            "shift": loe_max % 6,
            "estimate_when_omitted": float(loe[loe_max]),
            "estimates": [
                {
                    "omitted_environment_seed": seed,
                    "replication_index": index,
                    "shift": index % 6,
                    "effect": float(loe[index]),
                }
                for index, seed in enumerate(ENVIRONMENT_SEEDS)
            ],
        },
    }


def inferential_outputs(rows: list[dict[str, str]]) -> dict[str, Any]:
    learned, baselines = build_cubes(rows, "network_goodput")
    t_indices, e_indices, t_counts, e_counts = bootstrap_plan()
    plan_hash = hashlib.sha256(
        t_indices.astype("<i8").tobytes() + e_indices.astype("<i8").tobytes()
    ).hexdigest()
    comparators = (
        "phase_orthogonal", "history_free_dqn", "coordination_blind_dqn",
        "quality_greedy_persistent", "round_robin", "uniform_legal",
        "static_priority", "centralized_quality_matching",
    )
    outputs = {}
    for comparator_name in comparators:
        comparator = (
            learned[comparator_name]
            if comparator_name in learned
            else baselines[comparator_name]
        )
        difference = difference_cube(learned[FULL], comparator)
        point = float(difference.mean())
        draws = bootstrap_draws(difference, t_counts, e_counts)
        q05, q95 = np.quantile(draws, [0.05, 0.95])
        seed_effects = difference.mean(axis=(1, 2))
        margin = 0.05 if comparator_name == "phase_orthogonal" else 0.0
        sign_flip = None
        if comparator_name != "centralized_quality_matching":
            sign_flip = exact_sign_flip(seed_effects, margin)
        result = {
            "point_effect": point,
            "percentile_one_sided_95_lower": float(q05),
            "percentile_one_sided_95_upper": float(q95),
            "basic_one_sided_95_lower": float(2.0 * point - q95),
            "basic_one_sided_95_upper": float(2.0 * point - q05),
            "bootstrap_draw_mean": float(draws.mean()),
            "bootstrap_draw_standard_deviation_ddof1": float(draws.std(ddof=1)),
            "sign_flip": sign_flip,
        }
        if comparator_name in {"phase_orthogonal", "history_free_dqn"}:
            result["sensitivity"] = sensitivities(difference, point)
        outputs[comparator_name] = result
    secondary = {
        name: outputs[name]["sign_flip"]["p_value"]
        for name in (
            "quality_greedy_persistent", "round_robin", "uniform_legal", "static_priority"
        )
    }
    return {
        "method": {
            "replicates": BOOTSTRAP_REPLICATES,
            "generator": "numpy.random.Generator(PCG64)",
            "seed": BOOTSTRAP_SEED,
            "quantile_method": "numpy.quantile default method='linear'",
            "training_resample": "20 complete paired training units with replacement",
            "environment_resample": (
                "within each of six shifts, four complete environment blocks with replacement; "
                "same resamples across policies, training units, and fixed equally weighted cells"
            ),
            "factor_weight": "1/8 each",
            "shift_weight": "1/6 each",
            "sign_flip": (
                "fully enumerated 2^20 independent sign assignments; inclusive >= tail, "
                "boundary-centred at the stated margin, floating tolerance matching frozen code"
            ),
            "bootstrap_plan_sha256": plan_hash,
        },
        "comparisons": outputs,
        "secondary_holm_adjusted": holm(secondary),
    }


def optional_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def operational_record(row: dict[str, str]) -> dict[str, float | None]:
    steps = int(row["measurement_steps"])
    if steps != MEASUREMENT_STEPS:
        raise ValueError("measurement duration drift")
    start = int(row["starting_backlog"])
    arrivals = int(row["arrivals"])
    deliveries = int(row["deliveries"])
    ending = int(row["ending_backlog"])
    attempted = int(row["attempted"])
    singleton = int(row["singleton_attempts"])
    collisions = int(row["collision_attempts"])
    failures = int(row["link_failures"])
    voluntary = int(row["voluntary_deferrals"])
    forced = int(row["forced_deferrals"])
    eligible = int(row["priority_eligible_decisions"])
    if start + arrivals != deliveries + ending:
        raise ValueError("packet conservation failure")
    if attempted != singleton + collisions or singleton != deliveries + failures:
        raise ValueError("attempt accounting failure")
    if eligible != attempted + voluntary:
        raise ValueError("eligible decision partition failure")
    capacity_count_float = finite(row["capacity_ceiling_per_slot"], "capacity") * steps
    capacity_count = round(capacity_count_float)
    if not math.isclose(capacity_count_float, capacity_count, abs_tol=1e-7):
        raise ValueError("nonintegral capacity count")
    fairness = None if row["delivery_jain_fairness"] == "" else finite(
        row["delivery_jain_fairness"], "fairness"
    )
    if (fairness is None) != (deliveries == 0):
        raise ValueError("fairness missingness mismatch")
    computed = {
        "goodput": deliveries / steps,
        "attempts_per_node_slot": attempted / (N_NODES * steps),
        "eligible_per_node_slot": eligible / (N_NODES * steps),
        "collision_attempt_rate": optional_ratio(collisions, attempted),
        "conditional_link_failure_rate": optional_ratio(failures, singleton),
        "voluntary_deferrals_per_node_slot": voluntary / (N_NODES * steps),
        "forced_deferrals_per_node_slot": forced / (N_NODES * steps),
        "finite_window_service_fraction": optional_ratio(deliveries, start + arrivals),
        "opportunity_utilization": optional_ratio(deliveries, capacity_count),
        "mean_network_backlog": finite(row["mean_network_backlog"], "backlog"),
        "ending_backlog": float(ending),
        "delivery_jain_fairness_all_15_nodes": fairness,
        "zero_delivery_run": float(deliveries == 0),
    }
    checks = {
        "goodput": "network_goodput",
        "finite_window_service_fraction": "finite_window_service_fraction",
        "opportunity_utilization": "opportunity_utilization",
        "collision_attempt_rate": "collision_attempt_rate",
        "conditional_link_failure_rate": "conditional_link_failure_rate",
    }
    for ours, source in checks.items():
        expected = computed[ours]
        text = row[source]
        # Frozen collision-rate convention is 0.0 on zero-attempt runs; the
        # descriptive estimand correctly treats those ratios as undefined.
        if expected is None:
            if source == "collision_attempt_rate" and text == "0.0":
                continue
            if text != "":
                raise ValueError(f"expected missing {source}")
        elif not math.isclose(expected, finite(text, source), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"metric mismatch: {source}")
    return computed


def cell_mean(
    records: list[tuple[dict[str, str], dict[str, float | None]]],
    metric: str,
    learned: bool,
) -> float | None:
    if learned:
        by_seed = defaultdict(list)
        for row, values in records:
            value = values[metric]
            if value is not None:
                by_seed[int(row["training_seed"])].append(value)
        means = []
        for seed in TRAINING_SEEDS:
            if not by_seed[seed]:
                return None
            means.append(fmean(by_seed[seed]))
        return fmean(means)
    values = [values[metric] for _, values in records if values[metric] is not None]
    return None if not values else fmean(values)


def equal_factor_mean(
    records: list[tuple[dict[str, str], dict[str, float | None]]],
    metric: str,
    learned: bool,
) -> float | None:
    factor_means = []
    represented = sorted({row["factor_id"] for row, _ in records})
    for factor in represented:
        cell = [item for item in records if item[0]["factor_id"] == factor]
        value = cell_mean(cell, metric, learned)
        if value is None:
            return None
        factor_means.append(value)
    return None if not factor_means else fmean(factor_means)


def operational_outputs(rows: list[dict[str, str]]) -> dict[str, Any]:
    parsed = [(row, operational_record(row)) for row in rows]
    metrics = tuple(parsed[0][1])
    outputs = {}
    for policy in POLICIES:
        policy_rows = [(row, values) for row, values in parsed if row["policy"] == policy]
        learned = policy in LEARNED
        by_load = {}
        for label, arrival in (("low_arrival_0.10", "0.1"), ("high_arrival_0.25", "0.25")):
            group = [item for item in policy_rows if item[0]["arrival_probability"] == arrival]
            by_load[label] = {
                metric: equal_factor_mean(group, metric, learned) for metric in metrics
            }
            by_load[label]["run_count"] = len(group)
        macro = {metric: equal_factor_mean(policy_rows, metric, learned) for metric in metrics}
        totals = {
            field: sum(int(row[field]) for row, _ in policy_rows)
            for field in (
                "attempted", "collision_attempts", "singleton_attempts", "link_failures",
                "voluntary_deferrals", "forced_deferrals", "priority_eligible_decisions",
                "deliveries", "starting_backlog", "arrivals", "ending_backlog",
            )
        }
        macro["run_count"] = len(policy_rows)
        macro["defined_fairness_runs"] = sum(
            values["delivery_jain_fairness_all_15_nodes"] is not None
            for _, values in policy_rows
        )
        macro["pooled_voluntary_deferral_given_eligible"] = optional_ratio(
            totals["voluntary_deferrals"], totals["priority_eligible_decisions"]
        )
        macro["pooled_attempt_fraction_given_eligible"] = optional_ratio(
            totals["attempted"], totals["priority_eligible_decisions"]
        )
        macro["pooled_collision_attempt_rate"] = optional_ratio(
            totals["collision_attempts"], totals["attempted"]
        )
        macro["pooled_conditional_link_failure_rate"] = optional_ratio(
            totals["link_failures"], totals["singleton_attempts"]
        )
        macro["totals"] = totals
        outputs[policy] = {"macro": macro, "by_arrival": by_load}
    return {
        "aggregation": (
            "Equal factor-cell weights. Learned policies: arithmetic mean across environment "
            "blocks within each training seed, then equal mean across 20 seeds. Nonlearning "
            "policies: one mean across 24 environment blocks. Ratios are run-level ratios; "
            "undefined ratios remain missing. Pooled fields are explicitly labelled."
        ),
        "fairness_source_audit": {
            "formula": "(sum_i D_i)^2 / (15 * sum_i D_i^2), i=1,...,15",
            "zero_delivery_handling": "undefined/missing",
            "source_summary_contains_per_node_delivery_vector": False,
            "independent_recalculation_from_summary_possible": False,
            "note": (
                "The sealed summary retains the computed all-node Jain value but not the 15 "
                "delivery counts; the generation formula can be source-audited, while a fully "
                "independent numerical reconstruction requires deterministic replay or retained "
                "per-node counts."
            ),
        },
        "policies": outputs,
    }


def expected_max_quality() -> tuple[float, float]:
    sigma = 0.10
    lower, upper = 0.05, 0.99

    def conditional(c: float) -> float:
        tail, _ = quad(
            lambda x: 1.0 - ndtr((x - c) / sigma) ** N_NODES,
            lower,
            upper,
            epsabs=2e-13,
            epsrel=2e-13,
            limit=200,
        )
        return lower + tail

    expectation, error = quad(
        lambda c: 30.0 * c**4 * (1.0 - c) * conditional(c),
        0.0,
        1.0,
        epsabs=2e-12,
        epsrel=2e-12,
        limit=200,
    )
    return expectation, error


def load_outputs() -> dict[str, Any]:
    eqmax, error = expected_max_quality()
    cells = []
    for rho, arrival in itertools.product((0.30, 0.60), (0.10, 0.25)):
        offered = N_NODES * arrival
        idle_bound = 6.0 * (1.0 - rho)
        quality_bound = idle_bound * eqmax
        cells.append(
            {
                "occupancy_probability": rho,
                "arrival_probability": arrival,
                "offered_packets_per_slot": offered,
                "universal_idle_channel_upper_bound": idle_bound,
                "quality_aware_optimistic_service_upper_bound": quality_bound,
                "overloaded_by_quality_bound": bool(offered > quality_bound),
                "conclusion": (
                    "overloaded under verified bound"
                    if offered > quality_bound
                    else "not ruled out as sustainable by this bound"
                ),
            }
        )
    return {
        "expected_best_of_15_clipped_link_qualities": eqmax,
        "numerical_quadrature_absolute_error_estimate": error,
        "integral": (
            "Integral c=0..1 of 30*c^4*(1-c)*[0.05 + Integral x=0.05..0.99 "
            "{1-Phi((x-c)/0.10)^15} dx] dc"
        ),
        "upper_bound": "6*(1-rho)*E[max_i Q_i]",
        "assumptions_source_audited": [
            "common per-channel quality C~Beta(5,2)",
            "independent node perturbations epsilon_i~Normal(0,0.10^2)",
            "Q_i=clip(C+epsilon_i,0.05,0.99)",
            "quality draws independent of occupancy draws",
            (
                "current link-success uniforms drawn independently after observations "
                "and unavailable to scheduler"
            ),
        ],
        "bound_is_optimistic_because": (
            "it ignores queue availability and the one-channel-per-node matching constraint"
        ),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.summary_csv)
    payload = {
        "input": {
            "path": str(args.summary_csv),
            "sha256": sha256(args.summary_csv),
            "row_count": len(rows),
        },
        "inference": inferential_outputs(rows),
        "load_bound": load_outputs(),
        "operational": operational_outputs(rows),
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "input_sha256": payload["input"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
