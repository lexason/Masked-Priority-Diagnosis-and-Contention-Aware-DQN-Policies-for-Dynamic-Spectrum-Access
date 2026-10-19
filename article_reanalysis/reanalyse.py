#!/usr/bin/env python3
"""Independent read-only reanalysis of the prospective collision study.

All outputs are written under the disposable /tmp task directory.  The script
uses only the released aggregate cube, episode logs, metadata, and frozen
source/configuration; it does not import the study's analysis implementation.
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

import numpy as np
from scipy.special import ndtr


PARSER = argparse.ArgumentParser(
    description=(
        "Independently reproduce the logical priority audit, coordinate "
        "control, operational summaries, load bound, and training-log checks."
    )
)
PARSER.add_argument(
    "--study-root",
    type=Path,
    required=True,
    help="Root containing results/, main/revision/, and manuscript_assets/.",
)
PARSER.add_argument(
    "--out-dir",
    type=Path,
    required=True,
    help="New or existing directory for generated JSON/CSV outputs.",
)
ARGS = PARSER.parse_args()
ROOT = ARGS.study_root.resolve()
OUT_DIR = ARGS.out_dir.resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results/crn_ca_pr1_20261017"
SUMMARY = RESULTS / "evaluation/summary.csv"
OUT = OUT_DIR / "reanalysis_output.json"
AUDIT_CSV = OUT_DIR / "attempted_only_logical_audit.csv"
FIGURE_CSV = OUT_DIR / "mechanism_audit_figure_data.csv"

N = 6
ABSTAIN = 6
ACTIONS = 7
TRAINING_SEEDS = (
    71101, 71209, 71317, 71423, 71531, 71639, 71747, 71851, 71959,
    72067, 72173, 72281, 72389, 72497, 72601, 72709, 72817, 72923,
    73031, 73139,
)
FACTORS = (
    "iid_o30_a10", "iid_o30_a25", "iid_o60_a10", "iid_o60_a25",
    "markov_o30_a10", "markov_o30_a25", "markov_o60_a10",
    "markov_o60_a25",
)
LEARNED = (
    "collision_aware_dqn", "history_free_dqn", "coordination_blind_dqn",
)
BASELINES = (
    "phase_orthogonal", "quality_greedy_persistent", "round_robin",
    "uniform_legal", "static_priority", "centralized_quality_matching",
)
POLICIES = LEARNED + BASELINES
CAL = tuple(range(12))
VAL = tuple(range(12, 24))
ORDERS = tuple(itertools.permutations(range(N)))
PRED = np.full((len(ORDERS), 1 << N), ABSTAIN, dtype=np.int8)
for oi, order in enumerate(ORDERS):
    for mask in range(1, 1 << N):
        PRED[oi, mask] = next(c for c in order if mask & (1 << c))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_counts(value: str) -> np.ndarray:
    values = np.asarray([int(x) for x in value.split("-")], dtype=np.int64)
    assert values.size == 64 * 7
    return values.reshape(64, 7)


def order_scores(groups: np.ndarray, attempted_only: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-group scores and pooled matches/totals for all 720 orders."""
    assert groups.ndim == 3 and groups.shape[1:] == (64, 7)
    if attempted_only:
        totals = groups[:, :, :6].sum(axis=(1, 2), dtype=np.int64)
    else:
        totals = groups.sum(axis=(1, 2), dtype=np.int64)
    assert np.all(totals > 0)
    matches = np.empty((720, groups.shape[0]), dtype=np.int64)
    masks = np.arange(64)
    for gi, group in enumerate(groups):
        matches[:, gi] = group[masks[None, :], PRED].sum(axis=1, dtype=np.int64)
    scores = np.mean(matches / totals[None, :], axis=1)
    return scores, matches.sum(axis=1, dtype=np.int64), totals


def best(scores: np.ndarray) -> np.ndarray:
    m = float(scores.max())
    return np.flatnonzero(np.isclose(scores, m, rtol=0.0, atol=1e-15))


def graph_certificate(counts: np.ndarray) -> dict:
    """Certificate for attempted actions only; counts may be 64x6 or 64x7."""
    attempted = counts[:, :6]
    edge_support = np.zeros((N, N), dtype=np.int64)
    for mask in range(1, 64):
        legal = [c for c in range(N) if mask & (1 << c)]
        for chosen in legal:
            support = int(attempted[mask, chosen])
            if support:
                for other in legal:
                    if other != chosen:
                        edge_support[chosen, other] += support
    adjacency = edge_support > 0
    indegree = adjacency.sum(axis=0).astype(int)
    frontier = sorted(np.flatnonzero(indegree == 0).tolist())
    topo = []
    unique = True
    while frontier:
        if len(frontier) != 1:
            unique = False
        node = frontier.pop(0)
        topo.append(int(node))
        for target in np.flatnonzero(adjacency[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(int(target))
                frontier.sort()
    acyclic = len(topo) == N
    if not acyclic:
        topo = []
        unique = False
    total = int(attempted.sum())
    matches = np.asarray(
        [sum(int(attempted[m, PRED[oi, m]]) for m in range(1, 64)) for oi in range(720)],
        dtype=np.int64,
    )
    representing = np.flatnonzero(matches == total)
    return {
        "attempts": total,
        "edge_count": int(adjacency.sum()),
        "acyclic": bool(acyclic),
        "unique_topological_order": bool(acyclic and unique),
        "topological_order": topo,
        "representing_order_count": int(representing.size),
        "representing_orders": ["-".join(map(str, ORDERS[i])) for i in representing],
        "counts_64x6_le_i8_sha256": hashlib.sha256(
            attempted.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }


with SUMMARY.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 12672, len(rows)
factor_i = {v: i for i, v in enumerate(FACTORS)}
seed_i = {v: i for i, v in enumerate(TRAINING_SEEDS)}
full_rows = [r for r in rows if r["policy"] == "collision_aware_dqn"]
assert len(full_rows) == 20 * 8 * 24
cube = np.full((20, 8, 24, 64, 7), -1, dtype=np.int64)
identity_violations = defaultdict(int)
for r in rows:
    ri = int(r["replication_index"])
    expected_perm = "-".join(str((logical + ri % 6) % 6) for logical in range(6))
    if r["logical_to_physical"] != expected_perm:
        identity_violations["permutation"] += 1
    measurement = int(r["measurement_steps"])
    if int(r["attempted"]) != int(r["singleton_attempts"]) + int(r["collision_attempts"]):
        identity_violations["attempt_partition"] += 1
    if int(r["singleton_attempts"]) != int(r["deliveries"]) + int(r["link_failures"]):
        identity_violations["singleton_partition"] += 1
    if int(r["starting_backlog"]) + int(r["arrivals"]) != int(r["deliveries"]) + int(r["ending_backlog"]):
        identity_violations["packet_conservation"] += 1
    if not math.isclose(float(r["network_goodput"]), int(r["deliveries"]) / measurement, rel_tol=0, abs_tol=1e-15):
        identity_violations["goodput"] += 1
    logical_actions = [int(x) for x in r["logical_action_counts"].split("-")]
    if sum(logical_actions) != 15 * measurement:
        identity_violations["logical_action_total"] += 1
    if sum(logical_actions[:6]) != int(r["attempted"]):
        identity_violations["attempt_action_total"] += 1
    counts = parse_counts(r["mask_action_counts"])
    if int(counts.sum()) != int(r["priority_eligible_decisions"]):
        identity_violations["mask_eligible_total"] += 1
    # In this common-availability environment eligible decisions partition
    # exactly into attempts and voluntary deferrals; forced deferrals have no
    # legal channel and are excluded from the audit denominator.
    if int(counts[:, :6].sum()) != int(r["attempted"]):
        identity_violations["mask_attempt_total"] += 1
    if int(counts[:, 6].sum()) != int(r["voluntary_deferrals"]):
        identity_violations["mask_voluntary_total"] += 1

for r in full_rows:
    si = seed_i[int(r["training_seed"])]
    fi = factor_i[r["factor_id"]]
    ei = int(r["replication_index"])
    assert np.all(cube[si, fi, ei] == -1)
    cube[si, fi, ei] = parse_counts(r["mask_action_counts"])
assert np.all(cube >= 0)


def checkpoint_audit(si: int, attempted_only: bool) -> dict:
    # The released analysis pools runs within factor and weights the eight
    # factor-specific agreement ratios equally.
    # Index the environment axis after selecting the checkpoint.  Doing both
    # in one expression would trigger NumPy advanced-index axis reordering.
    cal_groups = cube[si][:, CAL].sum(axis=1, dtype=np.int64)
    val_groups = cube[si][:, VAL].sum(axis=1, dtype=np.int64)
    full_counts = cube[si].sum(axis=(0, 1), dtype=np.int64)
    cs, cm, ct = order_scores(cal_groups, attempted_only)
    vs, vm, vt = order_scores(val_groups, attempted_only)
    cw = best(cs)
    envelope = vs[cw]
    selected_candidates = cw[np.isclose(envelope, envelope.max(), rtol=0, atol=1e-15)]
    selected = int(selected_candidates[0])
    vw = best(vs)
    denominator_slice = slice(0, 6) if attempted_only else slice(0, 7)
    cal_total = int(cal_groups[:, :, denominator_slice].sum())
    val_total = int(val_groups[:, :, denominator_slice].sum())
    record = {
        "training_seed": TRAINING_SEEDS[si],
        "calibration_count": cal_total,
        "validation_count": val_total,
        "calibration_mask_coverage": int(np.count_nonzero(cal_groups[:, :, :6].sum(axis=(0, 2))[1:])),
        "validation_mask_coverage": int(np.count_nonzero(val_groups[:, :, :6].sum(axis=(0, 2))[1:])),
        "calibration_optimal_agreement": float(cs[cw[0]]),
        "calibration_tie_count": int(cw.size),
        "calibration_tied_orders": ["-".join(map(str, ORDERS[i])) for i in cw],
        "selected_order_original_validation_tie_rule": "-".join(map(str, ORDERS[selected])),
        "selected_order_index": selected,
        "calibration_selected_validation_agreement": float(vs[selected]),
        "calibration_tie_validation_envelope": [float(envelope.min()), float(envelope.max())],
        "validation_optimal_agreement": float(vs[vw[0]]),
        "validation_optimal_tie_count": int(vw.size),
        "validation_optimal_orders": ["-".join(map(str, ORDERS[i])) for i in vw],
        "selected_validation_pooled_matches": int(vm[selected]),
        "selected_validation_pooled_agreement": float(vm[selected] / val_total),
        "calibration_certificate": graph_certificate(
            cal_groups.sum(axis=0, dtype=np.int64)
        ),
        "validation_certificate": graph_certificate(
            val_groups.sum(axis=0, dtype=np.int64)
        ),
    }
    cert = graph_certificate(full_counts)
    record.update({"full_support_certificate": cert})
    if not attempted_only:
        record["full_eligible_count"] = int(full_counts.sum())
        record["full_abstention_count"] = int(full_counts[:, 6].sum())
        record["full_attempt_fraction"] = float(full_counts[:, :6].sum() / full_counts.sum())
        record["full_64x7_le_i8_sha256"] = hashlib.sha256(
            full_counts.astype("<i8", copy=False).tobytes()
        ).hexdigest()
    return record


original = [checkpoint_audit(si, False) for si in range(20)]
attempted = [checkpoint_audit(si, True) for si in range(20)]

# Check exact reproduction against the released original seven-action file.
cert_path = RESULTS / "analysis/mechanism_certificates.csv"
with cert_path.open(newline="", encoding="utf-8") as f:
    released_cert = {int(r["training_seed"]): r for r in csv.DictReader(f)}
reproduction_errors = []
for record in original:
    rel = released_cert[record["training_seed"]]
    checks = {
        "calibration_count": int(rel["calibration_eligible_decisions"]),
        "validation_count": int(rel["validation_eligible_decisions"]),
        "calibration_tie_count": int(rel["calibration_order_tie_count"]),
        "selected_order_original_validation_tie_rule": rel["calibration_best_order"],
        "full_eligible_count": int(rel["full_eligible_decisions"]),
        "full_abstention_count": int(rel["eligible_abstentions"]),
    }
    for key, expected in checks.items():
        if record[key] != expected:
            reproduction_errors.append([record["training_seed"], key, record[key], expected])
    float_checks = {
        "calibration_optimal_agreement": float(rel["calibration_best_equal_cell_agreement"]),
        "calibration_selected_validation_agreement": float(rel["validation_agreement_for_calibration_order"]),
        "validation_optimal_agreement": float(rel["validation_best_equal_cell_agreement"]),
    }
    for key, expected in float_checks.items():
        if not math.isclose(record[key], expected, rel_tol=0, abs_tol=1e-15):
            reproduction_errors.append([record["training_seed"], key, record[key], expected])

all_counts = cube.sum(axis=(0, 1, 2), dtype=np.int64)
all_eligible = int(all_counts.sum())
all_attempts = int(all_counts[:, :6].sum())
all_abstain = int(all_counts[:, 6].sum())
ascending_matches = int(sum(all_counts[m, (m & -m).bit_length() - 1] for m in range(1, 64)))

# Identical-weight ceilings for the original validation statistic: equal
# training seeds and factors, with run records pooled within each factor.
val_attempt_fractions = []
val_selected_scores = []
val_selected_pooled_matches = 0
val_eligible_total = 0
val_attempt_total = 0
for si, record in enumerate(original):
    val_groups = cube[si][:, VAL].sum(axis=1, dtype=np.int64)
    val_attempt_fractions.append(float(np.mean(val_groups[:, :, :6].sum(axis=(1, 2)) / val_groups.sum(axis=(1, 2)))))
    val_selected_scores.append(record["calibration_selected_validation_agreement"])
    val_selected_pooled_matches += record["selected_validation_pooled_matches"]
    val_eligible_total += record["validation_count"]
    val_attempt_total += int(val_groups[:, :, :6].sum())


# 378-case coordinate-control construction.
control_logical = np.zeros((64, 6), dtype=np.int64)
control_physical = np.zeros((64, 6), dtype=np.int64)
for shift in range(6):
    perm = tuple((logical + shift) % 6 for logical in range(6))
    for logical_mask in range(1, 64):
        logical_action = (logical_mask & -logical_mask).bit_length() - 1
        control_logical[logical_mask, logical_action] += 1
        physical_mask = sum(1 << perm[c] for c in range(6) if logical_mask & (1 << c))
        physical_action = perm[logical_action]
        control_physical[physical_mask, physical_action] += 1
control_logical_cert = graph_certificate(control_logical)
control_physical_cert = graph_certificate(control_physical)
logical_best_matches = max(
    sum(int(control_logical[m, PRED[oi, m]]) for m in range(1, 64))
    for oi in range(720)
)
physical_best_matches = max(
    sum(int(control_physical[m, PRED[oi, m]]) for m in range(1, 64))
    for oi in range(720)
)
control = {
    "case_count": int(control_logical.sum()),
    "logical_exact_agreement": logical_best_matches / int(control_logical.sum()),
    "logical": control_logical_cert,
    "physical_masks_with_multiple_selected_actions": int(
        sum(np.count_nonzero(control_physical[m]) > 1 for m in range(1, 64))
    ),
    "physical_best_order_agreement": physical_best_matches / int(control_physical.sum()),
    "physical": control_physical_cert,
}


def numeric(r: dict[str, str], field: str, *, denominator: float | None = None) -> float:
    raw = r[field]
    if raw == "":
        return float("nan")
    value = float(raw)
    return value if denominator is None else value / denominator


metric_getters = {
    "goodput_packets_per_slot": lambda r: numeric(r, "network_goodput"),
    "attempts_per_slot": lambda r: numeric(r, "attempted", denominator=3000),
    "collision_per_attempt": lambda r: numeric(r, "collision_attempt_rate"),
    "link_failure_per_singleton": lambda r: numeric(r, "conditional_link_failure_rate"),
    "voluntary_defer_per_node_slot": lambda r: numeric(r, "voluntary_deferrals", denominator=45000),
    "forced_defer_per_node_slot": lambda r: numeric(r, "forced_deferrals", denominator=45000),
    "opportunity_utilization": lambda r: numeric(r, "opportunity_utilization"),
    "finite_window_service_fraction": lambda r: numeric(r, "finite_window_service_fraction"),
    "mean_network_backlog_packets": lambda r: numeric(r, "mean_network_backlog"),
    "ending_network_backlog_packets": lambda r: numeric(r, "ending_backlog"),
    "delivery_jain_fairness_all_15_nodes": lambda r: numeric(r, "delivery_jain_fairness"),
    "zero_delivery_run_fraction": lambda r: numeric(r, "zero_delivery_run"),
}


def macro(policy: str, getter, arrival: str | None = None) -> float | None:
    selected_factors = [f for f in FACTORS if arrival is None or f.endswith(arrival)]
    factor_values = []
    for factor in selected_factors:
        fr = [r for r in rows if r["policy"] == policy and r["factor_id"] == factor]
        if policy in LEARNED:
            seed_values = []
            for seed in TRAINING_SEEDS:
                vals = np.asarray([getter(r) for r in fr if int(r["training_seed"]) == seed], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    seed_values.append(float(vals.mean()))
            if seed_values:
                factor_values.append(float(np.mean(seed_values)))
        else:
            vals = np.asarray([getter(r) for r in fr], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                factor_values.append(float(vals.mean()))
    return None if not factor_values else float(np.mean(factor_values))


operational = {}
for policy in POLICIES:
    operational[policy] = {}
    for stratum, suffix in (("all", None), ("arrival_0.10", "a10"), ("arrival_0.25", "a25")):
        operational[policy][stratum] = {
            name: macro(policy, getter, suffix) for name, getter in metric_getters.items()
        }


# Quality-aware optimistic service upper bound by product Gauss-Legendre
# integration.  Conditional on common quality C=c, each local quality is
# clip(c + Normal(0,.1), .05, .99); E[max Q] is .05 plus its survival integral.
def max_quality_expectation(nodes: int, quadrature_order: int) -> float:
    zc, wc = np.polynomial.legendre.leggauss(quadrature_order)
    zq, wq = np.polynomial.legendre.leggauss(quadrature_order)
    c = (zc + 1.0) / 2.0
    wc = wc / 2.0
    q = 0.05 + (zq + 1.0) * (0.99 - 0.05) / 2.0
    wq = wq * (0.99 - 0.05) / 2.0
    beta_pdf = 30.0 * c**4 * (1.0 - c)  # Beta(5,2)
    conditional = 0.05 + np.sum(
        wq[:, None] * (1.0 - ndtr((q[:, None] - c[None, :]) / 0.10) ** nodes),
        axis=0,
    )
    return float(np.sum(wc * beta_pdf * conditional))


emq = {str(order): max_quality_expectation(15, order) for order in (100, 200, 400, 800)}
emq800 = emq["800"]
load = {}
for rho in (0.30, 0.60):
    bound = 6.0 * (1.0 - rho) * emq800
    for arrival in (0.10, 0.25):
        offered = 15.0 * arrival
        load[f"rho_{rho:.2f}_arrival_{arrival:.2f}"] = {
            "offered_packets_per_slot": offered,
            "optimistic_service_upper_bound": bound,
            "offered_exceeds_bound": bool(offered > bound),
        }


# Retained episode-level training evidence and metadata consistency.
training = {}
metadata_problems = []
for policy in LEARNED:
    early = defaultdict(list)
    late = defaultdict(list)
    all_ep = defaultdict(list)
    per_seed_changes = defaultdict(list)
    for seed in TRAINING_SEEDS:
        run_dir = RESULTS / f"training/{policy}/seed_{seed}"
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        for field, expected in {
            "global_environment_steps": 100000,
            "gradient_steps": 99992,
            "replay_total_insertions": 1500000,
            "replay_node_insertions": [100000] * 15,
            "selection_rule": "final_iterate_only",
        }.items():
            if metadata.get(field) != expected:
                metadata_problems.append([policy, seed, field, metadata.get(field), expected])
        with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as f:
            eps = list(csv.DictReader(f))
        assert len(eps) == 400
        fields = (
            "epsilon_final", "mean_node_reward", "deliveries",
            "collision_attempts", "mean_loss",
        )
        for field in fields:
            values = np.asarray([float(e[field]) for e in eps], dtype=float)
            all_ep[field].extend(values.tolist())
            early[field].extend(values[:96].tolist())
            late[field].extend(values[-96:].tolist())
            per_seed_changes[field].append(float(values[-96:].mean() - values[:96].mean()))
    training[policy] = {
        "retained_episode_rows": len(TRAINING_SEEDS) * 400,
        "first_96_episode_mean": {k: float(np.mean(v)) for k, v in early.items()},
        "last_96_episode_mean": {k: float(np.mean(v)) for k, v in late.items()},
        "last_minus_first_96_mean": {k: float(np.mean(late[k]) - np.mean(early[k])) for k in early},
        "per_seed_last_minus_first_96_range": {
            k: [float(np.min(v)), float(np.max(v))] for k, v in per_seed_changes.items()
        },
        "all_400_episode_mean": {k: float(np.mean(v)) for k, v in all_ep.items()},
        "logged_fields": [
            "episode", "factor_id", "environment_seed", "label_shift",
            "epsilon_final", "mean_node_reward", "deliveries",
            "collision_attempts", "mean_loss", "trace_sha256",
        ],
        "queue_fields_present": [],
    }


payload = {
    "source_hashes": {
        "evaluation_summary_csv": sha(SUMMARY),
        "released_analysis_summary_json": sha(RESULTS / "analysis/summary.json"),
        "released_mechanism_certificates_csv": sha(cert_path),
        "results_seal_json": sha(RESULTS / "RESULTS_SEAL.json"),
        "config_py": sha(ROOT / "main/revision/crn_ca_pr1_20261017/config.py"),
        "evaluate_py": sha(ROOT / "main/revision/crn_ca_pr1_20261017/evaluate.py"),
        "environment_py": sha(ROOT / "main/revision/crn_ca_pr1_20261017/environment.py"),
        "train_py": sha(ROOT / "main/revision/crn_ca_pr1_20261017/train.py"),
        "analyze_py": sha(ROOT / "main/revision/crn_ca_pr1_20261017/analyze.py"),
    },
    "row_counts": {
        "total": len(rows),
        "full_dqn": len(full_rows),
    },
    "summary_identity_violations": dict(identity_violations),
    "original_seven_action_reproduction_errors": reproduction_errors,
    "original_seven_action": {
        "coordinate_frame": "logical (mask and action are logged before logical-to-physical mapping)",
        "full_pooled_eligible": all_eligible,
        "full_pooled_abstentions": all_abstain,
        "full_pooled_attempts": all_attempts,
        "full_pooled_attempt_ceiling": all_attempts / all_eligible,
        "full_pooled_ascending_order_matches": ascending_matches,
        "full_pooled_ascending_order_agreement": ascending_matches / all_eligible,
        "validation_equal_checkpoint_factor_selected_agreement": float(np.mean(val_selected_scores)),
        "validation_equal_checkpoint_factor_attempt_ceiling_same_weights": float(np.mean(val_attempt_fractions)),
        "validation_pooled_selected_matches": val_selected_pooled_matches,
        "validation_pooled_eligible": val_eligible_total,
        "validation_pooled_selected_agreement": val_selected_pooled_matches / val_eligible_total,
        "validation_pooled_attempts": val_attempt_total,
        "validation_pooled_attempt_ceiling": val_attempt_total / val_eligible_total,
        "per_checkpoint": original,
    },
    "attempted_only_logical_audit": {
        "validation_equal_checkpoint_factor_selected_agreement": float(np.mean([r["calibration_selected_validation_agreement"] for r in attempted])),
        "validation_equal_checkpoint_factor_validation_optimal_agreement": float(np.mean([r["validation_optimal_agreement"] for r in attempted])),
        "per_checkpoint": attempted,
    },
    "coordinate_positive_control": control,
    "operational_equal_cell_macro": operational,
    "load_bound": {
        "expected_best_of_15_local_link_qualities_by_quadrature_order": emq,
        "formula": "6*(1-rho)*E[max_i clip(C+eps_i,0.05,0.99)], C~Beta(5,2), eps_i iid N(0,0.1^2)",
        "cells": load,
    },
    "training_logs": training,
    "training_metadata_problems": metadata_problems,
    "availability_limits": {
        "run_level_fairness_recalculation": "blocked: summary.csv retains only the scalar Jain index, not 15 per-node delivery counts; source code shows all 15 nodes, including zeros, enter _jain",
        "evaluation_queue_quantiles_or_fraction_ge_50": "blocked: aggregate summary has starting/mean/ending network backlog only, not raw per-node queues",
        "training_queue_exposure": "blocked: episodes.csv has no queue/backlog fields and final checkpoints cannot reconstruct original exploratory trajectories",
        "attempted_contextual_node_phase_audit": "blocked from aggregate mask_action_counts: records omit node and phase dimensions",
    },
}

OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

audit_fields = (
    "training_seed", "calibration_count", "validation_count",
    "calibration_mask_coverage", "validation_mask_coverage",
    "calibration_optimal_agreement", "calibration_tie_count",
    "selected_order_original_validation_tie_rule",
    "calibration_selected_validation_agreement",
    "calibration_tie_validation_envelope_min",
    "calibration_tie_validation_envelope_max",
    "validation_optimal_agreement", "validation_optimal_tie_count",
    "validation_optimal_orders", "calibration_edge_count",
    "calibration_acyclic", "calibration_representing_order_count",
    "validation_edge_count", "validation_acyclic",
    "validation_representing_order_count", "full_edge_count",
    "full_acyclic", "full_representing_order_count",
    "full_counts_64x6_le_i8_sha256",
)
with AUDIT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=audit_fields, lineterminator="\n")
    writer.writeheader()
    for r in attempted:
        writer.writerow({
            "training_seed": r["training_seed"],
            "calibration_count": r["calibration_count"],
            "validation_count": r["validation_count"],
            "calibration_mask_coverage": r["calibration_mask_coverage"],
            "validation_mask_coverage": r["validation_mask_coverage"],
            "calibration_optimal_agreement": r["calibration_optimal_agreement"],
            "calibration_tie_count": r["calibration_tie_count"],
            "selected_order_original_validation_tie_rule": r["selected_order_original_validation_tie_rule"],
            "calibration_selected_validation_agreement": r["calibration_selected_validation_agreement"],
            "calibration_tie_validation_envelope_min": r["calibration_tie_validation_envelope"][0],
            "calibration_tie_validation_envelope_max": r["calibration_tie_validation_envelope"][1],
            "validation_optimal_agreement": r["validation_optimal_agreement"],
            "validation_optimal_tie_count": r["validation_optimal_tie_count"],
            "validation_optimal_orders": ";".join(r["validation_optimal_orders"]),
            "calibration_edge_count": r["calibration_certificate"]["edge_count"],
            "calibration_acyclic": r["calibration_certificate"]["acyclic"],
            "calibration_representing_order_count": r["calibration_certificate"]["representing_order_count"],
            "validation_edge_count": r["validation_certificate"]["edge_count"],
            "validation_acyclic": r["validation_certificate"]["acyclic"],
            "validation_representing_order_count": r["validation_certificate"]["representing_order_count"],
            "full_edge_count": r["full_support_certificate"]["edge_count"],
            "full_acyclic": r["full_support_certificate"]["acyclic"],
            "full_representing_order_count": r["full_support_certificate"]["representing_order_count"],
            "full_counts_64x6_le_i8_sha256": r["full_support_certificate"]["counts_64x6_le_i8_sha256"],
        })

with FIGURE_CSV.open("w", newline="", encoding="utf-8") as f:
    fields = ("training_seed", "series", "agreement")
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for original_r, attempt_r in zip(original, attempted, strict=True):
        si = seed_i[original_r["training_seed"]]
        val_groups = cube[si][:, VAL].sum(axis=1, dtype=np.int64)
        equal_factor_attempt_ceiling = float(
            np.mean(
                val_groups[:, :, :6].sum(axis=(1, 2))
                / val_groups.sum(axis=(1, 2))
            )
        )
        for series, agreement in (
            ("seven_action_selected", original_r["calibration_selected_validation_agreement"]),
            ("seven_action_attempt_ceiling", equal_factor_attempt_ceiling),
            ("attempted_only_calibration_selected", attempt_r["calibration_selected_validation_agreement"]),
            ("attempted_only_validation_optimal", attempt_r["validation_optimal_agreement"]),
        ):
            writer.writerow({
                "training_seed": original_r["training_seed"],
                "series": series,
                "agreement": agreement,
            })

print(json.dumps({
    "output": str(OUT),
    "output_sha256": sha(OUT),
    "attempted_audit_csv": str(AUDIT_CSV),
    "attempted_audit_csv_sha256": sha(AUDIT_CSV),
    "figure_csv": str(FIGURE_CSV),
    "figure_csv_sha256": sha(FIGURE_CSV),
    "summary_sha256": sha(SUMMARY),
    "identity_violations": dict(identity_violations),
    "original_reproduction_errors": reproduction_errors,
    "pooled_eligible": all_eligible,
    "pooled_abstentions": all_abstain,
    "pooled_attempts": all_attempts,
    "control": control,
    "emq": emq,
}, indent=2, sort_keys=True))
