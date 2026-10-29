#!/usr/bin/env python3
"""Rebuild the archived-study Gold record from public package members."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

STUDY_ID = "crn-2026-003-finite-domain-dsa"
DEFAULT_OUTPUT = Path(f"studies/{STUDY_ID}/evidence/gold/archived/recomputed.json")
COMPACT_MEMBER = Path(
    "independent_replications/independent_masked_priority/"
    "CRN_MP_PR1_20261017_Compact_Replication.zip"
)
REPLICATION_MEMBER = "results/audits/priority_collapse/priority_collapse_audit.json"
ORIGINAL_COMPACT_SHA256 = "c3c71c75fbd735517602ab9973012465d286deee898835a8604a47c9ea3269eb"
PUBLIC_COMPACT_SHA256 = "d90fb6316b3f551ef2a31578b33c9608e65eaf0185473d0d28a9d6d3d75c55be"
INPUTS = {
    "priority_collapse": (
        Path("outputs/priority_collapse_audit.json"),
        "5f9e6b90feacf59d67f06e1e3bd5fe1278b7f8c111da4025cdf57ea7015a19d6",
    ),
    "priority_positive_controls": (
        Path("outputs/priority_positive_controls_audit.json"),
        "87dd6318d7dea9d98ba84c8309c4a9bb6d1ea8ff963076dcd113f30f6694eac1",
    ),
    "feature_adversarial": (
        Path("outputs/feature_adversarial_audit.json"),
        "ebfb66657c2944f393598bcd24b441d99442295e5ae166232f2b188c3a325dab",
    ),
    "generator_dispersion": (
        Path("outputs/generator_dispersion_path_audit.json"),
        "34bac673c417940c133e72dcf8f4990bdb2fcfcf36d910dbcbd43727c36f5e63",
    ),
    "archived_figure_data": (
        Path("figure_data/evidence_figure_data.json"),
        "985c718c5016d8e19df3dfc8a97a30d025b40dcbd51224eee56011662f31a0e5",
    ),
}
ARCHIVAL_RELEASE_ORIGINS = {
    "priority_collapse": "reproducibility/outputs/priority_collapse_audit.json",
    "priority_positive_controls": ("reproducibility/outputs/priority_positive_controls_audit.json"),
    "feature_adversarial": "reproducibility/outputs/feature_adversarial_audit.json",
    "generator_dispersion": ("reproducibility/outputs/generator_dispersion_path_audit.json"),
    "direct_replication_zip": COMPACT_MEMBER.as_posix(),
    "archived_figure_data": "figure_data/archived_evidence_figure_data.json",
}
CURRENT_PACKAGE_MEMBERS = {name: path.as_posix() for name, (path, _) in INPUTS.items()} | {
    "direct_replication_zip": COMPACT_MEMBER.as_posix()
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verified_bytes(path: Path, expected_sha256: str, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is absent or symbolic: {path}")
    payload = path.read_bytes()
    actual = sha256_bytes(payload)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return payload


def verified_json(root: Path, name: str) -> dict[str, Any]:
    relative, expected = INPUTS[name]
    return json.loads(verified_bytes(root / relative, expected, name))


def build_record(release_root: Path) -> dict[str, Any]:
    root = release_root.resolve()
    priority = verified_json(root, "priority_collapse")
    controls = verified_json(root, "priority_positive_controls")
    adversarial = verified_json(root, "feature_adversarial")
    dispersion = verified_json(root, "generator_dispersion")
    figure_data = verified_json(root, "archived_figure_data")

    compact_payload = verified_bytes(
        root / COMPACT_MEMBER, PUBLIC_COMPACT_SHA256, "public compact replication"
    )
    sanitization = json.loads((root / "PUBLIC_SANITIZATION.json").read_text(encoding="utf-8"))
    if sanitization.get("original_archive_sha256") != ORIGINAL_COMPACT_SHA256:
        raise ValueError("sanitization record does not bind the sealed compact archive")
    if sanitization.get("sanitized_archive_sha256") != sha256_bytes(compact_payload):
        raise ValueError("sanitization record does not bind the public compact derivative")
    with ZipFile(root / COMPACT_MEMBER) as archive:
        replication_payload = archive.read(REPLICATION_MEMBER)
    replication = json.loads(replication_payload)

    original_aggregate = priority["aggregate"]
    original_per_checkpoint = priority["per_checkpoint"]
    analytical = priority["analytical_references"]
    positive_records = controls["archived_controls"]["records"]
    untrained = controls["untrained_dqn_controls"]["records"]
    adversarial_aggregate = adversarial["aggregate"]
    stress = adversarial["seed_202_random_cube_stress"]

    full_core_node_cycles = 36_000_000
    forced_abstentions = full_core_node_cycles - original_aggregate["full_core_attempts"]
    constructed_states_per_checkpoint = {
        record["counterfactual_states"] for record in original_per_checkpoint
    }
    if len(constructed_states_per_checkpoint) != 1:
        raise ValueError("counterfactual state count is not constant across checkpoints")
    constructed_states_each = constructed_states_per_checkpoint.pop()

    predicted_by_rank = analytical["S4"]["predicted_acceptance_by_tone_rank"]
    tone_errors_pp = [
        abs(record["s4_acceptance"] - predicted_by_rank[str(record["tone_channel_rank"])]) * 100.0
        for record in original_per_checkpoint
    ]
    shared_mask = analytical["S1"]["shared_mask_destructive_collision"]
    sweep = analytical["S1_shared_mask_offered_load_sweep"]
    mixed_below_uniform_all_loads = all(
        row["fitted_heterogeneous_rotation_mean"]["expected_singleton_successes_per_cycle"]
        < row["uniform_random_legal"]["expected_singleton_successes_per_cycle"]
        for row in sweep
    )

    input_records = {
        name: {
            "archival_release_origin": ARCHIVAL_RELEASE_ORIGINS[name],
            "current_package_member": CURRENT_PACKAGE_MEMBERS[name],
            "sha256": expected,
        }
        for name, (_, expected) in sorted(INPUTS.items())
    }
    input_records["direct_replication_zip"] = {
        "archival_release_origin": ARCHIVAL_RELEASE_ORIGINS["direct_replication_zip"],
        "current_package_member": CURRENT_PACKAGE_MEMBERS["direct_replication_zip"],
        "original_archive_sha256": ORIGINAL_COMPACT_SHA256,
        "public_derivative_sha256": PUBLIC_COMPACT_SHA256,
    }
    input_records["direct_replication_member"] = {
        "archive_member": REPLICATION_MEMBER,
        "sha256": sha256_bytes(replication_payload),
    }

    return {
        "schema_version": 1,
        "study": "Study A: archived masked-priority evidence",
        "derivation": "hash-verified deterministic field selection from sealed legacy evidence",
        "inputs": input_records,
        "priority_certificate": {
            "checkpoint_count": len(original_per_checkpoint),
            "calibration_attempts": original_aggregate["calibration_attempts"],
            "heldout_attempts": original_aggregate["heldout_attempts"],
            "full_core_attempts": original_aggregate["full_core_attempts"],
            "full_core_node_cycles": full_core_node_cycles,
            "full_core_forced_abstentions": forced_abstentions,
            "all_nonempty_masks_per_checkpoint": min(
                record["full_core_distinct_nonempty_masks"] for record in original_per_checkpoint
            ),
            "all_agreements_exact": all(
                record[key] == 1.0
                for record in original_per_checkpoint
                for key in (
                    "calibration_agreement",
                    "heldout_agreement",
                    "full_core_agreement",
                )
            ),
            "s4_attempts": original_aggregate["s4_attempts"],
            "maximum_tone_acceptance_absolute_error_percentage_points": max(tone_errors_pp),
        },
        "direct_replication": {
            "checkpoint_count": len(replication["per_checkpoint"]),
            "heldout_attempts": replication["aggregate"]["heldout_attempts"],
            "full_core_attempts": replication["aggregate"]["full_core_attempts"],
            "all_agreements_exact": all(
                record[key] == 1.0
                for record in replication["per_checkpoint"]
                for key in (
                    "calibration_agreement",
                    "heldout_agreement",
                    "full_core_agreement",
                )
            ),
        },
        "assay_controls": {
            "quality_greedy_heldout_agreement": next(
                record["heldout_without_refitting"]["agreement"]
                for record in positive_records
                if record["policy"] == "quality_greedy"
            ),
            "round_robin_heldout_agreement": next(
                record["heldout_without_refitting"]["agreement"]
                for record in positive_records
                if record["policy"] == "round_robin"
            ),
            "positive_control_masks_with_multiple_actions": sorted(
                {
                    record["heldout_without_refitting"]["masks_with_multiple_observed_actions"]
                    for record in positive_records
                }
            ),
            "untrained_agreement_range": [
                min(record["agreement"] for record in untrained),
                max(record["agreement"] for record in untrained),
            ],
            "untrained_checkpoint_count": len(untrained),
            "untrained_feature_vectors_per_mask": untrained[0]["samples_per_nonempty_mask"],
        },
        "feature_falsification": {
            "constructed_states_per_checkpoint": constructed_states_each,
            "constructed_states_total": constructed_states_each * len(original_per_checkpoint),
            "balanced_random_cube_draws_total": adversarial_aggregate["balanced_random_cube_draws"],
            "balanced_random_cube_flips": adversarial_aggregate["balanced_random_cube_flips"],
            "seed_202_stress_draws": stress["draws"],
            "seed_202_stress_flips": stress["flips"],
            "seed_202_stress_distinct_masks": stress["distinct_flip_masks"],
            "targeted_problem_count": adversarial_aggregate["mask_target_problems"],
            "targeted_crossing_problem_count": adversarial_aggregate[
                "problems_with_nonpositive_margin_found"
            ],
            "targeted_crossing_checkpoint_mask_pairs": adversarial_aggregate[
                "checkpoint_mask_pairs_with_nonpositive_margin_found"
            ],
            "targeted_crossing_checkpoint_count": sum(
                1
                for record in adversarial["per_checkpoint"]
                if record["problems_with_nonpositive_margin_found"] > 0
            ),
            "minimum_pairwise_q_margin": adversarial_aggregate["minimum_pairwise_q_margin_found"],
            "bounded_dynamic_feature_count": 24,
        },
        "analytical_reconstruction": {
            "s1_homogeneous_expected_spread": analytical["S1"]["independent_masks"][
                "fixed_priority_expected_unique_channels"
            ],
            "s1_homogeneous_observed_spread": figure_data["s1_spread"]["observed"][
                "homogeneous_dqn"
            ],
            "s1_mixed_expected_spread": analytical["S1"]["independent_masks"][
                "fitted_heterogeneous_rotation_expected_unique_channels"
            ],
            "s1_mixed_observed_spread": figure_data["s1_spread"]["observed"]["mixed_checkpoints"],
            "s1_uniform_expected_spread": analytical["S1"]["independent_masks"][
                "uniform_legal_expected_unique_channels"
            ],
            "s1_quality_greedy_observed_spread": figure_data["s1_spread"]["observed"][
                "quality_greedy"
            ],
        },
        "generator_path": dispersion["aggregate"],
        "shared_mask_sensitivity": {
            "homogeneous_singleton_successes_per_cycle": shared_mask["homogeneous_fitted_priority"][
                "expected_singleton_successes_per_cycle"
            ],
            "mixed_singleton_successes_per_cycle": shared_mask[
                "fitted_heterogeneous_rotation_mean"
            ]["expected_singleton_successes_per_cycle"],
            "mixed_singleton_delivery_fraction": shared_mask["fitted_heterogeneous_rotation_mean"][
                "destructive_collision_delivery_fraction"
            ],
            "uniform_singleton_successes_per_cycle": shared_mask["uniform_random_legal"][
                "expected_singleton_successes_per_cycle"
            ],
            "uniform_singleton_delivery_fraction": shared_mask["uniform_random_legal"][
                "destructive_collision_delivery_fraction"
            ],
            "evaluated_load_count": len(sweep),
            "minimum_load": min(row["offered_load_probability"] for row in sweep),
            "maximum_load": max(row["offered_load_probability"] for row in sweep),
            "mixed_below_uniform_at_all_evaluated_loads": mixed_below_uniform_all_loads,
        },
    }


def render_record(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.release_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(render_record(build_record(args.release_root)))


if __name__ == "__main__":
    main()
