#!/usr/bin/env python3
"""Generate the article's result-bearing TeX macros from canonical Gold data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


STUDY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = STUDY_ROOT / "evidence/gold/manuscript/result_macros.tex"
DEFAULT_MANIFEST = STUDY_ROOT / "evidence/gold/manuscript/result_macros.manifest.json"
DEFAULT_MANUSCRIPT_COPY = STUDY_ROOT.parents[1] / "main/manuscript/result_macros.tex"

SOURCES = {
    "archived": (
        STUDY_ROOT / "evidence/gold/archived/recomputed.json",
        "8834e79f4624ea1ae402f119ef7e7ee1f07520fc29caa5de6088f077d70162ca",
    ),
    "inference": (
        STUDY_ROOT / "evidence/gold/inference/recomputed.json",
        "5199f52a44d119010207c2e0358217d2a14ed238f57bb9a743f76f7c189ccca3",
    ),
    "prospective_parent": (
        STUDY_ROOT / "evidence/gold/inference/parent_summary.json",
        "c3fc74b020a801baee303e1d6741256af89c8c53d540f4e18de6da1ddcc7d812",
    ),
    "mechanism": (
        STUDY_ROOT / "evidence/gold/reanalysis/reanalysis_output.json",
        "ab6ccafc932ddca28a2aec40d6e8f46091ab17992955aed988a764649affe2e0",
    ),
    "figure_data": (
        STUDY_ROOT / "evidence/gold/figure_inputs/collision_aware_figure_data.csv",
        "751647f671780d0fa0e38756c22485d0e70cf8d9ccea97e33e39f21dcb15ad6b",
    ),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_bytes(name: str) -> bytes:
    path, expected = SOURCES[name]
    payload = path.read_bytes()
    actual = sha256_bytes(payload)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return payload


def verified_json(name: str) -> dict[str, Any]:
    return json.loads(verified_bytes(name))


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def fixed(value: float, places: int) -> str:
    return f"{value:.{places}f}"


def percent(value: float, places: int) -> str:
    return fixed(100.0 * value, places) + r"\%"


def scientific(value: float, places: int) -> str:
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.{places}e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def minmax(values: list[float]) -> tuple[float, float]:
    return min(values), max(values)


def build_macros() -> dict[str, str]:
    archived = verified_json("archived")
    inference = verified_json("inference")
    parent = verified_json("prospective_parent")
    mechanism = verified_json("mechanism")
    figure_payload = verified_bytes("figure_data").decode("utf-8")
    figure_rows = list(csv.DictReader(figure_payload.splitlines()))

    comparisons = inference["inference"]["comparisons"]
    policies = inference["operational"]["policies"]
    phase = comparisons["phase_orthogonal"]
    history = comparisons["history_free_dqn"]
    quality = comparisons["quality_greedy_persistent"]
    coordination = comparisons["coordination_blind_dqn"]
    attempted = mechanism["attempted_only_logical_audit"]
    attempted_records = attempted["per_checkpoint"]
    seven = mechanism["original_seven_action"]
    seven_records = seven["per_checkpoint"]
    coordinate = mechanism["coordinate_positive_control"]
    archived_priority = archived["priority_certificate"]
    archived_replication = archived["direct_replication"]
    feature = archived["feature_falsification"]
    analytical = archived["analytical_reconstruction"]
    generator = archived["generator_path"]
    shared = archived["shared_mask_sensitivity"]
    gate = parent["confirmatory_gates"]["finite_support_mechanism_departure"]

    phase_cells = [
        float(row["value"])
        for row in figure_rows
        if row["record_type"] == "cell_effect"
        and row["comparator"] == "phase_orthogonal"
    ]
    history_cells = [
        float(row["value"])
        for row in figure_rows
        if row["record_type"] == "cell_effect"
        and row["comparator"] == "history_free_dqn"
    ]
    factor_cells = {row["factor_id"] for row in figure_rows if row["factor_id"]}
    phase_cell_min, phase_cell_max = minmax(phase_cells)
    history_cell_min, history_cell_max = minmax(history_cells)

    calibration_counts = [row["calibration_count"] for row in attempted_records]
    validation_counts = [row["validation_count"] for row in attempted_records]
    calibration_agreements = [
        row["calibration_optimal_agreement"] for row in attempted_records
    ]
    validation_selected_agreements = [
        row["calibration_selected_validation_agreement"]
        for row in attempted_records
    ]
    changed_validation_order_count = sum(
        row["selected_order_original_validation_tie_rule"]
        not in row["validation_optimal_orders"]
        for row in attempted_records
    )

    full = policies["collision_aware_dqn"]["macro"]
    history_policy = policies["history_free_dqn"]["macro"]
    phase_policy = policies["phase_orthogonal"]["macro"]
    quality_policy = policies["quality_greedy_persistent"]["macro"]
    coordination_policy = policies["coordination_blind_dqn"]["macro"]
    centralized_policy = policies["centralized_quality_matching"]["macro"]
    run_count = inference["input"]["row_count"]
    training_units = len(attempted_records)
    environment_blocks_per_cell = len(
        gate["calibration_indices"] + gate["validation_indices"]
    )
    learned_checkpoint_count = training_units * 3

    # Fail closed on cross-source identities central to the article.
    identities = [
        (full["goodput"], mechanism["operational_equal_cell_macro"]["collision_aware_dqn"]["all"]["goodput_packets_per_slot"]),
        (history_policy["goodput"], mechanism["operational_equal_cell_macro"]["history_free_dqn"]["all"]["goodput_packets_per_slot"]),
        (phase_policy["goodput"], mechanism["operational_equal_cell_macro"]["phase_orthogonal"]["all"]["goodput_packets_per_slot"]),
        (quality_policy["goodput"], mechanism["operational_equal_cell_macro"]["quality_greedy_persistent"]["all"]["goodput_packets_per_slot"]),
    ]
    if any(abs(left - right) > 1e-12 for left, right in identities):
        raise ValueError("cross-source operational identity failed")
    if sum(validation_counts) != seven["validation_pooled_attempts"]:
        raise ValueError("attempted-only validation total does not match seven-action source")

    phase_train_range = phase["sensitivity"]["leave_one_training_unit"]["range"]
    phase_env_range = phase["sensitivity"][
        "leave_one_environment_block_shift_preserving"
    ]["range"]
    history_train_range = history["sensitivity"]["leave_one_training_unit"]["range"]
    history_env_range = history["sensitivity"][
        "leave_one_environment_block_shift_preserving"
    ]["range"]
    history_checkpoint_range = history["sensitivity"]["checkpoint_effect_range"]

    macros = {
        # Study and support sizes.
        "ResultArchiveCheckpointCount": str(archived_priority["checkpoint_count"]),
        "ResultReplicationCheckpointCount": str(archived_replication["checkpoint_count"]),
        "ResultUntrainedCheckpointCount": str(
            archived["assay_controls"]["untrained_checkpoint_count"]
        ),
        "ResultUntrainedFeatureVectorsPerMask": thousands(
            archived["assay_controls"]["untrained_feature_vectors_per_mask"]
        ),
        "ResultTrainingUnitCount": str(training_units),
        "ResultFactorCellCount": str(len(factor_cells)),
        "ResultEnvironmentBlocksPerCell": str(environment_blocks_per_cell),
        "ResultLearnedCheckpointCount": str(learned_checkpoint_count),
        "ResultInterventionCheckpointCount": str(learned_checkpoint_count - training_units),
        "ResultAllReleasedCheckpointCount": str(
            archived_priority["checkpoint_count"]
            + archived_replication["checkpoint_count"]
            + learned_checkpoint_count
        ),
        "ResultProspectiveRunCount": thousands(run_count),
        # Study A finite-support and falsification results.
        "ResultArchiveCalibrationAttempts": thousands(archived_priority["calibration_attempts"]),
        "ResultArchiveHeldoutAttempts": thousands(archived_priority["heldout_attempts"]),
        "ResultArchiveFullCoreAttempts": thousands(archived_priority["full_core_attempts"]),
        "ResultArchiveForcedAbstentions": thousands(archived_priority["full_core_forced_abstentions"]),
        "ResultReplicationHeldoutAttempts": thousands(archived_replication["heldout_attempts"]),
        "ResultReplicationFullCoreAttempts": thousands(archived_replication["full_core_attempts"]),
        "ResultNonemptyMaskCount": str(archived_priority["all_nonempty_masks_per_checkpoint"]),
        "ResultQualityGreedyAgreement": fixed(archived["assay_controls"]["quality_greedy_heldout_agreement"], 6),
        "ResultRoundRobinAgreement": fixed(archived["assay_controls"]["round_robin_heldout_agreement"], 6),
        "ResultAssayMultipleActionMaskCount": str(archived["assay_controls"]["positive_control_masks_with_multiple_actions"][0]),
        "ResultUntrainedAgreementMinimum": fixed(archived["assay_controls"]["untrained_agreement_range"][0], 6),
        "ResultUntrainedAgreementMaximum": fixed(archived["assay_controls"]["untrained_agreement_range"][1], 6),
        "ResultConstructedStatesPerCheckpoint": thousands(feature["constructed_states_per_checkpoint"]),
        "ResultConstructedStatesTotal": thousands(feature["constructed_states_total"]),
        "ResultBalancedDrawCount": f"{feature['balanced_random_cube_draws_total'] / 1_000_000:.0f} million",
        "ResultBalancedFlipCount": str(feature["balanced_random_cube_flips"]),
        "ResultStressDrawCount": "ten-million",
        "ResultStressFlipCount": str(feature["seed_202_stress_flips"]),
        "ResultStressFlipMaskCount": str(feature["seed_202_stress_distinct_masks"]),
        "ResultTargetProblemCount": thousands(feature["targeted_problem_count"]),
        "ResultTargetProblemsPerCheckpoint": str(feature["targeted_problem_count"] // archived_priority["checkpoint_count"]),
        "ResultTargetCrossingCount": str(feature["targeted_crossing_problem_count"]),
        "ResultTargetCrossingPairCount": str(feature["targeted_crossing_checkpoint_mask_pairs"]),
        "ResultTargetCrossingCheckpointCount": str(feature["targeted_crossing_checkpoint_count"]),
        "ResultMinimumQMargin": fixed(feature["minimum_pairwise_q_margin"], 6),
        "ResultArchiveFeatureDimension": str(feature["bounded_dynamic_feature_count"]),
        "ResultHomogeneousSpreadExpected": fixed(analytical["s1_homogeneous_expected_spread"], 6),
        "ResultHomogeneousSpreadObserved": fixed(analytical["s1_homogeneous_observed_spread"], 6),
        "ResultMixedSpreadExpected": fixed(analytical["s1_mixed_expected_spread"], 6),
        "ResultMixedSpreadObserved": fixed(analytical["s1_mixed_observed_spread"], 6),
        "ResultUniformSpreadExpected": fixed(analytical["s1_uniform_expected_spread"], 6),
        "ResultQualitySpreadObserved": fixed(analytical["s1_quality_greedy_observed_spread"], 6),
        "ResultArchiveSFourAttempts": thousands(archived_priority["s4_attempts"]),
        "ResultToneMaximumErrorPercentagePoints": fixed(archived_priority["maximum_tone_acceptance_absolute_error_percentage_points"], 2),
        "ResultGeneratorPairCount": fixed(generator["rows"] / 1_000_000, 1) + " million",
        "ResultGeneratorQualityChanged": thousands(generator["selected_quality_changed"]),
        "ResultGeneratorMeanAbsoluteChange": fixed(generator["selected_quality_mean_absolute_delta"], 5),
        "ResultSharedMixedSingletons": fixed(shared["mixed_singleton_successes_per_cycle"], 6),
        "ResultSharedMixedDeliveryPercent": percent(shared["mixed_singleton_delivery_fraction"], 4),
        "ResultSharedUniformSingletons": fixed(shared["uniform_singleton_successes_per_cycle"], 6),
        "ResultSharedUniformDeliveryPercent": percent(shared["uniform_singleton_delivery_fraction"], 4),
        "ResultSharedLoadCount": str(shared["evaluated_load_count"]),
        "ResultSharedLoadMinimum": fixed(shared["minimum_load"], 2),
        "ResultSharedLoadMaximum": fixed(shared["maximum_load"], 2),
        # Study B primary inference.
        "ResultFullGoodput": fixed(full["goodput"], 4),
        "ResultHistoryFreeGoodput": fixed(history_policy["goodput"], 4),
        "ResultPhaseGoodput": fixed(phase_policy["goodput"], 4),
        "ResultQualityGoodput": fixed(quality_policy["goodput"], 4),
        "ResultCoordinationBlindGoodput": fixed(coordination_policy["goodput"], 4),
        "ResultCentralizedGoodput": fixed(centralized_policy["goodput"], 4),
        "ResultPhaseEffect": fixed(phase["point_effect"], 4),
        "ResultPhasePercentileLCB": fixed(phase["percentile_one_sided_95_lower"], 4),
        "ResultPhaseBasicLCB": fixed(phase["basic_one_sided_95_lower"], 5),
        "ResultPhasePercentileUCB": fixed(phase["percentile_one_sided_95_upper"], 5),
        "ResultPhaseBasicUCB": fixed(phase["basic_one_sided_95_upper"], 5),
        "ResultPhaseSignFlipP": fixed(phase["sign_flip"]["p_value"], 4),
        "ResultPhaseCellMinimum": fixed(phase_cell_min, 4),
        "ResultPhaseCellMaximum": fixed(phase_cell_max, 4),
        "ResultPhaseLeaveTrainingMinimum": fixed(phase_train_range[0], 5),
        "ResultPhaseLeaveTrainingMaximum": fixed(phase_train_range[1], 5),
        "ResultPhaseLeaveEnvironmentMinimum": fixed(phase_env_range[0], 5),
        "ResultPhaseLeaveEnvironmentMaximum": fixed(phase_env_range[1], 5),
        "ResultHistoryEffect": fixed(history["point_effect"], 4),
        "ResultHistoryPercentileLCB": fixed(history["percentile_one_sided_95_lower"], 4),
        "ResultHistoryBasicLCB": fixed(history["basic_one_sided_95_lower"], 5),
        "ResultHistoryRelativePercent": fixed(100.0 * history["point_effect"] / history_policy["goodput"], 1) + r"\%",
        "ResultHistoryPositiveCellCount": str(sum(value > 0 for value in history_cells)),
        "ResultHistoryCellMinimum": fixed(history_cell_min, 4),
        "ResultHistoryCellMaximum": fixed(history_cell_max, 4),
        "ResultHistoryPositiveCheckpointCount": str(history["sensitivity"]["positive_checkpoint_effect_count"]),
        "ResultHistoryCheckpointMinimum": fixed(history_checkpoint_range[0], 4),
        "ResultHistoryCheckpointMaximum": fixed(history_checkpoint_range[1], 4),
        "ResultHistorySignFlipP": scientific(history["sign_flip"]["p_value"], 3),
        "ResultHistorySignFlipNumerator": str(history["sign_flip"]["inclusive_tail_numerator"]),
        "ResultSignFlipDenominator": thousands(history["sign_flip"]["enumeration_denominator"]),
        "ResultHistoryLeaveTrainingMinimum": fixed(history_train_range[0], 5),
        "ResultHistoryLeaveTrainingMaximum": fixed(history_train_range[1], 5),
        "ResultHistoryLeaveEnvironmentMinimum": fixed(history_env_range[0], 5),
        "ResultHistoryLeaveEnvironmentMaximum": fixed(history_env_range[1], 5),
        "ResultMinimumSignFlipP": scientific(1.0 / history["sign_flip"]["enumeration_denominator"], 6),
        # Seven-action and attempted-only mechanism results.
        "ResultSevenActionAgreement": fixed(seven["validation_equal_checkpoint_factor_selected_agreement"], 4),
        "ResultSevenActionUCB": fixed(gate["one_sided_95_ucb"], 4),
        "ResultSevenActionUniqueCalibrationOptimaCount": str(
            sum(row["calibration_tie_count"] == 1 for row in seven_records)
        ),
        "ResultSevenActionNonrepresentableCount": str(sum(row["full_support_certificate"]["representing_order_count"] == 0 for row in seven_records)),
        "ResultSevenActionNonrepresentableFraction": f"{sum(row['full_support_certificate']['representing_order_count'] == 0 for row in seven_records)}/{len(seven_records)}",
        "ResultValidationAttemptCeiling": fixed(seven["validation_equal_checkpoint_factor_attempt_ceiling_same_weights"], 6),
        "ResultEligibleDecisionCount": thousands(seven["full_pooled_eligible"]),
        "ResultVoluntaryDeferralCount": thousands(seven["full_pooled_abstentions"]),
        "ResultAttemptCount": thousands(seven["full_pooled_attempts"]),
        "ResultPooledAttemptFraction": fixed(seven["full_pooled_attempt_ceiling"], 6),
        "ResultAttemptedCalibrationCount": thousands(sum(calibration_counts)),
        "ResultAttemptedCalibrationMinimum": thousands(min(calibration_counts)),
        "ResultAttemptedCalibrationMaximum": thousands(max(calibration_counts)),
        "ResultAttemptedValidationCount": thousands(sum(validation_counts)),
        "ResultAttemptedValidationMinimum": thousands(min(validation_counts)),
        "ResultAttemptedValidationMaximum": thousands(max(validation_counts)),
        "ResultAttemptedEdgeCount": str(min(row["validation_certificate"]["edge_count"] for row in attempted_records)),
        "ResultAttemptedAgreementMinimum": fixed(min(calibration_agreements), 10),
        "ResultAttemptedAgreementMaximum": fixed(max(calibration_agreements), 10),
        "ResultAttemptedValidationMinimumAgreement": fixed(min(validation_selected_agreements), 10),
        "ResultAttemptedValidationMaximumAgreement": fixed(max(validation_selected_agreements), 10),
        "ResultAttemptedValidationMacroAgreement": fixed(attempted["validation_equal_checkpoint_factor_selected_agreement"], 10),
        "ResultAttemptedValidationMacroAgreementShort": fixed(attempted["validation_equal_checkpoint_factor_selected_agreement"], 4),
        "ResultAttemptedValidationOptimalMacroAgreement": fixed(attempted["validation_equal_checkpoint_factor_validation_optimal_agreement"], 10),
        "ResultChangedValidationOrderCount": str(changed_validation_order_count),
        "ResultChangedValidationOrderSeed": str(next(row["training_seed"] for row in attempted_records if row["selected_order_original_validation_tie_rule"] not in row["validation_optimal_orders"])),
        # Coordinate positive control.
        "ResultCoordinateCaseCount": str(coordinate["case_count"]),
        "ResultCoordinateLogicalEdgeCount": str(coordinate["logical"]["edge_count"]),
        "ResultCoordinatePhysicalMultipleMaskCount": str(coordinate["physical_masks_with_multiple_selected_actions"]),
        "ResultCoordinatePhysicalEdgeCount": str(coordinate["physical"]["edge_count"]),
        "ResultCoordinatePhysicalAgreement": fixed(coordinate["physical_best_order_agreement"], 6),
        # Operating metrics and comparator family.
        "ResultFullCollisionMeanPercent": percent(full["collision_attempt_rate"], 2),
        "ResultHistoryCollisionMeanPercent": percent(history_policy["collision_attempt_rate"], 2),
        "ResultFullCollisionPooledPercent": percent(full["pooled_collision_attempt_rate"], 2),
        "ResultHistoryCollisionPooledPercent": percent(history_policy["pooled_collision_attempt_rate"], 2),
        "ResultFullDeferralNodeSlotPercent": percent(full["voluntary_deferrals_per_node_slot"], 2),
        "ResultFullDeferralEligiblePercent": percent(seven["full_pooled_abstentions"] / seven["full_pooled_eligible"], 2),
        "ResultFullOpportunityPercent": percent(full["opportunity_utilization"], 2),
        "ResultFullServicePercent": percent(full["finite_window_service_fraction"], 2),
        "ResultFullEndingBacklog": f"{full['ending_backlog']:,.2f}".replace(",", "{,}"),
        "ResultFullFairnessPercent": percent(full["delivery_jain_fairness_all_15_nodes"], 2),
        "ResultPhaseCollisionPercent": percent(phase_policy["collision_attempt_rate"], 0),
        "ResultPhaseOpportunityPercent": percent(phase_policy["opportunity_utilization"], 2),
        "ResultPhaseServicePercent": percent(phase_policy["finite_window_service_fraction"], 2),
        "ResultPhaseEndingBacklog": f"{phase_policy['ending_backlog']:,.2f}".replace(",", "{,}"),
        "ResultPhaseFairnessPercent": percent(phase_policy["delivery_jain_fairness_all_15_nodes"], 2),
        "ResultQualityFairnessPercent": percent(quality_policy["delivery_jain_fairness_all_15_nodes"], 2),
        "ResultQualityEffect": fixed(quality["point_effect"], 4),
        "ResultQualityLCB": fixed(quality["percentile_one_sided_95_lower"], 4),
        "ResultQualityRawP": fixed(quality["sign_flip"]["p_value"], 4),
        "ResultQualityHolmP": fixed(inference["inference"]["secondary_holm_adjusted"]["quality_greedy_persistent"], 4),
        "ResultRoundRobinEffect": fixed(comparisons["round_robin"]["point_effect"], 6),
        "ResultRoundRobinLCB": fixed(comparisons["round_robin"]["percentile_one_sided_95_lower"], 6),
        "ResultRoundRobinRawP": scientific(comparisons["round_robin"]["sign_flip"]["p_value"], 6),
        "ResultRoundRobinHolmP": scientific(inference["inference"]["secondary_holm_adjusted"]["round_robin"], 6),
        "ResultUniformEffect": fixed(comparisons["uniform_legal"]["point_effect"], 6),
        "ResultUniformLCB": fixed(comparisons["uniform_legal"]["percentile_one_sided_95_lower"], 6),
        "ResultUniformRawP": scientific(comparisons["uniform_legal"]["sign_flip"]["p_value"], 6),
        "ResultUniformHolmP": scientific(inference["inference"]["secondary_holm_adjusted"]["uniform_legal"], 6),
        "ResultStaticEffect": fixed(comparisons["static_priority"]["point_effect"], 6),
        "ResultStaticLCB": fixed(comparisons["static_priority"]["percentile_one_sided_95_lower"], 6),
        "ResultStaticRawP": scientific(comparisons["static_priority"]["sign_flip"]["p_value"], 6),
        "ResultStaticHolmP": scientific(inference["inference"]["secondary_holm_adjusted"]["static_priority"], 6),
        "ResultCoordinationEffect": fixed(coordination["point_effect"], 4),
        "ResultCoordinationLCB": fixed(coordination["percentile_one_sided_95_lower"], 4),
        "ResultCoordinationSignFlipP": scientific(coordination["sign_flip"]["p_value"], 6),
        # Analytical load bound.
        "ResultExpectedBestQuality": fixed(inference["load_bound"]["expected_best_of_15_clipped_link_qualities"], 10),
        "ResultOfferedLoadLow": fixed(inference["load_bound"]["cells"][0]["offered_packets_per_slot"], 2),
        "ResultOfferedLoadHigh": fixed(inference["load_bound"]["cells"][1]["offered_packets_per_slot"], 2),
        "ResultServiceBoundLowOccupancy": fixed(inference["load_bound"]["cells"][0]["quality_aware_optimistic_service_upper_bound"], 5),
        "ResultServiceBoundHighOccupancy": fixed(inference["load_bound"]["cells"][2]["quality_aware_optimistic_service_upper_bound"], 5),
    }
    return dict(sorted(macros.items()))


def render_macros(macros: dict[str, str]) -> bytes:
    lines = [
        "% GENERATED FILE -- DO NOT EDIT.",
        "% Source: canonical Gold records for crn-2026-003-finite-domain-dsa.",
        "% Regenerate with analysis/generate_result_macros.py.",
    ]
    lines.extend(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items())
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_manifest(output_payload: bytes) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study_id": "crn-2026-003-finite-domain-dsa",
        "artifact": "evidence/gold/manuscript/result_macros.tex",
        "artifact_sha256": sha256_bytes(output_payload),
        "manuscript_copy": "../../main/manuscript/result_macros.tex",
        "manuscript_copy_sha256": sha256_bytes(output_payload),
        "generator": "analysis/generate_result_macros.py",
        "generator_sha256": sha256_file(Path(__file__)),
        "sources": [
            {
                "name": name,
                "path": str(path.relative_to(STUDY_ROOT)),
                "sha256": expected,
            }
            for name, (path, expected) in sorted(SOURCES.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--manuscript-copy", type=Path, default=DEFAULT_MANUSCRIPT_COPY
    )
    args = parser.parse_args()
    payload = render_macros(build_macros())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    args.manuscript_copy.parent.mkdir(parents=True, exist_ok=True)
    args.manuscript_copy.write_bytes(payload)
    args.manifest.write_text(
        json.dumps(build_manifest(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
