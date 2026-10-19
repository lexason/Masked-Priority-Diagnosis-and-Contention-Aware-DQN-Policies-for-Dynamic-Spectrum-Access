"""Focused outcome-independent tests for the v2 inference implementation."""

from __future__ import annotations

import numpy as np
import pytest

from main.revision.crn_ca_pr1_20261017 import analyze as inference
from main.revision.crn_ca_pr1_20261017.analyze import (
    _best_order,
    _comparison_specifications,
    _effect_draws,
    _exact_sign_flip_p,
    _holm_adjust,
    _make_bootstrap_plan,
    _mechanism_analysis,
    _parse_mask_action_counts,
    _precedence_certificate,
)
from main.revision.crn_ca_pr1_20261017.config import (
    EVALUATION_SEEDS,
    FACTORS,
    N_ACTIONS,
    N_CHANNELS,
    ORACLE_POLICY_NAME,
    PRIORITY_CALIBRATION_INDICES,
    PRIORITY_VALIDATION_INDICES,
    TRAINING_SEEDS,
)


def _fixed_priority_counts(order: tuple[int, ...]) -> np.ndarray:
    counts = np.zeros((1 << N_CHANNELS, N_ACTIONS), dtype=np.int64)
    for mask in range(1, 1 << N_CHANNELS):
        selected = next(channel for channel in order if mask & (1 << channel))
        counts[mask, selected] = 1
    return counts


def test_crossed_bootstrap_preserves_every_label_shift_stratum() -> None:
    plan = _make_bootstrap_plan(
        rng=np.random.default_rng(12),
        replicates=25,
    )
    assert plan.training_indices.shape == (25, len(TRAINING_SEEDS))
    assert plan.environment_indices.shape == (25, len(EVALUATION_SEEDS))
    for draw in plan.environment_indices:
        np.testing.assert_array_equal(
            np.bincount(draw % N_CHANNELS, minlength=N_CHANNELS),
            np.full(N_CHANNELS, len(EVALUATION_SEEDS) // N_CHANNELS),
        )

    validation_plan = _make_bootstrap_plan(
        rng=np.random.default_rng(13),
        replicates=10,
        allowed_environment_indices=PRIORITY_VALIDATION_INDICES,
    )
    assert np.all(np.isin(validation_plan.environment_indices, PRIORITY_VALIDATION_INDICES))
    for draw in validation_plan.environment_indices:
        np.testing.assert_array_equal(
            np.bincount(draw % N_CHANNELS, minlength=N_CHANNELS),
            np.full(N_CHANNELS, len(PRIORITY_VALIDATION_INDICES) // N_CHANNELS),
        )


def test_crossed_effect_draws_retain_constant_paired_difference() -> None:
    shape = (len(TRAINING_SEEDS), len(FACTORS), len(EVALUATION_SEEDS))
    baseline = np.arange(len(FACTORS) * len(EVALUATION_SEEDS), dtype=np.float64)
    baseline = baseline.reshape(len(FACTORS), len(EVALUATION_SEEDS)) / 10_000.0
    treatment = np.broadcast_to(baseline, shape).copy() + 0.25
    plan = _make_bootstrap_plan(rng=np.random.default_rng(14), replicates=30)
    points, draws = _effect_draws(treatment, baseline, plan)
    np.testing.assert_allclose(points, 0.25, atol=1e-14)
    np.testing.assert_allclose(draws, 0.25, atol=1e-14)


def test_exact_sign_flip_and_holm_are_deterministic() -> None:
    assert _exact_sign_flip_p(np.ones(len(TRAINING_SEEDS))) == pytest.approx(
        1.0 / (2 ** len(TRAINING_SEEDS))
    )
    assert _exact_sign_flip_p(np.zeros(len(TRAINING_SEEDS))) == 1.0
    assert _holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06}
    )


def test_mask_action_parser_rejects_an_unavailable_action() -> None:
    counts = _fixed_priority_counts((2, 0, 1, 3, 4, 5))
    encoded = "-".join(str(int(value)) for value in counts.ravel())
    np.testing.assert_array_equal(_parse_mask_action_counts(encoded), counts)

    invalid = counts.copy()
    invalid[1, 2] = 1
    invalid_encoded = "-".join(str(int(value)) for value in invalid.ravel())
    with pytest.raises(ValueError, match="unavailable channel"):
        _parse_mask_action_counts(invalid_encoded)


def test_exhaustive_order_and_precedence_certificate_recover_known_order() -> None:
    order = (2, 0, 1, 3, 4, 5)
    counts = _fixed_priority_counts(order)
    best = _best_order(counts)
    certificate = _precedence_certificate(counts)
    assert best["order"] == order
    assert best["equal_group_agreement"] == 1.0
    assert best["tie_count"] == 1
    assert certificate["acyclic"]
    assert certificate["unique_topological_order"]
    assert certificate["topological_order"] == list(order)
    assert certificate["exact_priority_representable"]


def test_cycle_and_eligible_abstention_each_prevent_exact_representation() -> None:
    cyclic = np.zeros((1 << N_CHANNELS, N_ACTIONS), dtype=np.int64)
    cyclic[0b000011, 0] = 1
    cyclic[0b000011, 1] = 1
    cycle_certificate = _precedence_certificate(cyclic)
    assert not cycle_certificate["acyclic"]
    assert not cycle_certificate["exact_priority_representable"]

    abstaining = _fixed_priority_counts((0, 1, 2, 3, 4, 5))
    abstaining[0b000001, N_CHANNELS] = 1
    abstention_certificate = _precedence_certificate(abstaining)
    assert abstention_certificate["acyclic"]
    assert abstention_certificate["eligible_abstentions"] == 1
    assert not abstention_certificate["exact_priority_representable"]


def test_calibration_ties_use_most_conservative_validation_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reverse_order = tuple(reversed(range(N_CHANNELS)))
    singleton_only = np.zeros((1 << N_CHANNELS, N_ACTIONS), dtype=np.int64)
    for channel in range(N_CHANNELS):
        singleton_only[1 << channel, channel] = 1
    validation_counts = _fixed_priority_counts(reverse_order)
    cube = np.zeros(
        (
            len(TRAINING_SEEDS),
            len(FACTORS),
            len(EVALUATION_SEEDS),
            1 << N_CHANNELS,
            N_ACTIONS,
        ),
        dtype=np.int64,
    )
    for environment_index in range(len(EVALUATION_SEEDS)):
        cube[:, :, environment_index] = (
            singleton_only
            if environment_index in PRIORITY_CALIBRATION_INDICES
            else validation_counts
        )
    monkeypatch.setattr(inference, "BOOTSTRAP_REPLICATES", 10)
    rows, _ = _mechanism_analysis(cube, rng=np.random.default_rng(15))
    assert rows[0]["calibration_order_tie_count"] == 720
    assert rows[0]["calibration_best_order"] == "-".join(
        str(channel) for channel in reverse_order
    )
    assert rows[0]["validation_agreement_for_calibration_order"] == 1.0


def test_centralized_oracle_is_not_in_the_holm_family() -> None:
    specifications = _comparison_specifications()
    oracle = next(
        item for item in specifications if item["comparator"] == ORACLE_POLICY_NAME
    )
    assert oracle["role"] == "descriptive_centralized_oracle"
    assert not oracle["holm_family"]
