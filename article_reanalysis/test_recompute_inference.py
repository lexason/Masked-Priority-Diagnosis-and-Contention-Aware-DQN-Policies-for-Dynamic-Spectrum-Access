"""Focused determinism checks for the independent inference recomputation."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("recompute_inference.py")
SPEC = importlib.util.spec_from_file_location("recompute_inference", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
RECOMPUTE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOMPUTE)


def test_sign_flip_resolution_and_bootstrap_plan_identity() -> None:
    result = RECOMPUTE.exact_sign_flip(np.ones(20), margin=0.0)
    assert result["inclusive_tail_numerator"] == 1
    assert result["enumeration_denominator"] == 2**20
    assert result["p_value"] == 2**-20

    training, environment, _, _ = RECOMPUTE.bootstrap_plan()
    digest = hashlib.sha256(
        training.astype("<i8").tobytes() + environment.astype("<i8").tobytes()
    ).hexdigest()
    assert digest == "3b26f415c4992bb579afe0615649eaa08233f81f8ae7b3f48ce87c8b4fbd3245"


def test_quality_upper_bound_integral() -> None:
    expectation, error = RECOMPUTE.expected_max_quality()
    np.testing.assert_allclose(
        expectation,
        0.8629688770304695,
        rtol=0.0,
        atol=1e-13,
    )
    assert error < 1e-12
