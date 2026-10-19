"""AI Training — Reward Function.

Implements the 4-component composite reward signal defined in CLAUDE.md §4:

    R = 0.40·r_util + 0.30·r_fairness + 0.20·p_latency + 0.10·b_success

Components:
  r_util     — spectrum utilisation: fraction of available channels actively used
  r_fairness — Jain Fairness Index over per-node channel allocation counts
  p_latency  — latency reward: exponential decay penalising allocation delays
  b_success  — binary: 1.0 if the channel lease was granted, 0.0 otherwise

All components are normalised to [0, 1] before weighting so the composite
reward is always in [0, 1]. This keeps the DQN Q-value scale predictable
across scenarios with different absolute latency magnitudes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)

# ─── Reward weights (must sum to 1.0) ────────────────────────────────────────
_W_UTIL: float = 0.40
_W_FAIRNESS: float = 0.30
_W_LATENCY: float = 0.20
_W_SUCCESS: float = 0.10

# Normalisation anchor for latency: the 7-stage pipeline target is <100ms.
# Latency reward is 1.0 at 0ms and decays toward 0 at _LATENCY_MAX_MS.
_LATENCY_MAX_MS: float = 100.0


# ─── Data models ─────────────────────────────────────────────────────────────


class RewardComponents(BaseModel):
    """Decomposed reward signal for one allocation step.

    Stored alongside experience tuples in the replay buffer so that
    per-component trends can be logged and plotted during analysis.
    """

    r_util: float = Field(..., ge=0.0, le=1.0, description="Spectrum utilisation reward")
    r_fairness: float = Field(..., ge=0.0, le=1.0, description="Jain Fairness reward")
    p_latency: float = Field(..., ge=0.0, le=1.0, description="Latency reward (higher = faster)")
    b_success: float = Field(..., ge=0.0, le=1.0, description="Binary allocation success")
    total: float = Field(..., description="Weighted composite reward R ∈ [0, 1]")


# ─── Public API ───────────────────────────────────────────────────────────────


def compute_reward(
    *,
    n_allocated_channels: int,
    n_total_channels: int,
    allocation_counts: Sequence[int],
    allocation_latency_ms: float,
    success: bool,
    weights: tuple[float, float, float, float] | None = None,
) -> RewardComponents:
    """Compute the composite reward for one channel allocation step.

    Args:
        n_allocated_channels: Number of channels currently in active use across
            the entire network. Must be ≥ 0.
        n_total_channels: Total number of logical channels. Must be > 0.
        allocation_counts: Per-node allocation count over the current episode
            window (e.g. number of successful leases per node in the last N
            steps). Used to compute the Jain Fairness Index. Must be non-empty.
        allocation_latency_ms: End-to-end allocation latency in milliseconds
            (from request trigger to lease confirmation). Must be ≥ 0.
        success: True if the channel lease was granted; False if rejected
            (channel contention, all channels occupied, blockchain rejection).
        weights: Optional (w_util, w_fairness, w_latency, w_success) override,
            for the Tier B reward-weight sensitivity sweep. Defaults to the
            paper's (0.40, 0.30, 0.20, 0.10) when omitted — every existing
            trained checkpoint used exactly that default, unaffected by this
            parameter's addition.

    Returns:
        RewardComponents containing all four raw components and the weighted
        composite total.

    Raises:
        ValueError: If n_total_channels ≤ 0 or allocation_counts is empty.
    """
    if n_total_channels <= 0:
        raise ValueError(f"n_total_channels must be > 0; got {n_total_channels}")
    if not allocation_counts:
        raise ValueError("allocation_counts must be non-empty")

    w_util, w_fair_weight, w_lat, w_suc = weights or (
        _W_UTIL,
        _W_FAIRNESS,
        _W_LATENCY,
        _W_SUCCESS,
    )

    r_util = _utilisation_reward(n_allocated_channels, n_total_channels)
    r_fair = jain_fairness_index(allocation_counts)
    p_lat = _latency_reward(allocation_latency_ms)
    b_suc = 1.0 if success else 0.0

    total = w_util * r_util + w_fair_weight * r_fair + w_lat * p_lat + w_suc * b_suc

    components = RewardComponents(
        r_util=r_util,
        r_fairness=r_fair,
        p_latency=p_lat,
        b_success=b_suc,
        total=total,
    )

    log.debug(
        "reward_computed",
        r_util=round(r_util, 4),
        r_fairness=round(r_fair, 4),
        p_latency=round(p_lat, 4),
        b_success=b_suc,
        total=round(total, 4),
        latency_ms=round(allocation_latency_ms, 2),
        success=success,
    )

    return components


def jain_fairness_index(counts: Sequence[int | float]) -> float:
    """Compute the Jain Fairness Index for a sequence of resource allocations.

    J(x) = (Σ xᵢ)² / (N · Σ xᵢ²)  (Jain et al. 1984), returned unscaled, so
    this function's output is exactly the quantity the manuscript's reward
    formula (Section IV) cites and names — no additional transform is
    applied. Degenerately, J = 1/N when one node holds every allocation and
    J = 1.0 when all nodes are allocated equally; both bounds fall inside
    [0, 1] already since N ≥ 1, so no rescaling is needed to keep r_fairness
    within the composite reward's [0, 1] range.

    Args:
        counts: Per-node allocation counts or utilisation values. All values
            must be ≥ 0. A sequence of all-zeros returns 1.0 (degenerate fair).

    Returns:
        Fairness score in [1/N, 1.0] ⊆ [0.0, 1.0]. Returns 1.0 for a
        single-element sequence or all-zero counts (no contention → trivially
        fair).

    Raises:
        ValueError: If counts is empty.
    """
    if not counts:
        raise ValueError("counts must be non-empty")

    n = len(counts)
    if n == 1:
        return 1.0

    sum_x = sum(counts)
    if sum_x == 0.0:
        # No allocations yet — network is trivially fair
        return 1.0

    sum_x2 = sum(x * x for x in counts)
    if sum_x2 == 0.0:
        return 1.0

    raw_jfi = (sum_x**2) / (n * sum_x2)
    return float(max(0.0, min(1.0, raw_jfi)))


# ─── Private helpers ──────────────────────────────────────────────────────────


def _utilisation_reward(n_allocated: int, n_total: int) -> float:
    """Spectrum utilisation reward: fraction of channels in active use.

    Args:
        n_allocated: Number of channels currently occupied by active leases.
        n_total:     Total number of logical channels.

    Returns:
        Float in [0.0, 1.0].
    """
    return float(max(0.0, min(1.0, n_allocated / n_total)))


def _latency_reward(latency_ms: float) -> float:
    """Latency reward using an exponential decay over [0, _LATENCY_MAX_MS].

    Reward = exp(−k · latency_ms) where k is chosen so that:
      latency = 0ms    → reward = 1.0
      latency = 100ms  → reward ≈ 0.05 (near-zero at the SLA boundary)

    k = −ln(0.05) / 100 ≈ 0.02996

    Args:
        latency_ms: Allocation latency in milliseconds. Clamped to [0, ∞).

    Returns:
        Float in (0.0, 1.0].
    """
    latency_ms = max(0.0, latency_ms)
    k = -math.log(0.05) / _LATENCY_MAX_MS  # ≈ 0.02996
    return float(math.exp(-k * latency_ms))
