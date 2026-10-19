"""Prespecified non-learning comparators for the collision-aware study."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from main.revision.crn_ca_pr1_20261017.config import (
    ABSTAIN_ACTION,
    N_ACTIONS,
    N_CHANNELS,
    N_NODES,
    ORACLE_POLICY_NAME,
    PRIMARY_BASELINE_NAME,
    STATE_DIM,
    Factor,
)


class Policy(Protocol):
    def choose(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Choose one logical action per node."""


def _validate(states: np.ndarray, masks: np.ndarray) -> None:
    if states.shape != (N_NODES, STATE_DIM):
        raise ValueError("unexpected state shape")
    if masks.shape != (N_NODES, N_ACTIONS):
        raise ValueError("unexpected mask shape")


class UniformLegalPolicy:
    """Uniformly sample an available channel; abstain only when forced."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def choose(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        _validate(states, masks)
        actions = np.full(N_NODES, ABSTAIN_ACTION, dtype=np.int64)
        for node_id in range(N_NODES):
            legal_channels = np.flatnonzero(masks[node_id, :N_CHANNELS])
            if legal_channels.size:
                actions[node_id] = int(self._rng.choice(legal_channels))
        return actions


class StaticPriorityPolicy:
    """Choose the smallest currently legal logical channel."""

    def choose(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        _validate(states, masks)
        actions = np.full(N_NODES, ABSTAIN_ACTION, dtype=np.int64)
        for node_id in range(N_NODES):
            legal_channels = np.flatnonzero(masks[node_id, :N_CHANNELS])
            if legal_channels.size:
                actions[node_id] = int(legal_channels[0])
        return actions


class PhaseOrthogonalPolicy:
    """Collision-free schedule using only node ID, common phase, and availability.

    In a slot with phase ``p``, node ``i`` has cyclic rank ``(i-p) mod N``.
    The first ``K_idle`` ranks map one-to-one to the sorted idle logical
    channels.  A scheduled node transmits only when it has a queued packet.
    """

    def choose(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        _validate(states, masks)
        phase_block = states[:, -N_NODES:]
        phases = np.argmax(phase_block, axis=1)
        if (
            not np.allclose(np.sum(phase_block, axis=1), 1.0)
            or not np.all(phases == phases[0])
        ):
            raise ValueError("phase-orthogonal policy requires one common one-hot phase")
        phase = int(phases[0])
        available = states[:, :N_CHANNELS] > 0.5
        if not np.all(available == available[0]):
            raise ValueError("phase-orthogonal policy requires common availability")
        idle_channels = np.flatnonzero(available[0])

        actions = np.full(N_NODES, ABSTAIN_ACTION, dtype=np.int64)
        for node_id in range(N_NODES):
            rank = (node_id - phase) % N_NODES
            if rank >= idle_channels.size:
                continue
            channel = int(idle_channels[rank])
            if masks[node_id, channel]:
                actions[node_id] = channel
        return actions


class RoundRobinPolicy:
    """Node-phased round robin, with no hidden global coordination state."""

    def __init__(self) -> None:
        self._pointers = np.arange(N_NODES, dtype=np.int64) % N_CHANNELS

    def choose(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        _validate(states, masks)
        actions = np.full(N_NODES, ABSTAIN_ACTION, dtype=np.int64)
        for node_id in range(N_NODES):
            for offset in range(N_CHANNELS):
                candidate = int((self._pointers[node_id] + offset) % N_CHANNELS)
                if masks[node_id, candidate]:
                    actions[node_id] = candidate
                    self._pointers[node_id] = (candidate + 1) % N_CHANNELS
                    break
        return actions


class QualityGreedyPersistentPolicy:
    """Quality-greedy p-persistent access with a fixed population denominator."""

    def __init__(self, *, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def choose(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        _validate(states, masks)
        actions = np.full(N_NODES, ABSTAIN_ACTION, dtype=np.int64)
        quality_start = N_CHANNELS
        qualities = states[:, quality_start : quality_start + N_CHANNELS]
        available = states[:, :N_CHANNELS] > 0.5
        if not np.all(available == available[0]):
            raise ValueError("persistent baseline requires common availability")
        transmit_probability = min(1.0, float(np.sum(available[0])) / N_NODES)
        for node_id in range(N_NODES):
            legal = np.flatnonzero(masks[node_id, :N_CHANNELS])
            if not legal.size:
                continue
            if self._rng.random() <= transmit_probability:
                best_quality = np.max(qualities[node_id, legal])
                tied = legal[np.isclose(qualities[node_id, legal], best_quality)]
                actions[node_id] = int(self._rng.choice(tied))
        return actions


class CentralizedQualityMatchingPolicy:
    """Descriptive oracle maximizing current quality under one-to-one matching."""

    def choose(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        _validate(states, masks)
        actions = np.full(N_NODES, ABSTAIN_ACTION, dtype=np.int64)
        queued_nodes = np.flatnonzero(np.any(masks[:, :N_CHANNELS], axis=1))
        if not queued_nodes.size:
            return actions

        available = states[:, :N_CHANNELS] > 0.5
        if not np.all(available == available[0]):
            raise ValueError("centralized oracle requires common availability")
        idle_channels = np.flatnonzero(available[0])
        if not idle_channels.size:
            return actions

        qualities = states[:, N_CHANNELS : 2 * N_CHANNELS]
        weights = qualities[np.ix_(queued_nodes, idle_channels)].astype(
            np.float64, copy=False
        )
        row_indices, column_indices = linear_sum_assignment(weights, maximize=True)
        for row_index, column_index in zip(row_indices, column_indices, strict=True):
            node_id = int(queued_nodes[row_index])
            channel = int(idle_channels[column_index])
            if not masks[node_id, channel]:
                raise AssertionError("oracle matching selected an illegal action")
            actions[node_id] = channel
        return actions


def build_baseline(name: str, *, seed: int, factor: Factor) -> Policy:
    """Instantiate only a preregistered comparator."""
    if name == PRIMARY_BASELINE_NAME:
        return PhaseOrthogonalPolicy()
    if name == "quality_greedy_persistent":
        return QualityGreedyPersistentPolicy(seed=seed)
    if name == "uniform_legal":
        return UniformLegalPolicy(seed)
    if name == "static_priority":
        return StaticPriorityPolicy()
    if name == "round_robin":
        return RoundRobinPolicy()
    if name == ORACLE_POLICY_NAME:
        return CentralizedQualityMatchingPolicy()
    raise ValueError(f"unknown baseline: {name}")
