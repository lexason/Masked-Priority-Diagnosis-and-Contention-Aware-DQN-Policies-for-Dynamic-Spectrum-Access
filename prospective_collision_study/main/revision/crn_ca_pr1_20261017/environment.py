"""Shared-occupancy, simultaneous-action collision environment.

All policies receive the same exogenous stream for a given factor/seed/label
permutation.  Queue state and history are endogenous, but RNG consumption is
deliberately independent of actions so trace hashes remain policy-comparable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from main.revision.crn_ca_pr1_20261017.config import (
    ABSTAIN_ACTION,
    HISTORY_ALPHA,
    MARKOV_DEPARTURE_PROBABILITY,
    N_ACTIONS,
    N_CHANNELS,
    N_NODES,
    OUTCOME_CLASSES,
    REWARD_COLLISION,
    REWARD_DELIVERED,
    REWARD_FORCED_OR_NO_PACKET,
    REWARD_LINK_FAILURE,
    REWARD_VOLUNTARY_DEFER,
    STATE_DIM,
    Factor,
)

OUTCOME_NONE = 0
OUTCOME_DELIVERED = 1
OUTCOME_COLLISION = 2
OUTCOME_LINK_FAILURE = 3
OUTCOME_DEFERRED = 4


@dataclass(frozen=True, slots=True)
class CycleResult:
    """Resolved simultaneous outcomes for one slot."""

    rewards: np.ndarray
    delivered: np.ndarray
    collisions: np.ndarray
    link_failures: np.ndarray
    voluntary_deferrals: np.ndarray
    forced_deferrals: np.ndarray
    attempted: np.ndarray
    physical_actions: np.ndarray
    queue_before: np.ndarray
    queue_after: np.ndarray
    idle_channels: int
    arrivals: np.ndarray


class CollisionEnvironment:
    """A 15-node DSA environment with destructive co-channel collisions."""

    def __init__(
        self,
        *,
        factor: Factor,
        seed: int,
        logical_to_physical: tuple[int, ...],
    ) -> None:
        if sorted(logical_to_physical) != list(range(N_CHANNELS)):
            raise ValueError("logical_to_physical must be a channel permutation")
        if factor.occupancy_process not in {"iid", "markov"}:
            raise ValueError("unsupported occupancy process")
        self.factor = factor
        self.seed = int(seed)
        self.logical_to_physical = np.asarray(logical_to_physical, dtype=np.int64)
        self.physical_to_logical = np.argsort(self.logical_to_physical)
        self._rng = np.random.default_rng(seed)
        self._trace_digest = hashlib.sha256()
        self.slot_index = -1

        self.queues = np.zeros(N_NODES, dtype=np.int64)
        self.previous_action = np.full(N_NODES, ABSTAIN_ACTION, dtype=np.int64)
        self.previous_outcome = np.full(N_NODES, OUTCOME_NONE, dtype=np.int64)
        self.collision_ewma = np.zeros((N_NODES, N_CHANNELS), dtype=np.float32)
        self.delivery_ewma = np.zeros((N_NODES, N_CHANNELS), dtype=np.float32)

        self._occupied = self._rng.random(N_CHANNELS) < factor.occupancy_probability
        self._current_arrivals = np.zeros(N_NODES, dtype=np.int64)
        self._current_quality = np.zeros((N_NODES, N_CHANNELS), dtype=np.float32)
        self._current_link_uniform = np.zeros((N_NODES, N_CHANNELS), dtype=np.float32)
        self._slot_open = False

    @property
    def trace_sha256(self) -> str:
        """Digest of all exogenous arrays consumed so far."""
        return self._trace_digest.hexdigest()

    def begin_slot(self) -> tuple[np.ndarray, np.ndarray]:
        """Advance the common trace and return per-node states and action masks."""
        if self._slot_open:
            raise RuntimeError("resolve the current slot before beginning another")
        self.slot_index += 1
        self._step_occupancy()
        self._current_arrivals = (
            self._rng.random(N_NODES) < self.factor.arrival_probability
        ).astype(np.int64)
        self.queues += self._current_arrivals

        # A common channel component plus independent receiver variation yields
        # correlated but genuinely local channel-quality observations.
        common_quality = self._rng.beta(5.0, 2.0, size=N_CHANNELS)
        local_noise = self._rng.normal(0.0, 0.10, size=(N_NODES, N_CHANNELS))
        quality = np.clip(common_quality[None, :] + local_noise, 0.05, 0.99)
        quality[:, self._occupied] = 0.0
        self._current_quality = quality.astype(np.float32)
        self._current_link_uniform = self._rng.random(
            (N_NODES, N_CHANNELS)
        ).astype(np.float32)

        self._update_trace_digest()
        self._slot_open = True
        return self._build_states_and_masks()

    def resolve(self, logical_actions: np.ndarray) -> CycleResult:
        """Resolve simultaneous decisions; any co-channel multiplicity destroys all."""
        if not self._slot_open:
            raise RuntimeError("begin_slot must be called before resolve")
        actions = np.asarray(logical_actions, dtype=np.int64)
        if actions.shape != (N_NODES,):
            raise ValueError(f"expected {(N_NODES,)} actions; got {actions.shape}")
        if np.any((actions < 0) | (actions >= N_ACTIONS)):
            raise ValueError("action outside registered action space")

        queue_before = self.queues.copy()
        physical = np.full(N_NODES, -1, dtype=np.int64)
        choosing_channel = actions < N_CHANNELS
        physical[choosing_channel] = self.logical_to_physical[actions[choosing_channel]]
        attempted = choosing_channel & (queue_before > 0)

        if np.any(attempted & self._occupied[np.clip(physical, 0, N_CHANNELS - 1)]):
            raise ValueError("policy selected an occupied channel despite the action mask")
        if np.any(choosing_channel & (queue_before == 0)):
            raise ValueError("policy selected a channel with no queued packet")

        multiplicity = np.bincount(physical[attempted], minlength=N_CHANNELS)
        collisions = attempted & (multiplicity[np.clip(physical, 0, N_CHANNELS - 1)] > 1)
        singleton = attempted & ~collisions
        link_success = np.zeros(N_NODES, dtype=bool)
        singleton_nodes = np.flatnonzero(singleton)
        if singleton_nodes.size:
            singleton_channels = physical[singleton_nodes]
            link_success[singleton_nodes] = (
                self._current_link_uniform[singleton_nodes, singleton_channels]
                < self._current_quality[singleton_nodes, singleton_channels]
            )
        delivered = singleton & link_success
        link_failures = singleton & ~link_success

        abstained_with_packet = (actions == ABSTAIN_ACTION) & (queue_before > 0)
        any_idle = bool(np.any(~self._occupied))
        voluntary_deferrals = abstained_with_packet & any_idle
        forced_deferrals = abstained_with_packet & (not any_idle)

        rewards = np.full(N_NODES, REWARD_FORCED_OR_NO_PACKET, dtype=np.float32)
        rewards[delivered] = REWARD_DELIVERED
        rewards[collisions] = REWARD_COLLISION
        rewards[link_failures] = REWARD_LINK_FAILURE
        rewards[voluntary_deferrals] = REWARD_VOLUNTARY_DEFER

        self.queues[delivered] -= 1
        if np.any(self.queues < 0):
            raise AssertionError("queue conservation violated")
        self._update_history(
            actions=actions,
            physical=physical,
            delivered=delivered,
            collisions=collisions,
            link_failures=link_failures,
            deferred=voluntary_deferrals | forced_deferrals,
            queue_before=queue_before,
        )
        self._slot_open = False
        return CycleResult(
            rewards=rewards,
            delivered=delivered,
            collisions=collisions,
            link_failures=link_failures,
            voluntary_deferrals=voluntary_deferrals,
            forced_deferrals=forced_deferrals,
            attempted=attempted,
            physical_actions=physical,
            queue_before=queue_before,
            queue_after=self.queues.copy(),
            idle_channels=int(np.sum(~self._occupied)),
            arrivals=self._current_arrivals.copy(),
        )

    def _step_occupancy(self) -> None:
        rho = self.factor.occupancy_probability
        if self.factor.occupancy_process == "iid":
            self._occupied = self._rng.random(N_CHANNELS) < rho
            return
        departure = MARKOV_DEPARTURE_PROBABILITY
        arrival = rho * departure / max(1.0 - rho, 1e-12)
        draws = self._rng.random(N_CHANNELS)
        next_occupied = self._occupied.copy()
        next_occupied[self._occupied & (draws < departure)] = False
        next_occupied[~self._occupied & (draws < arrival)] = True
        self._occupied = next_occupied

    def _build_states_and_masks(self) -> tuple[np.ndarray, np.ndarray]:
        available_physical = ~self._occupied
        available_logical = available_physical[self.logical_to_physical]
        quality_logical = self._current_quality[:, self.logical_to_physical]
        collision_logical = self.collision_ewma[:, self.logical_to_physical]
        delivery_logical = self.delivery_ewma[:, self.logical_to_physical]

        previous_action_one_hot = np.eye(N_ACTIONS, dtype=np.float32)[
            self.previous_action
        ]
        previous_outcome_one_hot = np.eye(OUTCOME_CLASSES, dtype=np.float32)[
            self.previous_outcome
        ]
        queue_feature = np.clip(
            np.log1p(self.queues.astype(np.float32)) / np.log(51.0), 0.0, 1.0
        )[:, None]
        node_one_hot = np.eye(N_NODES, dtype=np.float32)
        phase_one_hot = np.broadcast_to(
            np.eye(N_NODES, dtype=np.float32)[self.slot_index % N_NODES],
            (N_NODES, N_NODES),
        )
        states = np.concatenate(
            (
                np.broadcast_to(available_logical.astype(np.float32), (N_NODES, N_CHANNELS)),
                quality_logical,
                collision_logical,
                delivery_logical,
                previous_action_one_hot,
                previous_outcome_one_hot,
                queue_feature,
                node_one_hot,
                phase_one_hot,
            ),
            axis=1,
        ).astype(np.float32)
        if states.shape != (N_NODES, STATE_DIM):
            raise AssertionError(f"state shape drift: {states.shape}")

        masks = np.zeros((N_NODES, N_ACTIONS), dtype=bool)
        queued = self.queues > 0
        masks[queued, :N_CHANNELS] = available_logical
        masks[:, ABSTAIN_ACTION] = True
        return states, masks

    def _update_history(
        self,
        *,
        actions: np.ndarray,
        physical: np.ndarray,
        delivered: np.ndarray,
        collisions: np.ndarray,
        link_failures: np.ndarray,
        deferred: np.ndarray,
        queue_before: np.ndarray,
    ) -> None:
        self.collision_ewma *= 1.0 - HISTORY_ALPHA
        self.delivery_ewma *= 1.0 - HISTORY_ALPHA
        attempted_nodes = np.flatnonzero(actions < N_CHANNELS)
        if attempted_nodes.size:
            attempted_channels = physical[attempted_nodes]
            self.collision_ewma[attempted_nodes, attempted_channels] += (
                HISTORY_ALPHA * collisions[attempted_nodes]
            )
            self.delivery_ewma[attempted_nodes, attempted_channels] += (
                HISTORY_ALPHA * delivered[attempted_nodes]
            )
        self.previous_action = actions.copy()
        outcome = np.full(N_NODES, OUTCOME_NONE, dtype=np.int64)
        outcome[delivered] = OUTCOME_DELIVERED
        outcome[collisions] = OUTCOME_COLLISION
        outcome[link_failures] = OUTCOME_LINK_FAILURE
        outcome[deferred] = OUTCOME_DEFERRED
        outcome[queue_before == 0] = OUTCOME_NONE
        self.previous_outcome = outcome

    def _update_trace_digest(self) -> None:
        self._trace_digest.update(np.asarray([self.slot_index], dtype="<i8").tobytes())
        self._trace_digest.update(self._occupied.astype(np.uint8).tobytes())
        self._trace_digest.update(self._current_arrivals.astype(np.uint8).tobytes())
        self._trace_digest.update(self._current_quality.astype("<f4").tobytes())
        self._trace_digest.update(self._current_link_uniform.astype("<f4").tobytes())
