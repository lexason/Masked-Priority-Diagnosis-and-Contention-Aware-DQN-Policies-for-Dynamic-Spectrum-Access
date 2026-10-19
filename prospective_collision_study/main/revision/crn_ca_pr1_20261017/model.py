"""Compact shared Double-DQN and memory-efficient all-node replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from main.revision.crn_ca_pr1_20261017.config import (
    BATCH_SIZE,
    COORDINATION_BLIND_POLICY_NAME,
    GAMMA,
    HIDDEN_DIM,
    HISTORY_FREE_POLICY_NAME,
    LEARNED_POLICIES,
    LEARNED_POLICY_NAME,
    LEARNING_RATE,
    N_ACTIONS,
    N_CHANNELS,
    N_NODES,
    REPLAY_CAPACITY,
    STATE_DIM,
    TARGET_UPDATE_STEPS,
)


def observation_for_policy(states: np.ndarray, policy_name: str) -> np.ndarray:
    """Apply a preregistered feature-block ablation without changing shape.

    Keeping the architecture, optimiser, random seeds, and replay schedule
    identical makes the history-free and coordination-blind comparisons
    seed-paired interventions on observation content alone.
    """

    values = np.asarray(states, dtype=np.float32)
    if values.shape != (N_NODES, STATE_DIM):
        raise ValueError("unexpected state batch for observation intervention")
    if policy_name not in LEARNED_POLICIES:
        raise ValueError(f"unknown learned policy: {policy_name}")
    if policy_name == LEARNED_POLICY_NAME:
        return values

    transformed = values.copy()
    if policy_name == HISTORY_FREE_POLICY_NAME:
        # availability K, quality K, then collision and delivery EWMA blocks
        transformed[:, 2 * N_CHANNELS : 4 * N_CHANNELS] = 0.0
        # previous action (K+1) follows those blocks; zero only the subsequent
        # previous-outcome block, retaining previous action as preregistered.
        outcome_start = 4 * N_CHANNELS + N_ACTIONS
        outcome_stop = outcome_start + 5
        transformed[:, outcome_start:outcome_stop] = 0.0
    elif policy_name == COORDINATION_BLIND_POLICY_NAME:
        # Node identity and common phase are the final two N-node blocks.
        transformed[:, STATE_DIM - 2 * N_NODES :] = 0.0
    return transformed


class QNetwork(nn.Module):
    """Small MLP sized for deterministic CPU experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, N_ACTIONS),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    next_masks: torch.Tensor
    dones: torch.Tensor


class ArrayReplay:
    """Preallocated circular replay; every slot inserts exactly all 15 nodes."""

    def __init__(self, *, capacity: int = REPLAY_CAPACITY, seed: int) -> None:
        if capacity < N_NODES:
            raise ValueError("replay capacity must fit at least one all-node cycle")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._states = np.empty((capacity, STATE_DIM), dtype=np.float32)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._next_states = np.empty((capacity, STATE_DIM), dtype=np.float32)
        self._next_masks = np.empty((capacity, N_ACTIONS), dtype=bool)
        self._dones = np.empty(capacity, dtype=bool)
        self._node_ids = np.empty(capacity, dtype=np.int16)
        self._position = 0
        self._size = 0
        self.total_inserted = 0
        self.node_insertions = np.zeros(N_NODES, dtype=np.int64)

    def add_all_nodes(
        self,
        *,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        next_masks: np.ndarray,
        done: bool,
    ) -> None:
        expected_state_shape = (N_NODES, STATE_DIM)
        if states.shape != expected_state_shape or next_states.shape != expected_state_shape:
            raise ValueError("all-node replay requires exactly 15 state transitions")
        if actions.shape != (N_NODES,) or rewards.shape != (N_NODES,):
            raise ValueError("all-node action/reward shape mismatch")
        if next_masks.shape != (N_NODES, N_ACTIONS):
            raise ValueError("all-node next-mask shape mismatch")
        for node_id in range(N_NODES):
            index = self._position
            self._states[index] = states[node_id]
            self._actions[index] = actions[node_id]
            self._rewards[index] = rewards[node_id]
            self._next_states[index] = next_states[node_id]
            self._next_masks[index] = next_masks[node_id]
            self._dones[index] = done
            self._node_ids[index] = node_id
            self._position = (self._position + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
            self.total_inserted += 1
            self.node_insertions[node_id] += 1

    def sample(self, batch_size: int = BATCH_SIZE) -> ReplayBatch:
        if self._size < batch_size:
            raise ValueError("replay is not ready")
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return ReplayBatch(
            states=torch.from_numpy(self._states[indices]),
            actions=torch.from_numpy(self._actions[indices]),
            rewards=torch.from_numpy(self._rewards[indices]),
            next_states=torch.from_numpy(self._next_states[indices]),
            next_masks=torch.from_numpy(self._next_masks[indices]),
            dones=torch.from_numpy(self._dones[indices].astype(np.float32)),
        )

    def ready(self, batch_size: int = BATCH_SIZE) -> bool:
        return self._size >= batch_size

    def __len__(self) -> int:
        return self._size


class CollisionAwareDQN:
    """Shared-parameter Double-DQN with explicit legal-action masking."""

    def __init__(self, *, seed: int, training: bool) -> None:
        self.seed = int(seed)
        self.training = bool(training)
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed + 1)
        self.online = QNetwork()
        self.target = QNetwork()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.gradient_steps = 0

    def choose(
        self,
        states: np.ndarray,
        masks: np.ndarray,
        *,
        epsilon: float,
    ) -> np.ndarray:
        if states.shape != (N_NODES, STATE_DIM):
            raise ValueError("unexpected state batch")
        if masks.shape != (N_NODES, N_ACTIONS) or np.any(~np.any(masks, axis=1)):
            raise ValueError("every node needs at least one legal action")
        self.online.eval()
        with torch.no_grad():
            q_values = self.online(torch.from_numpy(states)).numpy()
        q_values[~masks] = -np.inf
        actions = np.argmax(q_values, axis=1).astype(np.int64)
        if epsilon > 0.0:
            exploring = self._rng.random(N_NODES) < epsilon
            for node_id in np.flatnonzero(exploring):
                actions[node_id] = int(self._rng.choice(np.flatnonzero(masks[node_id])))
        return actions

    def update(self, replay: ArrayReplay) -> float | None:
        if not replay.ready():
            return None
        batch = replay.sample()
        self.online.train()
        q_selected = self.online(batch.states).gather(1, batch.actions[:, None]).squeeze(1)
        with torch.no_grad():
            online_next = self.online(batch.next_states)
            online_next = online_next.masked_fill(~batch.next_masks, -torch.inf)
            next_actions = online_next.argmax(dim=1, keepdim=True)
            target_next = self.target(batch.next_states).gather(1, next_actions).squeeze(1)
            targets = batch.rewards + GAMMA * (1.0 - batch.dones) * target_next
        loss = functional.smooth_l1_loss(q_selected, targets)
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimiser.step()
        self.gradient_steps += 1
        if self.gradient_steps % TARGET_UPDATE_STEPS == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def save(self, path: Path, *, metadata: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "seed": self.seed,
                "state_dim": STATE_DIM,
                "n_actions": N_ACTIONS,
                "gradient_steps": self.gradient_steps,
                "online_state_dict": self.online.state_dict(),
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> tuple[CollisionAwareDQN, dict[str, object]]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported collision-aware checkpoint schema")
        if int(payload.get("state_dim", -1)) != STATE_DIM:
            raise ValueError("checkpoint state dimension mismatch")
        if int(payload.get("n_actions", -1)) != N_ACTIONS:
            raise ValueError("checkpoint action dimension mismatch")
        agent = cls(seed=int(payload["seed"]), training=False)
        agent.online.load_state_dict(payload["online_state_dict"])
        agent.target.load_state_dict(payload["online_state_dict"])
        agent.gradient_steps = int(payload["gradient_steps"])
        metadata = dict(payload.get("metadata", {}))
        if int(metadata.get("model_seed", -1)) != agent.seed:
            raise ValueError("checkpoint top-level seed differs from embedded metadata")
        if int(metadata.get("gradient_steps", -1)) != agent.gradient_steps:
            raise ValueError(
                "checkpoint top-level gradient count differs from embedded metadata"
            )
        return agent, metadata
