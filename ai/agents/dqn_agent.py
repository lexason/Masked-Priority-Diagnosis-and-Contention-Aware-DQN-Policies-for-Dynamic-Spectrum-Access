"""AI Agents — Deep Q-Network (DQN) Agent.

Implements the full DQN agent as specified in CLAUDE.md §4:

  Architecture:
    Input(128) → Linear(256) → ReLU → Dropout(0.2)
               → Linear(256) → ReLU → Dropout(0.2)
               → Linear(128) → ReLU
               → Linear(K+1) → K channel Q-values plus forced abstention

  Training:
    Optimiser:     Adam, lr=1e-3
    Discount:      γ = 0.99
    Exploration:   ε-greedy, 1.0 → 0.05 over 500 episodes
    Replay buffer: capacity 10,000, mini-batch 32
    Target network: hard copy every 100 steps

  State vector (128-dim) layout — CLAUDE.md §4:
    [0  : N ]   occupancy    — from ScanResult (N = n_channels = 6)
    [N  : 2N]   interference — from ScanResult
    [2N : 3N]   power_norm   — from ScanResult
    [3N : 4N]   channel history  — rolling mean occupancy over last W steps
    [4N : 5N]   per-channel acceptance rate — accepted / attempted allocations
    [5N : 6N]   neighbour features  — injected by GraphRLAgent; zeros here
    [6N : 7N]   spatial intf graph  — injected by GraphRLAgent; zeros here
    [7N : 128]  zero padding

  Platform note: CPU-only (Dell Tower Intel Arc Xe, no CUDA). All tensors
  stay on torch.device('cpu'). Inference measured at < 1ms on Core Ultra 9 285.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from ai.agents.base_agent import BaseAgent
from ai.training.replay_buffer import Experience, ReplayBuffer
from sdr.sensing.channel_state import STATE_DIM, ScanResult

log = structlog.get_logger(__name__)

# State vector segment widths (n_channels = 6 by default)
_HISTORY_WINDOW: int = 20  # rolling window length for occupancy history
_CHECKPOINT_SCHEMA_VERSION: int = 2


def _validate_finite_tensors(value: object, path: str) -> int:
    """Recursively reject non-finite floating/complex tensors.

    Returns the number of floating or complex tensors checked. Integer and
    boolean tensors are valid but do not need a finiteness predicate.
    """
    if isinstance(value, torch.Tensor):
        if not (value.is_floating_point() or value.is_complex()):
            return 0
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Checkpoint contains a non-finite tensor at {path}")
        return 1
    if isinstance(value, Mapping):
        return sum(
            _validate_finite_tensors(item, f"{path}.{key}")
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            _validate_finite_tensors(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    return 0


def validate_dqn_checkpoint_payload(
    payload: Mapping[str, object], *, expected_train_steps: int | None = None
) -> dict[str, int]:
    """Validate the numerical integrity of a DQN checkpoint payload.

    The check is intentionally independent of the camera-ready lifecycle so
    artifact producers and validators can share one implementation. It
    enforces online/target key and tensor-shape parity, recursively checks all
    floating and complex tensors in both networks and the optimiser, and can
    bind the checkpoint's update counter to independently recorded metadata.

    Args:
        payload: Mapping returned by ``torch.load(..., weights_only=True)``.
        expected_train_steps: Optional independently recorded gradient-update
            count. When supplied it must exactly equal ``payload['train_steps']``.

    Returns:
        The validated train-step count and number of floating/complex tensors
        inspected, for optional audit logging.

    Raises:
        ValueError: If a required state is absent/malformed, network keys or
            shapes differ, a floating/complex tensor is non-finite, or the
            expected update count does not match.
    """
    states: dict[str, Mapping[object, object]] = {}
    for state_name in (
        "online_state_dict",
        "target_state_dict",
        "optimiser_state_dict",
    ):
        state = payload.get(state_name)
        if not isinstance(state, Mapping):
            raise ValueError(f"Checkpoint {state_name} must be a mapping")
        states[state_name] = state

    online = states["online_state_dict"]
    target = states["target_state_dict"]
    if set(online) != set(target):
        online_only = sorted(str(key) for key in set(online) - set(target))
        target_only = sorted(str(key) for key in set(target) - set(online))
        raise ValueError(
            "Checkpoint online/target parameter keys differ: "
            f"online_only={online_only}, target_only={target_only}"
        )
    if not online:
        raise ValueError("Checkpoint online/target state dictionaries are empty")
    for key in online:
        online_value = online[key]
        target_value = target[key]
        if not isinstance(online_value, torch.Tensor) or not isinstance(
            target_value, torch.Tensor
        ):
            raise ValueError(f"Checkpoint network state {key!r} is not a tensor")
        if online_value.shape != target_value.shape:
            raise ValueError(
                f"Checkpoint online/target shape mismatch at {key!r}: "
                f"{tuple(online_value.shape)} != {tuple(target_value.shape)}"
            )

    raw_train_steps = payload.get("train_steps")
    if not isinstance(raw_train_steps, int) or isinstance(raw_train_steps, bool):
        raise ValueError("Checkpoint train_steps must be an integer")
    if raw_train_steps < 0:
        raise ValueError("Checkpoint train_steps must be non-negative")
    if expected_train_steps is not None and raw_train_steps != expected_train_steps:
        raise ValueError(
            "Checkpoint train_steps does not match metadata gradient-update count: "
            f"{raw_train_steps} != {expected_train_steps}"
        )

    checked_tensors = sum(
        _validate_finite_tensors(state, state_name)
        for state_name, state in states.items()
    )
    return {
        "train_steps": raw_train_steps,
        "checked_floating_tensors": checked_tensors,
    }


# ─── Neural network ───────────────────────────────────────────────────────────


class DQNNetwork(nn.Module):
    """Feed-forward DQN network mapping a 128-dim state to per-channel Q-values.

    Architecture as specified in CLAUDE.md §4:
      Input(128) → FC(256) → ReLU → Dropout(0.2)
                → FC(256) → ReLU → Dropout(0.2)
                → FC(128) → ReLU
                → FC(n_actions) → Q-values

    Args:
        state_dim:  Input dimensionality (default 128).
        n_actions:  Number of output Q-values (one per channel, default 6).
        dropout:    Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """He (Kaiming) initialisation for ReLU activations."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of state vectors.

        Args:
            x: FloatTensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            FloatTensor of shape (batch_size, n_actions) or (n_actions,).
        """
        return self.net(x)


# ─── DQN Agent ────────────────────────────────────────────────────────────────


class DQNAgent(BaseAgent):
    """Deep Q-Network agent for spectrum channel allocation.

    Maintains two networks (online + frozen target), an experience replay buffer,
    a rolling context window for channel history, and per-channel acceptance-rate
    tracking. Fills state vector segments [3N:5N] before each forward pass.

    Args:
        node_id:             Unique node identity string (e.g. 'node_a').
        n_channels:          Number of logical spectrum channels (default 6).
        state_dim:           DQN input dimensionality (default 128).
        buffer_capacity:     Replay buffer size (default 10,000).
        batch_size:          Mini-batch size for training (default 32).
        learning_rate:       Adam learning rate (default 1e-3).
        gamma:               Discount factor γ (default 0.99).
        target_update_steps: Steps between hard target network updates (default 100).
        dropout:             Dropout probability in hidden layers (default 0.2).
        seed:                Optional random seed for reproducibility.
        double_q:            If True, decouple action selection (online network)
                              from value evaluation (target network) in the TD
                              target, per Double DQN (van Hasselt et al. 2016).
                              If False (default), standard DQN: both selection
                              and evaluation use the target network.
    """

    def __init__(
        self,
        node_id: str,
        n_channels: int = 6,
        state_dim: int = STATE_DIM,
        buffer_capacity: int = 10_000,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        target_update_steps: int = 100,
        dropout: float = 0.2,
        seed: int | None = None,
        double_q: bool = False,
    ) -> None:
        super().__init__(node_id, n_channels)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self._state_dim = state_dim
        # Internal action K is an explicit forced-abstention/no-op.  It is
        # legal iff all K physical channels are occupied.  Public scheduler
        # APIs continue to return physical channel IDs (or raise
        # NoFreeChannelError), while replay can now represent every cycle
        # without relabelling abstention as channel zero.
        self._abstain_action = n_channels
        self._n_actions = n_channels + 1
        self._batch_size = batch_size
        self._gamma = gamma
        self._target_update_steps = target_update_steps
        self._double_q = double_q
        self._train_steps: int = 0

        # Networks — CPU only (no CUDA on Dell Tower)
        self._online = DQNNetwork(state_dim, self._n_actions, dropout)
        self._target = DQNNetwork(state_dim, self._n_actions, dropout)
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()  # target network is never trained directly

        self._optimiser = optim.Adam(self._online.parameters(), lr=learning_rate)
        self._buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

        # Rolling context: per-channel occupancy history
        # Shape: deque of n_channels boolean arrays, length _HISTORY_WINDOW
        self._occ_history: deque[np.ndarray] = deque(maxlen=_HISTORY_WINDOW)

        # Per-channel cumulative lease-outcome tracking.  An attempt is counted
        # only when a legal channel was selected; ``accepted`` is the same
        # externally determined boolean used by both training and evaluation.
        self._channel_attempts = np.zeros(n_channels, dtype=np.float64)
        self._channel_acceptances = np.zeros(n_channels, dtype=np.float64)

        log.info(
            "dqn_agent_ready",
            node_id=node_id,
            n_channels=n_channels,
            state_dim=state_dim,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            lr=learning_rate,
            gamma=gamma,
            target_update_steps=target_update_steps,
            double_q=double_q,
        )

    # ─── Action selection ──────────────────────────────────────────────────────

    def select_action(
        self,
        scan: ScanResult,
        epsilon: float = 0.0,
        **kwargs: Any,
    ) -> int:
        """Select a channel using ε-greedy DQN policy with action masking.

        At inference time (epsilon=0.0) the agent is in eval mode: dropout
        is disabled and the forward pass takes < 1ms on Core Ultra 9 285.
        During training (epsilon > 0) random actions explore the free channels.

        Args:
            scan:    Current ScanResult from the sensing pipeline.
            epsilon: Exploration probability. 0.0 = pure greedy inference.
            **kwargs: Ignored (present for Graph RL subclass compatibility).

        Returns:
            Integer channel ID. Always in ``scan.free_channel_ids()``.

        Raises:
            NoFreeChannelError: If all channels are currently occupied.
        """
        # Every sensing cycle advances temporal context, including a cycle in
        # which every channel is occupied and the policy must abstain.
        self._record_observation(scan)
        state_vec = self._enrich_state(scan)
        free_channels = scan.require_free_channels()

        # ε-greedy exploration
        if epsilon > 0.0 and np.random.random() < epsilon:
            selected = int(np.random.choice(free_channels))
            log.debug(
                "dqn_explore",
                node_id=self._node_id,
                selected_channel=selected,
                epsilon=round(epsilon, 4),
            )
            return selected

        # Greedy: forward pass through online network
        state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)  # (1, 128)

        self._online.eval()
        with torch.no_grad():
            q_values = self._online(state_t).squeeze(0)  # (n_channels + 1,)
        self._online.train()

        # Mask occupied channels: set their Q-values to −∞
        mask = torch.full((self._n_actions,), float("-inf"))
        for ch_id in free_channels:
            mask[ch_id] = 0.0
        q_masked = q_values + mask

        selected = int(q_masked.argmax().item())

        log.debug(
            "dqn_exploit",
            node_id=self._node_id,
            selected_channel=selected,
            q_value=round(float(q_values[selected].item()), 4),
            free_channels=free_channels,
        )
        return selected

    def raw_q_values(self, scan: ScanResult) -> np.ndarray:
        """Return unmasked greedy Q-values for every channel.

        Used by arbitration schedulers (e.g. AdaptGate) that need this
        agent's raw value estimate per channel rather than a selected
        action. Like :meth:`select_action`, this represents one policy
        observation and therefore advances occupancy history exactly once.
        This keeps arbitration wrappers such as AdaptGate state-equivalent to
        standalone DQN inference.

        Args:
            scan: Current ScanResult from the sensing pipeline.

        Returns:
            Float32 numpy array of shape (n_channels,), one Q-value per
            channel, unmasked (occupied channels are not set to -inf).
        """
        self._record_observation(scan)
        state_vec = self._enrich_state(scan)
        state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        self._online.eval()
        with torch.no_grad():
            q_values = self._online(state_t).squeeze(0)
        self._online.train()
        return q_values[: self._n_channels].numpy().astype(np.float32)

    # ─── Experience storage ────────────────────────────────────────────────────

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push one transition into the replay buffer.

        Outcome statistics are intentionally *not* updated here.  Call
        :meth:`observe_outcome` once per environment decision instead, so
        replay insertion and evaluation use identical feature semantics.

        Args:
            state:      128-dim float32 numpy array (enriched state).
            action:     Physical channel ID, or ``abstain_action`` for an
                all-occupied forced no-op.
            reward:     Scalar composite reward R ∈ [0, 1].
            next_state: 128-dim float32 numpy array for the next step.
            done:       True if this ends the episode.
        """
        if not 0 <= action < self._n_actions:
            raise ValueError(
                f"replay action must be in [0, {self._n_actions - 1}]; got {action}"
            )
        self._buffer.push(
            Experience(
                state=state.astype(np.float32),
                action=action,
                reward=float(reward),
                next_state=next_state.astype(np.float32),
                done=done,
            )
        )

    def observe_outcome(self, action: int | None, accepted: bool) -> None:
        """Update the per-channel lease-acceptance feature.

        ``accepted`` is the simulator/lease-layer acceptance decision, not a
        threshold applied to the composite reward.  Abstentions do not create
        a fictitious attempt on channel zero.
        """
        if action is None:
            return
        if not 0 <= action < self._n_channels:
            raise ValueError(
                f"action must be in [0, {self._n_channels - 1}] or None; got {action}"
            )
        self._channel_attempts[action] += 1.0
        if accepted:
            self._channel_acceptances[action] += 1.0

    # ─── Training step ─────────────────────────────────────────────────────────

    def train_step(self) -> float | None:
        """One mini-batch gradient update of the online DQN network.

        Implements the temporal-difference loss. Standard DQN:
          L = MSE(r + γ · max_a' Q_target(s', a') · (1 − done),  Q_online(s, a))
        Double DQN (``double_q=True``), decoupling action selection from value
        evaluation to reduce overestimation bias (van Hasselt et al. 2016):
          a* = argmax_a' Q_online(s', a')
          L = MSE(r + γ · Q_target(s', a*) · (1 − done),  Q_online(s, a))

        The target network is hard-copied from the online network every
        ``target_update_steps`` training calls.

        Returns:
            Scalar loss value, or None if the buffer is not yet full enough
            to sample a mini-batch.
        """
        if not self._buffer.is_ready(self._batch_size):
            return None

        batch = self._buffer.sample(self._batch_size)

        # Q(s, a) from online network — gather the Q-value for the taken action
        q_online = self._online(batch.states)  # (B, n_channels + 1)
        q_taken = q_online.gather(1, batch.actions.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            q_target_next = self._bootstrap_next_values(batch.next_states)
            td_target = batch.rewards + self._gamma * q_target_next * (1.0 - batch.dones)

        loss = functional.mse_loss(q_taken, td_target)

        self._optimiser.zero_grad()
        loss.backward()
        # Gradient clipping: prevents exploding gradients on long episodes
        nn.utils.clip_grad_norm_(self._online.parameters(), max_norm=10.0)
        self._optimiser.step()

        self._train_steps += 1

        # Hard target network update
        if self._train_steps % self._target_update_steps == 0:
            self._target.load_state_dict(self._online.state_dict())
            log.debug(
                "target_network_updated",
                node_id=self._node_id,
                step=self._train_steps,
            )

        loss_val = float(loss.item())
        log.debug(
            "dqn_train_step",
            node_id=self._node_id,
            loss=round(loss_val, 6),
            step=self._train_steps,
            buffer_size=len(self._buffer),
        )
        return loss_val

    def _legal_next_action_mask(self, next_states: torch.Tensor) -> torch.Tensor:
        """Return legal K+1 actions for each next-state row."""
        legal_channels = next_states[:, : self._n_channels] < 0.5
        has_legal_channel = legal_channels.any(dim=1, keepdim=True)
        return torch.cat([legal_channels, ~has_legal_channel], dim=1)

    def _bootstrap_next_values(self, next_states: torch.Tensor) -> torch.Tensor:
        """Compute action-masked DQN/Double-DQN continuation values."""
        legal_next = self._legal_next_action_mask(next_states)
        if self._double_q:
            # Double DQN: select with online, evaluate with target.
            online_next = self._online(next_states).masked_fill(
                ~legal_next, float("-inf")
            )
            next_actions = online_next.argmax(dim=1)
            return self._target(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

        # Standard DQN: both selection and evaluation use target network.
        target_next = self._target(next_states).masked_fill(
            ~legal_next, float("-inf")
        )
        return target_next.max(dim=1).values

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save_checkpoint(self, path: Path | str) -> None:
        """Save online network weights, optimiser state, and training metadata.

        Args:
            path: Destination .pt file path.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state_dict": self._online.state_dict(),
                "target_state_dict": self._target.state_dict(),
                "optimiser_state_dict": self._optimiser.state_dict(),
                "train_steps": self._train_steps,
                "node_id": self._node_id,
                "n_channels": self._n_channels,
                "n_actions": self._n_actions,
                "state_dim": self._state_dim,
                "checkpoint_schema_version": _CHECKPOINT_SCHEMA_VERSION,
                "action_schema": "K physical channels plus forced-abstain action K",
            },
            dest,
        )
        log.info("checkpoint_saved", path=str(dest), step=self._train_steps)

    def load_checkpoint(
        self, path: Path | str, *, allow_legacy_action_head: bool = False
    ) -> None:
        """Load network weights and optimiser state from a checkpoint file.

        Args:
            path: Source .pt file path.
            allow_legacy_action_head: Explicit audit-only opt-in for a legacy
                K-output checkpoint with no abstention action.  The K channel
                rows are copied and the new abstention row is left at its
                deterministic initial value.  Such a checkpoint is not valid
                for corrected confirmatory evaluation and must not be used by
                default.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"DQN checkpoint not found: {src}")

        ckpt = torch.load(src, map_location="cpu", weights_only=True)
        checkpoint_n_actions = int(ckpt.get("n_actions", self._n_channels))
        legacy_action_head = checkpoint_n_actions != self._n_actions
        if legacy_action_head and not allow_legacy_action_head:
            raise ValueError(
                f"Checkpoint {src} has action_dim={checkpoint_n_actions}; corrected "
                f"agents require action_dim={self._n_actions} (K channels plus forced "
                "abstention). Retrain the checkpoint, or pass "
                "allow_legacy_action_head=True for audit-only reproduction."
            )

        if legacy_action_head:
            self._load_legacy_network_state(self._online, ckpt["online_state_dict"])
            self._load_legacy_network_state(self._target, ckpt["target_state_dict"])
            log.warning(
                "legacy_checkpoint_action_head_expanded",
                path=str(src),
                checkpoint_actions=checkpoint_n_actions,
                corrected_actions=self._n_actions,
                confirmatory_valid=False,
            )
        else:
            self._online.load_state_dict(ckpt["online_state_dict"])
            self._target.load_state_dict(ckpt["target_state_dict"])
            self._optimiser.load_state_dict(ckpt["optimiser_state_dict"])
        self._train_steps = ckpt.get("train_steps", 0)
        log.info("checkpoint_loaded", path=str(src), step=self._train_steps)

    @staticmethod
    def _load_legacy_network_state(
        network: nn.Module, legacy_state: dict[str, torch.Tensor]
    ) -> None:
        """Copy a K-output state dict into a K+1 network for audit only."""
        migrated = network.state_dict()
        for key, old_value in legacy_state.items():
            if key not in migrated:
                raise ValueError(f"Legacy checkpoint contains unexpected tensor {key!r}")
            new_value = migrated[key]
            if old_value.shape == new_value.shape:
                migrated[key] = old_value
                continue
            is_expanded_output = (
                new_value.ndim in (1, 2)
                and old_value.ndim == new_value.ndim
                and old_value.shape[0] == new_value.shape[0] - 1
                and old_value.shape[1:] == new_value.shape[1:]
            )
            if not is_expanded_output:
                raise ValueError(
                    f"Cannot migrate legacy tensor {key!r}: "
                    f"{tuple(old_value.shape)} -> {tuple(new_value.shape)}"
                )
            expanded = new_value.clone()
            expanded[: old_value.shape[0]] = old_value
            expanded[-1].zero_()
            migrated[key] = expanded
        network.load_state_dict(migrated)

    # ─── Internal: state enrichment ───────────────────────────────────────────

    def _enrich_state(self, scan: ScanResult) -> np.ndarray:
        """Build the full 128-dim state vector for one scan.

        Fills the sensor segments from the scan's ``to_state_vector()`` output
        and injects the rolling context into slots [3N:5N]:

          [3N:4N] — channel history: rolling mean occupancy over last W scans
          [4N:5N] — per-channel acceptance rate: accepted attempts / attempts

        The Graph RL segments [5N:7N] are left as zeros here for every agent
        type, including GraphRLAgent: this method is not overridden. Instead,
        ``GraphDQNNetwork.forward`` (ai/agents/graph_rl_agent.py) recomputes
        the GNN embedding fresh from segments [0:3N] on every forward pass
        and overwrites the zero placeholder in [5N:7N] before its head runs,
        so the value stored in the replay buffer is always the raw,
        pre-embedding state.

        Args:
            scan: Current ScanResult.

        Returns:
            Float32 numpy array of shape (128,).
        """
        vec = scan.to_state_vector().copy()  # already shape (128,), fills [0:3N]
        n = self._n_channels

        # Segment [3N:4N] — rolling mean occupancy history
        if self._occ_history:
            history_arr = np.stack(list(self._occ_history), axis=0)  # (W, N)
            vec[3 * n : 4 * n] = history_arr.mean(axis=0)
        # else: no history yet → stays zero (correct for first step)

        # Segment [4N:5N] — per-channel lease-acceptance rate
        with np.errstate(invalid="ignore", divide="ignore"):
            acceptance_rate = np.where(
                self._channel_attempts > 0,
                self._channel_acceptances / self._channel_attempts,
                0.0,
            )
        vec[4 * n : 5 * n] = acceptance_rate.astype(np.float32)

        return vec

    def _record_observation(self, scan: ScanResult) -> None:
        """Append one scan to the rolling occupancy history.

        Callers invoke this before checking for free channels so an
        all-occupied sensing cycle remains visible to future decisions.
        """
        occ_vec = np.array(
            [ch.occupied for ch in sorted(scan.channels, key=lambda c: c.id)],
            dtype=np.float32,
        )
        self._occ_history.append(occ_vec)

    def reset_episode_context(self) -> None:
        """Reset deployment-initialised context at an episode boundary."""
        self._occ_history.clear()
        self._channel_attempts.fill(0.0)
        self._channel_acceptances.fill(0.0)

    # ─── Properties ───────────────────────────────────────────────────────────

    @property
    def train_steps(self) -> int:
        """Total number of gradient update steps performed."""
        return self._train_steps

    @property
    def buffer_size(self) -> int:
        """Current number of experiences in the replay buffer."""
        return len(self._buffer)

    @property
    def abstain_action(self) -> int:
        """Internal replay index for an all-channels-occupied no-op."""
        return self._abstain_action
