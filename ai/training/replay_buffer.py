"""AI Training — Experience Replay Buffer.

A fixed-capacity circular buffer that stores (s, a, r, s', done) tuples
and supports uniform random mini-batch sampling for off-policy DQN training.

Design decisions:
  - Uses collections.deque for O(1) append and automatic eviction of oldest
    experience when capacity is exceeded.
  - Stores state vectors as numpy float32 arrays (not tensors) to keep the
    buffer CPU-resident. Tensors are constructed only at sample time to avoid
    pinning GPU memory unnecessarily (we are CPU-only on the Dell Tower).
  - Thread safety is NOT provided. The buffer is single-producer single-consumer
    in the offline training loop. Add a threading.Lock if concurrent access
    is ever needed.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import structlog
import torch

log = structlog.get_logger(__name__)


# ─── Experience tuple ─────────────────────────────────────────────────────────

class Experience(NamedTuple):
    """One (s, a, r, s', done) transition from the environment.

    Attributes:
        state:      Current observation — float32 numpy array, shape (128,).
        action:     Integer channel index selected by the agent.
        reward:     Scalar composite reward R ∈ [0, 1].
        next_state: Next observation after taking the action — same shape.
        done:       True if this transition ends the episode.
    """

    state: np.ndarray       # shape (STATE_DIM,), dtype float32
    action: int
    reward: float
    next_state: np.ndarray  # shape (STATE_DIM,), dtype float32
    done: bool


# ─── Sampled mini-batch ───────────────────────────────────────────────────────

class Batch(NamedTuple):
    """Collated mini-batch tensors ready for the DQN loss computation.

    All tensors are on CPU (no .cuda() calls — Intel Arc Xe, no CUDA).

    Attributes:
        states:      FloatTensor of shape (batch_size, 128).
        actions:     LongTensor  of shape (batch_size,).
        rewards:     FloatTensor of shape (batch_size,).
        next_states: FloatTensor of shape (batch_size, 128).
        dones:       FloatTensor of shape (batch_size,) — 1.0 = terminal.
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-capacity uniform experience replay buffer for DQN training.

    Stores up to ``capacity`` experiences and evicts the oldest when full.
    Sampling is uniform random without replacement.

    Args:
        capacity: Maximum number of experiences to retain. Older experiences
            are silently discarded when this limit is reached.
        seed:     Optional random seed for reproducible mini-batch sampling.

    Example:
        >>> buf = ReplayBuffer(capacity=10_000, seed=42)
        >>> buf.push(Experience(state, action, reward, next_state, done=False))
        >>> batch = buf.sample(batch_size=32)
    """

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0; got {capacity}")
        self._capacity = capacity
        self._buffer: deque[Experience] = deque(maxlen=capacity)
        if seed is not None:
            random.seed(seed)
        log.info("replay_buffer_created", capacity=capacity, seed=seed)

    # ─── Core operations ──────────────────────────────────────────────────────

    def push(self, experience: Experience) -> None:
        """Add one experience to the buffer, evicting the oldest if full.

        Args:
            experience: A completed (s, a, r, s', done) transition tuple.
        """
        self._buffer.append(experience)

    def sample(self, batch_size: int) -> Batch:
        """Sample a uniformly random mini-batch of experiences.

        Args:
            batch_size: Number of experiences to sample. Must be ≤ len(self).

        Returns:
            Batch of collated CPU tensors ready for the DQN loss computation.

        Raises:
            ValueError: If batch_size > current buffer size (not enough data).
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Cannot sample {batch_size} from buffer with only "
                f"{len(self._buffer)} experiences. "
                "Wait until the buffer has at least batch_size entries."
            )

        experiences = random.sample(self._buffer, batch_size)

        states = torch.tensor(
            np.stack([e.state for e in experiences]), dtype=torch.float32
        )
        actions = torch.tensor(
            [e.action for e in experiences], dtype=torch.long
        )
        rewards = torch.tensor(
            [e.reward for e in experiences], dtype=torch.float32
        )
        next_states = torch.tensor(
            np.stack([e.next_state for e in experiences]), dtype=torch.float32
        )
        dones = torch.tensor(
            [float(e.done) for e in experiences], dtype=torch.float32
        )

        return Batch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

    # ─── Inspection ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the current number of stored experiences."""
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Return True when the buffer holds at least ``batch_size`` experiences.

        Use this as a guard before calling :meth:`sample` to avoid
        premature training at the start of an episode.

        Args:
            batch_size: Minimum required buffer size.

        Returns:
            True if sampling is possible without error.
        """
        return len(self._buffer) >= batch_size

    @property
    def capacity(self) -> int:
        """Maximum buffer capacity."""
        return self._capacity

    @property
    def fill_ratio(self) -> float:
        """Current fill level as a fraction of capacity, in [0.0, 1.0]."""
        return len(self._buffer) / self._capacity
