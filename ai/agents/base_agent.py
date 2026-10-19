"""AI Agents — Abstract Base Class.

Defines the interface that every spectrum scheduling agent must implement,
whether RL-based (DQN, Graph RL) or rule-based (Round Robin, Auction).

This contract is enforced via Python abstract methods so that the experiment
runner in experiments/run_scenario.py can invoke any scheduler polymorphically
without knowing the concrete type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel

from sdr.sensing.channel_state import ScanResult

log = structlog.get_logger(__name__)


class AgentMetrics(BaseModel):
    """Snapshot of per-episode agent performance metrics."""

    episode: int
    total_reward: float
    mean_reward: float
    epsilon: float | None  # None for non-RL baselines
    steps: int
    successful_allocations: int
    failed_allocations: int


class BaseAgent(ABC):
    """Abstract base for all ChainRadioNet spectrum scheduling agents.

    Concrete subclasses must implement :meth:`select_action`. RL-based
    subclasses additionally implement :meth:`store_experience` and
    :meth:`train_step`. Rule-based baselines may leave those as no-ops.

    Args:
        node_id:    Unique identity of the node this agent serves.
        n_channels: Number of logical spectrum channels.
    """

    def __init__(self, node_id: str, n_channels: int) -> None:
        if not node_id:
            raise ValueError("node_id must be a non-empty string")
        if n_channels <= 0:
            raise ValueError(f"n_channels must be > 0; got {n_channels}")
        self._node_id = node_id
        self._n_channels = n_channels
        log.info(
            "agent_initialised",
            agent_type=type(self).__name__,
            node_id=node_id,
            n_channels=n_channels,
        )

    # ─── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def select_action(
        self,
        scan: ScanResult,
        epsilon: float = 0.0,
        **kwargs: Any,
    ) -> int:
        """Select a channel to allocate given the current spectrum state.

        Args:
            scan:    Current ScanResult from the sensing pipeline.
            epsilon: Exploration probability for epsilon-greedy strategies.
                     Rule-based agents may ignore this parameter.
            **kwargs: Subclass-specific parameters (e.g. graph edge features).

        Returns:
            Integer channel ID to allocate. Must be in ``scan.free_channel_ids()``.

        Raises:
            NoFreeChannelError: If all channels are occupied.
        """

    # ─── Optional RL hooks (no-ops in base; override in RL subclasses) ────────

    def store_experience(  # noqa: B027
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer (RL agents only).

        Rule-based baselines call this as a no-op.

        Args:
            state:      Observation before the action (128-dim numpy array).
            action:     Channel index that was selected.
            reward:     Scalar reward received after the action.
            next_state: Observation after the action.
            done:       True if this is the final step of the episode.
        """

    def train_step(self) -> float | None:
        """Perform one mini-batch gradient update (RL agents only).

        Rule-based baselines return None.

        Returns:
            Scalar training loss, or None if no update was performed.
        """
        return None

    def observe_outcome(  # noqa: B027
        self, action: int | None, accepted: bool
    ) -> None:
        """Record the externally determined outcome of one decision.

        This hook is deliberately separate from :meth:`store_experience`.
        Evaluation does not insert replay experiences, but stateful policy
        features (for example, a per-channel lease-acceptance rate) must evolve
        under exactly the same semantics in training and evaluation.

        Args:
            action: Selected channel, or ``None`` when the node abstained
                because no legal channel was available.
            accepted: Whether the attempted allocation was accepted. This is
                always false for an abstention.
        """

    def reset_episode_context(self) -> None:  # noqa: B027
        """Reset transient observation context at an episode boundary.

        Learned parameters remain untouched. Stateless schedulers may inherit
        this no-op implementation.
        """

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save_checkpoint(self, path: Path | str) -> None:  # noqa: B027
        """Save agent state to disk (RL agents only; no-op for baselines).

        Args:
            path: Destination file path (should end in .pt for PyTorch agents).
        """

    def load_checkpoint(self, path: Path | str) -> None:  # noqa: B027
        """Load agent state from disk (RL agents only; no-op for baselines).

        Args:
            path: Source file path.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """

    # ─── Common accessors ─────────────────────────────────────────────────────

    @property
    def node_id(self) -> str:
        """The node identity this agent serves."""
        return self._node_id

    @property
    def n_channels(self) -> int:
        """Number of logical spectrum channels."""
        return self._n_channels
