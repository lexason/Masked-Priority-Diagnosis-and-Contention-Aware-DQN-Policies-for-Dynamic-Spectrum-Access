"""SDR Sensing — Channel State Pydantic Models.

Defines the canonical data contracts for all spectrum sensing output.
Every layer boundary (SDR → AI, SDR → Blockchain, AI → Bridge) uses
these models exclusively — never raw dicts or ad-hoc JSON.

The JSON schema below is the single source of truth from CLAUDE.md Section 4.
No deviations are permitted without updating CLAUDE.md simultaneously.
"""

from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel, Field, field_validator, model_validator

log = structlog.get_logger(__name__)

# ─── Module constants ─────────────────────────────────────────────────────────

STATE_DIM: int = 128
"""DQN input state vector dimensionality (CLAUDE.md Section 4 MDP formulation)."""

POWER_MIN_DBM: float = -120.0
"""Lower bound for received power in dBm; used for normalisation to [0, 1]."""

POWER_MAX_DBM: float = 0.0
"""Upper bound for received power in dBm; used for normalisation to [0, 1]."""

_POWER_RANGE: float = POWER_MAX_DBM - POWER_MIN_DBM  # = 120.0


# ─── Exceptions ───────────────────────────────────────────────────────────────


class NoFreeChannelError(RuntimeError):
    """Raised when all channels are occupied and no allocation is possible.

    The calling agent must handle this by deferring the request, increasing
    its wait time, or returning a 'no resource' response to the bridge.
    """


class ChannelStateError(ValueError):
    """Raised when channel state data fails structural validation.

    Indicates a bug in the IQ generator or a corrupted message from the
    sensing pipeline — never a normal operating condition.
    """


# ─── Pydantic Models ──────────────────────────────────────────────────────────


class ChannelState(BaseModel):
    """State of a single logical spectrum channel at one point in time.

    Corresponds to one element of the ``channels`` array in the canonical
    ScanResult JSON schema defined in CLAUDE.md Section 4.
    """

    id: int = Field(
        ...,
        ge=0,
        le=5,
        description="Channel index (0–5 for the 6-channel 2.4 GHz configuration)",
    )
    freq_mhz: float = Field(
        ...,
        gt=0.0,
        description="Centre frequency of this channel in MHz",
    )
    occupied: bool = Field(
        ...,
        description="True when measured energy exceeds the detection threshold",
    )
    power_dbm: float = Field(
        ...,
        ge=-120.0,
        le=0.0,
        description="Mean received power in dBm over the sensing window",
    )
    interference: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised aggregate interference level [0 = clean, 1 = maximum]",
    )


class ScanResult(BaseModel):
    """Complete spectrum scan result from one 100ms sensing cycle.

    This is the canonical handoff payload between the SDR sensing pipeline
    and the AI decision layer. It is also committed to the Hyperledger Fabric
    ledger as the ``sensingHash`` field of a ChannelLease transaction.

    Schema source: CLAUDE.md Section 4 — Channel State JSON Schema.
    """

    node_id: str = Field(
        ...,
        min_length=1,
        description="Identity of the scanning node (e.g. 'node_a')",
    )
    scan_timestamp: datetime = Field(
        ...,
        description="UTC timestamp marking the start of this sensing scan",
    )
    sensing_duration_ms: float = Field(
        ...,
        gt=0.0,
        description="Actual sensing window duration in milliseconds",
    )
    channels: list[ChannelState] = Field(
        ...,
        min_length=1,
        description="Per-channel states, one entry per logical channel",
    )

    # ─── Validators ───────────────────────────────────────────────────────────

    @field_validator("scan_timestamp", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime:
        """Attach UTC timezone to naive datetimes rather than rejecting them.

        The IQ generator and GNU Radio pipeline may produce naive timestamps;
        this validator normalises them silently to UTC.
        """
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @model_validator(mode="after")
    def _validate_channel_set(self) -> "ScanResult":
        """Enforce that channel IDs form a contiguous zero-based set.

        Raises:
            ChannelStateError: On duplicate IDs or non-contiguous ID range.
        """
        ids = [ch.id for ch in self.channels]
        if len(ids) != len(set(ids)):
            raise ChannelStateError(f"Duplicate channel IDs in scan from '{self.node_id}': {ids}")
        n = len(ids)
        if sorted(ids) != list(range(n)):
            raise ChannelStateError(
                f"Channel IDs must be contiguous 0…{n - 1}. "
                f"Got: {sorted(ids)} in scan from '{self.node_id}'"
            )
        return self

    # ─── Read-only accessors ──────────────────────────────────────────────────

    def free_channel_ids(self) -> list[int]:
        """Return IDs of all currently unoccupied channels.

        Returns:
            List of integer channel IDs sorted ascending. Empty list if all
            channels are occupied.
        """
        return [ch.id for ch in self.channels if not ch.occupied]

    def occupied_channel_ids(self) -> list[int]:
        """Return IDs of all currently occupied channels.

        Returns:
            List of integer channel IDs sorted ascending. Empty list if all
            channels are free.
        """
        return [ch.id for ch in self.channels if ch.occupied]

    def occupancy_rate(self) -> float:
        """Fraction of channels currently occupied.

        Returns:
            Float in [0.0, 1.0]. Returns 0.0 for an empty channel list.
        """
        if not self.channels:
            return 0.0
        return sum(ch.occupied for ch in self.channels) / len(self.channels)

    def channel_by_id(self, channel_id: int) -> ChannelState:
        """Retrieve a single channel by its integer ID.

        Args:
            channel_id: The channel index to look up.

        Returns:
            The matching ChannelState object.

        Raises:
            KeyError: If channel_id is not present in this scan result.
        """
        for ch in self.channels:
            if ch.id == channel_id:
                return ch
        raise KeyError(
            f"Channel {channel_id} not found in scan from '{self.node_id}'. "
            f"Available IDs: {[ch.id for ch in self.channels]}"
        )

    def require_free_channels(self) -> list[int]:
        """Return free channel IDs, raising if none are available.

        Convenience wrapper for agent code that cannot proceed without a
        free channel — eliminates repetitive guard checks in call sites.

        Returns:
            Non-empty list of free channel IDs.

        Raises:
            NoFreeChannelError: If every channel in this scan is occupied.
        """
        free = self.free_channel_ids()
        if not free:
            raise NoFreeChannelError(
                f"All {len(self.channels)} channels occupied at "
                f"{self.scan_timestamp.isoformat()} on '{self.node_id}'"
            )
        return free

    # ─── State vector conversion ──────────────────────────────────────────────

    def to_state_vector(self) -> np.ndarray:
        """Convert this scan result to the 128-dimensional DQN input tensor.

        Packs sensor-derived features into the first 18 slots of the state
        vector. The remaining 110 values (context window, Graph RL features,
        zero-padding) are initialised to zero here and filled by the agent's
        rolling context manager before the network forward pass.

        Vector layout (from CLAUDE.md Section 4 MDP formulation):
            Slot [0 : N]      — occupancy binary  (1.0 = occupied, 0.0 = free)
            Slot [N : 2N]     — interference       normalised [0, 1]
            Slot [2N : 3N]    — received power     normalised over [-120, 0] dBm
            Slot [3N : 4N]    — channel history    (zeros; filled by agent)
            Slot [4N : 5N]    — per-channel SR     (zeros; filled by agent)
            Slot [5N : 6N]    — neighbour features (zeros; filled by Graph RL)
            Slot [6N : 7N]    — spatial intf graph (zeros; filled by Graph RL)
            Slot [7N : 128]   — zero padding

        where N = len(self.channels) = 6 for the standard configuration.

        Channels are sorted by ID before packing to guarantee order-independent
        consistency across calls.

        Returns:
            Float32 numpy array of shape (128,). Contains no NaN or Inf values.

        Raises:
            ChannelStateError: If this scan has no channels.
        """
        if not self.channels:
            raise ChannelStateError(
                f"Cannot build state vector from empty scan on '{self.node_id}'"
            )

        sorted_chs = sorted(self.channels, key=lambda c: c.id)
        n = len(sorted_chs)

        vec = np.zeros(STATE_DIM, dtype=np.float32)

        # Segment 0 — occupancy
        vec[0:n] = [float(ch.occupied) for ch in sorted_chs]

        # Segment 1 — interference (already normalised to [0, 1])
        vec[n : 2 * n] = [ch.interference for ch in sorted_chs]

        # Segment 2 — normalised power: map [-120, 0] dBm → [0, 1]
        vec[2 * n : 3 * n] = [
            float(
                np.clip(
                    (ch.power_dbm - POWER_MIN_DBM) / _POWER_RANGE,
                    0.0,
                    1.0,
                )
            )
            for ch in sorted_chs
        ]

        # Segments 3–6 remain zero; the DQN agent injects context before
        # calling the network. See ai/agents/dqn_agent.py::_enrich_state().

        log.debug(
            "state_vector_built",
            node_id=self.node_id,
            n_channels=n,
            occupancy_rate=round(self.occupancy_rate(), 3),
            free_channels=self.free_channel_ids(),
        )

        return vec
