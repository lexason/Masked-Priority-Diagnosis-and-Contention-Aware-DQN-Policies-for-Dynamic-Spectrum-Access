"""AI Agents — Graph RL Agent (GNN + DQN, jointly trained end-to-end).

Extends DQNAgent with a Graph Neural Network encoder that captures spatial
interference correlations between channels. The GNN encoder is a genuine
submodule of the online/target Q-network: its forward pass runs inside the
same autograd graph as the DQN head, so a single ``loss.backward()`` call in
``train_step`` updates both the GNN's and the head's parameters together.
This module previously computed the GNN embedding under ``torch.no_grad()``
and converted it to a detached NumPy array before it ever reached the
replay buffer, which meant the GNN's parameters received no gradient
whatsoever, and ``save_checkpoint``/``load_checkpoint`` (inherited from
DQNAgent) never persisted the GNN's weights at all. The two defects
compounded: even had gradients flowed, evaluation would have loaded a
freshly, randomly initialised GNN every time regardless. Both are fixed
here by making the GNN an internal submodule of the network assigned to
``_online``/``_target``, so the existing, unmodified
``DQNAgent.save_checkpoint``/``load_checkpoint``/``train_step`` machinery
saves, loads and trains it correctly with no further changes required
elsewhere.

Graph construction:
  - Nodes: one per logical channel (N = 6 for standard config)
  - Node features: [occupied, interference, power_norm] — shape (N, 3),
    reconstructed at each forward pass from the state vector's own
    occupancy/interference/power segments (slots [0:3N]), not stored
    separately
  - Edges: fully connected (all channel pairs, no self-loops)

GNN architecture:
  GCNConv(3 → 32) → ReLU → GCNConv(32 → 6) → ReLU → global mean pool → FC(6 → 12)

The 12-dim output fills segments [5N:6N] (neighbour embeddings, 6-dim) and
[6N:7N] (spatial interference graph summary, 6-dim) of a *copy* of the input
state before the combined DQN head processes it, so the head sees the same
128-dim layout as before.

torch_geometric is a required project dependency (pyproject.toml); this
module no longer degrades to plain DQN if it is missing, since a silent
fallback was one of the reasons the earlier defect went unnoticed for as
long as it did.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

from ai.agents.dqn_agent import STATE_DIM, DQNAgent, DQNNetwork

log = structlog.get_logger(__name__)


# ─── Combined, jointly-trained GNN + DQN network ──────────────────────────────

class GraphDQNNetwork(nn.Module):
    """DQN network augmented with a jointly trained GCN encoder.

    Reconstructs the six-channel interference graph from the input state's
    occupancy/interference/power segments, refines it through a two-layer
    GCN, splices the resulting embedding into the reserved state segments,
    and feeds the augmented state through the same MLP head architecture
    ``DQNNetwork`` uses. Nothing here is wrapped in ``torch.no_grad()``, so
    calling code controls differentiability the normal PyTorch way: a
    ``train_step``-style ``loss.backward()`` updates the GCN and the head
    together; ``select_action``-style inference wraps the whole forward
    pass in ``torch.no_grad()`` exactly as it already does for plain DQN.

    Args:
        state_dim:      Input/output state dimensionality (default 128).
        n_actions:      Number of output Q-values (default 6).
        n_channels:     Number of logical channels / graph nodes (default 6).
        gnn_hidden_dim: GCN hidden layer width (default 32).
        gnn_out_dim:    GNN output dimensionality injected into state (default 12).
        dropout:        Dropout probability in the MLP head.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = 6,
        n_channels: int = 6,
        gnn_hidden_dim: int = 32,
        gnn_out_dim: int = 12,
        dropout: float = 0.2,
        ablate_embedding: bool = False,
        use_graph_conv: bool = True,
    ) -> None:
        super().__init__()
        self._n = n_channels
        self.ablate_embedding = ablate_embedding
        self.use_graph_conv = use_graph_conv

        if use_graph_conv:
            self.conv1 = GCNConv(3, gnn_hidden_dim)
            self.conv2 = GCNConv(gnn_hidden_dim, gnn_out_dim // 2)
        else:
            # Parameter-matched control (no message passing): identical
            # layer shapes to conv1/conv2 above, applied independently per
            # channel with no cross-channel adjacency term, isolating
            # whether graph convolution specifically drives the encoder's
            # negligible marginal contribution (Section~sec:gcn) or whether
            # any small extra transform of the same size would do the same.
            self.conv1 = nn.Linear(3, gnn_hidden_dim)
            self.conv2 = nn.Linear(gnn_hidden_dim, gnn_out_dim // 2)
        self.gnn_fc = nn.Linear(gnn_out_dim // 2, gnn_out_dim)
        nn.init.kaiming_uniform_(self.gnn_fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.gnn_fc.bias)

        self.head = DQNNetwork(state_dim, n_actions, dropout)

        edge_index = _build_edge_index(n_channels)
        self.register_buffer("edge_index", edge_index)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of states.

        Args:
            state: FloatTensor of shape (B, state_dim).

        Returns:
            FloatTensor of shape (B, n_actions).
        """
        n = self._n
        augmented = state.clone()

        if self.ablate_embedding:
            # Embedding-zeroed ablation (Tier A): leave the GNN's reserved
            # state segments at zero — whatever _enrich_state already put
            # there — instead of computing and injecting the GCN embedding.
            # Isolates the trained head's reliance on genuinely-learned
            # spatial information, using a checkpoint that WAS trained with
            # that information available at training time.
            return self.head(augmented)

        occ = state[:, 0:n]
        interf = state[:, n : 2 * n]
        power = state[:, 2 * n : 3 * n]
        node_feats = torch.stack([occ, interf, power], dim=-1)  # (B, N, 3)

        if self.use_graph_conv:
            batch_size = state.shape[0]
            data_list = [
                Data(x=node_feats[i], edge_index=self.edge_index) for i in range(batch_size)
            ]
            batch = Batch.from_data_list(data_list)

            x = torch.relu(self.conv1(batch.x, batch.edge_index))
            x = torch.relu(self.conv2(x, batch.edge_index))
            pooled = global_mean_pool(x, batch.batch)  # (B, gnn_out_dim // 2)
        else:
            # Same per-node transform shapes, no adjacency: each channel's
            # (B, 3) features are mapped independently, then averaged across
            # channels exactly as global_mean_pool would, so the only
            # difference from the graph-convolution branch is the absence
            # of neighbour message passing in conv1/conv2 themselves.
            x = torch.relu(self.conv1(node_feats))  # (B, N, gnn_hidden_dim)
            x = torch.relu(self.conv2(x))  # (B, N, gnn_out_dim // 2)
            pooled = x.mean(dim=1)  # (B, gnn_out_dim // 2)
        embedding = self.gnn_fc(pooled)  # (B, gnn_out_dim)

        augmented[:, 5 * n : 6 * n] = embedding[:, :6]
        augmented[:, 6 * n : 7 * n] = embedding[:, 6:]
        return self.head(augmented)


def _build_edge_index(n: int) -> torch.Tensor:
    """Build a fully-connected undirected edge index (no self-loops).

    Args:
        n: Number of nodes.

    Returns:
        LongTensor of shape (2, n*(n-1)) — source and target node indices.
    """
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


# ─── Graph RL Agent ───────────────────────────────────────────────────────────

class GraphRLAgent(DQNAgent):
    """Graph RL agent: GNN spatial encoder + DQN policy, trained jointly.

    Inherits all DQNAgent functionality unchanged (state enrichment, replay
    buffer, ``train_step``, ``save_checkpoint``/``load_checkpoint``); the
    only difference from plain DQN is that ``_online``/``_target`` are
    ``GraphDQNNetwork`` instances instead of plain ``DQNNetwork`` instances,
    so no other method needs to know the difference.

    Args:
        node_id:        Unique node identity string.
        n_channels:     Number of logical spectrum channels (default 6).
        gnn_hidden_dim: GCN hidden layer width (default 32).
        gnn_out_dim:    GNN output dimensionality injected into state (default 12).
        See DQNAgent for remaining keyword arguments.
    """

    def __init__(
        self,
        node_id: str,
        n_channels: int = 6,
        gnn_hidden_dim: int = 32,
        gnn_out_dim: int = 12,
        ablate_embedding: bool = False,
        use_graph_conv: bool = True,
        **dqn_kwargs: Any,
    ) -> None:
        super().__init__(node_id, n_channels, **dqn_kwargs)

        state_dim = dqn_kwargs.get("state_dim", STATE_DIM)
        self._online = GraphDQNNetwork(
            state_dim=state_dim,
            n_actions=self._n_actions,
            n_channels=n_channels,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_out_dim=gnn_out_dim,
            dropout=dqn_kwargs.get("dropout", 0.2),
            ablate_embedding=ablate_embedding,
            use_graph_conv=use_graph_conv,
        )
        self._target = GraphDQNNetwork(
            state_dim=state_dim,
            n_actions=self._n_actions,
            n_channels=n_channels,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_out_dim=gnn_out_dim,
            dropout=dqn_kwargs.get("dropout", 0.2),
            ablate_embedding=ablate_embedding,
            use_graph_conv=use_graph_conv,
        )
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()

        self._optimiser = optim.Adam(
            self._online.parameters(), lr=dqn_kwargs.get("learning_rate", 1e-3)
        )

        log.info(
            "graph_rl_agent_ready",
            node_id=node_id,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_out_dim=gnn_out_dim,
        )
