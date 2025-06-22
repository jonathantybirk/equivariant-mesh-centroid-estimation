import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from .base_model import BaseModel


class BasicGNN(BaseModel):
    """
    Non-equivariant GNN with FAIR ACCESS to the same features as EquivariantGNN.

    This model has access to:
    1. Node positions (x) - same as EquivariantGNN input
    2. Edge vectors (r) - same geometric information
    3. Edge distances (d) - same scalar distances
    4. Average neighbor distances - same aggregated distance info
    5. Rich geometric features - non-equivariant version of SH features

    Target: ~7,000-8,000 parameters to match EquivariantGNN complexity
    """

    def __init__(
        self,
        input_dim=3,
        hidden_dim=16,  # Reduced to match parameter count
        message_passing_steps=2,  # Match EquivariantGNN
        edge_dim=3,
        final_mlp_dims=[32, 16],  # Reduced to match parameter count
        message_mlp_dims=[32, 16],  # Reduced to match parameter count
        dropout=0.1,
        seed=42,
        lr=1e-3,
        weight_decay=1e-5,  # Match EquivariantGNN
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        torch.manual_seed(seed)

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.message_passing_steps = message_passing_steps
        self.message_mlp_dims = message_mlp_dims

        # Rich node feature encoder - extract same geometric info as SH features
        # Input: [x, y, z, x², y², z², xy, xz, yz, ||x||, ||x||²] = 11 features
        self.node_encoder = nn.Sequential(
            nn.Linear(11, hidden_dim),  # Rich geometric features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Rich edge feature encoder - same info as spherical harmonics
        # Input: [r_x, r_y, r_z, r_x², r_y², r_z², r_xy, r_xz, r_yz, ||r||, ||r||²] = 11 features
        self.edge_encoder = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for _ in range(message_passing_steps):
            self.mp_layers.append(
                GNNLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    message_mlp_dims=message_mlp_dims,
                    dropout=dropout,
                )
            )

        # Final MLP - same architecture as EquivariantGNN
        final_layers = []
        final_layers.append(nn.LayerNorm(hidden_dim))

        prev_dim = hidden_dim
        for dim in final_mlp_dims:
            final_layers.append(nn.Linear(prev_dim, dim))
            final_layers.append(nn.LayerNorm(dim))
            final_layers.append(nn.ReLU())
            if dropout > 0:
                final_layers.append(nn.Dropout(dropout))
            prev_dim = dim

        final_layers.append(nn.Linear(prev_dim, 1))  # Output scalar weights
        self.final_mlp = nn.Sequential(*final_layers)

        # Initialize weights - same as EquivariantGNN
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training - same as EquivariantGNN"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _extract_rich_geometric_features(self, pos):
        """
        Extract rich geometric features equivalent to spherical harmonics.

        This gives the non-equivariant model access to the same geometric
        information that EquivariantGNN gets from spherical harmonics.
        """
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

        # Polynomial features (similar to SH expansion)
        x2, y2, z2 = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z

        # Radial features
        r = torch.norm(pos, dim=1)
        r2 = r * r

        # Stack all features: [x, y, z, x², y², z², xy, xz, yz, ||x||, ||x||²]
        features = torch.stack([x, y, z, x2, y2, z2, xy, xz, yz, r, r2], dim=1)
        return features

    def forward(self, x, edge_index, r, batch=None):
        """
        Forward pass with same interface as EquivariantGNN.

        Args:
            x: Node positions [N, 3]
            edge_index: Graph connectivity [2, E]
            r: Edge vectors [E, 3] (r_ij = x_j - x_i)
            batch: Batch assignment [N]
        """
        # Extract rich geometric features from positions (same info as SH)
        node_features = self._extract_rich_geometric_features(x)  # [N, 11]
        h = self.node_encoder(node_features)  # [N, hidden_dim]

        # Extract rich geometric features from edge vectors (same info as edge SH)
        edge_features = self._extract_rich_geometric_features(r)  # [E, 11]
        edge_h = self.edge_encoder(edge_features)  # [E, hidden_dim]

        # Compute edge distances (same as EquivariantGNN)
        d = torch.norm(r, dim=1, keepdim=True)  # [E, 1]

        # Message passing with distance information
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_h, d)

        # Compute node-level logits (same as EquivariantGNN final step)
        node_logits = self.final_mlp(h).squeeze(-1)  # [N]

        # Centroid estimation with softmax weights (same as EquivariantGNN)
        if batch is not None:
            # Batched processing
            unique_batches = torch.unique(batch)
            centroid_predictions = []

            for b in unique_batches:
                mask = batch == b
                batch_logits = node_logits[mask]
                batch_positions = x[mask]

                # Softmax weights (same as EquivariantGNN)
                batch_weights = torch.softmax(batch_logits, dim=0)

                # Weighted centroid (same as EquivariantGNN)
                weighted_centroid = torch.sum(
                    batch_weights.unsqueeze(-1) * batch_positions, dim=0
                )
                centroid_predictions.append(weighted_centroid)

            centroid_prediction = torch.stack(centroid_predictions, dim=0)
        else:
            # Single graph case
            weights = torch.softmax(node_logits, dim=0)
            centroid_prediction = torch.sum(weights.unsqueeze(-1) * x, dim=0).unsqueeze(
                0
            )

        return centroid_prediction


class GNNLayer(MessagePassing):
    """
    GNN layer with fair access to same information as EquivariantGNN layer.

    Includes:
    - Rich message computation using node + edge + distance features
    - MLP gating similar to EquivariantGNN's ψ_f function
    - Average distance computation per node
    """

    def __init__(self, hidden_dim, edge_dim, message_mlp_dims=[64, 32], dropout=0.1):
        super().__init__(aggr="add")

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Message MLP: [node_i + node_j + edge + distance] -> hidden_dim
        message_input_dim = 2 * hidden_dim + edge_dim + 1  # +1 for distance
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Gating MLP (similar to EquivariantGNN's ψ_f)
        # Input: [current_node + aggregated_messages + avg_distance]
        gate_input_dim = 2 * hidden_dim + 1  # node + messages + avg_distance

        # Build configurable gating MLP (same structure as EquivariantGNN)
        gate_layers = []
        prev_dim = gate_input_dim

        for dim in message_mlp_dims:
            gate_layers.append(nn.Linear(prev_dim, dim))
            gate_layers.append(nn.ReLU())
            if dropout > 0:
                gate_layers.append(nn.Dropout(dropout))
            prev_dim = dim

        gate_layers.append(nn.Linear(prev_dim, hidden_dim))  # Output gate
        gate_layers.append(nn.Sigmoid())

        self.gate_mlp = nn.Sequential(*gate_layers)

    def forward(self, x, edge_index, edge_attr, edge_distances):
        """Forward pass with gated updates like EquivariantGNN"""
        # Message passing
        messages = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, edge_distances=edge_distances
        )

        # Compute average distances per node (same as EquivariantGNN)
        source_idx, target_idx = edge_index
        node_distances = torch.zeros(x.size(0), 1, device=x.device)
        node_distances.index_add_(0, target_idx, edge_distances)
        neighbor_counts = torch.zeros(x.size(0), device=x.device)
        neighbor_counts.index_add_(
            0, target_idx, torch.ones(edge_distances.size(0), device=x.device)
        )
        avg_distances = node_distances / (neighbor_counts.unsqueeze(1) + 1e-8)

        # Gated update (same pattern as EquivariantGNN)
        gate_input = torch.cat([x, messages, avg_distances], dim=1)
        gates = self.gate_mlp(gate_input)  # [N, hidden_dim]

        # Gated residual update: x_new = x_old + gate * messages
        updated_x = x + gates * messages

        return updated_x

    def message(self, x_i, x_j, edge_attr, edge_distances):
        """Compute messages with same information as EquivariantGNN"""
        # Concatenate: source node + target node + edge features + distance
        inputs = torch.cat([x_i, x_j, edge_attr, edge_distances], dim=-1)
        return self.message_mlp(inputs)
