import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .base_model import BaseModel


class BasicGNN(BaseModel):
    """
    Simple, efficient GNN with exactly ~7,000 parameters for fair comparison.
    No attention mechanisms, just clean message passing with good architectural choices.

    Target: ~7,000 parameters to match EquivariantGNN
    """

    def __init__(
        self,
        input_dim=3,
        hidden_dim=24,  # Carefully chosen to hit ~7K params
        message_passing_steps=3,
        edge_dim=3,
        final_mlp_dims=[32, 16],
        dropout=0.1,
        seed=42,
        lr=1e-3,
        weight_decay=1e-4,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        torch.manual_seed(seed)

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Node feature encoder: 3 -> 24 (3*24 + 24 = 96 params)
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Edge feature encoder: 3 -> 24 (3*24 + 24 = 96 params)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Message passing layers (bulk of parameters)
        self.mp_layers = nn.ModuleList()
        for _ in range(message_passing_steps):
            self.mp_layers.append(
                BasicGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                )
            )

        # Final MLP for center of mass prediction
        final_layers = []
        prev_dim = hidden_dim

        for dim in final_mlp_dims:
            final_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
            prev_dim = dim

        final_layers.append(nn.Linear(prev_dim, 3))  # Output 3D coordinates
        self.final_mlp = nn.Sequential(*final_layers)

        # Initialize weights
        self._init_weights()

        # Print parameter count for verification
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"BasicGNN total parameters: {total_params:,}")

    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Simple forward pass through the network.

        Args:
            x: Node features [N, 3] (positions)
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge attributes [E, 3] (edge vectors)
            batch: Batch assignment for nodes [N]

        Returns:
            Center of mass prediction [B, 3] where B is batch size
        """
        # Encode initial node and edge features
        h = self.node_encoder(x)  # [N, hidden_dim]
        edge_h = self.edge_encoder(edge_attr)  # [E, hidden_dim]

        # Message passing
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_h)

        # Global aggregation (simple mean pooling)
        if batch is not None:
            graph_repr = self._global_mean_pool(h, batch)  # [B, hidden_dim]
        else:
            graph_repr = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]

        # Final prediction
        com_prediction = self.final_mlp(graph_repr)  # [B, 3]

        return com_prediction

    def _global_mean_pool(self, x, batch):
        """Global mean pooling for batched graphs"""
        unique_batches = torch.unique(batch)
        graph_reprs = []

        for b in unique_batches:
            mask = batch == b
            batch_x = x[mask]  # [N_batch, hidden_dim]
            graph_repr = batch_x.mean(dim=0)  # [hidden_dim]
            graph_reprs.append(graph_repr)

        return torch.stack(graph_reprs, dim=0)  # [B, hidden_dim]


class BasicGNNLayer(MessagePassing):
    """
    Simple GNN layer with efficient message passing and minimal parameters
    """

    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__(aggr="add")

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Message MLP: [node_i + node_j + edge] -> hidden_dim
        # Input: 24 + 24 + 24 = 72, Output: 24
        # Params: 72*24 + 24 = 1,752 per layer
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Update MLP: [old_node + message] -> hidden_dim
        # Input: 24 + 24 = 48, Output: 24
        # Params: 48*24 + 24 = 1,176 per layer
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Simple residual connection
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, edge_index, edge_attr):
        """Forward pass with residual connection"""
        # Store residual
        residual = x

        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Simple residual connection with learnable weight
        x = (1 - self.residual_weight) * residual + self.residual_weight * out

        return x

    def message(self, x_i, x_j, edge_attr):
        """Compute messages using simple concatenation"""
        # Concatenate source features, target features, and edge attributes
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(inputs)

    def update(self, aggr_out, x):
        """Update node features with aggregated messages"""
        inputs = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(inputs)


# Keep the original ImprovedGNN class for backward compatibility
class ImprovedGNN(BaseModel):
    """
    Improved Graph Neural Network with layer normalization, residual connections,
    and better architectural choices for mesh centroid estimation.

    Key improvements over LargeGNN:
    1. Layer normalization for stable training
    2. Residual connections for better gradient flow
    3. Better edge feature utilization with improved MLPs
    4. Simple mean pooling for final aggregation
    5. More sophisticated node initialization
    6. Gated mechanisms for better information flow
    7. Better weight initialization
    """

    def __init__(
        self,
        input_dim=3,
        hidden_dim=32,
        message_passing_steps=3,
        num_attention_heads=4,
        edge_dim=3,
        final_mlp_dims=[64, 32],
        dropout=0.1,
        seed=42,
        lr=1e-3,
        weight_decay=1e-4,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        torch.manual_seed(seed)

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads

        # Ensure hidden_dim is divisible by num_attention_heads
        assert (
            hidden_dim % num_attention_heads == 0
        ), f"hidden_dim ({hidden_dim}) must be divisible by num_attention_heads ({num_attention_heads})"

        # Node feature encoder with better initialization
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Message passing layers without attention
        self.mp_layers = nn.ModuleList()
        for _ in range(message_passing_steps):
            self.mp_layers.append(
                ImprovedGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                )
            )

        # Final MLP for center of mass prediction
        final_layers = []
        prev_dim = hidden_dim

        for dim in final_mlp_dims:
            final_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
            prev_dim = dim

        final_layers.append(nn.Linear(prev_dim, 3))  # Output 3D coordinates
        self.final_mlp = nn.Sequential(*final_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass through the improved network.

        Args:
            x: Node features [N, 3] (positions)
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge attributes [E, 3] (edge vectors)
            batch: Batch assignment for nodes [N]

        Returns:
            Center of mass prediction [B, 3] where B is batch size
        """
        # Encode initial node features
        h = self.node_encoder(x)  # [N, hidden_dim]

        # Encode edge features
        edge_h = self.edge_encoder(edge_attr)  # [E, hidden_dim]

        # Message passing without attention
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_h)

        # Simple global aggregation (mean pooling)
        if batch is not None:
            graph_repr = self._global_mean_pool(h, batch)  # [B, hidden_dim]
        else:
            graph_repr = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]

        # Final prediction
        com_prediction = self.final_mlp(graph_repr)  # [B, 3]

        return com_prediction

    def _global_mean_pool(self, x, batch):
        """Global mean pooling for batched graphs"""
        unique_batches = torch.unique(batch)
        graph_reprs = []

        for b in unique_batches:
            mask = batch == b
            batch_x = x[mask]  # [N_batch, hidden_dim]
            graph_repr = batch_x.mean(dim=0)  # [hidden_dim]
            graph_reprs.append(graph_repr)

        return torch.stack(graph_reprs, dim=0)  # [B, hidden_dim]


class ImprovedGNNLayer(MessagePassing):
    """
    Improved GNN layer with normalization, residual connections, and better edge handling
    """

    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__(aggr="add")

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Message MLP - improved from LargeGNN
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Update MLP - improved from LargeGNN
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Normalization and residual
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Gating mechanism
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

    def forward(self, x, edge_index, edge_attr):
        """Forward pass with residual connections and normalization"""
        # Store residual
        residual = x

        # Apply layer norm
        x_norm = self.norm1(x)

        # Message passing
        out = self.propagate(edge_index, x=x_norm, edge_attr=edge_attr)

        # Residual connection
        x = residual + out

        # Second residual block with FFN
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)

        # Gating mechanism
        gate_input = torch.cat([x, ffn_out], dim=-1)
        gate_values = self.gate(gate_input)
        x = residual + gate_values * ffn_out

        return x

    def message(self, x_i, x_j, edge_attr):
        """Compute messages using improved MLPs"""
        # Concatenate source features, target features, and edge attributes
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(inputs)

    def update(self, aggr_out, x):
        """Update node features with aggregated messages"""
        inputs = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(inputs)
