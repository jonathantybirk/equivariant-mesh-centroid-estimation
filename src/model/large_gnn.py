import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from .base_model import BaseModel


class LargeGNN(BaseModel):
    """
    Heavily optimized Graph Neural Network for mesh centroid estimation

    Key improvements for better performance:
    1. Attention-based node weighting for center-of-mass prediction
    2. Better feature learning with more sophisticated architecture
    3. Multiple prediction heads with ensemble
    4. Stronger geometric priors
    """

    def __init__(
        self,
        hidden_dim=64,  # Increased significantly for better capacity
        message_passing_steps=4,  # More layers for better learning
        message_mlp_dims=[128, 64],  # Larger MLPs
        update_mlp_dims=[64],
        final_mlp_dims=[64, 32],
        dropout=0.15,  # More dropout for regularization
        seed=42,
        lr=1e-3,
        weight_decay=1e-4,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        torch.manual_seed(seed)

        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # FIXED: Much better initial node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # Additional layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Better edge encoding
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
        )

        # GNN layers with better architecture
        self.gnn_layers = nn.ModuleList()
        for i in range(message_passing_steps):
            self.gnn_layers.append(
                AdvancedGNNLayer(
                    hidden_dim=hidden_dim,
                    message_mlp_dims=message_mlp_dims,
                    update_mlp_dims=update_mlp_dims,
                    dropout=dropout,
                    layer_idx=i,
                )
            )

        # FIXED: Attention mechanism for better node weighting
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Multiple prediction heads for ensemble
        self.prediction_heads = nn.ModuleList(
            [
                self._make_prediction_head(hidden_dim, final_mlp_dims, dropout)
                for _ in range(3)  # 3 prediction heads
            ]
        )

        # Initialize weights properly
        self._init_weights()

    def _make_prediction_head(self, hidden_dim, final_mlp_dims, dropout):
        """Create a prediction head"""
        layers = []
        prev_dim = hidden_dim

        for dim in final_mlp_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 3))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Better weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Advanced forward pass with attention-based prediction.
        """
        # Store original positions
        node_positions = x

        # Apply initial embedding
        h = self.node_encoder(x)

        # Edge features
        edge_h = self.edge_encoder(edge_attr)

        # Apply GNN layers with skip connections
        h_layers = [h]
        for i, layer in enumerate(self.gnn_layers):
            h_new = layer(h, edge_index, edge_h)

            # Skip connections from all previous layers
            if i > 0:
                h = h_new + 0.1 * sum(h_layers)  # Weighted sum of all previous
            else:
                h = h_new
            h_layers.append(h)

        # FIXED: Much better prediction strategy
        if batch is not None:
            com_predictions = []
            for b in torch.unique(batch):
                mask = batch == b
                batch_features = h[mask]  # [N_batch, hidden_dim]
                batch_positions = node_positions[mask]  # [N_batch, 3]

                # Attention-based weighting
                attention_scores = self.attention_net(batch_features)  # [N_batch, 1]
                attention_weights = torch.softmax(
                    attention_scores, dim=0
                )  # [N_batch, 1]

                # Weighted feature aggregation
                weighted_features = (attention_weights * batch_features).sum(
                    dim=0, keepdim=True
                )  # [1, hidden_dim]

                # Ensemble predictions from multiple heads
                head_predictions = []
                for head in self.prediction_heads:
                    head_pred = head(weighted_features)  # [1, 3]
                    head_predictions.append(head_pred)

                # Average ensemble predictions
                ensemble_offset = torch.stack(head_predictions, dim=0).mean(
                    dim=0
                )  # [1, 3]

                # Geometric center as strong prior
                geometric_center = batch_positions.mean(dim=0, keepdim=True)  # [1, 3]

                # FIXED: More sophisticated combination
                # Use attention-weighted center + learned offset
                attention_weighted_center = (attention_weights * batch_positions).sum(
                    dim=0, keepdim=True
                )

                # Final prediction: combination of geometric, attention-weighted, and learned
                final_prediction = (
                    0.4 * geometric_center  # Strong geometric prior
                    + 0.4 * attention_weighted_center  # Attention-weighted center
                    + 0.2 * ensemble_offset  # Learned refinement
                )

                com_predictions.append(final_prediction.squeeze(0))

            com_prediction = torch.stack(com_predictions, dim=0)  # [B, 3]
        else:
            # Single graph case
            attention_scores = self.attention_net(h)
            attention_weights = torch.softmax(attention_scores, dim=0)
            weighted_features = (attention_weights * h).sum(dim=0, keepdim=True)

            # Ensemble predictions
            ensemble_offset = torch.stack(
                [head(weighted_features) for head in self.prediction_heads], dim=0
            ).mean(dim=0)

            geometric_center = node_positions.mean(dim=0, keepdim=True)
            attention_weighted_center = (attention_weights * node_positions).sum(
                dim=0, keepdim=True
            )

            com_prediction = (
                0.4 * geometric_center
                + 0.4 * attention_weighted_center
                + 0.2 * ensemble_offset
            )

        return com_prediction


class AdvancedGNNLayer(MessagePassing):
    """
    Advanced GNN layer with better message passing and information flow
    """

    def __init__(
        self, hidden_dim, message_mlp_dims, update_mlp_dims, dropout=0.15, layer_idx=0
    ):
        super().__init__(aggr="add")

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.layer_idx = layer_idx

        # More sophisticated message MLP
        in_dim = 3 * hidden_dim  # [source, target, edge]
        message_layers = []
        prev_dim = in_dim

        for i, dim in enumerate(message_mlp_dims):
            message_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    (
                        nn.ReLU() if i < len(message_mlp_dims) - 1 else nn.Tanh()
                    ),  # Tanh for last layer
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = dim

        message_layers.append(nn.Linear(prev_dim, hidden_dim))
        self.message_mlp = nn.Sequential(*message_layers)

        # Better update MLP
        update_layers = []
        prev_dim = 2 * hidden_dim

        for i, dim in enumerate(update_mlp_dims):
            update_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = dim

        update_layers.append(nn.Linear(prev_dim, hidden_dim))
        self.update_mlp = nn.Sequential(*update_layers)

        # Advanced gating with multiple gates
        self.input_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid()
        )

        self.forget_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid()
        )

        # Layer-specific normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        """Forward pass with advanced gating."""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """Advanced message computation."""
        # Concatenate all features
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(inputs)

    def update(self, aggr_out, x):
        """Advanced update with LSTM-style gating."""
        combined = torch.cat([x, aggr_out], dim=-1)

        # Compute gates
        input_gate = self.input_gate(combined)
        forget_gate = self.forget_gate(combined)

        # Apply update MLP
        candidate = self.update_mlp(combined)

        # LSTM-style update
        updated = forget_gate * x + input_gate * candidate

        # Layer normalization
        return self.layer_norm(updated)
