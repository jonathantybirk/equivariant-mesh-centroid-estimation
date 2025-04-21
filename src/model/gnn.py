import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GNN(torch.nn.Module):
    """
    Graph Neural Network for center of mass estimation
    """
    def __init__(
        self, 
        hidden_dim=16, 
        message_passing_steps=3,
        message_mlp_dims=[70, 140, 20],
        update_mlp_dims=[70],
        final_mlp_dims=[64, 32],
        seed=42
    ):
        super().__init__()
        torch.manual_seed(seed)
        
        # Initial node embedding
        self.node_encoder = nn.Linear(hidden_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(message_passing_steps):
            self.gnn_layers.append(
                GNNLayer(
                    hidden_dim=hidden_dim,
                    message_mlp_dims=message_mlp_dims,
                    update_mlp_dims=update_mlp_dims
                )
            )
        
        # Final MLP for per-node prediction
        final_layers = []
        prev_dim = hidden_dim
        
        for dim in final_mlp_dims:
            final_layers.append(nn.Linear(prev_dim, dim))
            final_layers.append(nn.ReLU())
            prev_dim = dim
        
        final_layers.append(nn.Linear(prev_dim, 3))  # Output 3D coordinates
        self.final_mlp = nn.Sequential(*final_layers)
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass through the network.
        
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge attributes [E, 3]
            batch: Batch assignment for nodes [N]
            
        Returns:
            Center of mass prediction [B, 3] where B is batch size
        """
        # Apply initial embedding
        x = self.node_encoder(x)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)
        
        # Apply final MLP to get per-node predictions
        node_predictions = self.final_mlp(x)
        
        # Average predictions across nodes in each graph
        if batch is not None:
            # Custom implementation of scatter_mean
            com_prediction = self._scatter_mean(node_predictions, batch)
        else:
            # If single graph, just average all node predictions
            com_prediction = node_predictions.mean(dim=0, keepdim=True)
            
        return com_prediction
    
    def _scatter_mean(self, src, index, dim_size=None):
        """Custom function to replace torch_scatter.scatter_mean for pooling"""
        if dim_size is None:
            dim_size = index.max().item() + 1
            
        # Create output tensor
        out = torch.zeros(dim_size, src.size(1), device=src.device)
        
        # Count elements per target index
        ones = torch.ones(src.size(0), 1, device=src.device)
        count = torch.zeros(dim_size, 1, device=src.device)
        count.index_add_(0, index, ones)
        
        # Sum elements with same target
        out.index_add_(0, index, src)
        
        # Avoid division by zero
        count = torch.clamp(count, min=1.0)
        
        # Compute mean
        out = out / count
        
        return out


class GNNLayer(MessagePassing):
    """
    Basic Graph Neural Network layer with message passing
    """
    def __init__(self, hidden_dim, message_mlp_dims, update_mlp_dims):
        super().__init__(aggr='mean')  # Use mean aggregation
        
        # Message MLP - transforms source, target node features and edge attributes
        # Input: [source_features, target_features, edge_attributes]
        in_dim = 2 * hidden_dim + 3  # (hidden_dim + hidden_dim + 3D positions)
        message_layers = []
        prev_dim = in_dim
        
        for dim in message_mlp_dims:
            message_layers.append(nn.Linear(prev_dim, dim))
            message_layers.append(nn.ReLU())
            prev_dim = dim
        
        message_layers.append(nn.Linear(prev_dim, hidden_dim))
        self.message_mlp = nn.Sequential(*message_layers)
        
        # Update MLP - updates node features after aggregation
        # Input: [node_features, aggregated_messages]
        update_layers = []
        prev_dim = 2 * hidden_dim
        
        for dim in update_mlp_dims:
            update_layers.append(nn.Linear(prev_dim, dim))
            update_layers.append(nn.ReLU())
            prev_dim = dim
        
        update_layers.append(nn.Linear(prev_dim, hidden_dim))
        self.update_mlp = nn.Sequential(*update_layers)
        
    def forward(self, x, edge_index, edge_attr):
        """Forward pass through the layer."""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
    def message(self, x_i, x_j, edge_attr):
        """Compute messages from source to target nodes."""
        # Concatenate source features, target features, and edge attributes
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(inputs)
    
    def update(self, aggr_out, x):
        """Update node features with aggregated messages."""
        inputs = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(inputs)
    