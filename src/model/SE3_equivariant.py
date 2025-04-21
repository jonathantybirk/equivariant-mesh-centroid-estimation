import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from src.model.gnn import GNN
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing
from torch import nn
from torch_geometric.data import Data

class SteerableConv(MessagePassing):
    def __init__(self, radial_hidden_dims):
        # sum‐aggregation; you could also try mean/max
        super().__init__(aggr='add')
        # radial MLP: input= r, output=[a,b]
        layers = []
        dims = [1] + radial_hidden_dims + [2]
        for d_in, d_out in zip(dims, dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            if d_out != dims[-1]:
                layers.append(nn.ReLU(inplace=True))
        self.radial_mlp = nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_attr):
        # Print shape of incoming edge_attr
        print(f"SteerableConv received edge_attr with shape {edge_attr.shape}, mean norm: {edge_attr.norm(dim=1).mean().item():.4f}")
        
        # Print first few entries to verify the values are changing with rotation
        print(f"First 3 edge_attr entries: {edge_attr[:3].detach().cpu().numpy()}")
        
        # x: (N,3) or whatever, but unused here
        # edge_attr: (E,3) = Δx
        return self.propagate(edge_index, edge_attr=edge_attr)

    def message(self, edge_attr):
        """
        edge_attr: [E,3] = Δx_ij
        returns message m_ij: [E,3]
        """
        dx = edge_attr                      # [E,3]
        r = dx.norm(p=2, dim=-1, keepdim=True)       # [E,1]
        hatd = dx / (r + 1e-8)             # [E,3]

        # radial MLP predicts [a,b] per edge
        a_b = self.radial_mlp(r)           # [E,2]
        a, b = a_b[:, 0:1], a_b[:, 1:2]     # each [E,1]

        # K(x) x = a I·dx + b (hatd hatdᵀ) dx
        # but (hatd hatdᵀ) dx = hatd (hatd·dx) = hatd * r
        msg = a * dx + b * (hatd * r)      # [E,3]
        return msg
    


class GNN(nn.Module):
    def __init__(self, hidden_dim, message_passing_steps, radial_hidden_dims):
        # Removed update_mlp_dims, final_mlp_dims
        super().__init__()
        self.convs = nn.ModuleList([
            SteerableConv(radial_hidden_dims) 
            for _ in range(message_passing_steps)
        ])
        # REMOVED self.update_mlps
        # REMOVED self.final_mlp

    def forward(self, x, edge_index, edge_attr, batch):
        # Initial features 'h' can start as zero vectors, as SteerableConv uses edge_attr
        # Ensure h has the correct shape [N, 3] for vector features
        num_nodes = x.shape[0]
        h = torch.zeros((num_nodes, 3), device=x.device, dtype=edge_attr.dtype)
        
        for i, conv in enumerate(self.convs):
            # Calculate equivariant messages and aggregate them
            m = conv(h, edge_index, edge_attr)    # m is equivariant [N, 3]
            
            # Use the aggregated messages directly as the node features for the next layer
            # Or, if it's the last layer, this 'm' contains the final node-level equivariant features.
            h = m # Replacing h = upd(m)
            
            # Optional: Add debug print for the node features h after each layer
            # print(f"--- GNN Layer {i} Features 'h' (mean norm): {h.norm(dim=1).mean().item():.4f}")

        # Global SUM pooling (preserves equivariance for vectors)
        # Use manual aggregation if torch_scatter is unavailable
        if batch is None:
            # If no batch, sum over all nodes
            hg = h.sum(dim=0, keepdim=True) # Shape [1, 3]
        else:
            # Manual batch sum calculation
            batch = batch.to(x.device)
            num_batches = int(batch.max().item() + 1)
            output_shape = (num_batches, h.shape[1]) # Output shape [batch_size, 3]
            hg = torch.zeros(output_shape, device=x.device, dtype=h.dtype)
            hg.index_add_(0, batch, h) # Sum node features 'h' per batch

        # Return the equivariant graph-level vector directly
        return hg



class GNNLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for center of mass estimation using a GNN
    """
    def __init__(
        self,
        hidden_dim=16,
        message_passing_steps=3,
        message_mlp_dims=[70, 140, 20],
        lr=0.001
    ):
        super().__init__()
        # Save relevant hyperparameters
        self.save_hyperparameters('hidden_dim', 'message_passing_steps', 'message_mlp_dims', 'lr')
        
        # Create the modified GNN model
        self.model = GNN(
            hidden_dim=self.hparams.hidden_dim,
            message_passing_steps=self.hparams.message_passing_steps,
            radial_hidden_dims=self.hparams.message_mlp_dims
        )
        
        # Define loss function
        self.loss_fn = torch.nn.MSELoss()
        
    def forward(self, data):
        """Forward pass through the model"""
        return self.model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )
    
    def _common_step(self, batch, batch_idx):
        """Common operations for training, validation, and test steps"""
        # Forward pass
        pred = self(batch)
        
        # Calculate loss
        loss = self.loss_fn(pred, batch.y)
        
        # Calculate MSE and RMSE
        mse = torch.nn.functional.mse_loss(pred, batch.y)
        rmse = torch.sqrt(mse)
        
        return {
            'loss': loss,
            'mse': mse,
            'rmse': rmse,
            'preds': pred,
            'targets': batch.y
        }
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        results = self._common_step(batch, batch_idx)
        self.log('train_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', results['mse'], on_epoch=True)
        self.log('train_rmse', results['rmse'], on_epoch=True)
        return results['loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        results = self._common_step(batch, batch_idx)
        self.log('val_loss', results['loss'], on_epoch=True, prog_bar=True)
        self.log('val_mse', results['mse'], on_epoch=True)
        self.log('val_rmse', results['rmse'], on_epoch=True)
        return results
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        results = self._common_step(batch, batch_idx)
        self.log('test_loss', results['loss'], on_epoch=True)
        self.log('test_mse', results['mse'], on_epoch=True)
        self.log('test_rmse', results['rmse'], on_epoch=True)
        return results
    
    def configure_optimizers(self):
        """Configure optimizers"""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def on_validation_epoch_end(self):
        """Log visualizations at the end of validation epochs"""
        if self.current_epoch % 10 == 0:  # Every 10 epochs
            self._visualize_predictions()
    
    def _visualize_predictions(self):
        """Create and save 3D visualizations of predictions"""
        if not self.trainer.is_global_zero:
            return
            
        # Get a batch from validation loader
        val_batch = next(iter(self.trainer.datamodule.val_dataloader()))
        val_batch = val_batch.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            preds = self(val_batch)
        
        # Ensure visualization directory exists    
        os.makedirs('visualizations', exist_ok=True)
        
        # Visualize 3 examples
        for i in range(min(3, val_batch.num_graphs)):
            # Get example data from the batch
            example = val_batch.get_example(i)
            point_cloud = example.pos.cpu().numpy()
            true_com = val_batch.y[i].cpu().numpy()
            pred_com = preds[i].cpu().numpy()
            
            # Create 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot point cloud with small dots
            ax.scatter(
                point_cloud[:, 0], 
                point_cloud[:, 1], 
                point_cloud[:, 2], 
                c='blue', s=10, alpha=0.5, marker='o',
                label='Point Cloud'
            )
            
            # Plot true center of mass
            ax.scatter(
                true_com[0], true_com[1], true_com[2],
                c='green', s=100, marker='*',
                label='True Center of Mass'
            )
            
            # Plot predicted center of mass
            ax.scatter(
                pred_com[0], pred_com[1], pred_com[2],
                c='red', s=100, marker='x',
                label='Predicted Center of Mass'
            )
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Center of Mass Prediction (Epoch {self.current_epoch})')
            ax.legend()
            
            # Save figure
            plt.savefig(
                f'visualizations/epoch_{self.current_epoch}_example_{i+1}.png',
                dpi=150, bbox_inches='tight'
            )
            plt.close(fig)