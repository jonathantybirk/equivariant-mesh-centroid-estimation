import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from src.model.gnn import GNN


class GNNLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for center of mass estimation using a GNN
    """
    def __init__(
        self,
        hidden_dim=16,
        message_passing_steps=3,
        message_mlp_dims=[70, 140, 20],
        update_mlp_dims=[70],
        final_mlp_dims=[64, 32],
        lr=0.001,
        seed=42
    ):
        super().__init__()
        # Save all constructor arguments as hyperparameters
        self.save_hyperparameters()
        
        # Create the GNN model
        self.model = GNN(
            hidden_dim=hidden_dim,
            message_passing_steps=message_passing_steps,
            message_mlp_dims=message_mlp_dims,
            update_mlp_dims=update_mlp_dims,
            final_mlp_dims=final_mlp_dims,
            seed=seed
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
        
        # Create visualization directory
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