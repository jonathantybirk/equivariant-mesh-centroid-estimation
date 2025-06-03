import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, List
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.eq_gnn import GNN


class EquivariantGNNLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the Equivariant GNN
    """

    def __init__(
        self,
        input_a_l: List[int] = [0, 1, 2],
        input_h_l: List[int] = [0, 1],
        h_l_out: List[int] = [0, 1],
        hidden_dim: int = 16,
        message_passing_steps: int = 3,
        message_mlp_dims: Optional[List[int]] = None,  # For compatibility with config
        final_mlp_dims: List[int] = [64, 32],
        max_sh_degree: int = 2,
        init_method: str = "xavier",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create the underlying GNN model
        self.gnn = GNN(
            input_a_l=input_a_l,
            input_h_l=input_h_l,
            h_l_out=h_l_out,
            hidden_dim=hidden_dim,
            message_passing_steps=message_passing_steps,
            final_mlp_dims=final_mlp_dims,
            max_sh_degree=max_sh_degree,
            init_method=init_method,
            seed=seed,
        )

        # Store hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Forward pass through the model"""
        return self.gnn(x, edge_index, edge_attr, batch)

    def compute_loss(self, pred, target):
        """Compute MSE loss between prediction and target center of mass"""
        return F.mse_loss(pred, target)

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, edge_index, edge_attr, batch_indices, target_com = batch

        # Forward pass
        pred_com = self.forward(x, edge_index, edge_attr, batch_indices)

        # Compute loss
        loss = self.compute_loss(pred_com, target_com)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Additional metrics
        mae = F.l1_loss(pred_com, target_com)
        self.log("train_mae", mae, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, edge_index, edge_attr, batch_indices, target_com = batch

        # Forward pass
        pred_com = self.forward(x, edge_index, edge_attr, batch_indices)

        # Compute loss
        loss = self.compute_loss(pred_com, target_com)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Additional metrics
        mae = F.l1_loss(pred_com, target_com)
        self.log("val_mae", mae, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step"""
        x, edge_index, edge_attr, batch_indices, target_com = batch

        # Forward pass
        pred_com = self.forward(x, edge_index, edge_attr, batch_indices)

        # Compute loss
        loss = self.compute_loss(pred_com, target_com)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # Additional metrics
        mae = F.l1_loss(pred_com, target_com)
        self.log("test_mae", mae, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Learning rate scheduler (optional)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        # Get the current learning rate
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        """Prediction step for inference"""
        x, edge_index, edge_attr, batch_indices, _ = batch
        pred_com = self.forward(x, edge_index, edge_attr, batch_indices)
        return pred_com


# Legacy compatibility - alias for the old naming convention
class EquivariantGNN(EquivariantGNNLightning):
    """Alias for backward compatibility"""

    pass
