#!/usr/bin/env python3
"""
Simplified Multi-Model GNN Training with Lightning CLI

Usage:
    python train_gnn_optimized.py fit --model EquivariantGNN --data PointCloudData
    python train_gnn_optimized.py fit --model BasicGNN --data PointCloudData
    python train_gnn_optimized.py fit --model ZeroBaseline --data PointCloudData

With Weights & Biases logging:
    python train_gnn_optimized.py fit --model EquivariantGNN --data PointCloudData --trainer.logger.class_path=lightning.pytorch.loggers.WandbLogger --trainer.logger.init_args.project="gnn-optimization" --trainer.logger.init_args.name="equivariant-gnn-run"
"""

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from torch_geometric.data import Data, DataLoader
import os
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import core models
from src.model.eq_gnn import EquivariantGNN
from src.model.gnn import GNN


class CustomLightningModule(pl.LightningModule):
    """
    Base class with all common training/validation/logging logic

    Key design decisions based on research:
    - MSE as loss function for better optimization (smooth gradients)
    - MAE as primary interpretable metric (same units as target)
    - Per-epoch logging only (since epochs are fast now)
    - Per-component MAE for detailed analysis (x, y, z coordinates)
    """

    def training_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Use MSE as loss function for better optimization (smooth gradients)
        loss = F.mse_loss(pred, batch.y)

        # Track MAE as the primary interpretable metric
        mae = F.l1_loss(pred, batch.y)

        # Per-component MAE for detailed analysis
        mae_per_component = F.l1_loss(pred, batch.y, reduction="none").mean(dim=0)

        # Log metrics per epoch only (since epochs are fast now)
        batch_size = batch.y.size(0)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train_mae",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # Log per-component MAE
        for i, component in enumerate(["x", "y", "z"]):
            self.log(
                f"train_mae_{component}",
                mae_per_component[i],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Use MSE as validation loss for consistency with training
        loss = F.mse_loss(pred, batch.y)

        # Track MAE as the primary interpretable metric
        mae = F.l1_loss(pred, batch.y)

        # Per-component MAE for detailed analysis
        mae_per_component = F.l1_loss(pred, batch.y, reduction="none").mean(dim=0)

        # Log metrics per epoch only
        batch_size = batch.y.size(0)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val_mae",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Log per-component MAE
        for i, component in enumerate(["x", "y", "z"]):
            self.log(
                f"val_mae_{component}",
                mae_per_component[i],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        return mae  # Return MAE for monitoring (more interpretable)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mae",  # Monitor MAE instead of loss for interpretability
                "frequency": 1,
            },
        }


class EquivariantGNN(CustomLightningModule):
    def __init__(self, hidden_dim=128, num_layers=4, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = EquivariantGNN(
            input_dim=3,
            hidden_dim=hidden_dim,
            message_passing_steps=num_layers,
            final_mlp_dims=[64, 32],
            max_sh_degree=1,
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.model(x, edge_index, edge_attr, batch, node_pos=x)


class BasicGNN(CustomLightningModule):
    def __init__(self, hidden_dim=128, num_layers=4, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = GNN(
            hidden_dim=hidden_dim,
            message_passing_steps=num_layers,
            message_mlp_dims=[70, 140, 20],
            update_mlp_dims=[70],
            final_mlp_dims=[64, 32],
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.model(x, edge_index, edge_attr, batch)


class ZeroBaseline(CustomLightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        batch_size = batch.max().item() + 1 if batch is not None else 1
        return torch.zeros(batch_size, 3, device=x.device)


class PointCloudData(pl.LightningDataModule):
    def __init__(
        self, data_dir="data/processed_sh", batch_size=16, val_split=0.1, num_workers=0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_cache = []

    def setup(self, stage=None):
        print(f"Loading dataset from {self.hparams.data_dir}...")

        # Load all .pt files
        file_list = list(Path(self.hparams.data_dir).glob("**/*.pt"))

        for file_path in tqdm(file_list, desc="Loading data"):
            try:
                data = torch.load(file_path, weights_only=False)
                x = data["node_features"].float().contiguous()
                edge_index = data["edge_index"].long().contiguous()
                edge_attr = data.get("edge_attr")
                if edge_attr is not None:
                    edge_attr = edge_attr.float().contiguous()
                target = data["target"].squeeze().float().contiguous()

                self.data_cache.append(
                    Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=target.unsqueeze(0),
                    )
                )
            except:
                continue

        print(f"Loaded {len(self.data_cache)} samples")

        # Split data
        train_size = int((1 - self.hparams.val_split) * len(self.data_cache))
        self.train_data, self.val_data = torch.utils.data.random_split(
            self.data_cache,
            [train_size, len(self.data_cache) - train_size],
            generator=torch.Generator().manual_seed(43),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


if __name__ == "__main__":
    """
    Lightning CLI with multiple model support and W&B integration
    
    Examples:
    
    1. Basic training with TensorBoard (default):
        python train_gnn_optimized.py fit --model.class_path=EquivariantGNN --data.class_path=PointCloudData
    
    2. Training with Weights & Biases (command line):
        python train_gnn_optimized.py fit \
            --model.class_path=EquivariantGNN \
            --data.class_path=PointCloudData \
            --trainer.logger.class_path=lightning.pytorch.loggers.WandbLogger \
            --trainer.logger.init_args.project="gnn-optimization" \
            --trainer.logger.init_args.name="equivariant-gnn-experiment"
    
    3. Using a config file (recommended for W&B):
        python train_gnn_optimized.py fit --config config_wandb.yaml
    
    4. Quick testing (fast_dev_run):
        python train_gnn_optimized.py fit \
            --model.class_path=EquivariantGNN \
            --data.class_path=PointCloudData \
            --trainer.fast_dev_run=true
    
    5. Model variants:
        - EquivariantGNN (recommended - ultra-fast equivariant model)
        - BasicGNN (standard GNN without equivariance)
        - ZeroBaseline (always predicts zero for comparison)
    """
    # Lightning CLI with auto-discovery of both models and datamodules
    # Allow overwriting config files from previous runs
    cli = LightningCLI(save_config_kwargs={"overwrite": True})
