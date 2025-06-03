#!/usr/bin/env python3
"""
Production Training Script for Equivariant GNN Center of Mass Estimation

This script uses:
- Simplified edge connectivity (efficient)
- PyTorch Lightning with debugging features
- Real dataset from data/processed/
- Automatic GPU usage
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import (
    DataLoader,
)  # Updated import to fix deprecation warning
import os
import sys
from pathlib import Path
import glob
import argparse
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.model.eq_gnn import GNN


class PointCloudDataset:
    """Point cloud dataset using preprocessed graph structure with optimized PyTorch tensors"""

    def __init__(self, data_dir):
        print(f"🔍 DEBUG: Initializing dataset from {data_dir}")
        start_time = time.time()
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        init_time = time.time() - start_time
        print(
            f"📁 Found {len(self.file_list)} files in {data_dir} (took {init_time:.3f}s)"
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # DEBUG: Time data loading
        load_start = time.time()

        # Load the data - already in optimized PyTorch tensor format
        data = torch.load(self.file_list[idx], weights_only=False)
        load_time = time.time() - load_start

        # DEBUG: Time data processing
        process_start = time.time()

        # Extract point cloud positions and target
        pos = data["point_cloud"].float()  # [N, 3]
        target = data["target"].squeeze().float()  # [3]

        # Use 3D positions directly as node features (PyG best practice for spatial data)
        x = data["node_features"].float()  # Already 3D positions [N, 3]

        # Use the preprocessed graph structure with displacement vectors on edges
        edge_index = data["edge_index"]  # Use original edge connectivity
        edge_attr = data[
            "edge_attr"
        ]  # Displacement vectors (optimized for message passing)

        # Create PyG data object
        graph = Data(
            x=x,  # 3D positions as node features
            edge_index=edge_index,
            edge_attr=edge_attr,  # Displacement vectors
            y=target.unsqueeze(0),  # Fix shape: [1, 3] to match model output
            pos=pos,
        )

        process_time = time.time() - process_start
        total_time = load_time + process_time

        # DEBUG: Print timing for slow samples
        if total_time > 0.1:  # If a single sample takes more than 100ms
            print(
                f"🐌 SLOW SAMPLE {idx}: load={load_time:.3f}s, process={process_time:.3f}s, "
                f"total={total_time:.3f}s, nodes={x.size(0)}, edges={edge_index.size(1)}"
            )

        return graph


class EquivariantGNN(pl.LightningModule):
    """Lightning module for equivariant center of mass estimation with debugging"""

    def __init__(
        self,
        input_dim=3,
        hidden_dim=4,
        max_sh_degree=1,
        message_passing_steps=2,
        final_mlp_dims=[16, 8],
        lr=1e-3,
        weight_decay=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        print(
            f"🤖 DEBUG: Creating model with {hidden_dim} hidden dim, {message_passing_steps} steps"
        )
        model_start = time.time()

        self.gnn = GNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_sh_degree=max_sh_degree,
            message_passing_steps=message_passing_steps,
            final_mlp_dims=final_mlp_dims,
        )

        model_time = time.time() - model_start
        print(f"🤖 Model created in {model_time:.3f}s")

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, edge_index, edge_attr, batch):
        return self.gnn(x, edge_index, edge_attr, batch)

    def training_step(self, batch, batch_idx):
        # DEBUG: Time each component of training step
        step_start = time.time()

        # Forward pass timing
        forward_start = time.time()
        pred = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        forward_time = time.time() - forward_start

        # Loss computation timing
        loss_start = time.time()
        target = batch.y
        loss = F.mse_loss(pred, target)
        loss_time = time.time() - loss_start

        # Metrics timing
        metrics_start = time.time()
        mae = F.l1_loss(pred, target)
        metrics_time = time.time() - metrics_start

        total_step_time = time.time() - step_start

        # DEBUG: Print detailed timing for slow steps
        if total_step_time > 1.0 or batch_idx < 5:  # First 5 batches or slow ones
            print(
                f"🕐 STEP {batch_idx}: forward={forward_time:.3f}s, loss={loss_time:.3f}s, "
                f"metrics={metrics_time:.3f}s, total={total_step_time:.3f}s"
            )
            print(
                f"     Batch size: {batch.y.size(0)}, Total nodes: {batch.x.size(0)}, "
                f"Total edges: {batch.edge_index.size(1)}"
            )

        # Fix batch_size warning by providing explicit batch_size
        batch_size = batch.y.size(0)

        # Log metrics with explicit batch_size
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log("train_mae", mae, on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        target = batch.y
        loss = F.mse_loss(pred, target)

        # Fix batch_size warning by providing explicit batch_size
        batch_size = batch.y.size(0)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        mae = F.l1_loss(pred, target)
        self.log("val_mae", mae, on_epoch=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


def main():
    parser = argparse.ArgumentParser(description="Train Equivariant GNN")

    # Data parameters
    parser.add_argument(
        "--data_dir", default="data/processed/train", help="Training data directory"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension")
    parser.add_argument(
        "--max_sh_degree", type=int, default=1, help="Max spherical harmonic degree"
    )
    parser.add_argument(
        "--message_steps", type=int, default=2, help="Message passing steps"
    )
    parser.add_argument(
        "--mlp_dims", nargs="+", type=int, default=[64, 32], help="Final MLP dimensions"
    )

    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")

    # Debugging parameters (PyTorch Lightning built-in)
    parser.add_argument(
        "--fast_dev_run", type=int, default=None, help="Quick debug run with N batches"
    )
    parser.add_argument(
        "--limit_train_batches", type=float, default=None, help="Limit training batches"
    )
    parser.add_argument(
        "--limit_val_batches", type=float, default=None, help="Limit validation batches"
    )
    parser.add_argument(
        "--overfit_batches", type=float, default=None, help="Overfit on subset"
    )

    args = parser.parse_args()

    print("🚀 Equivariant GNN Center of Mass Training")
    print("=" * 60)

    # Create dataset
    print("📊 Loading dataset...")
    dataset_start = time.time()
    full_dataset = PointCloudDataset(args.data_dir)
    dataset_time = time.time() - dataset_start
    print(f"📊 Dataset loaded in {dataset_time:.3f}s")

    # Split into train/val
    print("🔀 Splitting dataset...")
    split_start = time.time()
    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    split_time = time.time() - split_start
    print(f"🔀 Split completed in {split_time:.3f}s")

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")

    # Create dataloaders with num_workers to fix performance warning
    print("🔄 Creating dataloaders...")
    loader_start = time.time()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    loader_time = time.time() - loader_start
    print(f"🔄 Dataloaders created in {loader_time:.3f}s")

    # Test one batch
    print("🧪 Testing data loading...")
    batch_start = time.time()
    batch = next(iter(train_loader))
    batch_time = time.time() - batch_start
    print(f"🧪 First batch loaded in {batch_time:.3f}s")
    print(
        f"   Batch: x={batch.x.shape}, edges={batch.edge_index.size(1)}, y={batch.y.shape}"
    )

    # Create model
    print("🤖 Creating model...")
    model = EquivariantGNN(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        max_sh_degree=args.max_sh_degree,
        message_passing_steps=args.message_steps,
        final_mlp_dims=args.mlp_dims,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"   Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    print("🧪 Testing forward pass...")
    forward_test_start = time.time()
    with torch.no_grad():
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        print(f"   Prediction shape: {pred.shape}")
    forward_test_time = time.time() - forward_test_start
    print(f"🧪 Forward pass test completed in {forward_test_time:.3f}s")

    # Create trainer
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "accelerator": "auto",
        "enable_checkpointing": True,
        "logger": True,
        "enable_progress_bar": True,
    }

    # Add debugging parameters if specified
    if args.fast_dev_run:
        trainer_kwargs["fast_dev_run"] = args.fast_dev_run
        trainer_kwargs["logger"] = False  # Disable logging for debug runs

    if args.limit_train_batches:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    if args.overfit_batches:
        trainer_kwargs["overfit_batches"] = args.overfit_batches

    trainer = pl.Trainer(**trainer_kwargs)

    print("🚀 Starting training...")
    print("⏱️  Watch for detailed timing information above...")
    trainer.fit(model, train_loader, val_loader)

    print("✅ Training complete!")


if __name__ == "__main__":
    main()
