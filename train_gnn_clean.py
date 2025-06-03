#!/usr/bin/env python3
"""
Clean Training Script for Equivariant GNN Center of Mass Estimation

This script provides:
- Streamlined PyTorch Lightning training
- Efficient data loading with minimal overhead
- Clean progress reporting
- Production-ready configuration
"""

import warnings

# Suppress torch-scatter warnings (the module works despite these warnings)
warnings.filterwarnings(
    "ignore", message="An issue occurred while importing 'torch-scatter'"
)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.typing")

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import sys
from pathlib import Path
import glob
import argparse
import time
import json

# Weights & Biases for experiment tracking
try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.model.eq_gnn import GNN


class PointCloudDataset:
    """Optimized point cloud dataset for fast loading"""

    def __init__(self, data_dir, debug=False):
        if debug:
            print(f"Loading dataset from {data_dir}...")
        else:
            print("Loading dataset...")

        # Load metadata if available
        metadata_file = os.path.join(data_dir, "dataset_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
            if debug:
                print(
                    f"Loaded metadata: {self.metadata['total_files']} files, "
                    f"edge_attr_dim={self.metadata['edge_attr_dim']}, "
                    f"SH={self.metadata['use_spherical_harmonics']}"
                )
            # Use files from metadata
            self.file_list = [
                os.path.join(data_dir, file_info["filename"])
                for file_info in self.metadata["files"]
            ]
        else:
            # Fallback to scanning directory
            if debug:
                print("No metadata found, scanning directory...")
            self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
            self.metadata = None

        if debug:
            print(f"Found {len(self.file_list)} samples")
        else:
            print(f"Found {len(self.file_list)} samples")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load preprocessed data (already optimized)
        data = torch.load(self.file_list[idx], weights_only=False)

        # Extract components and ensure proper tensor formats
        x = data["node_features"].float().clone()  # [N, 3] positions as features
        edge_index = data["edge_index"].long().clone()
        edge_attr = data["edge_attr"].float().clone()
        target = data["target"].squeeze().float().clone()  # [3]
        pos = data["point_cloud"].float().clone()

        # Create PyG data object
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=target.unsqueeze(0),  # [1, 3] for batching
            pos=pos,
        )


class EquivariantGNN(pl.LightningModule):
    """Clean Lightning module for center of mass estimation"""

    def __init__(
        self,
        input_dim=3,
        hidden_dim=32,
        max_sh_degree=1,
        message_passing_steps=2,
        final_mlp_dims=[64, 32],
        lr=1e-3,
        weight_decay=1e-5,
        debug=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gnn = GNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_sh_degree=max_sh_degree,
            message_passing_steps=message_passing_steps,
            final_mlp_dims=final_mlp_dims,
            debug=debug,
        )

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, edge_index, edge_attr, batch):
        return self.gnn(x, edge_index, edge_attr, batch)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.mse_loss(pred, batch.y)
        mae = F.l1_loss(pred, batch.y)

        # Calculate additional metrics for better tracking
        mse_per_component = F.mse_loss(pred, batch.y, reduction="none").mean(dim=0)
        mae_per_component = F.l1_loss(pred, batch.y, reduction="none").mean(dim=0)

        batch_size = batch.y.size(0)

        # Log main metrics
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log("train_mae", mae, on_step=True, on_epoch=True, batch_size=batch_size)

        # Log per-component metrics for wandb
        self.log(
            "train_mse_x",
            mse_per_component[0],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_mse_y",
            mse_per_component[1],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_mse_z",
            mse_per_component[2],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_mae_x",
            mae_per_component[0],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_mae_y",
            mae_per_component[1],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_mae_z",
            mae_per_component[2],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.mse_loss(pred, batch.y)
        mae = F.l1_loss(pred, batch.y)

        # Calculate additional metrics
        mse_per_component = F.mse_loss(pred, batch.y, reduction="none").mean(dim=0)
        mae_per_component = F.l1_loss(pred, batch.y, reduction="none").mean(dim=0)

        batch_size = batch.y.size(0)

        # Log main metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val_mae", mae, on_epoch=True, batch_size=batch_size)

        # Log per-component metrics for wandb
        self.log(
            "val_mse_x", mse_per_component[0], on_epoch=True, batch_size=batch_size
        )
        self.log(
            "val_mse_y", mse_per_component[1], on_epoch=True, batch_size=batch_size
        )
        self.log(
            "val_mse_z", mse_per_component[2], on_epoch=True, batch_size=batch_size
        )
        self.log(
            "val_mae_x", mae_per_component[0], on_epoch=True, batch_size=batch_size
        )
        self.log(
            "val_mae_y", mae_per_component[1], on_epoch=True, batch_size=batch_size
        )
        self.log(
            "val_mae_z", mae_per_component[2], on_epoch=True, batch_size=batch_size
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True, min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Train Equivariant GNN")

    # Data parameters
    parser.add_argument(
        "--data_dir",
        default="data/processed_sh",
        help="Training data directory (use processed_sh for SH features)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (optimized for highest throughput)",
    )
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader workers (8 recommended for high-end systems)",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        default=True,
        help="Use persistent workers to reduce dataloader overhead",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Pin memory for faster GPU transfer",
    )

    # Model parameters (increased for better GPU utilization)
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension (increased for better GPU utilization)",
    )
    parser.add_argument(
        "--max_sh_degree",
        type=int,
        default=1,
        help="Max spherical harmonic degree (must match preprocessed data)",
    )
    parser.add_argument(
        "--message_steps",
        type=int,
        default=4,
        help="Message passing steps (increased for deeper model)",
    )
    parser.add_argument(
        "--mlp_dims",
        nargs="+",
        type=int,
        default=[256, 128, 64],
        help="Final MLP dimensions (increased for better capacity)",
    )

    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16", "16-mixed", "bf16-mixed"],
        help="Training precision (16-mixed for better GPU utilization)",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Use torch.compile for faster training (PyTorch 2.0+)",
    )

    # Debug parameters
    parser.add_argument(
        "--fast_dev_run", type=int, default=None, help="Quick debug run"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Weights & Biases parameters
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=True,
        help="Use Weights & Biases for logging (default: True)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="equivariant-center-of-mass",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--wandb_tags", nargs="+", default=None, help="W&B tags for the run"
    )
    parser.add_argument(
        "--wandb_notes", type=str, default=None, help="Notes for the W&B run"
    )

    args = parser.parse_args()

    # Setup Weights & Biases logging
    logger = None
    if args.use_wandb and WANDB_AVAILABLE:
        # Auto-generate run name if not provided
        if args.wandb_run_name is None:
            args.wandb_run_name = (
                f"gnn_h{args.hidden_dim}_mp{args.message_steps}_b{args.batch_size}"
            )

        # Create wandb logger
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            log_model=True,  # Log model checkpoints to W&B
        )

        # Log hyperparameters
        logger.experiment.config.update(
            {
                "model": {
                    "hidden_dim": args.hidden_dim,
                    "max_sh_degree": args.max_sh_degree,
                    "message_passing_steps": args.message_steps,
                    "final_mlp_dims": args.mlp_dims,
                    "input_dim": 3,
                },
                "training": {
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "max_epochs": args.max_epochs,
                    "precision": args.precision,
                },
                "data": {
                    "data_dir": args.data_dir,
                    "val_split": args.val_split,
                    "num_workers": args.num_workers,
                    "persistent_workers": args.persistent_workers,
                    "pin_memory": args.pin_memory,
                },
                "system": {
                    "compile_model": args.compile_model,
                    "gpu_available": torch.cuda.is_available(),
                    "gpu_count": (
                        torch.cuda.device_count() if torch.cuda.is_available() else 0
                    ),
                },
            }
        )

        if args.debug:
            print(f"W&B logging enabled: {args.wandb_project}/{args.wandb_run_name}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print(
            "Warning: W&B requested but not available. Install with: pip install wandb"
        )

    if not args.debug:
        print("Starting training...")
    else:
        print("Equivariant GNN Training")
        print("=" * 40)

    # Load dataset
    start_time = time.time()
    full_dataset = PointCloudDataset(args.data_dir, args.debug)

    # Split dataset
    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    if args.debug:
        print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val")

    # Log dataset info to wandb
    if logger is not None:
        logger.experiment.config.update(
            {
                "dataset": {
                    "total_samples": len(full_dataset),
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "edge_attr_dim": (
                        full_dataset.metadata.get("edge_attr_dim", "unknown")
                        if hasattr(full_dataset, "metadata") and full_dataset.metadata
                        else "unknown"
                    ),
                    "use_spherical_harmonics": (
                        full_dataset.metadata.get("use_spherical_harmonics", "unknown")
                        if hasattr(full_dataset, "metadata") and full_dataset.metadata
                        else "unknown"
                    ),
                }
            }
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True,  # Ensures consistent batch sizes for better GPU utilization
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    setup_time = time.time() - start_time
    if args.debug:
        print(f"Setup completed in {setup_time:.2f}s")

    # Test performance if debug mode
    if args.debug:
        print("\nTesting model performance...")
        test_batch = next(iter(train_loader))
        print(
            f"Sample batch: {test_batch.x.shape[0]} nodes, {test_batch.edge_index.size(1)} edges"
        )

        # Test model creation and forward pass
        test_model = EquivariantGNN(
            hidden_dim=args.hidden_dim,
            max_sh_degree=args.max_sh_degree,
            message_passing_steps=args.message_steps,
            final_mlp_dims=args.mlp_dims,
            debug=args.debug,
        )

        test_start = time.time()
        with torch.no_grad():
            pred = test_model(
                test_batch.x,
                test_batch.edge_index,
                test_batch.edge_attr,
                test_batch.batch,
            )
        test_time = time.time() - test_start

        print(
            f"Forward pass: {test_time:.3f}s ({test_time/test_batch.edge_index.size(1)*1000:.2f}ms per edge)"
        )
        print(f"Model parameters: {sum(p.numel() for p in test_model.parameters())}")

    # Create model
    model = EquivariantGNN(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        max_sh_degree=args.max_sh_degree,
        message_passing_steps=args.message_steps,
        final_mlp_dims=args.mlp_dims,
        lr=args.lr,
        weight_decay=args.weight_decay,
        debug=args.debug,
    )

    # Log model information to wandb
    if logger is not None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.experiment.config.update(
            {
                "model_info": {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "model_size_mb": total_params
                    * 4
                    / (1024 * 1024),  # Assuming float32
                }
            }
        )

        if args.debug:
            print(
                f"Logged model info: {total_params} parameters ({total_params * 4 / (1024 * 1024):.2f} MB)"
            )

    # Optionally compile model for faster training
    if args.compile_model:
        if args.debug:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Configure trainer with GPU optimizations
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "accelerator": "auto",
        "precision": args.precision,
        "enable_checkpointing": True,
        "logger": logger if logger is not None else True,
        "enable_progress_bar": True,
        # Optimization settings
        "gradient_clip_val": 1.0,  # Prevent gradient explosion
        "gradient_clip_algorithm": "norm",
        "sync_batchnorm": True if torch.cuda.device_count() > 1 else False,
    }

    if args.fast_dev_run:
        trainer_kwargs["fast_dev_run"] = args.fast_dev_run
        trainer_kwargs["logger"] = False  # Disable logging for fast dev runs

    trainer = pl.Trainer(**trainer_kwargs)

    if args.debug:
        print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    if not args.debug:
        print("Training complete!")
    else:
        print("Training complete!")

    # Finish wandb run
    if logger is not None:
        # Log final model performance
        if hasattr(trainer, "callback_metrics"):
            final_metrics = {
                k: v.item() if hasattr(v, "item") else v
                for k, v in trainer.callback_metrics.items()
            }
            logger.experiment.summary.update(final_metrics)

        # Close wandb run
        wandb.finish()
        if args.debug:
            print("W&B run finished and logged.")


if __name__ == "__main__":
    main()
