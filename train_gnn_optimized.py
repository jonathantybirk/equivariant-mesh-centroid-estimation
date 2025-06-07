#!/usr/bin/env python3
"""
Optimized Training Script for Equivariant GNN Center of Mass Estimation

Based on performance testing results:
- Uses PyG's native GCN layers (fastest performance)
- Optimized data loading with memory efficiency
- Simplified architecture for better speed
- Fallback for torch.compile when triton not available
- Weights & Biases logging for experiment tracking
"""

import warnings

warnings.filterwarnings(
    "ignore", message="An issue occurred while importing 'torch-scatter'"
)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.typing")

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import os
import sys
from pathlib import Path
import glob
import argparse
import time
import json
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Wandb logging
try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not available. Install with: pip install wandb")


class OptimizedPointCloudDataset:
    """Memory-optimized dataset with contiguous tensor storage"""

    def __init__(self, data_dir, debug=False):
        if debug:
            print(f"Loading optimized dataset from {data_dir}...")

        # Load metadata if available
        metadata_file = os.path.join(data_dir, "dataset_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
            file_list = [
                os.path.join(data_dir, file_info["filename"])
                for file_info in self.metadata["files"]
            ]
        else:
            file_list = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
            self.metadata = None

        if debug:
            print(f"Found {len(file_list)} samples")

        # Load all data with memory optimization
        self.data_cache = []
        iterator = (
            tqdm(file_list, desc="Loading optimized data") if debug else file_list
        )

        for file_path in iterator:
            try:
                data = torch.load(file_path, weights_only=False)

                # Create contiguous tensors for better memory efficiency
                x = data["node_features"].float().contiguous()
                edge_index = data["edge_index"].long().contiguous()
                edge_attr = (
                    data["edge_attr"].float().contiguous()
                    if "edge_attr" in data
                    else None
                )
                target = data["target"].squeeze().float().contiguous()

                # Create optimized PyG data object
                data_obj = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=target.unsqueeze(0),
                )

                self.data_cache.append(data_obj)

            except Exception as e:
                if debug:
                    print(f"Warning: Failed to load {file_path}: {e}")
                continue

        if debug:
            print(
                f"Successfully loaded {len(self.data_cache)} samples with memory optimization"
            )

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx]


class OptimizedGNN(pl.LightningModule):
    """Optimized GNN using PyG's native GCN layers for best performance"""

    def __init__(
        self,
        input_dim=3,
        hidden_dim=128,
        output_dim=3,
        num_layers=4,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        use_compile=True,
        debug=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_compile = use_compile

        # Build GCN layers (proven fastest in our tests)
        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(GCNConv(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer
        self.layers.append(GCNConv(hidden_dim, hidden_dim))

        # Final prediction head
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 4, output_dim),
        )

        if debug:
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Created optimized GCN model with {total_params:,} parameters")

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Apply GCN layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)

            if i < len(self.layers) - 1:  # Don't apply activation after last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling to get graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)

        # Final prediction
        return self.predictor(x)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.l1_loss(pred, batch.y)  # Use MAE as primary loss
        mse = F.mse_loss(pred, batch.y)  # Keep MSE as secondary metric

        # Calculate per-component metrics
        mae_per_component = F.l1_loss(pred, batch.y, reduction="none").mean(dim=0)
        mse_per_component = F.mse_loss(pred, batch.y, reduction="none").mean(dim=0)

        batch_size = batch.y.size(0)

        # Log metrics (loss is now MAE)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train_mae", loss, on_step=True, on_epoch=True, batch_size=batch_size
        )  # Same as loss now
        self.log(
            "train_mse", mse, on_step=True, on_epoch=True, batch_size=batch_size
        )  # Secondary metric

        # Log per-component metrics
        for i, component in enumerate(["x", "y", "z"]):
            self.log(
                f"train_mae_{component}",
                mae_per_component[i],
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"train_mse_{component}",
                mse_per_component[i],
                on_epoch=True,
                batch_size=batch_size,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.l1_loss(pred, batch.y)  # Use MAE as primary loss
        mse = F.mse_loss(pred, batch.y)  # Keep MSE as secondary metric

        # Calculate per-component metrics
        mae_per_component = F.l1_loss(pred, batch.y, reduction="none").mean(dim=0)
        mse_per_component = F.mse_loss(pred, batch.y, reduction="none").mean(dim=0)

        batch_size = batch.y.size(0)

        # Log metrics (loss is now MAE)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(
            "val_mae", loss, on_epoch=True, batch_size=batch_size
        )  # Same as loss now
        self.log(
            "val_mse", mse, on_epoch=True, batch_size=batch_size
        )  # Secondary metric

        # Log per-component metrics
        for i, component in enumerate(["x", "y", "z"]):
            self.log(
                f"val_mae_{component}",
                mae_per_component[i],
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"val_mse_{component}",
                mse_per_component[i],
                on_epoch=True,
                batch_size=batch_size,
            )

        return loss

    def on_validation_epoch_end(self):
        """Log additional metrics to wandb at the end of each validation epoch"""
        # This runs after all validation steps are complete
        if hasattr(self.logger, "experiment") and hasattr(
            self.logger.experiment, "log"
        ):
            # Log learning rate
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.logger.experiment.log({"learning_rate": current_lr})

            # Log epoch number
            self.logger.experiment.log({"epoch": self.current_epoch})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Learning rate scheduler
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


def create_optimized_dataloader(dataset, batch_size, num_workers=0, debug=False):
    """Create optimized dataloader based on our testing results"""

    # Optimized settings based on performance tests
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 4 if num_workers > 0 else None,
        "drop_last": True,  # Consistent batch sizes for better GPU utilization
    }

    if debug:
        print(f"Optimized DataLoader config: {loader_kwargs}")

    return loader_kwargs


def main():
    parser = argparse.ArgumentParser(description="Optimized Equivariant GNN Training")

    # Data parameters
    parser.add_argument(
        "--data_dir", default="data/processed_sh", help="Training data directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (optimized for performance)",
    )
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Model parameters (optimized based on testing)
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of GCN layers"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--precision", type=str, default="16-mixed", help="Training precision"
    )

    # Optimization parameters
    parser.add_argument(
        "--use_compile", action="store_true", help="Use torch.compile (if available)"
    )
    parser.add_argument(
        "--fast_dev_run", type=int, default=None, help="Quick debug run"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Wandb logging parameters
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="equivariant-center-of-mass",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Wandb experiment name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--wandb_tags", type=str, nargs="*", default=[], help="Wandb experiment tags"
    )

    args = parser.parse_args()

    print("🚀 Optimized Equivariant GNN Training")
    print("=" * 50)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")

    # Load optimized dataset
    start_time = time.time()
    full_dataset = OptimizedPointCloudDataset(args.data_dir, args.debug)

    if len(full_dataset) == 0:
        print("❌ No data loaded - cannot train")
        return

    # Split dataset
    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    if args.debug:
        print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    # Create optimized dataloaders
    loader_kwargs = create_optimized_dataloader(
        train_dataset, args.batch_size, args.num_workers, args.debug
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **{k: v for k, v in loader_kwargs.items() if k != "drop_last"},
        drop_last=False,
    )

    setup_time = time.time() - start_time
    if args.debug:
        print(f"Setup completed in {setup_time:.2f}s")

    # Create optimized model
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.shape[1]

    model = OptimizedGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_compile=args.use_compile,
        debug=args.debug,
    )

    # Try to compile model for better performance
    if args.use_compile:
        try:
            if args.debug:
                print("Attempting model compilation...")

            # Suppress errors and fallback to eager mode if compilation fails
            try:
                import torch._dynamo as dynamo_module

                dynamo_module.config.suppress_errors = True
            except ImportError:
                pass

            model = torch.compile(model, dynamic=True)
            if args.debug:
                print("✅ Model compilation successful")
        except Exception as e:
            if args.debug:
                print(f"⚠️ Model compilation failed: {e}")
                print("Continuing without compilation...")

    # Setup wandb logger if requested
    logger = None
    if args.use_wandb and WANDB_AVAILABLE:
        # Generate experiment name if not provided
        if args.wandb_name is None:
            args.wandb_name = f"optimized-gnn-{int(time.time())}"

        # Create wandb logger
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=args.wandb_tags,
            log_model=True,  # Log model checkpoints
        )

        # Log hyperparameters and configuration
        logger.experiment.config.update(
            {
                "model_type": "OptimizedGNN",
                "architecture": "GCN",
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "max_epochs": args.max_epochs,
                "precision": args.precision,
                "loss_function": "MAE",
                "dataset_size": len(full_dataset),
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "num_workers": args.num_workers,
                "use_compile": args.use_compile,
            }
        )

        if args.debug:
            print(f"📊 Wandb logging enabled: {args.wandb_project}/{args.wandb_name}")

    elif args.use_wandb and not WANDB_AVAILABLE:
        print("❌ Wandb requested but not available. Install with: pip install wandb")

    # Configure trainer
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "accelerator": "auto",
        "precision": args.precision,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "norm",
        "logger": logger,  # Add wandb logger
    }

    if args.fast_dev_run:
        trainer_kwargs["fast_dev_run"] = args.fast_dev_run

    trainer = pl.Trainer(**trainer_kwargs)

    if args.debug:
        print("Starting optimized training...")

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    print("🎉 Optimized training complete!")

    # Finish wandb run if active
    if args.use_wandb and WANDB_AVAILABLE and logger:
        print(f"📊 View results at: {logger.experiment.url}")
        wandb.finish()


if __name__ == "__main__":
    main()
