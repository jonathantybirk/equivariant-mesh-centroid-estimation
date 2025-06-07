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

        # Ensure data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Load metadata if available
        metadata_file = os.path.join(data_dir, "dataset_metadata.json")
        if os.path.exists(metadata_file):
            if debug:
                print("Found dataset metadata file")
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
            file_list = [
                os.path.join(data_dir, file_info["filename"])
                for file_info in self.metadata["files"]
            ]
            if debug:
                print(f"Metadata indicates {len(file_list)} files")
        else:
            if debug:
                print("No metadata file found, scanning directory for .pt files")
            # Find ALL .pt files recursively to ensure we get everything
            file_list = []
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(".pt"):
                        file_list.append(os.path.join(root, file))
            file_list = sorted(file_list)  # Sort for reproducibility
            self.metadata = None

        if debug:
            print(f"Total .pt files discovered: {len(file_list)}")

        if len(file_list) == 0:
            raise ValueError(
                f"No .pt files found in {data_dir}. Please check your data directory."
            )

        # Load all data with memory optimization
        self.data_cache = []
        successful_loads = 0
        failed_loads = 0

        iterator = (
            tqdm(file_list, desc="Loading complete dataset") if debug else file_list
        )

        for i, file_path in enumerate(iterator):
            try:
                data = torch.load(file_path, weights_only=False)

                # Validate data structure
                required_keys = ["node_features", "edge_index", "target"]
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    if debug:
                        print(f"Skipping {file_path}: missing keys {missing_keys}")
                    failed_loads += 1
                    continue

                # Create contiguous tensors for better memory efficiency
                x = data["node_features"].float().contiguous()
                edge_index = data["edge_index"].long().contiguous()
                edge_attr = (
                    data["edge_attr"].float().contiguous()
                    if "edge_attr" in data
                    else None
                )
                target = data["target"].squeeze().float().contiguous()

                # Validate tensor shapes
                if x.dim() != 2 or edge_index.dim() != 2 or target.dim() != 1:
                    if debug:
                        print(f"Skipping {file_path}: invalid tensor dimensions")
                    failed_loads += 1
                    continue

                if target.shape[0] != 3:  # Should be 3D center of mass
                    if debug:
                        print(
                            f"Skipping {file_path}: target not 3D (shape: {target.shape})"
                        )
                    failed_loads += 1
                    continue

                # Create optimized PyG data object
                data_obj = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=target.unsqueeze(0),
                )

                self.data_cache.append(data_obj)
                successful_loads += 1

            except Exception as e:
                if debug:
                    print(f"Warning: Failed to load {file_path}: {e}")
                failed_loads += 1
                continue

        if debug:
            print(f"Dataset loading complete:")
            print(f"  ✅ Successfully loaded: {successful_loads} samples")
            print(f"  ❌ Failed to load: {failed_loads} samples")
            print(f"  📊 Final dataset size: {len(self.data_cache)} samples")

            if len(self.data_cache) > 0:
                # Show dataset statistics
                sample = self.data_cache[0]
                print(f"  📋 Sample structure:")
                print(f"    - Node features: {sample.x.shape}")
                print(f"    - Edge index: {sample.edge_index.shape}")
                print(
                    f"    - Edge attr: {sample.edge_attr.shape if sample.edge_attr is not None else 'None'}"
                )
                print(f"    - Target: {sample.y.shape}")

        if len(self.data_cache) == 0:
            raise ValueError(
                f"No valid data loaded from {data_dir}. Check your data files."
            )

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx]

    def get_statistics(self):
        """Get detailed dataset statistics"""
        if len(self.data_cache) == 0:
            return {}

        num_nodes = [data.x.shape[0] for data in self.data_cache]
        num_edges = [data.edge_index.shape[1] for data in self.data_cache]

        stats = {
            "total_samples": len(self.data_cache),
            "avg_nodes_per_graph": np.mean(num_nodes),
            "avg_edges_per_graph": np.mean(num_edges),
            "min_nodes": np.min(num_nodes),
            "max_nodes": np.max(num_nodes),
            "min_edges": np.min(num_edges),
            "max_edges": np.max(num_edges),
            "node_feature_dim": self.data_cache[0].x.shape[1],
            "edge_feature_dim": (
                self.data_cache[0].edge_attr.shape[1]
                if self.data_cache[0].edge_attr is not None
                else 0
            ),
            "target_dim": self.data_cache[0].y.shape[1],
        }

        return stats


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
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Training precision (32, 16, bf16, 16-mixed, bf16-mixed)",
    )

    # Checkpoint parameters
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory"
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
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable wandb logging", default=False
    )
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
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"💾 Checkpoint directory: {args.checkpoint_dir}")
    print()

    # Create checkpoint directory
    startup_timer = time.time()
    print("⏱️  STARTUP PHASE 1: Setting up directories...")
    checkpoint_timer = time.time()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(
        f"   ✅ Created checkpoint directory: {args.checkpoint_dir} ({time.time() - checkpoint_timer:.2f}s)"
    )

    # Load optimized dataset
    print("\n⏱️  STARTUP PHASE 2: Loading dataset...")
    dataset_timer = time.time()
    full_dataset = OptimizedPointCloudDataset(args.data_dir, args.debug)

    if len(full_dataset) == 0:
        print("❌ No data loaded - cannot train")
        return

    dataset_load_time = time.time() - dataset_timer
    print(f"   ✅ Dataset loaded in {dataset_load_time:.2f}s")

    # Get comprehensive dataset statistics
    stats_timer = time.time()
    dataset_stats = full_dataset.get_statistics()
    stats_time = time.time() - stats_timer
    print(f"   ✅ Dataset statistics computed in {stats_time:.2f}s")

    print("\n📊 COMPLETE DATASET STATISTICS")
    print("=" * 50)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"🔢 Total samples: {dataset_stats['total_samples']:,}")
    print(
        f"📈 Graphs: {dataset_stats['min_nodes']}-{dataset_stats['max_nodes']} nodes (avg: {dataset_stats['avg_nodes_per_graph']:.1f})"
    )
    print(
        f"🔗 Edges: {dataset_stats['min_edges']}-{dataset_stats['max_edges']} edges (avg: {dataset_stats['avg_edges_per_graph']:.1f})"
    )
    print(
        f"📋 Features: {dataset_stats['node_feature_dim']}D nodes, {dataset_stats['edge_feature_dim']}D edges"
    )
    print(f"🎯 Target: {dataset_stats['target_dim']}D center of mass")

    # Split dataset
    print("\n⏱️  STARTUP PHASE 3: Splitting dataset...")
    split_timer = time.time()

    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(43),
    )

    split_time = time.time() - split_timer
    print(f"   ✅ Dataset split completed in {split_time:.2f}s")

    print(f"\n📂 DATASET SPLIT")
    print(
        f"🚂 Training samples: {len(train_dataset):,} ({len(train_dataset)/len(full_dataset)*100:.1f}%)"
    )
    print(
        f"✅ Validation samples: {len(val_dataset):,} ({len(val_dataset)/len(full_dataset)*100:.1f}%)"
    )
    print(
        f"💯 Total samples used: {len(train_dataset) + len(val_dataset):,} (100% of available data)"
    )

    # Verify we're using the entire dataset
    assert len(train_dataset) + len(val_dataset) == len(
        full_dataset
    ), "Dataset split error - not using full dataset!"

    # Create optimized dataloaders
    print("\n⏱️  STARTUP PHASE 4: Creating data loaders...")
    loader_timer = time.time()

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

    loader_time = time.time() - loader_timer
    print(f"   ✅ Data loaders created in {loader_time:.2f}s")
    print(f"   📊 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    print("\n⏱️  STARTUP PHASE 5: Initializing model...")
    model_timer = time.time()

    # Warm up CUDA early if available
    if torch.cuda.is_available():
        warmup_timer = time.time()
        _ = torch.tensor([1.0]).cuda()  # Simple CUDA warmup
        print(f"   🔥 CUDA warmed up in {time.time() - warmup_timer:.2f}s")

    # Get input dimension more efficiently
    input_dim_timer = time.time()
    # Instead of loading a full batch, just check the first sample
    first_sample = train_dataset[0]
    input_dim = first_sample.x.shape[1]
    input_dim_time = time.time() - input_dim_timer
    print(f"   📐 Input dimension ({input_dim}D) determined in {input_dim_time:.2f}s")

    # Model creation timing
    model_creation_timer = time.time()
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
    model_creation_time = time.time() - model_creation_timer
    print(f"   🏗️  Model created in {model_creation_time:.2f}s")

    # Move to device timing
    device_timer = time.time()
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"   🖥️  Moved to CUDA in {time.time() - device_timer:.2f}s")

    model_time = time.time() - model_timer
    print(f"   ✅ Model initialized in {model_time:.2f}s")

    # Try to compile model for better performance
    if args.use_compile:
        print("\n⏱️  STARTUP PHASE 6: Compiling model...")
        compile_timer = time.time()

        try:
            if args.debug:
                print("   🔄 Attempting model compilation...")

            # Suppress errors and fallback to eager mode if compilation fails
            try:
                import torch._dynamo as dynamo_module

                dynamo_module.config.suppress_errors = True
            except ImportError:
                pass

            model = torch.compile(model, dynamic=True)
            compile_time = time.time() - compile_timer
            print(f"   ✅ Model compilation successful in {compile_time:.2f}s")
        except Exception as e:
            compile_time = time.time() - compile_timer
            print(f"   ⚠️ Model compilation failed in {compile_time:.2f}s: {e}")
            print("   🔄 Continuing without compilation...")
    else:
        print("\n⏱️  STARTUP PHASE 6: Skipping model compilation (disabled)")

    # Setup logging
    print("\n⏱️  STARTUP PHASE 7: Setting up logging...")
    logging_timer = time.time()

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
                "val_split": args.val_split,
                "num_workers": args.num_workers,
                "use_compile": args.use_compile,
                # Comprehensive dataset information
                "data_directory": args.data_dir,
                "total_dataset_size": dataset_stats["total_samples"],
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
                "dataset_utilization": "100%",  # We use the entire dataset
                # Graph statistics
                "avg_nodes_per_graph": dataset_stats["avg_nodes_per_graph"],
                "avg_edges_per_graph": dataset_stats["avg_edges_per_graph"],
                "min_nodes": dataset_stats["min_nodes"],
                "max_nodes": dataset_stats["max_nodes"],
                "min_edges": dataset_stats["min_edges"],
                "max_edges": dataset_stats["max_edges"],
                "node_feature_dim": dataset_stats["node_feature_dim"],
                "edge_feature_dim": dataset_stats["edge_feature_dim"],
                "target_dim": dataset_stats["target_dim"],
            }
        )

        if args.debug:
            print(f"📊 Wandb logging enabled: {args.wandb_project}/{args.wandb_name}")

    elif args.use_wandb and not WANDB_AVAILABLE:
        print("❌ Wandb requested but not available. Install with: pip install wandb")

    logging_time = time.time() - logging_timer
    print(f"   ✅ Logging setup completed in {logging_time:.2f}s")

    # Setup checkpoint callback
    print("\n⏱️  STARTUP PHASE 8: Configuring training...")
    training_setup_timer = time.time()

    # Create checkpoint callback
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best-model-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=False,  # Set to True for faster saving if you don't need optimizer states
        verbose=True,
        every_n_epochs=1,  # Save every epoch (default)
        save_on_train_epoch_end=False,  # Only save after validation
        auto_insert_metric_name=False,  # Faster filename generation
    )

    print(f"   ✅ Checkpoint callback configured")
    print(f"   💾 Saving checkpoints to: {args.checkpoint_dir}")
    print(f"   🏆 Monitoring: val_loss (save top 3 + last)")

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
        "callbacks": [checkpoint_callback],  # Add checkpoint callback
        # Optimizations to reduce startup overhead
        "enable_model_summary": True,  # Keep model summary but make it efficient
        "num_sanity_val_steps": 0,  # Skip sanity validation to speed up startup
        "detect_anomaly": False,  # Disable anomaly detection for speed
        # Optimizations to reduce epoch delays
        "check_val_every_n_epoch": 1,  # Validate every epoch (default, but explicit)
        "log_every_n_steps": 10,  # Reduce logging frequency to speed up training
    }

    if args.fast_dev_run:
        trainer_kwargs["fast_dev_run"] = args.fast_dev_run
        # For fast dev run, disable some optimizations that might interfere
        trainer_kwargs["log_every_n_steps"] = 1

    trainer_creation_timer = time.time()
    trainer = pl.Trainer(**trainer_kwargs)
    trainer_creation_time = time.time() - trainer_creation_timer
    print(f"   🏃 Trainer created in {trainer_creation_time:.2f}s")

    training_setup_time = time.time() - training_setup_timer
    print(f"   ✅ Trainer configured in {training_setup_time:.2f}s")

    # Final startup summary
    total_startup_time = time.time() - startup_timer
    print(f"\n🎯 STARTUP COMPLETE")
    print("=" * 50)
    print(f"⏱️  Total startup time: {total_startup_time:.2f}s")
    print(f"🔧 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(
        f"🤖 Model: {args.num_layers}-layer GCN ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params)"
    )
    print(f"📊 Data: {len(train_dataset):,} train + {len(val_dataset):,} val samples")
    print(f"💾 Checkpoints: {args.checkpoint_dir}")
    print(f"🎯 Precision: {args.precision}")
    if args.precision in ["16-mixed", "bf16-mixed"]:
        print("⚠️  Mixed precision may cause slow first epoch due to AMP compilation")
    if args.use_wandb and WANDB_AVAILABLE and logger:
        print(f"📈 Wandb: {args.wandb_project}/{args.wandb_name}")

    # CUDA Kernel Warmup
    if torch.cuda.is_available() and not args.fast_dev_run:
        print(f"\n🔥 CUDA KERNEL WARMUP")
        print("=" * 50)
        print(
            "⏱️  Pre-compiling CUDA kernels (this may take 20-30s but only happens once)..."
        )
        warmup_start = time.time()

        try:
            # Get a sample batch for warmup
            sample_batch = next(iter(train_loader))
            sample_batch = (
                sample_batch.cuda() if torch.cuda.is_available() else sample_batch
            )

            # Set model to training mode and do multiple forward/backward passes
            model.train()

            print("   🔥 Warming up forward pass...")
            forward_start = time.time()
            with torch.no_grad():
                for _ in range(3):  # Multiple passes to ensure all kernels compile
                    _ = model(
                        sample_batch.x,
                        sample_batch.edge_index,
                        sample_batch.edge_attr,
                        sample_batch.batch,
                    )
            forward_time = time.time() - forward_start
            print(f"   ✅ Forward pass warmed up in {forward_time:.2f}s")

            print("   🔄 Warming up backward pass...")
            backward_start = time.time()
            # Do a backward pass to compile backward kernels too
            model.zero_grad()
            pred = model(
                sample_batch.x,
                sample_batch.edge_index,
                sample_batch.edge_attr,
                sample_batch.batch,
            )
            loss = F.l1_loss(pred, sample_batch.y)
            loss.backward()
            backward_time = time.time() - backward_start
            print(f"   ✅ Backward pass warmed up in {backward_time:.2f}s")

            print("   🏃 Warming up optimizer step...")
            optimizer_start = time.time()
            # Create a temporary optimizer to warm up optimizer kernels
            temp_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            temp_optimizer.step()
            temp_optimizer.zero_grad()
            optimizer_time = time.time() - optimizer_start
            print(f"   ✅ Optimizer warmed up in {optimizer_time:.2f}s")

            warmup_time = time.time() - warmup_start
            print(f"✅ CUDA kernels pre-compiled in {warmup_time:.2f}s")
            print("🚀 First epoch should now be much faster!")

        except Exception as e:
            warmup_time = time.time() - warmup_start
            print(f"⚠️  Kernel warmup failed in {warmup_time:.2f}s: {e}")
            print("🔄 Continuing without warmup...")

    print("\n🚀 Starting training...")

    if args.debug:
        print("Starting optimized training...")

    # Train the model
    print("⏱️  Initializing trainer and starting fit...")
    fit_timer = time.time()

    # Add a callback to time training phases
    class TimingCallback(pl.Callback):
        def __init__(self):
            self.setup_start = None
            self.train_start = None
            self.epoch_start = None
            self.val_start = None
            self.checkpoint_start = None

        def on_fit_start(self, trainer, pl_module):
            print(f"   🎬 Training fit started at {time.time() - fit_timer:.2f}s")

        def setup(self, trainer, pl_module, stage):
            if stage == "fit":
                self.setup_start = time.time()
                print(f"   ⚙️  Lightning setup phase started...")

        def on_train_start(self, trainer, pl_module):
            if self.setup_start:
                setup_time = time.time() - self.setup_start
                print(f"   ⚙️  Lightning setup completed in {setup_time:.2f}s")
            self.train_start = time.time()
            print(f"   🏃 First epoch starting...")

        def on_train_epoch_start(self, trainer, pl_module):
            self.epoch_start = time.time()
            if trainer.current_epoch == 0 and self.train_start:
                first_epoch_start_time = time.time() - self.train_start
                print(f"   🥇 First epoch began after {first_epoch_start_time:.2f}s")
            elif trainer.current_epoch > 0:
                print(f"   📈 Epoch {trainer.current_epoch} starting...")

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if trainer.current_epoch == 0 and batch_idx == 0:
                self.first_batch_start = time.time()
                print(f"   🔥 First batch of first epoch starting...")
            elif trainer.current_epoch == 0 and batch_idx == 1:
                if hasattr(self, "first_batch_start"):
                    first_batch_time = time.time() - self.first_batch_start
                    print(f"   ⚡ First batch completed in {first_batch_time:.2f}s")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.current_epoch == 0 and batch_idx == 0:
                if hasattr(self, "first_batch_start"):
                    first_batch_time = time.time() - self.first_batch_start
                    print(f"   ⚡ First batch processing took {first_batch_time:.2f}s")

        def on_train_epoch_end(self, trainer, pl_module):
            if self.epoch_start:
                epoch_time = time.time() - self.epoch_start
                if trainer.current_epoch == 0:
                    print(
                        f"   🐌 First epoch completed in {epoch_time:.2f}s (includes Lightning overhead)"
                    )
                else:
                    print(
                        f"   🏁 Epoch {trainer.current_epoch} total time: {epoch_time:.2f}s"
                    )

        def on_validation_start(self, trainer, pl_module):
            self.val_start = time.time()
            print(f"   🔍 Validation starting...")

        def on_validation_end(self, trainer, pl_module):
            if self.val_start:
                val_time = time.time() - self.val_start
                print(f"   ✅ Validation completed in {val_time:.2f}s")

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            self.checkpoint_start = time.time()
            print(f"   💾 Saving checkpoint...")

        def on_validation_epoch_end(self, trainer, pl_module):
            if self.checkpoint_start:
                checkpoint_time = time.time() - self.checkpoint_start
                print(f"   💾 Checkpoint saved in {checkpoint_time:.2f}s")
                self.checkpoint_start = None

    # Add timing callback temporarily
    timing_callback = TimingCallback()
    trainer.callbacks.append(timing_callback)

    trainer.fit(model, train_loader, val_loader)

    total_training_time = time.time() - fit_timer
    print(f"⏱️  Training completed in {total_training_time:.2f}s")

    print("🎉 Optimized training complete!")

    # Finish wandb run if active
    if args.use_wandb and WANDB_AVAILABLE and logger:
        print(f"📊 View results at: {logger.experiment.url}")
        wandb.finish()


if __name__ == "__main__":
    main()
