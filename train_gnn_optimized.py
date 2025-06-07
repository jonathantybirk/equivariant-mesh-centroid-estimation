#!/usr/bin/env python3
"""
Clean Lightning CLI Setup - Models inherit directly from BaseModel

Usage:
    python train_gnn_optimized.py fit --model.class_path=src.model.eq_gnn.EquivariantGNN --data.class_path=PointCloudData
    python train_gnn_optimized.py fit --model.class_path=src.model.gnn.BasicGNN --data.class_path=PointCloudData

With Weights & Biases:
    python train_gnn_optimized.py fit --model.class_path=src.model.eq_gnn.EquivariantGNN --data.class_path=PointCloudData \
        --trainer.logger.class_path=lightning.pytorch.loggers.WandbLogger \
        --trainer.logger.init_args.project="gnn-optimization"
"""

import warnings

warnings.filterwarnings("ignore")

import torch
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from torch_geometric.data import Data, DataLoader
from pathlib import Path
import sys
from tqdm import tqdm
from src.model.eq_gnn import EquivariantGNN
from src.model.gnn import BasicGNN

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


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
    Lightning CLI - models inherit from BaseModel directly
    
    Examples:
    
    1. Basic training:
        python train_gnn_optimized.py fit --model.class_path=src.model.eq_gnn.EquivariantGNN --data.class_path=PointCloudData
    
    2. With Weights & Biases:
        python train_gnn_optimized.py fit --model.class_path=src.model.eq_gnn.EquivariantGNN --data.class_path=PointCloudData \
            --trainer.logger.class_path=lightning.pytorch.loggers.WandbLogger \
            --trainer.logger.init_args.project="gnn-optimization"
    
    3. Using config file:
        python train_gnn_optimized.py fit --config config_wandb.yaml
    
    4. Fast dev run:
        python train_gnn_optimized.py fit --model.class_path=src.model.eq_gnn.EquivariantGNN --data.class_path=PointCloudData \
            --trainer.fast_dev_run=true
    """

    # Lightning CLI with no restrictions (auto-discovery)
    cli = LightningCLI(
        save_config_kwargs={"overwrite": True}, datamodule_class=PointCloudData
    )
