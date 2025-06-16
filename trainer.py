#!/usr/bin/env python3
"""
Clean Lightning CLI Setup - Models inherit directly from BaseModel

Usage:
    python trainer.py fit --model.class_path=src.model.eq_gnn.EquivariantGNN --data.class_path=PointCloudData
    python trainer.py fit --model.class_path=src.model.gnn.BasicGNN --data.class_path=PointCloudData

With Weights & Biases:
    python trainer.py fit --model.class_path=src.model.eq_gnn.EquivariantGNN --data.class_path=PointCloudData \
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
import numpy as np
from src.model.eq_gnn import EquivariantGNN


# from src.model.eq_gnn_fast import EquivariantGNNFast
from src.model.gnn import BasicGNN
from src.model.baseline_zero import BaselineZero
from src.model.eq_e3nn import EquivariantE3NN

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def random_rotation_matrix():
    """Generate a random 3D rotation matrix using Rodrigues' rotation formula."""
    # Random axis (normalized)
    axis = torch.randn(3)
    axis = axis / torch.norm(axis)
    
    # Random angle between 0 and 2π
    angle = torch.rand(1) * 2 * np.pi
    
    # Rodrigues' rotation formula
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
    return R


def apply_data_augmentation(data, rotation_prob=0.5):
    """
    Apply random rotation to graph data.
    
    For graph-based models, rotation is the key augmentation since it preserves
    geometric relationships in edge attributes while improving rotation robustness.
    
    Args:
        data: PyTorch Geometric Data object
        rotation_prob: Probability of applying rotation
    
    Returns:
        Augmented Data object
    """
    augmented_data = data.clone()
    
    # Apply random rotation with given probability
    if torch.rand(1) < rotation_prob:
        R = random_rotation_matrix()
        # Rotate node positions (used as initial embeddings)
        augmented_data.x = torch.matmul(augmented_data.x, R.T)
        # Rotate edge attributes (geometric relationships between nodes)
        if augmented_data.edge_attr is not None:
            augmented_data.edge_attr = torch.matmul(augmented_data.edge_attr, R.T)
        # Rotate target (center of mass)
        augmented_data.y = torch.matmul(augmented_data.y, R.T)
    
    return augmented_data


class PointCloudData(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir="data/processed_sh", 
        batch_size=16, 
        val_split=0.2, 
        num_workers=0,
        # Data augmentation parameters
        use_augmentation=False,
        rotation_prob=0.5
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_cache = []

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print(f"Loading dataset from {self.hparams.data_dir}...")
        if self.hparams.use_augmentation:
            print(f"Data augmentation enabled: rotation_prob={self.hparams.rotation_prob}")

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
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        if self.hparams.use_augmentation:
            # Apply augmentation only to training data
            class AugmentedDataset(torch.utils.data.Dataset):
                def __init__(self, dataset, rotation_prob):
                    self.dataset = dataset
                    self.rotation_prob = rotation_prob
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    data = self.dataset[idx]
                    return apply_data_augmentation(
                        data, 
                        rotation_prob=self.rotation_prob
                    )
            
            augmented_dataset = AugmentedDataset(
                self.train_data, 
                self.hparams.rotation_prob
            )
            
            return DataLoader(
                augmented_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.hparams.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )
        else:
            return DataLoader(
                self.train_data,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.hparams.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )

    def val_dataloader(self):
        # Never apply augmentation to validation data
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
