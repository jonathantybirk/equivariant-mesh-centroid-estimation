import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random


class PointCloudBaselineDataset(Dataset):
    """Dataset that loads pointclouds directly from .npy files for baselines"""
    
    def __init__(self, pointcloud_dirs):
        self.pointcloud_dirs = pointcloud_dirs
        
    def __len__(self):
        return len(self.pointcloud_dirs)
        
    def __getitem__(self, idx):
        pointcloud_dir = self.pointcloud_dirs[idx]
        
        # Load pointcloud and target
        pointcloud = np.load(pointcloud_dir / "pointcloud.npy")
        target = np.load(pointcloud_dir / "mesh_centroid.npy")
        
        # Convert to tensors
        pointcloud = torch.from_numpy(pointcloud).float()  # (N, 3)
        target = torch.from_numpy(target).float()  # (3,)
        
        # Create a simple data object with pos and y (target)
        # This mimics the structure that the baselines expect
        class SimpleData:
            def __init__(self, pos, y):
                self.pos = pos
                self.y = y
                
        return SimpleData(pos=pointcloud, y=target)


class PointCloudBaselineDataModule(pl.LightningDataModule):
    """Data module for baselines that works directly with pointcloud .npy files"""
    
    def __init__(self, data_dir="data/pointclouds", batch_size=16, val_split=0.2, num_workers=0):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        print(f"Loading pointcloud dataset from {self.hparams.data_dir}...")
        
        # Find all pointcloud directories
        data_path = Path(self.hparams.data_dir)
        pointcloud_dirs = []
        
        for item in data_path.iterdir():
            if item.is_dir():
                pointcloud_file = item / "pointcloud.npy"
                target_file = item / "mesh_centroid.npy"
                if pointcloud_file.exists() and target_file.exists():
                    pointcloud_dirs.append(item)
        
        print(f"Found {len(pointcloud_dirs)} pointcloud samples")
        
        # Split data
        random.seed(42)
        random.shuffle(pointcloud_dirs)
        
        train_size = int((1 - self.hparams.val_split) * len(pointcloud_dirs))
        self.train_dirs = pointcloud_dirs[:train_size]
        self.val_dirs = pointcloud_dirs[train_size:]
        
        self.train_dataset = PointCloudBaselineDataset(self.train_dirs)
        self.val_dataset = PointCloudBaselineDataset(self.val_dirs)
        
        print(f"Train: {len(self.train_dataset)} samples")
        print(f"Val: {len(self.val_dataset)} samples")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn
        )
        
    def collate_fn(self, batch):
        """Custom collate function to handle variable-sized pointclouds"""
        from torch_geometric.data import Batch, Data
        
        # Convert to PyG Data objects
        data_list = []
        for item in batch:
            data = Data(pos=item.pos, y=item.y.unsqueeze(0))  # Add batch dimension to y
            data_list.append(data)
            
        # Use PyG's batching
        return Batch.from_data_list(data_list)
