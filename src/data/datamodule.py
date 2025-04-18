import os
import torch
import numpy as np
import random
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import logging
import glob

logger = logging.getLogger(__name__)

class SimpleGraphDataset(InMemoryDataset):
    """Simple dataset for graph objects using PyG's InMemoryDataset"""
    
    def __init__(self, graphs, transform=None):
        super().__init__(None, transform)
        self.data_list = graphs
        self.data, self.slices = self.collate(graphs)
    
    def __len__(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]


class PointCloudGraphDataset(Dataset):
    """Dataset for GNN-based center of mass estimation"""
    
    def __init__(self, data_path, node_feature_dim=16, transform=None):
        super().__init__(transform=transform)
        self.data_path = data_path
        self.node_feature_dim = node_feature_dim
        
        # Check if it's a directory or a file
        if os.path.isdir(data_path):
            logger.info(f"Loading data from directory {data_path}")
            self.file_list = sorted(glob.glob(os.path.join(data_path, "*.pt")))
            if not self.file_list:
                raise FileNotFoundError(f"No .pt files found in directory {data_path}")
            logger.info(f"Found {len(self.file_list)} data files")
            self.is_directory = True
        else:
            logger.info(f"Loading data from file {data_path}")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file {data_path} not found")
            self.is_directory = False
            self.data = torch.load(data_path)
    
    def len(self):
        """Return the number of samples in the dataset"""
        if self.is_directory:
            return len(self.file_list)
        else:
            return 1  # Single object per file
    
    def get(self, idx):
        """Get a single graph data object"""
        if self.is_directory:
            # Load the file for this index
            file_path = self.file_list[idx]
            data = torch.load(file_path)
        else:
            # Use the already loaded data
            data = self.data
        
        # Create a PyG Data object from the loaded data
        point_cloud = data['point_cloud']
        node_features = data['node_features']
        edge_index = data['edge_index']
        edge_attr = data['edge_attr']
        com = data['target']  # This should consistently be [1, 3] from preprocessing
        
        # No reshape needed - the preprocessing ensures consistent [1, 3] shape
        
        # Create PyG Data object
        graph = Data(
            x=node_features,          # Node features
            edge_index=edge_index,    # Graph connectivity
            edge_attr=edge_attr,      # Edge attributes (3D vector differences)
            pos=point_cloud,          # Original point cloud positions
            y=com                     # Target center of mass [1, 3]
        )
        
        # Add object name if available
        if 'name' in data:
            graph.name = data['name']
        
        return graph


class PointCloudDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for center of mass estimation"""
    
    def __init__(
        self, 
        processed_dir, 
        train_dir=None,
        test_dir=None,
        batch_size=32, 
        num_workers=4, 
        pin_memory=True,
        node_feature_dim=16,
        val_split=0.1  # Validation split from train data
    ):
        super().__init__()
        self.processed_dir = processed_dir
        self.train_dir = train_dir or processed_dir
        self.test_dir = test_dir or processed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.node_feature_dim = node_feature_dim
        self.val_split = val_split
    
    def setup(self, stage=None):
        """Setup train, validation and test datasets"""
        if stage == 'fit' or stage is None:
            # Load train dataset
            self.full_train_dataset = PointCloudGraphDataset(
                self.train_dir, 
                node_feature_dim=self.node_feature_dim
            )
            
            # Split train dataset into train and validation
            train_size = int(len(self.full_train_dataset) * (1 - self.val_split))
            val_size = len(self.full_train_dataset) - train_size
            
            self.train_dataset, self.val_dataset = random_split(
                self.full_train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = PointCloudGraphDataset(
                self.test_dir, 
                node_feature_dim=self.node_feature_dim
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )