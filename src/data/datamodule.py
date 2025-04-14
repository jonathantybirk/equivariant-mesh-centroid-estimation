import os
import torch
import numpy as np
import random
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
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found")
            
        self.data = torch.load(data_path)
        
    def len(self):
        """Required by PyG Dataset"""
        return 1  # Single object per file
    
    def get(self, idx):
        # Create a PyG Data object from the loaded data
        point_cloud = self.data['point_cloud']
        node_features = self.data['node_features']
        edge_index = self.data['edge_index']
        edge_attr = self.data['edge_attr']
        com = self.data['target']
        
        # Create PyG Data object
        graph = Data(
            x=node_features,          # Node features
            edge_index=edge_index,    # Graph connectivity
            edge_attr=edge_attr,      # Edge attributes (3D vector differences)
            pos=point_cloud,          # Original point cloud positions
            y=com                     # Target center of mass
        )
        
        return graph


class PointCloudDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for center of mass estimation"""
    
    def __init__(
        self,
        processed_dir="data/processed",
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        node_feature_dim=16,
        train_ratio=1.0,
        val_ratio=0.0,
        seed=42
    ):
        super().__init__()
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.node_feature_dim = node_feature_dim
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        
    def setup(self, stage=None):
        """Load all object files and create splits"""
        # Find all .pt files in the processed directory
        object_files = glob.glob(os.path.join(self.processed_dir, "*.pt"))
        if not object_files:
            raise FileNotFoundError(f"No .pt files found in {self.processed_dir}")
        
        logger.info(f"Found {len(object_files)} object files")
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        
        # Load all graphs directly
        all_graphs = []
        for obj_file in object_files:
            try:
                # Load the raw data
                data = torch.load(obj_file)
                
                # Create a PyG Data object from the loaded data
                point_cloud = data['point_cloud']
                node_features = data['node_features']
                edge_index = data['edge_index']
                edge_attr = data['edge_attr']
                com = data['target']
                
                # Create PyG Data object
                graph = Data(
                    x=node_features,          # Node features
                    edge_index=edge_index,    # Graph connectivity
                    edge_attr=edge_attr,      # Edge attributes (3D vector differences)
                    pos=point_cloud,          # Original point cloud positions
                    y=com                     # Target center of mass
                )
                
                all_graphs.append(graph)
            except Exception as e:
                logger.error(f"Error loading {obj_file}: {e}")
        
        if stage == 'fit' or stage is None:
            # Use all data for training
            self.train_dataset = SimpleGraphDataset(all_graphs)
            
            # For validation, use a minimal subset (just 1 sample)
            self.val_dataset = SimpleGraphDataset([all_graphs[0]])
        
        if stage == 'test' or stage is None:
            # For now, use the same data for testing
            self.test_dataset = SimpleGraphDataset(all_graphs)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=min(self.batch_size, len(self.train_dataset)),
            shuffle=True,
            num_workers=0,  # Using 0 to avoid multiprocessing issues
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=min(self.batch_size, len(self.test_dataset)),
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory
        )