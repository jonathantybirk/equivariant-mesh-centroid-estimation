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
    """Dataset for GNN-based center of mass estimation with optimized PyTorch tensor loading"""

    def __init__(self, data_path, transform=None):
        super().__init__(transform=transform)
        self.data_path = data_path

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
        """Get a single graph data object with optimized tensor loading"""
        if self.is_directory:
            # Load the file for this index - already in PyTorch format for efficiency
            file_path = self.file_list[idx]
            data = torch.load(file_path)
        else:
            # Use the already loaded data
            data = self.data

        # Create a PyG Data object from the loaded data
        point_cloud = data["point_cloud"]
        node_features = data["node_features"]  # Now directly 3D positions
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]  # Displacement vectors
        com = data["target"]

        # CRITICAL FIX: Ensure consistency between tensors
        num_nodes = node_features.size(0)

        # Verify edge_index references valid nodes
        if edge_index.size(1) > 0:  # Only if there are edges
            # Make sure edge_index doesn't reference non-existent nodes
            max_node_idx = edge_index.max().item()
            if max_node_idx >= num_nodes:
                # Filter out invalid edges (those referencing non-existent nodes)
                valid_edges = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                edge_index = edge_index[:, valid_edges]
                if edge_attr is not None and edge_attr.size(0) > 0:
                    edge_attr = edge_attr[valid_edges]

        # Create PyG Data object with verified tensors
        graph = Data(
            x=node_features,  # Node features (3D positions)
            edge_index=edge_index,  # Graph connectivity
            edge_attr=edge_attr,  # Edge attributes (displacement vectors)
            pos=point_cloud,  # Original point cloud positions
            y=com,  # Target center of mass [1, 3]
        )

        # Verify graph integrity - key step to ensure data consistency
        assert (
            graph.num_nodes == num_nodes
        ), f"Node count mismatch in graph {idx}: {graph.num_nodes} vs {num_nodes}"
        if edge_index.size(1) > 0:
            assert (
                edge_index.max().item() < num_nodes
            ), f"Edge index out of bounds in graph {idx}"
            assert edge_attr.size(0) == edge_index.size(
                1
            ), f"Edge attr count mismatch in graph {idx}"

        # Add object name if available
        if "name" in data:
            graph.name = data["name"]

        return graph


class PointCloudDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for center of mass estimation with simplified interface"""

    def __init__(
        self,
        processed_dir,
        train_dir=None,
        test_dir=None,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        val_split=0.1,  # Validation split from train data
        sample_balanced=False,  # Whether to balance samples across different meshes
    ):
        super().__init__()
        self.processed_dir = processed_dir
        self.train_dir = train_dir or processed_dir
        self.test_dir = test_dir or processed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.sample_balanced = sample_balanced

    def setup(self, stage=None):
        """Setup train, validation and test datasets"""
        if stage == "fit" or stage is None:
            # Load train dataset
            self.full_train_dataset = PointCloudGraphDataset(self.train_dir)

            if self.sample_balanced:
                # Group samples by mesh name and select one point cloud per mesh
                self._create_balanced_dataset()

            # Split train dataset into train and validation
            train_size = int(len(self.full_train_dataset) * (1 - self.val_split))
            val_size = len(self.full_train_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                self.full_train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test" or stage is None:
            self.test_dataset = PointCloudGraphDataset(self.test_dir)

    def _create_balanced_dataset(self):
        """
        Create a balanced dataset by grouping multiple samples of the same mesh
        and selecting one representative sample per mesh.
        This is useful when we have multiple point clouds generated per mesh.
        """
        if not self.sample_balanced:
            return

        # Extract base object name from file paths
        file_list = self.full_train_dataset.file_list
        file_dict = {}

        for file_path in file_list:
            base_name = os.path.basename(file_path)
            # Extract mesh name from filenames like "Chair_sample1.pt" -> "Chair"
            if "_sample" in base_name:
                mesh_name = base_name.split("_sample")[0]
            else:
                mesh_name = os.path.splitext(base_name)[0]

            if mesh_name not in file_dict:
                file_dict[mesh_name] = []
            file_dict[mesh_name].append(file_path)

        # Select one sample per mesh (randomly)
        balanced_files = []
        for mesh_name, files in file_dict.items():
            # Randomly select one file per mesh
            balanced_files.append(random.choice(files))

        # Replace the original file list with the balanced one
        self.full_train_dataset.file_list = balanced_files
        logger.info(
            f"Balanced dataset created: {len(balanced_files)} samples from {len(file_dict)} unique meshes"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
