import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
from tqdm.auto import tqdm

class GraphDataset(Dataset):
    def __init__(self, processed_data_path, dim=32):
        """
        Dataset for 3D point cloud graphs.
        
        Args:
            processed_data_path: Path to processed .pt file containing point clouds and labels
            dim: Dimension of node features
        """
        self.dim = dim
        data = torch.load(processed_data_path)
        self.point_clouds = data["point_clouds"]
        self.centers_of_mass = data["centers_of_mass"]
        
        cache_file = processed_data_path.replace('.pt', '_graph_cache.pt')
        if os.path.exists(cache_file):
            self.graph_data = torch.load(cache_file)
        else:
            self.graph_data = self._preprocess_graphs()
            torch.save(self.graph_data, cache_file)
            
    def _preprocess_graphs(self):
        """Create graph representations for all point clouds"""
        graph_data = []
        for i, (point_cloud, com) in enumerate(tqdm(zip(self.point_clouds, self.centers_of_mass), 
                                               desc="Creating graph data", total=len(self.point_clouds))):
            edge_index, edge_attr = self._create_fully_connected_graph(point_cloud)
            # Initialize node features with zeros
            node_features = torch.zeros(len(point_cloud), self.dim)
            graph_data.append((point_cloud, node_features, edge_index, edge_attr, com))
        return graph_data
    
    def _create_fully_connected_graph(self, point_cloud):
        """Create a fully connected graph from a point cloud"""
        num_nodes = len(point_cloud)
        
        # Create edge indices for fully connected graph (without self-loops)
        source = torch.arange(num_nodes).repeat_interleave(num_nodes - 1)
        target = torch.cat([torch.cat([torch.arange(i), torch.arange(i+1, num_nodes)]) 
                            for i in range(num_nodes)])
        edge_index = torch.stack([source, target], dim=0)
        
        # Compute 3D pairwise differences
        edge_attr = point_cloud[source] - point_cloud[target]
        
        return edge_index, edge_attr
        
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        point_cloud, node_features, edge_index, edge_attr, com = self.graph_data[idx]
        
        # Create a PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=com,  # center of mass as target
            pos=point_cloud  # store original point cloud
        )
        return data

def collate_fn(data_list):
    """Custom collate function to batch PyG Data objects"""
    return Batch.from_data_list(data_list)

class PointCloudGraphDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/processed", batch_size=16, dim=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dim = dim
        
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_data_path = os.path.join(self.data_dir, "train.pt")
            val_data_path = os.path.join(self.data_dir, "val.pt")
            
            self.train_dataset = GraphDataset(train_data_path, self.dim)
            self.val_dataset = GraphDataset(val_data_path, self.dim)
            
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            test_data_path = os.path.join(self.data_dir, "test.pt")
            self.test_dataset = GraphDataset(test_data_path, self.dim)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=4
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=4
        )