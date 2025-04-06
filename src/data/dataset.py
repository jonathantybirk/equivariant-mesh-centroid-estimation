import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class LiDARPointCloudDataset(Dataset):
    def __init__(self, data_path, num_points=1024):
        """
        Expects data_path to have subdirectories for each object, with files:
          - pointcloud_combined.npy
          - center_of_mass.npy
        """
        self.samples = []
        for obj_dir in os.listdir(data_path):
            full_path = os.path.join(data_path, obj_dir)
            if os.path.isdir(full_path):
                pc_file = os.path.join(full_path, "pointcloud_combined.npy")
                com_file = os.path.join(full_path, "center_of_mass.npy")
                if os.path.exists(pc_file) and os.path.exists(com_file):
                    self.samples.append((pc_file, com_file))
        self.num_points = num_points
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pc_file, com_file = self.samples[idx]
        point_cloud = np.load(pc_file)  # shape: (N, 3)
        center_of_mass = np.load(com_file)  # shape: (3,)
        # Sample exactly num_points from the point cloud (with replacement if needed)
        if point_cloud.shape[0] >= self.num_points:
            indices = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
        else:
            indices = np.random.choice(point_cloud.shape[0], self.num_points, replace=True)
        sampled_pc = point_cloud[indices]
        sampled_pc = torch.from_numpy(sampled_pc.astype(np.float32))
        center_of_mass = torch.tensor(center_of_mass.astype(np.float32))
        return sampled_pc, center_of_mass

class LiDARDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, num_points=1024, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_points = num_points
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        self.dataset = LiDARPointCloudDataset(self.data_path, num_points=self.num_points)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
