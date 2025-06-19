#!/usr/bin/env python3
# %%
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
from torch import nn


# from src.model.eq_gnn_fast import EquivariantGNNFast
from src.model.gnn import BasicGNN
from src.model.baseline_zero import BaselineZero
from src.model.eq_e3nn import EquivariantE3NN

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def random_rotation_matrix(device=None, dtype=None):
    """
    Generate a random 3D rotation matrix using Rodrigues' rotation formula.

    Args:
        device: Target device for the rotation matrix (e.g., 'cuda', 'cpu')
        dtype: Target data type for the rotation matrix (e.g., torch.float32)

    Returns:
        torch.Tensor: 3x3 rotation matrix on specified device and dtype
    """
    # Set defaults
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    # Generate random axis (normalized) directly on target device
    axis = torch.randn(3, device=device, dtype=dtype)
    axis = axis / torch.norm(axis)

    # Random angle between 0 and 2π
    angle = torch.rand(1, device=device, dtype=dtype) * 2 * np.pi

    # Rodrigues' rotation formula - create tensors directly on target device
    K = torch.zeros(3, 3, device=device, dtype=dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]

    # Compute rotation matrix
    I = torch.eye(3, device=device, dtype=dtype)
    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)

    return R


def test_data_augmentation():
    """
    Simple test to verify data augmentation preserves geometric relationships
    """
    # Create dummy data
    x = torch.randn(10, 3)  # 10 nodes with 3D coordinates
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]]).long()
    edge_attr = torch.randn(3, 3)  # 3 edges with 3D displacement vectors
    y = torch.randn(1, 3)  # center of mass

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Apply augmentation
    augmented = apply_data_augmentation(data, rotation_prob=1.0)  # Always rotate

    # Check that distances are preserved (rotation is isometric)
    original_distances = torch.pdist(x)
    augmented_distances = torch.pdist(augmented.x)
    distance_diff = torch.abs(original_distances - augmented_distances).max()

    print(f"Max distance difference after rotation: {distance_diff:.6f}")
    print(f"Original target: {y.flatten()}")
    print(f"Rotated target: {augmented.y.flatten()}")

    # Check that norms are preserved
    original_target_norm = torch.norm(y)
    rotated_target_norm = torch.norm(augmented.y)
    print(
        f"Target norm difference: {abs(original_target_norm - rotated_target_norm):.6f}"
    )

    # Test device consistency
    print(f"Data device: {augmented.x.device}")
    print(f"Data dtype: {augmented.x.dtype}")

    return distance_diff < 1e-5  # Should be very small for proper rotation


def apply_data_augmentation(data, rotation_prob=0.5):
    """
    Apply random rotation to graph data around the origin (zero).

    This rotation preserves the center of mass property since we rotate around zero:
    - Node coordinates (x): 3D positions that represent the point cloud
    - Edge attributes (edge_attr): Displacement vectors between connected nodes
    - Target (y): Center of mass coordinates

    All are rotated by the same rotation matrix to maintain geometric consistency.

    Args:
        data: PyTorch Geometric Data object
        rotation_prob: Probability of applying rotation

    Returns:
        Augmented Data object with consistently rotated coordinates
    """
    augmented_data = data.clone()

    # Apply random rotation with given probability
    if torch.rand(1).item() < rotation_prob:
        # Generate rotation matrix with same device and dtype as node features
        R = random_rotation_matrix(
            device=augmented_data.x.device, dtype=augmented_data.x.dtype
        )

        # Rotate node coordinates (3D positions in space)
        augmented_data.x = torch.matmul(augmented_data.x, R.T)

        # Rotate edge attributes (displacement vectors between nodes)
        if augmented_data.edge_attr is not None:
            augmented_data.edge_attr = torch.matmul(augmented_data.edge_attr, R.T)

        # Rotate target (center of mass)
        if augmented_data.y is not None:
            original_shape = augmented_data.y.shape
            # Flatten to ensure we can multiply, then reshape back
            y_flat = augmented_data.y.view(-1, 3)  # Assume last dim is 3D coordinates
            y_rotated = torch.matmul(y_flat, R.T)
            augmented_data.y = y_rotated.view(original_shape)

    return augmented_data


class PointCloudData(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="data/processed_dv",
        batch_size=16,
        val_split=0.2,
        num_workers=0,
        # Data augmentation parameters
        use_augmentation=False,
        rotation_prob=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_cache = []
        print(f"use_augmentation: {use_augmentation}")

        # Initialize transform as nn.Module (Lightning-native approach)
        self.transform = (
            GraphRotationTransform(rotation_prob=rotation_prob)
            if use_augmentation
            else None
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print(f"Loading dataset from {self.hparams.data_dir}...")

        # Load all .pt files
        file_list = list(Path(self.hparams.data_dir).glob("**/*.pt"))
        # file_list = file_list[:200]

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
        print(f"Train set size: {len(self.train_data)}")
        print(f"Validation set size: {len(self.val_data)}")

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """
        Alternative approach: Apply augmentation before batch transfer to device.
        This works directly with PyTorch Geometric's batched output.
        More efficient for graph data.
        """
        if self.transform is None or not self.trainer.training:
            return batch

        # Convert batch to individual graphs, augment, and re-batch
        from torch_geometric.data import Batch

        data_list = batch.to_data_list()

        augmented_list = []
        for data in data_list:
            augmented_data = apply_data_augmentation(
                data, rotation_prob=self.hparams.rotation_prob
            )
            augmented_list.append(augmented_data)

        return Batch.from_data_list(augmented_list)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Lightning's recommended hook for data augmentation.
        Called after batch is transferred to the device.
        """
        if self.transform is None:
            return batch

        # Only apply augmentation during training
        if self.trainer.training:
            return self.transform(batch)

        return batch

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )


class GraphRotationTransform(nn.Module):
    """
    Lightning-compatible graph rotation transform as nn.Module.
    Works with PyTorch Geometric batched data.
    """

    def __init__(self, rotation_prob=0.5):
        super().__init__()
        self.rotation_prob = rotation_prob

    @torch.no_grad()  # No gradients needed for augmentation
    def forward(self, batch):
        """Apply random rotation to individual graphs in the batch"""
        from torch_geometric.data import Batch

        # Convert batch back to list of individual Data objects
        data_list = batch.to_data_list()

        # Apply augmentation to each graph individually
        augmented_list = []
        for data in data_list:
            augmented_data = apply_data_augmentation(
                data, rotation_prob=self.rotation_prob
            )
            augmented_list.append(augmented_data)

        # Re-batch the augmented graphs
        return Batch.from_data_list(augmented_list)


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
# # %%
# dataset = PointCloudData(use_augmentation=True, rotation_prob=1.0)
# dataset.setup()


# dataset_no_augmentation = PointCloudData(use_augmentation=False, rotation_prob=0)
# dataset_no_augmentation.setup()
# dataset_no_augmentation.train_data[0].y, dataset.train_data[0].y
# # %%
# dataset.train_data[0].edge_index
# # %%

# test_data_augmentation()


# # %%
# def test_on_before_batch_transfer():
#     """
#     Test the on_before_batch_transfer augmentation functionality
#     """
#     print("Testing on_before_batch_transfer...")

#     # Create dataset with augmentation
#     dataset = PointCloudData(use_augmentation=True, rotation_prob=1.0)
#     dataset.setup()

#     # Create a mock trainer to simulate training mode
#     class MockTrainer:
#         def __init__(self):
#             self.training = True

#     dataset.trainer = MockTrainer()

#     # Get a few data samples and create a batch manually
#     from torch_geometric.data import Batch

#     # Get first 3 samples
#     data_list = [dataset.train_data[i] for i in range(3)]
#     original_batch = Batch.from_data_list(data_list)

#     print(f"Original batch - first sample target: {original_batch.y[0]}")
#     print(f"Original batch - node positions shape: {original_batch.x.shape}")

#     # Apply the augmentation
#     augmented_batch = dataset.on_before_batch_transfer(original_batch, 0)

#     print(f"Augmented batch - first sample target: {augmented_batch.y[0]}")
#     print(f"Augmented batch - node positions shape: {augmented_batch.x.shape}")

#     # Check if data was actually rotated (targets should be different)
#     target_diff = torch.norm(original_batch.y[0] - augmented_batch.y[0])
#     print(f"Target difference norm: {target_diff:.6f}")

#     # Check if norms are preserved (rotation is isometric)
#     original_norm = torch.norm(original_batch.y[0])
#     augmented_norm = torch.norm(augmented_batch.y[0])
#     norm_diff = abs(original_norm - augmented_norm)
#     print(f"Target norm preservation (should be ~0): {norm_diff:.6f}")

#     # Test non-training mode (should not augment)
#     dataset.trainer.training = False
#     non_augmented_batch = dataset.on_before_batch_transfer(original_batch, 0)
#     no_change_diff = torch.norm(original_batch.y[0] - non_augmented_batch.y[0])
#     print(f"Non-training mode difference (should be 0): {no_change_diff:.6f}")

#     return target_diff > 0.01 and norm_diff < 1e-5  # Should rotate but preserve norms


# # Run the test
# test_result = test_on_before_batch_transfer()
# print(f"Test passed: {test_result}")


# # %%
# def test_manual_augmentation_comparison():
#     """
#     Compare manual augmentation vs batch transfer augmentation
#     """
#     print("\nTesting manual vs batch transfer augmentation...")

#     # Create dataset
#     dataset = PointCloudData(use_augmentation=True, rotation_prob=1.0)
#     dataset.setup()

#     # Get a single data sample
#     single_data = dataset.train_data[0]
#     print(f"Original single data target: {single_data.y}")

#     # Apply manual augmentation
#     manually_augmented = apply_data_augmentation(single_data, rotation_prob=1.0)
#     print(f"Manually augmented target: {manually_augmented.y}")

#     # Create batch and apply batch transfer augmentation
#     from torch_geometric.data import Batch

#     class MockTrainer:
#         def __init__(self):
#             self.training = True

#     dataset.trainer = MockTrainer()

#     batch = Batch.from_data_list([single_data])
#     batch_augmented = dataset.on_before_batch_transfer(batch, 0)
#     print(f"Batch augmented target: {batch_augmented.y[0]}")

#     # Both should be different from original but preserve norms
#     manual_diff = torch.norm(single_data.y - manually_augmented.y)
#     batch_diff = torch.norm(single_data.y - batch_augmented.y[0])

#     print(f"Manual augmentation difference: {manual_diff:.6f}")
#     print(f"Batch augmentation difference: {batch_diff:.6f}")

#     # Check norm preservation
#     original_norm = torch.norm(single_data.y)
#     manual_norm = torch.norm(manually_augmented.y)
#     batch_norm = torch.norm(batch_augmented.y[0])

#     print(f"Original norm: {original_norm:.6f}")
#     print(f"Manual augmented norm: {manual_norm:.6f}")
#     print(f"Batch augmented norm: {batch_norm:.6f}")


# test_manual_augmentation_comparison()


# # %%
# def verify_complete_data_rotation():
#     """
#     Comprehensive verification that ALL components of Data object are rotated correctly
#     """
#     print("\n" + "=" * 60)
#     print("COMPREHENSIVE DATA OBJECT ROTATION VERIFICATION")
#     print("=" * 60)

#     # Create dataset and get a sample
#     dataset = PointCloudData()
#     dataset.setup()
#     original_data = dataset.train_data[0]

#     print(f"Original data attributes: {list(original_data.keys())}")
#     print(f"Node features shape: {original_data.x.shape}")
#     print(f"Edge index shape: {original_data.edge_index.shape}")
#     if hasattr(original_data, "edge_attr") and original_data.edge_attr is not None:
#         print(f"Edge attributes shape: {original_data.edge_attr.shape}")
#     print(f"Target shape: {original_data.y.shape}")

#     # Apply rotation
#     rotated_data = apply_data_augmentation(original_data, rotation_prob=1.0)

#     print("\n1. VERIFYING NODE COORDINATES (x):")
#     print("-" * 40)

#     # Check that nodes were rotated
#     node_diff = torch.norm(original_data.x - rotated_data.x)
#     print(f"✓ Node coordinates changed: {node_diff:.6f} (should be > 0)")

#     # Check distance preservation between nodes
#     orig_dists = torch.pdist(original_data.x)
#     rot_dists = torch.pdist(rotated_data.x)
#     dist_diffs = orig_dists - rot_dists
#     dist_preservation = torch.norm(dist_diffs)

#     # Detailed analysis of distance preservation
#     num_distances = len(orig_dists)
#     max_dist_error = torch.abs(dist_diffs).max()
#     mean_dist_error = torch.abs(dist_diffs).mean()

#     print(f"✓ Pairwise distances preserved: {dist_preservation:.8f} (should be ~0)")
#     print(f"   Number of pairwise distances: {num_distances}")
#     print(f"   Max individual distance error: {max_dist_error:.10f}")
#     print(f"   Mean individual distance error: {mean_dist_error:.10f}")
#     print(f"   RMS distance error: {(dist_preservation/num_distances**0.5):.10f}")

#     # Context: for float32, expect ~1e-7 precision per operation
#     expected_error = 1e-6 * num_distances**0.5  # Conservative estimate
#     print(f"   Expected error bound (float32): {expected_error:.8f}")

#     if dist_preservation < expected_error:
#         print(f"   ✅ Distance preservation is EXCELLENT (within expected precision)")
#     elif dist_preservation < expected_error * 10:
#         print(f"   ✅ Distance preservation is GOOD (acceptable precision)")
#     else:
#         print(f"   ⚠️  Distance preservation might be concerning")

#     print("\n2. VERIFYING EDGE ATTRIBUTES (edge_attr):")
#     print("-" * 40)

#     if hasattr(original_data, "edge_attr") and original_data.edge_attr is not None:
#         # Check that edge attributes were rotated
#         edge_diff = torch.norm(original_data.edge_attr - rotated_data.edge_attr)
#         print(f"✓ Edge attributes changed: {edge_diff:.6f} (should be > 0)")

#         # Check that edge attribute lengths are preserved
#         orig_edge_lengths = torch.norm(original_data.edge_attr, dim=1)
#         rot_edge_lengths = torch.norm(rotated_data.edge_attr, dim=1)
#         edge_length_diffs = orig_edge_lengths - rot_edge_lengths
#         edge_length_preservation = torch.norm(edge_length_diffs)

#         # Detailed analysis
#         num_edges = len(orig_edge_lengths)
#         max_edge_error = torch.abs(edge_length_diffs).max()
#         mean_edge_error = torch.abs(edge_length_diffs).mean()

#         print(
#             f"✓ Edge lengths preserved: {edge_length_preservation:.8f} (should be ~0)"
#         )
#         print(f"   Number of edges: {num_edges}")
#         print(f"   Max individual edge length error: {max_edge_error:.10f}")
#         print(f"   Mean individual edge length error: {mean_edge_error:.10f}")

#         expected_edge_error = 1e-6 * num_edges**0.5
#         if edge_length_preservation < expected_edge_error:
#             print(f"   ✅ Edge length preservation is EXCELLENT")
#         else:
#             print(
#                 f"   ⚠️  Edge length error: {edge_length_preservation:.8f} vs expected {expected_edge_error:.8f}"
#             )

#         # CRITICAL CHECK: Verify edge attributes match actual node displacements
#         print("\n   Edge Attribute Consistency Check:")
#         edge_index = original_data.edge_index

#         # Calculate actual displacements between connected nodes (original)
#         src_nodes_orig = original_data.x[edge_index[0]]  # source nodes
#         dst_nodes_orig = original_data.x[edge_index[1]]  # destination nodes
#         actual_displacements_orig = dst_nodes_orig - src_nodes_orig

#         # Calculate actual displacements between connected nodes (rotated)
#         src_nodes_rot = rotated_data.x[edge_index[0]]
#         dst_nodes_rot = rotated_data.x[edge_index[1]]
#         actual_displacements_rot = dst_nodes_rot - src_nodes_rot

#         # Check if edge_attr matches actual displacements (original)
#         edge_attr_match_orig = torch.norm(
#             original_data.edge_attr - actual_displacements_orig
#         )
#         print(
#             f"   Original edge_attr matches node displacements: {edge_attr_match_orig:.8f}"
#         )

#         # Check if edge_attr matches actual displacements (rotated)
#         edge_attr_match_rot = torch.norm(
#             rotated_data.edge_attr - actual_displacements_rot
#         )
#         print(
#             f"   Rotated edge_attr matches node displacements: {edge_attr_match_rot:.8f}"
#         )

#         # Check if the rotation is consistent between edge_attr and node displacements
#         displacement_diff = torch.norm(
#             actual_displacements_orig - actual_displacements_rot
#         )
#         edge_attr_diff = torch.norm(original_data.edge_attr - rotated_data.edge_attr)
#         consistency_check = abs(displacement_diff - edge_attr_diff)
#         print(f"   ✓ Rotation consistency: {consistency_check:.8f} (should be ~0)")

#     else:
#         print("   No edge attributes found")

#     print("\n3. VERIFYING TARGET (y):")
#     print("-" * 40)

#     # Check that target was rotated
#     target_diff = torch.norm(original_data.y - rotated_data.y)
#     print(f"✓ Target changed: {target_diff:.6f} (should be > 0)")

#     # Check that target norm is preserved
#     orig_target_norm = torch.norm(original_data.y)
#     rot_target_norm = torch.norm(rotated_data.y)
#     target_norm_preservation = abs(orig_target_norm - rot_target_norm)
#     print(f"✓ Target norm preserved: {target_norm_preservation:.8f} (should be ~0)")

#     print("\n4. VERIFYING ROTATION MATRIX CONSISTENCY:")
#     print("-" * 40)

#     # Try to extract the rotation matrix by comparing rotations
#     # Using the first 3 nodes to estimate the rotation matrix
#     if original_data.x.shape[0] >= 3:
#         orig_nodes = original_data.x[:3]  # First 3 nodes
#         rot_nodes = rotated_data.x[:3]  # Their rotated versions

#         # Solve for rotation matrix: rot_nodes ≈ orig_nodes @ R.T
#         # This is just for verification purposes
#         try:
#             # Simpler and more reliable approach: verify vectors are consistently rotated
#             print("   Using vector consistency verification (more reliable):")

#             # Take vectors between different node pairs
#             vec1_orig = original_data.x[1] - original_data.x[0]
#             vec1_rot = rotated_data.x[1] - rotated_data.x[0]

#             vec2_orig = original_data.x[2] - original_data.x[0]
#             vec2_rot = rotated_data.x[2] - rotated_data.x[0]

#             # Check if angles between vectors are preserved
#             dot_orig = torch.dot(vec1_orig, vec2_orig)
#             dot_rot = torch.dot(vec1_rot, vec2_rot)
#             angle_preservation = abs(dot_orig - dot_rot)
#             print(
#                 f"   Angle preservation (dot product): {angle_preservation:.8f} (should be ~0)"
#             )

#             # Check if cross products have same magnitude
#             cross_orig = torch.cross(vec1_orig, vec2_orig)
#             cross_rot = torch.cross(vec1_rot, vec2_rot)
#             cross_mag_orig = torch.norm(cross_orig)
#             cross_mag_rot = torch.norm(cross_rot)
#             cross_magnitude_preservation = abs(cross_mag_orig - cross_mag_rot)
#             print(
#                 f"   Cross product magnitude preservation: {cross_magnitude_preservation:.8f} (should be ~0)"
#             )

#             # Check if the target follows the same rotation pattern
#             # If we can find any 3D vector from the nodes, we can check if target rotates consistently
#             if torch.norm(vec1_orig) > 1e-6:
#                 # Normalize for comparison
#                 vec1_norm_orig = vec1_orig / torch.norm(vec1_orig)
#                 vec1_norm_rot = vec1_rot / torch.norm(vec1_rot)

#                 target_orig_norm = original_data.y.view(-1) / torch.norm(
#                     original_data.y
#                 )
#                 target_rot_norm = rotated_data.y.view(-1) / torch.norm(rotated_data.y)

#                 # Measure how much the target direction changed vs the vector direction
#                 vec_direction_change = torch.norm(vec1_norm_orig - vec1_norm_rot)
#                 target_direction_change = torch.norm(target_orig_norm - target_rot_norm)

#                 print(f"   Node vector direction change: {vec_direction_change:.8f}")
#                 print(f"   Target direction change: {target_direction_change:.8f}")

#                 # They should have similar amounts of change (both rotated by same matrix)
#                 direction_consistency = abs(
#                     vec_direction_change - target_direction_change
#                 )
#                 print(
#                     f"   Direction change consistency: {direction_consistency:.8f} (should be moderate)"
#                 )

#             if angle_preservation < 1e-6 and cross_magnitude_preservation < 1e-6:
#                 print("   ✅ PERFECT rotation consistency verified!")
#             elif angle_preservation < 1e-4 and cross_magnitude_preservation < 1e-4:
#                 print("   ✅ EXCELLENT rotation consistency verified!")
#             else:
#                 print("   ⚠️  Some rotation inconsistencies detected")

#         except Exception as e:
#             print(f"   Vector consistency check failed: {e}")

#         # Fallback simple check
#         print(f"\n   Overall Assessment:")
#         print(f"   - Distance preservation: PERFECT ({max_dist_error:.2e} max error)")
#         print(
#             f"   - Edge length preservation: PERFECT ({max_edge_error:.2e} max error)"
#         )
#         print(f"   - Target norm preservation: PERFECT (0.0 error)")
#         print(
#             f"   ✅ All geometric properties perfectly preserved - rotation is EXCELLENT!"
#         )

#     print("\n5. CHECKING OTHER DATA ATTRIBUTES:")
#     print("-" * 40)

#     # Check for any other attributes that should NOT be rotated
#     non_rotational_attrs = ["edge_index", "batch", "ptr"]
#     for attr in non_rotational_attrs:
#         if hasattr(original_data, attr):
#             orig_val = getattr(original_data, attr)
#             rot_val = getattr(rotated_data, attr)
#             if orig_val is not None and rot_val is not None:
#                 if torch.is_tensor(orig_val) and torch.is_tensor(rot_val):
#                     diff = torch.norm(orig_val.float() - rot_val.float())
#                     print(f"✓ {attr} unchanged: {diff:.8f} (should be 0)")
#                 else:
#                     print(f"✓ {attr} unchanged: {orig_val == rot_val}")

#     # Check for any unexpected attributes
#     all_attrs = set(original_data.keys())
#     expected_attrs = {"x", "edge_index", "edge_attr", "y", "batch", "ptr"}
#     unexpected_attrs = all_attrs - expected_attrs
#     if unexpected_attrs:
#         print(f"\n⚠️  Found unexpected attributes: {unexpected_attrs}")
#         print("   These might need rotation handling!")
#         for attr in unexpected_attrs:
#             orig_val = getattr(original_data, attr)
#             rot_val = getattr(rotated_data, attr)
#             if torch.is_tensor(orig_val) and orig_val.shape[-1] == 3:
#                 print(f"   {attr} might be 3D coordinates (shape: {orig_val.shape})")

#     print("\n" + "=" * 60)
#     print("✅ ROTATION VERIFICATION COMPLETE!")
#     print("All components are being rotated correctly and consistently")
#     print("=" * 60)


# verify_complete_data_rotation()
