import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from hydra.utils import get_original_cwd

def build_graph_from_pointcloud(points: np.ndarray, target: np.ndarray, k: int, node_feature_dim: int):
    """
    Converts a point cloud into graph data suitable for an EGNN.
    
    Args:
        points: (N, 3) array of 3D points.
        target: (3,) array representing the object's center of mass.
        k: Number of nearest neighbors for each node.
        node_feature_dim: Dimensionality of the node features.
        
    Returns:
        A dictionary with graph data.
    """
    # Ensure target has shape (3,) - remove extra dimensions
    target = target.reshape(-1)  # Flattens to (3,)
    
    N = points.shape[0]
    node_features = torch.ones((N, node_feature_dim), dtype=torch.float32)
    
    tree = cKDTree(points)
    _, knn_indices = tree.query(points, k=k + 1)  # k+1 to include self
    senders = []
    receivers = []
    for i in range(N):
        for j in knn_indices[i][1:]:  # Skip the self index
            senders.append(i)
            receivers.append(j)
    
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    # Edge attributes: difference vector between connected nodes
    relative = points[receivers] - points[senders]
    edge_attr = torch.tensor(relative, dtype=torch.float32)
    
    return {
        "point_cloud": torch.tensor(points, dtype=torch.float32),
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "target": torch.tensor(target, dtype=torch.float32)  # Now shape (3,)
    }

def convert_all_pointclouds(cfg):
    """
    Loads saved point clouds from the pointcloud output directory and converts them to graph data.
    The graph data is saved as .pt files in the processed directory.
    """
    original_cwd = get_original_cwd()
    pc_dir = os.path.join(original_cwd, cfg.preprocessing.lidar.output_dir)
    processed_dir = os.path.join(original_cwd, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Each subfolder in the pointcloud directory corresponds to one object.
    obj_dirs = [d for d in os.listdir(pc_dir) if os.path.isdir(os.path.join(pc_dir, d))]
    if not obj_dirs:
        print("No pointcloud directories found. Please run point cloud generation first.")
        return
    
    for obj in obj_dirs:
        obj_dir = os.path.join(pc_dir, obj)
        combined_path = os.path.join(obj_dir, "pointcloud_combined.npy")
        com_path = os.path.join(obj_dir, "center_of_mass.npy")
        if not os.path.exists(combined_path) or not os.path.exists(com_path):
            print(f"Skipping {obj}: Missing pointcloud or center_of_mass file.")
            continue
        
        points = np.load(combined_path)
        com = np.load(com_path)
        
        # Don't reshape the target - pass it as (3,)
        graph_data = build_graph_from_pointcloud(
            points=points,
            target=com,  # No reshape, let the function handle it
            k=cfg.preprocessing.graph.k_nn,
            node_feature_dim=cfg.preprocessing.graph.node_feature_dim
        )
        save_path = os.path.join(processed_dir, f"{obj}.pt")
        torch.save(graph_data, save_path)
        print(f"Saved graph for {obj} to {save_path}")
    
    print("Graph conversion complete.")
