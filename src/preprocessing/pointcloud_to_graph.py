import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from hydra.utils import get_original_cwd
import glob

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
    # Ensure target has shape (3,)
    target = target.reshape(-1)
    
    # Get number of nodes
    N = points.shape[0]
    node_features = torch.ones((N, node_feature_dim), dtype=torch.float32)
    
    # Build k-nearest neighbors graph with fixed number of edges per node
    tree = cKDTree(points)
    # Query for k+1 neighbors (including self) but limit max neighbors to available points
    k_query = min(k + 1, N)
    _, knn_indices = tree.query(points, k=k_query)
    
    # Create edge connections with a deterministic count
    senders = []
    receivers = []
    
    for i in range(N):
        # Skip the first index (self)
        for j_idx in range(1, len(knn_indices[i])):
            j = knn_indices[i][j_idx]
            senders.append(i)
            receivers.append(j)
    
    # Convert to tensors
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    
    # Create edge attributes (difference vectors)
    # This ensures that edge_attr dimensions match edge_index
    edge_attr = torch.tensor(points[receivers] - points[senders], dtype=torch.float32)
    
    # Format target for PyTorch Geometric
    target_tensor = torch.tensor(target, dtype=torch.float32).view(1, 3)
    
    return {
        "point_cloud": torch.tensor(points, dtype=torch.float32),
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "target": target_tensor
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
    
    # Get number of samples per mesh (default to 1 for backward compatibility)
    num_samples = cfg.preprocessing.lidar.get("num_samples", 1)
    
    for obj in obj_dirs:
        obj_dir = os.path.join(pc_dir, obj)
        
        # Handle multiple samples if they exist
        if num_samples > 1:
            # Process each sample
            for sample_idx in range(1, num_samples + 1):
                combined_path = os.path.join(obj_dir, f"pointcloud_combined_sample{sample_idx}.npy")
                com_path = os.path.join(obj_dir, "center_of_mass.npy")  # COM is the same for all samples
                
                if not os.path.exists(combined_path) or not os.path.exists(com_path):
                    print(f"Skipping {obj} sample {sample_idx}: Missing pointcloud or center_of_mass file.")
                    continue
                
                points = np.load(combined_path)
                com = np.load(com_path)
                
                # Process the sample point cloud
                graph_data = build_graph_from_pointcloud(
                    points=points,
                    target=com,
                    k=cfg.preprocessing.graph.k_nn,
                    node_feature_dim=cfg.preprocessing.graph.node_feature_dim
                )
                
                # Save with sample index in filename
                save_path = os.path.join(processed_dir, f"{obj}_sample{sample_idx}.pt")
                torch.save(graph_data, save_path)
                print(f"Saved graph for {obj} sample {sample_idx} to {save_path}")
        else:
            # Original single-sample behavior for backward compatibility
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
