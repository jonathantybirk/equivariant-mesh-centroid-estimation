import os
import numpy as np
import torch
from scipy.spatial import cKDTree
import glob
from e3nn.o3 import spherical_harmonics
from tqdm import tqdm
import logging


def compute_spherical_harmonics_preprocessing(vectors, max_l=2):
    """
    Compute spherical harmonics for 3D vectors up to degree max_l during preprocessing

    Args:
        vectors: [N, 3] tensor of 3D vectors
        max_l: maximum spherical harmonic degree

    Returns:
        Concatenated tensor of all SH features [N, total_sh_dim]
        where total_sh_dim = sum((2*l+1) for l in range(max_l+1))
    """
    # Ensure float32 and normalize vectors
    vectors = (
        torch.tensor(vectors, dtype=torch.float32)
        if not isinstance(vectors, torch.Tensor)
        else vectors.float()
    )
    norms = torch.norm(vectors, dim=1, keepdim=True)
    # Avoid division by zero
    norms = torch.clamp(norms, min=1e-8)
    normalized_vectors = vectors / norms

    # Compute spherical harmonics for each degree and concatenate
    sh_features_list = []
    for l in range(max_l + 1):
        sh_l = spherical_harmonics(l, normalized_vectors, normalize=True)
        sh_features_list.append(sh_l.float())

    # Concatenate all SH features into single tensor
    # Shape: [N, sum(2*l+1 for l in range(max_l+1))]
    sh_features = torch.cat(sh_features_list, dim=1)
    return sh_features


def build_graph_from_pointcloud(
    points: np.ndarray,
    target: np.ndarray,
    k: int,
    use_spherical_harmonics: bool = False,
    max_sh_degree: int = 1,
    debug: bool = False,
    normalize_pointcloud: bool = True,
):
    """
    Converts a point cloud into a sparse graph data suitable for PyTorch Geometric.

    Args:
        points: (N, 3) array of 3D points.
        target: (3,) array representing the object's mesh centroid.
        k: Number of nearest neighbors for each node.
        use_spherical_harmonics: Whether to compute SH features for edge attributes
        max_sh_degree: Maximum spherical harmonic degree for edge features
        debug: Whether to show debug information
        normalize_pointcloud: Whether to center each point cloud at the origin

    Returns:
        A dictionary with graph data in PyTorch tensor format.
    """
    # Ensure target has shape (3,)
    target = target.reshape(-1)

    # Get number of nodes
    N = points.shape[0]

    if debug:
        print(f"  [GRAPH] Building graph: {N} nodes, k={k} neighbors")
        if use_spherical_harmonics:
            print(f"  [SH] Computing spherical harmonics with max_l={max_sh_degree}")

    # Normalize point cloud by centering it at origin
    original_points = points.copy()
    if normalize_pointcloud:
        # Center the point cloud at the origin
        pointcloud_center = np.mean(points, axis=0)
        points = points - pointcloud_center

        # Also adjust the target mesh centroid relative to the new centered point cloud
        target = target - pointcloud_center

        if debug:
            print(f"  [NORM] Original point cloud center: {pointcloud_center}")
            print(f"  [NORM] Centered point cloud at origin")
            print(f"  [NORM] Adjusted target mesh centroid: {target}")

    # Build k-nearest neighbor graph using cKDTree for efficiency
    tree = cKDTree(points)

    edge_list = []
    displacement_vectors = []

    # For each node, find k nearest neighbors
    for i in range(N):
        # Query k+1 neighbors (including self) and exclude self
        distances, neighbor_indices = tree.query(points[i], k=k + 1)

        # Exclude self (first neighbor is always the point itself)
        neighbor_indices = neighbor_indices[1:]

        for j in neighbor_indices:
            edge_list.append([i, j])
            # Compute displacement vector: target - source
            displacement = points[j] - points[i]
            displacement_vectors.append(displacement)

    # Convert to PyTorch tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).T  # [2, E]
    displacement_vectors = torch.tensor(
        displacement_vectors, dtype=torch.float32
    )  # [E, 3]

    if debug:
        print(f"  [STATS] Graph statistics: {edge_index.size(1)} edges")

    # Compute edge attributes
    if use_spherical_harmonics:
        # Compute spherical harmonics features for displacement vectors
        edge_attr = compute_spherical_harmonics_preprocessing(
            displacement_vectors, max_l=max_sh_degree
        )
        if debug:
            print(f"  [SH] SH edge features shape: {edge_attr.shape}")
            print(
                f"  [SH] SH features per edge: {edge_attr.size(1)} (sum of 2*l+1 for l=0 to {max_sh_degree})"
            )
    else:
        # Use raw displacement vectors
        edge_attr = displacement_vectors
        if debug:
            print(f"  [RAW] Raw displacement vectors shape: {edge_attr.shape}")

    # Use 3D positions directly as node features (PyG best practice)
    node_features = torch.tensor(points, dtype=torch.float32)  # [N, 3]

    # Convert target to PyTorch tensor
    target_tensor = torch.tensor(target, dtype=torch.float32)

    # Create graph data dictionary
    graph_data = {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "point_cloud": torch.tensor(points, dtype=torch.float32),
        "target": target_tensor,
        "num_nodes": N,
        "num_edges": edge_index.size(1),
        "metadata": {
            "k_neighbors": k,
            "use_spherical_harmonics": use_spherical_harmonics,
            "max_sh_degree": max_sh_degree if use_spherical_harmonics else None,
            "edge_attr_dim": edge_attr.size(1),
            "normalized_pointcloud": normalize_pointcloud,
        },
    }

    return graph_data


def process_point_cloud_files(
    cfg=None,
    input_dir=None,
    output_dir=None,
    k_nn=2,
    use_sh=True,
    max_sh_degree=1,
    normalize_pointcloud=True,
):
    """Process all point cloud files and convert them to graph format"""

    # Handle both Hydra config and direct parameters
    if cfg is not None:
        # Using Hydra config
        try:
            from hydra.utils import get_original_cwd

            base_dir = get_original_cwd()
        except:
            # Fallback to current working directory
            base_dir = os.getcwd()

        input_dir = os.path.join(base_dir, cfg.preprocessing.lidar.output_dir)

        # Determine output directory based on spherical harmonics usage
        base_output_dir = cfg.preprocessing.processed_dir
        k_nn = cfg.preprocessing.graph.k_nn
        use_sh = cfg.preprocessing.graph.get("use_spherical_harmonics", False)
        max_sh_degree = cfg.preprocessing.graph.get("max_sh_degree", 1)
        normalize_pointcloud = cfg.preprocessing.graph.get("normalize_pointcloud", True)

        # Modify output directory based on SH usage
        if use_sh:
            output_dir = os.path.join(base_dir, base_output_dir + "_sh")
        else:
            output_dir = os.path.join(base_dir, base_output_dir + "_dv")

        debug = cfg.get("debug", False)
    else:
        # Using direct parameters
        if input_dir is None:
            input_dir = "data/pointclouds"
        if output_dir is None:
            if use_sh:
                output_dir = "data/processed_sh"
            else:
                output_dir = "data/processed_dv"
        debug = False

    print("Loading point cloud data...")

    # Control logging levels based on debug mode
    if not debug:
        # Suppress various library INFO messages when not in debug mode
        logging.getLogger("trimesh").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
    else:
        # Allow all messages in debug mode
        logging.getLogger("trimesh").setLevel(logging.INFO)
        logging.getLogger("torch").setLevel(logging.INFO)

    if debug:
        print(f"[PROCESSING] Processing point clouds to graphs...")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")
        print(f"   k-NN: {k_nn}")
        print(f"   Spherical Harmonics: {use_sh}")
        print(f"   Normalize Point Clouds: {normalize_pointcloud}")
        if use_sh:
            print(f"   Max SH degree: {max_sh_degree}")
            print(f"   Note: Using SH-specific output directory")
        else:
            print(f"   Note: Using raw displacement vectors")

    os.makedirs(output_dir, exist_ok=True)

    # No longer create train/test subdirectories - save all files directly
    processed_count = 0

    # Track files for metadata
    file_metadata = []

    # Get list of object directories
    obj_dirs = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ]

    print("Converting point clouds to graphs...")

    # Process each object directory
    for obj_name in tqdm(obj_dirs, desc="Processing objects"):
        obj_dir = os.path.join(input_dir, obj_name)

        if debug:
            print(f"[OBJ] Processing {obj_name}...")

        # Load mesh centroid (target) in the new format
        mesh_centroid_file = os.path.join(obj_dir, "mesh_centroid.npy")
        if not os.path.exists(mesh_centroid_file):
            if debug:
                print(f"  [SKIP] Skipping {obj_name}: no mesh_centroid.npy")
            continue

        mesh_centroid = np.load(mesh_centroid_file)

        # Find point cloud files in the new format
        pointcloud_files = []

        # Check for main pointcloud file
        main_pc_file = os.path.join(obj_dir, "pointcloud.npy")
        if os.path.exists(main_pc_file):
            pointcloud_files.append(main_pc_file)

        # Check for sample-based pointcloud files (if multiple samples exist)
        sample_pc_files = glob.glob(os.path.join(obj_dir, "pointcloud_sample*.npy"))
        pointcloud_files.extend(sample_pc_files)

        if not pointcloud_files:
            if debug:
                print(f"  [SKIP] Skipping {obj_name}: no pointcloud files found")
            continue

        for pc_file in tqdm(
            pointcloud_files,
            desc=f"Point clouds for {obj_name}",
            leave=False,
            disable=not debug or len(pointcloud_files) == 1,
        ):
            # Load point cloud
            points = np.load(pc_file)

            # Convert to graph
            graph_data = build_graph_from_pointcloud(
                points,
                mesh_centroid,
                k_nn,
                use_sh,
                max_sh_degree,
                debug=debug,
                normalize_pointcloud=normalize_pointcloud,
            )

            # Determine output filename - save directly in main directory
            pc_basename = os.path.basename(pc_file).replace(".npy", "")

            # Handle different pointcloud file naming conventions
            if pc_basename == "pointcloud":
                # Main pointcloud file
                filename = f"{obj_name}.pt"
            elif pc_basename.startswith("pointcloud_sample"):
                # Sample-based pointcloud file
                sample_part = pc_basename.replace("pointcloud_sample", "sample")
                filename = f"{obj_name}_{sample_part}.pt"
            else:
                # Fallback for other naming patterns
                filename = f"{obj_name}_{pc_basename}.pt"

            output_path = os.path.join(output_dir, filename)

            # Save graph data
            torch.save(graph_data, output_path)
            processed_count += 1

            # Add to metadata
            file_metadata.append(
                {
                    "filename": filename,
                    "object_name": obj_name,
                    "num_nodes": graph_data["num_nodes"],
                    "num_edges": graph_data["num_edges"],
                    "edge_attr_dim": graph_data["edge_attr"].size(1),
                    "use_spherical_harmonics": use_sh,
                    "max_sh_degree": max_sh_degree if use_sh else None,
                    "k_neighbors": k_nn,
                    "normalized_pointcloud": normalize_pointcloud,
                }
            )

            if debug:
                print(f"  [SAVE] Saved: {filename}")

    # Save metadata file
    import json

    metadata_file = os.path.join(output_dir, "dataset_metadata.json")
    metadata = {
        "total_files": processed_count,
        "edge_attr_dim": file_metadata[0]["edge_attr_dim"] if file_metadata else None,
        "use_spherical_harmonics": use_sh,
        "max_sh_degree": max_sh_degree if use_sh else None,
        "k_neighbors": k_nn,
        "normalized_pointcloud": normalize_pointcloud,
        "files": file_metadata,
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Processing complete! Converted {processed_count} point clouds to graphs")
    print(f"Saved metadata to: {metadata_file}")
    if normalize_pointcloud:
        print(
            "Point clouds normalized: Each centered at origin, preserving relative structure"
        )
    else:
        print("Point clouds not normalized: Using absolute coordinates")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.1", config_path="../../configs", config_name="config")
    def main(cfg: DictConfig):
        process_point_cloud_files(cfg)

    main()
