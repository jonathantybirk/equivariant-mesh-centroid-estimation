import sys
import os
from pathlib import Path

# Add the project root to Python path so we can import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
import argparse
from omegaconf import DictConfig, OmegaConf
from src.preprocessing.mesh_to_pointcloud import process_all_meshes_optimized
from src.preprocessing.pointcloud_to_graph import process_point_cloud_files
import numpy as np
import glob
import torch
from collections import defaultdict, deque


def check_connectivity_from_edges(edge_index, num_nodes):
    """Check if graph is connected using BFS from edge_index tensor"""
    if num_nodes == 0:
        return True, 1, []

    # Convert edge_index to adjacency list
    adj = defaultdict(list)
    if hasattr(edge_index, "numpy"):
        edges = edge_index.numpy()
    else:
        edges = edge_index

    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        adj[u].append(v)
        adj[v].append(u)  # Undirected graph

    visited = set()
    components = []

    for start_node in range(num_nodes):
        if start_node not in visited:
            # BFS to find connected component
            component = []
            queue = deque([start_node])
            visited.add(start_node)

            while queue:
                node = queue.popleft()
                component.append(node)

                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

    is_connected = len(components) == 1
    return is_connected, len(components), [len(comp) for comp in components]


def analyze_graph_connectivity(processed_dir, debug=False):
    """Analyze connectivity of processed graph files"""
    if not os.path.exists(processed_dir):
        return None

    pt_files = glob.glob(os.path.join(processed_dir, "*.pt"))
    if not pt_files:
        return None

    results = []
    errors = []

    # Sample up to 50 files for analysis
    sample_files = pt_files[:50] if len(pt_files) > 50 else pt_files

    for file_path in sample_files:
        try:
            data = torch.load(file_path, map_location="cpu", weights_only=False)

            if isinstance(data, dict) and "edge_index" in data:
                edge_index = data["edge_index"]
                num_nodes = data.get("num_nodes", int(edge_index.max().item()) + 1)

                # Check connectivity
                is_connected, num_components, component_sizes = (
                    check_connectivity_from_edges(edge_index, num_nodes)
                )

                results.append(
                    {
                        "file": os.path.basename(file_path),
                        "num_nodes": num_nodes,
                        "num_edges": edge_index.shape[1],
                        "is_connected": is_connected,
                        "num_components": num_components,
                        "component_sizes": component_sizes,
                        "edge_attr_dim": (
                            data.get("edge_attr", torch.tensor([])).shape[-1]
                            if "edge_attr" in data
                            else 0
                        ),
                    }
                )

                if debug:
                    status = (
                        "✓ Connected"
                        if is_connected
                        else f"✗ {num_components} components"
                    )
                    print(
                        f"   {os.path.basename(file_path)[:40]:40} | {num_nodes:3d} nodes | {edge_index.shape[1]:4d} edges | {status}"
                    )

            else:
                errors.append(
                    f"Invalid data structure in {os.path.basename(file_path)}"
                )

        except Exception as e:
            errors.append(f"Error loading {os.path.basename(file_path)}: {str(e)}")

    return {
        "total_files": len(pt_files),
        "analyzed_files": len(sample_files),
        "results": results,
        "errors": errors,
    }


def compute_preprocessing_statistics(cfg: DictConfig):
    """
    Compute and display comprehensive statistics about the preprocessed data
    """
    debug = cfg.get("debug", False)

    print("Camera positions statistics:")
    print("=" * 50)
    file_list_dv = list(
        Path(cfg.preprocessing.lidar.output_dir).glob("**/camera_positions.npy")
    )

    camera_positions = []
    for file in file_list_dv:
        _camera_positions = np.load(file)[1:3]
        camera_positions.extend(_camera_positions)
    camera_positions = np.array(camera_positions)

    from scipy import stats

    # Perform t-tests for each coordinate
    print("T-tests for camera positions coordinates:")
    print("=" * 50)

    for i, coord_name in enumerate(["x", "y", "z"]):
        # Extract the coordinate values
        coord_values = camera_positions[:, i]

        # Perform one-sample t-test against null hypothesis (mean = 0)
        t_stat, p_value = stats.ttest_1samp(coord_values, 0)

        print(f"\n{coord_name.upper()}-coordinate:")
        print(f"  Mean: {np.mean(coord_values):.4f}")
        print(f"  Std:  {np.std(coord_values):.4f}")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

    print("\n" + "=" * 50)
    print("Note: P-value < 0.05 indicates the mean is significantly different from 0")

    if debug:
        print("\n" + "=" * 60)
        print("COMPUTING PREPROCESSING STATISTICS")
        print("=" * 60)

    # Determine input directory based on configuration
    try:
        from hydra.utils import get_original_cwd

        base_dir = get_original_cwd()
    except:
        base_dir = os.getcwd()

    input_dir = os.path.join(base_dir, cfg.preprocessing.lidar.output_dir)

    if not os.path.exists(input_dir):
        print("ERROR: Input directory not found for statistics computation")
        return

    # Collect all mesh centroid data
    centroid_data = []
    object_names = []

    # Get list of object directories
    obj_dirs = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ]

    if debug:
        print(f"Scanning {len(obj_dirs)} object directories...")

    for obj_name in obj_dirs:
        obj_dir = os.path.join(input_dir, obj_name)
        centroid_file = os.path.join(obj_dir, "mesh_centroid.npy")

        if os.path.exists(centroid_file):
            try:
                centroid = np.load(centroid_file)
                centroid_data.append(centroid)
                object_names.append(obj_name)
                if debug:
                    print(
                        f"   SUCCESS {obj_name}: Mesh Centroid = [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]"
                    )
            except Exception as e:
                if debug:
                    print(f"   FAILED to load {obj_name}: {e}")

    if len(centroid_data) == 0:
        print("ERROR: No mesh centroid data found!")
        return

    # Convert to numpy array for analysis
    centroid_array = np.array(centroid_data)  # Shape: [N, 3]

    print(f"\nMESH CENTROID STATISTICS")
    print("=" * 60)
    print(f"Total objects analyzed: {len(centroid_data)}")
    print(f"Data directory: {input_dir}")

    # Basic statistics
    print(f"\nMESH CENTROID COORDINATES")
    print("-" * 40)
    for i, axis in enumerate(["X", "Y", "Z"]):
        axis_data = centroid_array[:, i]
        print(f"   {axis}-axis:")
        print(f"      Mean: {np.mean(axis_data):8.4f}")
        print(f"      Std:  {np.std(axis_data):8.4f}")
        print(f"      Min:  {np.min(axis_data):8.4f}")
        print(f"      Max:  {np.max(axis_data):8.4f}")
        print(f"      Range:{np.max(axis_data) - np.min(axis_data):8.4f}")

    # Average centroid displacement from origin
    mean_centroid = np.mean(centroid_array, axis=0)
    print(f"\nAVERAGE MESH CENTROID DISPLACEMENT FROM ORIGIN")
    print("-" * 40)
    print(
        f"   Average centroid vector: [{mean_centroid[0]:8.4f}, {mean_centroid[1]:8.4f}, {mean_centroid[2]:8.4f}]"
    )

    # Norm of average displacement
    mean_centroid_norm = np.linalg.norm(mean_centroid)
    print(f"   Average displacement norm: {mean_centroid_norm:.4f}")

    # Individual centroid norms
    centroid_norms = np.linalg.norm(centroid_array, axis=1)
    print(f"\nMESH CENTROID DISTANCE FROM ORIGIN")
    print("-" * 40)
    print(f"   Mean distance: {np.mean(centroid_norms):8.4f}")
    print(f"   Std distance:  {np.std(centroid_norms):8.4f}")
    print(f"   Min distance:  {np.min(centroid_norms):8.4f}")
    print(f"   Max distance:  {np.max(centroid_norms):8.4f}")
    print(f"   Median distance: {np.median(centroid_norms):6.4f}")

    # Distribution analysis
    print(f"\nDISTRIBUTION ANALYSIS")
    print("-" * 40)

    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("   Distance percentiles:")
    for p in percentiles:
        value = np.percentile(centroid_norms, p)
        print(f"      {p:2d}th: {value:8.4f}")

    # Objects closest and farthest from origin
    min_idx = np.argmin(centroid_norms)
    max_idx = np.argmax(centroid_norms)

    print(f"\nEXTREME CASES")
    print("-" * 40)
    print(f"   Closest to origin:")
    print(f"      Object: {object_names[min_idx]}")
    print(
        f"      Centroid: [{centroid_array[min_idx, 0]:.4f}, {centroid_array[min_idx, 1]:.4f}, {centroid_array[min_idx, 2]:.4f}]"
    )
    print(f"      Distance: {centroid_norms[min_idx]:.4f}")

    print(f"   Farthest from origin:")
    print(f"      Object: {object_names[max_idx]}")
    print(
        f"      Centroid: [{centroid_array[max_idx, 0]:.4f}, {centroid_array[max_idx, 1]:.4f}, {centroid_array[max_idx, 2]:.4f}]"
    )
    print(f"      Distance: {centroid_norms[max_idx]:.4f}")

    # Coordinate system bias analysis
    print(f"\nCOORDINATE SYSTEM ANALYSIS")
    print("-" * 40)

    # Check if there's bias towards positive/negative directions
    for i, axis in enumerate(["X", "Y", "Z"]):
        axis_data = centroid_array[:, i]
        positive_count = np.sum(axis_data > 0)
        negative_count = np.sum(axis_data < 0)
        zero_count = np.sum(axis_data == 0)

        print(f"   {axis}-axis distribution:")
        print(
            f"      Positive: {positive_count:3d} ({100*positive_count/len(axis_data):5.1f}%)"
        )
        print(
            f"      Negative: {negative_count:3d} ({100*negative_count/len(axis_data):5.1f}%)"
        )
        if zero_count > 0:
            print(
                f"      Zero:     {zero_count:3d} ({100*zero_count/len(axis_data):5.1f}%)"
            )

    # Covariance analysis
    print(f"\nCOVARIANCE ANALYSIS")
    print("-" * 40)
    cov_matrix = np.cov(centroid_array.T)
    print("   Covariance matrix:")
    print("        X        Y        Z")
    for i, axis in enumerate(["X", "Y", "Z"]):
        row_str = f"   {axis}: "
        for j in range(3):
            row_str += f"{cov_matrix[i, j]:8.4f} "
        print(row_str)

    # Correlation analysis
    print(f"\nCORRELATION ANALYSIS")
    print("-" * 40)
    corr_matrix = np.corrcoef(centroid_array.T)
    print("   Correlation matrix:")
    print("        X        Y        Z")
    for i, axis in enumerate(["X", "Y", "Z"]):
        row_str = f"   {axis}: "
        for j in range(3):
            row_str += f"{corr_matrix[i, j]:8.4f} "
        print(row_str)

    # GRAPH CONNECTIVITY ANALYSIS
    print(f"\nGRAPH CONNECTIVITY ANALYSIS")
    print("=" * 60)

    # Determine processed graph directory
    use_sh = cfg.preprocessing.graph.get("use_spherical_harmonics", False)
    edge_sh_degree = cfg.preprocessing.graph.get("edge_sh_degree", 1)
    base_processed_dir = cfg.preprocessing.processed_dir

    if use_sh:
        processed_dir = os.path.join(
            base_dir, base_processed_dir + "_sh" + str(edge_sh_degree)
        )
    else:
        processed_dir = os.path.join(base_dir, base_processed_dir + "_dv")

    print(f"Analyzing graphs in: {processed_dir}")

    if debug:
        print("\nDetailed connectivity analysis:")

    connectivity_data = analyze_graph_connectivity(processed_dir, debug=debug)

    if connectivity_data is None:
        print("ERROR: No processed graph files found for connectivity analysis")
    else:
        results = connectivity_data["results"]
        errors = connectivity_data["errors"]

        print(f"\nCONNECTIVITY STATISTICS")
        print("-" * 40)
        print(f"   Total graph files: {connectivity_data['total_files']}")
        print(f"   Files analyzed: {connectivity_data['analyzed_files']}")
        print(f"   Successfully loaded: {len(results)}")
        print(f"   Errors: {len(errors)}")

        if results:
            connected_count = sum(1 for r in results if r["is_connected"])
            disconnected_count = len(results) - connected_count

            print(
                f"\n   Connected graphs: {connected_count} ({100*connected_count/len(results):.1f}%)"
            )
            print(
                f"   Disconnected graphs: {disconnected_count} ({100*disconnected_count/len(results):.1f}%)"
            )

            # Node/edge statistics
            node_counts = [r["num_nodes"] for r in results]
            edge_counts = [r["num_edges"] for r in results]
            component_counts = [r["num_components"] for r in results]

            print(f"\nGRAPH SIZE STATISTICS")
            print("-" * 40)
            print(f"   Nodes per graph:")
            print(f"      Mean: {np.mean(node_counts):8.1f}")
            print(f"      Std:  {np.std(node_counts):8.1f}")
            print(f"      Min:  {np.min(node_counts):8.0f}")
            print(f"      Max:  {np.max(node_counts):8.0f}")

            print(f"   Edges per graph:")
            print(f"      Mean: {np.mean(edge_counts):8.1f}")
            print(f"      Std:  {np.std(edge_counts):8.1f}")
            print(f"      Min:  {np.min(edge_counts):8.0f}")
            print(f"      Max:  {np.max(edge_counts):8.0f}")

            print(
                f"   Avg edges per node: {np.mean(edge_counts)/np.mean(node_counts):8.1f}"
            )

            # Edge features
            edge_dims = [r["edge_attr_dim"] for r in results if r["edge_attr_dim"] > 0]
            if edge_dims:
                print(
                    f"   Edge feature dimension: {edge_dims[0]} (spherical harmonics)"
                )

            print(f"   Components per graph:")
            print(f"      Mean: {np.mean(component_counts):8.1f}")
            print(f"      Max:  {np.max(component_counts):8.0f}")

            # Show disconnected examples
            disconnected = [r for r in results if not r["is_connected"]]
            if disconnected:
                print(f"\nDISCONNECTED GRAPH EXAMPLES")
                print("-" * 40)
                for r in disconnected[:5]:
                    print(
                        f"   {r['file']}: {r['num_components']} components, sizes: {r['component_sizes']}"
                    )

        # Show errors
        if errors:
            print(f"\nERRORS")
            print("-" * 40)
            for error in errors[:5]:
                print(f"   {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more errors")

    print(f"\n" + "=" * 60)
    print("PREPROCESSING STATISTICS COMPLETE")
    print("=" * 60)


@hydra.main(config_path="../configs", config_name="preprocess_only", version_base=None)
def main(cfg: DictConfig):
    """Main preprocessing pipeline"""

    # Get skip flags from config
    skip_pointcloud = cfg.get("skip_pointcloud", False)
    skip_graph = cfg.get("skip_graph", False)
    no_save = cfg.get("no_save", False)

    # Override save settings if no_save is enabled
    if no_save:
        cfg.preprocessing.lidar.save = False
        print("No-save mode enabled - data will not be saved to disk")

    debug = cfg.get("debug", False)

    if debug:
        print("[DEBUG] Debug mode enabled")
        print("Configuration:")
        print(f"   Mesh directory: {cfg.preprocessing.lidar.mesh_dir}")
        print(f"   Output directory: {cfg.preprocessing.lidar.output_dir}")
        print(f"   Processed directory: {cfg.preprocessing.processed_dir}")
        print(f"   Number of cameras: {cfg.preprocessing.lidar.num_cameras}")
        print(f"   Number of samples: {cfg.preprocessing.lidar.get('num_samples', 1)}")
        print(f"   k-NN neighbors: {cfg.preprocessing.graph.k_nn}")
        print(
            f"   Spherical harmonics: {cfg.preprocessing.graph.get('use_spherical_harmonics', False)}"
        )
        if cfg.preprocessing.lidar.get("visualize_first_n", 0) > 0:
            print(
                f"   Visualizing first: {cfg.preprocessing.lidar.visualize_first_n} pointclouds"
            )
        print(f"   Save enabled: {cfg.preprocessing.lidar.save}")
        print("")
        if skip_pointcloud:
            print("   Skipping point cloud generation")
        if skip_graph:
            print("   Skipping graph generation")
        if no_save:
            print("   No-save mode enabled")
        print("")

    print("Starting preprocessing pipeline...")

    # Use optimized preprocessing by default
    use_optimized = cfg.get("use_optimized_preprocessing", True)

    if use_optimized:
        print("Using optimized preprocessing pipeline...")
        from src.preprocessing.optimized_preprocessing import (
            run_optimized_preprocessing,
        )

        run_optimized_preprocessing(cfg)
    else:
        print("Using original preprocessing pipeline...")
        # Step 1: Generate and save point clouds (if not skipped)
        if not skip_pointcloud:
            process_all_meshes_optimized(cfg)
        else:
            print("Skipping point cloud generation step...")

        # Step 2: Convert saved point clouds to graph data (if not skipped)
        if not skip_graph:
            process_point_cloud_files(cfg)
        else:
            print("Skipping graph generation step...")

    print("Preprocessing pipeline complete!")

    # Step 3: Compute and display comprehensive statistics
    compute_preprocessing_statistics(cfg)


if __name__ == "__main__":
    main()
