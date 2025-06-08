import sys
import os
from pathlib import Path

# Add the project root to Python path so we can import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig
from src.preprocessing.mesh_to_pointcloud import process_all_meshes
from src.preprocessing.pointcloud_to_graph import process_point_cloud_files
import numpy as np
import glob


def compute_preprocessing_statistics(cfg: DictConfig):
    """
    Compute and display comprehensive statistics about the preprocessed data
    """
    debug = cfg.get("debug", False)

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

    # Collect all center of mass data
    com_data = []
    object_names = []

    # Get list of object directories
    obj_dirs = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ]

    if debug:
        print(f"Scanning {len(obj_dirs)} object directories...")

    for obj_name in obj_dirs:
        obj_dir = os.path.join(input_dir, obj_name)
        com_file = os.path.join(obj_dir, "center_of_mass.npy")

        if os.path.exists(com_file):
            try:
                com = np.load(com_file)
                com_data.append(com)
                object_names.append(obj_name)
                if debug:
                    print(
                        f"   SUCCESS {obj_name}: COM = [{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}]"
                    )
            except Exception as e:
                if debug:
                    print(f"   FAILED to load {obj_name}: {e}")

    if len(com_data) == 0:
        print("ERROR: No center of mass data found!")
        return

    # Convert to numpy array for analysis
    com_array = np.array(com_data)  # Shape: [N, 3]

    print(f"\nCENTER OF MASS STATISTICS")
    print("=" * 60)
    print(f"Total objects analyzed: {len(com_data)}")
    print(f"Data directory: {input_dir}")

    # Basic statistics
    print(f"\nCENTER OF MASS COORDINATES")
    print("-" * 40)
    for i, axis in enumerate(["X", "Y", "Z"]):
        axis_data = com_array[:, i]
        print(f"   {axis}-axis:")
        print(f"      Mean: {np.mean(axis_data):8.4f}")
        print(f"      Std:  {np.std(axis_data):8.4f}")
        print(f"      Min:  {np.min(axis_data):8.4f}")
        print(f"      Max:  {np.max(axis_data):8.4f}")
        print(f"      Range:{np.max(axis_data) - np.min(axis_data):8.4f}")

    # Average COM displacement from origin
    mean_com = np.mean(com_array, axis=0)
    print(f"\nAVERAGE COM DISPLACEMENT FROM ORIGIN")
    print("-" * 40)
    print(
        f"   Average COM vector: [{mean_com[0]:8.4f}, {mean_com[1]:8.4f}, {mean_com[2]:8.4f}]"
    )

    # Norm of average displacement
    mean_com_norm = np.linalg.norm(mean_com)
    print(f"   Average displacement norm: {mean_com_norm:.4f}")

    # Individual COM norms
    com_norms = np.linalg.norm(com_array, axis=1)
    print(f"\nCOM DISTANCE FROM ORIGIN")
    print("-" * 40)
    print(f"   Mean distance: {np.mean(com_norms):8.4f}")
    print(f"   Std distance:  {np.std(com_norms):8.4f}")
    print(f"   Min distance:  {np.min(com_norms):8.4f}")
    print(f"   Max distance:  {np.max(com_norms):8.4f}")
    print(f"   Median distance: {np.median(com_norms):6.4f}")

    # Distribution analysis
    print(f"\nDISTRIBUTION ANALYSIS")
    print("-" * 40)

    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("   Distance percentiles:")
    for p in percentiles:
        value = np.percentile(com_norms, p)
        print(f"      {p:2d}th: {value:8.4f}")

    # Objects closest and farthest from origin
    min_idx = np.argmin(com_norms)
    max_idx = np.argmax(com_norms)

    print(f"\nEXTREME CASES")
    print("-" * 40)
    print(f"   Closest to origin:")
    print(f"      Object: {object_names[min_idx]}")
    print(
        f"      COM: [{com_array[min_idx, 0]:.4f}, {com_array[min_idx, 1]:.4f}, {com_array[min_idx, 2]:.4f}]"
    )
    print(f"      Distance: {com_norms[min_idx]:.4f}")

    print(f"   Farthest from origin:")
    print(f"      Object: {object_names[max_idx]}")
    print(
        f"      COM: [{com_array[max_idx, 0]:.4f}, {com_array[max_idx, 1]:.4f}, {com_array[max_idx, 2]:.4f}]"
    )
    print(f"      Distance: {com_norms[max_idx]:.4f}")

    # Coordinate system bias analysis
    print(f"\nCOORDINATE SYSTEM ANALYSIS")
    print("-" * 40)

    # Check if there's bias towards positive/negative directions
    for i, axis in enumerate(["X", "Y", "Z"]):
        axis_data = com_array[:, i]
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
    cov_matrix = np.cov(com_array.T)
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
    corr_matrix = np.corrcoef(com_array.T)
    print("   Correlation matrix:")
    print("        X        Y        Z")
    for i, axis in enumerate(["X", "Y", "Z"]):
        row_str = f"   {axis}: "
        for j in range(3):
            row_str += f"{corr_matrix[i, j]:8.4f} "
        print(row_str)

    print(f"\n" + "=" * 60)
    print("PREPROCESSING STATISTICS COMPLETE")
    print("=" * 60)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main preprocessing pipeline"""
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
        # Step 1: Generate and save point clouds
        process_all_meshes(cfg)

        # Step 2: Convert saved point clouds to graph data
        process_point_cloud_files(cfg)

    print("Preprocessing pipeline complete!")

    # Step 3: Compute and display comprehensive statistics
    compute_preprocessing_statistics(cfg)


if __name__ == "__main__":
    main()
