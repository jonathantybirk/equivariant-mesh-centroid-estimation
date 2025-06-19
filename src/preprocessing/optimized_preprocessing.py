"""
Optimized preprocessing pipeline for mesh-to-graph conversion

Key optimizations:
1. Multiprocessing for parallel mesh processing
2. Batched I/O operations
3. Memory-efficient data structures
4. Vectorized operations where possible
"""

import os
import numpy as np
from hydra.utils import get_original_cwd
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
import glob
import torch
from scipy.spatial import cKDTree
from e3nn.o3 import spherical_harmonics

from src.utils import geometry, mesh, lidar
from src.preprocessing.pointcloud_to_graph import (
    compute_spherical_harmonics_preprocessing,
    build_graph_from_pointcloud,
)


def random_rotation_matrix_np():
    """Generate a random 3D rotation matrix using Rodrigues' rotation formula."""
    # Random axis (normalized)
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    # Random angle between 0 and 2π
    angle = np.random.rand() * 2 * np.pi

    # Rodrigues' rotation formula
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R


def apply_random_rotation_to_mesh(mesh_obj):
    """
    Apply a random 3D rotation to the mesh vertices.
    This removes orientation bias from the dataset.

    Args:
        mesh_obj: Trimesh object

    Returns:
        mesh_obj: The same mesh object with rotated vertices (modified in-place)
    """
    # Generate random rotation matrix
    R = random_rotation_matrix_np()

    # Apply rotation to mesh vertices
    mesh_obj.vertices = np.dot(mesh_obj.vertices, R.T)

    return mesh_obj


def process_cameras_vectorized(m, radius, cfg):
    """
    Vectorized camera processing for a single mesh
    """
    num_cameras = cfg.preprocessing.lidar.num_cameras
    camera_positions = []
    target_points = []
    all_points = []

    # Try to generate all camera positions first
    for i in range(num_cameras):
        try:
            target_point, camera_pos = mesh.sample_visible_target_and_camera(m, radius)
            camera_positions.append(camera_pos)
            target_points.append(target_point)
        except RuntimeError:
            continue

    if not camera_positions:
        return {"success": False}

    # Process cameras in parallel if possible
    for i, (camera_pos, target_point) in enumerate(
        zip(camera_positions, target_points)
    ):
        try:
            pts = lidar.simulate_lidar(
                m,
                camera_pos,
                target_point,
                h_fov_deg=cfg.preprocessing.lidar.h_fov_deg,
                v_fov_deg=cfg.preprocessing.lidar.v_fov_deg,
                h_steps=cfg.preprocessing.lidar.h_steps,
                v_steps=cfg.preprocessing.lidar.v_steps,
                max_distance=radius,
                include_misses=cfg.preprocessing.lidar.include_missed_rays,
            )
            all_points.append(pts)
        except Exception:
            continue

    if all_points:
        combined_points = np.concatenate(all_points, axis=0)
        return {
            "success": True,
            "combined_points": combined_points,
            "camera_positions": np.array(camera_positions),
        }
    else:
        return {"success": False}


def batch_save_pointclouds(batch_results_list, cfg):
    """
    Efficiently save multiple batches of point cloud data
    """
    original_cwd = get_original_cwd()

    for batch_results in batch_results_list:
        for object_data in batch_results:
            obj_name = object_data["obj_name"]
            obj_output_dir = os.path.join(
                original_cwd, cfg.preprocessing.lidar.output_dir, obj_name
            )
            os.makedirs(obj_output_dir, exist_ok=True)

            # Save center of mass
            np.save(
                os.path.join(obj_output_dir, "center_of_mass.npy"),
                object_data["center_of_mass"],
            )

            # Save samples
            for sample_data in object_data["samples"]:
                sample_idx = sample_data["sample_idx"]
                num_samples = len(object_data["samples"])

                if num_samples > 1:
                    pc_file = f"pointcloud_combined_sample{sample_idx+1}.npy"
                    cam_file = f"camera_positions_sample{sample_idx+1}.npy"
                else:
                    pc_file = "pointcloud_combined.npy"
                    cam_file = "camera_positions.npy"

                np.save(
                    os.path.join(obj_output_dir, pc_file),
                    sample_data["combined_points"],
                )
                np.save(
                    os.path.join(obj_output_dir, cam_file),
                    sample_data["camera_positions"],
                )


def process_pointclouds_to_graphs_optimized(cfg):
    """
    Optimized point cloud to graph conversion with parallel processing
    """
    # Setup directories
    try:
        base_dir = get_original_cwd()
    except:
        base_dir = os.getcwd()

    input_dir = os.path.join(base_dir, cfg.preprocessing.lidar.output_dir)
    k_nn = cfg.preprocessing.graph.k_nn
    use_sh = cfg.preprocessing.graph.get("use_spherical_harmonics", False)
    max_sh_degree = cfg.preprocessing.graph.get("max_sh_degree", 1)
    normalize_pointcloud = cfg.preprocessing.graph.get("normalize_pointcloud", True)
    save_enabled = cfg.preprocessing.lidar.get("save", True)

    # Output directory based on edge type
    base_output_dir = cfg.preprocessing.processed_dir
    if use_sh:
        output_dir = os.path.join(
            base_dir, base_output_dir + "_sh" + str(max_sh_degree)
        )
    else:
        output_dir = os.path.join(base_dir, base_output_dir + "_dv")

    if save_enabled:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Converting point clouds to graphs...")
    print(f"   Input: {input_dir}")
    if save_enabled:
        print(f"   Output: {output_dir}")
    else:
        print(f"   Output: [No-save mode - graphs will not be saved]")
    print(
        f"   Edge type: {'Spherical Harmonics' if use_sh else 'Displacement Vectors'}"
    )

    # Get all point cloud files
    obj_dirs = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ]

    # Prepare arguments for parallel processing
    args_list = []
    for obj_name in obj_dirs:
        obj_dir = os.path.join(input_dir, obj_name)
        target_file = os.path.join(obj_dir, "mesh_centroid.npy")

        if not os.path.exists(target_file):
            continue

        pointcloud_files = glob.glob(os.path.join(obj_dir, "pointcloud.npy"))
        if not pointcloud_files:
            continue

        center_of_mass = np.load(target_file)

        for pc_file in pointcloud_files:
            args_list.append(
                (
                    obj_name,
                    pc_file,
                    center_of_mass,
                    k_nn,
                    use_sh,
                    max_sh_degree,
                    normalize_pointcloud,
                    output_dir,
                    save_enabled,
                )
            )

    # Process in parallel
    n_processes = max(min(cpu_count() - 1, len(args_list)), 1)
    print(f"Processing {len(args_list)} point clouds using {n_processes} processes...")

    with Pool(processes=n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_pointcloud_to_graph, args_list),
                total=len(args_list),
                desc="Converting to graphs",
            )
        )

    # Count successful conversions
    successful = sum(1 for r in results if r is not None)
    if save_enabled:
        print(
            f"Successfully converted {successful}/{len(args_list)} point clouds to graphs"
        )
    else:
        print(
            f"Successfully processed {successful}/{len(args_list)} point clouds (no-save mode)"
        )


def process_single_pointcloud_to_graph(args):
    """
    Convert a single point cloud to graph format
    """
    (
        obj_name,
        pc_file,
        center_of_mass,
        k_nn,
        use_sh,
        max_sh_degree,
        normalize_pointcloud,
        output_dir,
        save_enabled,
    ) = args

    try:
        # Load point cloud
        points = np.load(pc_file)

        # Convert to graph
        graph_data = build_graph_from_pointcloud(
            points,
            center_of_mass,
            k_nn,
            use_sh,
            max_sh_degree,
            debug=False,
            normalize_pointcloud=normalize_pointcloud,
        )

        # Generate output filename
        filename = f"{obj_name}_{os.path.basename(pc_file).replace('.npy', '')}.pt"
        clean_filename = filename.replace("pointcloud_combined_", "").replace(
            "pointcloud_combined", ""
        )

        # Only save if saving is enabled
        if save_enabled:
            output_path = os.path.join(output_dir, clean_filename)
            torch.save(graph_data, output_path)

        return clean_filename

    except Exception as e:
        print(f"Error processing {pc_file}: {e}")
        return None


def optimized_preprocessing_pipeline(cfg):
    """
    Main optimized preprocessing pipeline
    """
    print("=== OPTIMIZED PREPROCESSING PIPELINE ===")

    # Step 1: Optimized mesh to point cloud conversion (with visualization support)
    if not cfg.get("skip_pointcloud", False) and False:
        print("\nStep 1: Converting meshes to point clouds...")
        # Use the mesh_to_pointcloud_optimized module which has better visualization support
        from src.preprocessing.mesh_to_pointcloud import process_all_meshes_optimized

        process_all_meshes_optimized(cfg)
    else:
        print("\nStep 1: Skipping point cloud generation...")

    # Step 2: Optimized point cloud to graph conversion
    if not cfg.get("skip_graph", False):
        print("\nStep 2: Converting point clouds to graphs...")
        process_pointclouds_to_graphs_optimized(cfg)
    else:
        print("\nStep 2: Skipping graph generation...")

    print("\n=== PREPROCESSING COMPLETE ===")


# Main entry point
def run_optimized_preprocessing(cfg):
    """
    Run the complete optimized preprocessing pipeline
    """
    optimized_preprocessing_pipeline(cfg)
