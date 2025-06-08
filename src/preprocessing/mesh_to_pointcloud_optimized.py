import os
import numpy as np
import matplotlib.pyplot as plt
from hydra.utils import get_original_cwd
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle

from src.utils import geometry, mesh, lidar


def process_mesh_optimized(args):
    """
    Optimized mesh processing function for multiprocessing
    Returns data instead of saving immediately
    """
    mesh_path, cfg, obj_idx, total_objs = args
    obj_name = os.path.splitext(os.path.basename(mesh_path))[0]

    try:
        # Load mesh
        m = mesh.load_mesh(mesh_path, debug=False)
        com = mesh.compute_center_of_mass(m)

        # Determine camera distance
        radius = (
            m.bounding_sphere.primitive.radius
            * cfg.preprocessing.lidar.max_lidar_distance_factor
        )

        num_samples = cfg.preprocessing.lidar.get("num_samples", 1)

        # Collect all data for this object
        object_data = {
            "obj_name": obj_name,
            "center_of_mass": com,
            "radius": radius,
            "samples": [],
        }

        # Process all samples for this object
        for sample_idx in range(num_samples):
            sample_seed = cfg.preprocessing.lidar.seed + sample_idx
            np.random.seed(sample_seed)

            camera_positions = []
            target_points = []
            all_points_by_camera = []

            # Process all cameras for this sample
            for i in range(cfg.preprocessing.lidar.num_cameras):
                try:
                    target_point, camera_pos = mesh.sample_visible_target_and_camera(
                        m, radius
                    )
                    camera_positions.append(camera_pos)
                    target_points.append(target_point)

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
                    all_points_by_camera.append(pts)

                except RuntimeError:
                    continue  # Skip failed cameras

            if all_points_by_camera:
                all_points = np.concatenate(all_points_by_camera, axis=0)

                sample_data = {
                    "sample_idx": sample_idx,
                    "combined_points": all_points,
                    "camera_positions": np.array(camera_positions),
                    "individual_cameras": (
                        all_points_by_camera if cfg.preprocessing.lidar.save else None
                    ),
                }
                object_data["samples"].append(sample_data)

        return object_data, obj_idx, total_objs

    except Exception as e:
        print(f"Error processing {obj_name}: {e}")
        return None, obj_idx, total_objs


def save_object_data_batch(object_data_list, cfg):
    """
    Batch save multiple objects' data at once
    """
    original_cwd = get_original_cwd()

    for object_data in object_data_list:
        if object_data is None:
            continue

        obj_name = object_data["obj_name"]
        obj_output_dir = os.path.join(
            original_cwd, cfg.preprocessing.lidar.output_dir, obj_name
        )
        os.makedirs(obj_output_dir, exist_ok=True)

        # Save center of mass once per object
        np.save(
            os.path.join(obj_output_dir, "center_of_mass.npy"),
            object_data["center_of_mass"],
        )

        # Save all samples for this object
        for sample_data in object_data["samples"]:
            sample_idx = sample_data["sample_idx"]
            num_samples = len(object_data["samples"])

            if num_samples > 1:
                # Multi-sample naming
                np.save(
                    os.path.join(
                        obj_output_dir, f"pointcloud_combined_sample{sample_idx+1}.npy"
                    ),
                    sample_data["combined_points"],
                )
                np.save(
                    os.path.join(
                        obj_output_dir, f"camera_positions_sample{sample_idx+1}.npy"
                    ),
                    sample_data["camera_positions"],
                )

                # Save individual cameras if requested
                if sample_data["individual_cameras"] is not None:
                    for cam_idx, cam_points in enumerate(
                        sample_data["individual_cameras"]
                    ):
                        np.save(
                            os.path.join(
                                obj_output_dir,
                                f"pointcloud_sample{sample_idx+1}_cam{cam_idx+1}.npy",
                            ),
                            cam_points,
                        )
            else:
                # Single sample naming
                np.save(
                    os.path.join(obj_output_dir, "pointcloud_combined.npy"),
                    sample_data["combined_points"],
                )
                np.save(
                    os.path.join(obj_output_dir, "camera_positions.npy"),
                    sample_data["camera_positions"],
                )

                # Save individual cameras if requested
                if sample_data["individual_cameras"] is not None:
                    for cam_idx, cam_points in enumerate(
                        sample_data["individual_cameras"]
                    ):
                        np.save(
                            os.path.join(
                                obj_output_dir, f"pointcloud_cam{cam_idx+1}.npy"
                            ),
                            cam_points,
                        )


def update_metadata_batch(object_data_list, cfg):
    """
    Batch update metadata for multiple objects
    """
    original_cwd = get_original_cwd()
    metadata_path = os.path.join(original_cwd, cfg.preprocessing.lidar.metadata_path)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Prepare all metadata lines
    lines = []
    header = "object,sample,center_of_mass,camera_distance,num_cameras\n"

    for object_data in object_data_list:
        if object_data is None:
            continue

        obj_name = object_data["obj_name"]
        com = object_data["center_of_mass"]
        radius = object_data["radius"]

        for sample_data in object_data["samples"]:
            sample_idx = sample_data["sample_idx"]
            num_cameras = len(sample_data["camera_positions"])
            line = f"{obj_name},{sample_idx+1},{com.tolist()},{radius},{num_cameras}\n"
            lines.append(line)

    # Write all at once
    mode = "w" if not os.path.exists(metadata_path) else "a"
    with open(metadata_path, mode) as f:
        if mode == "w":
            f.write(header)
        f.writelines(lines)


def process_all_meshes_optimized(cfg):
    """
    Optimized parallel processing of all meshes
    """
    print("Loading mesh data...")
    original_cwd = get_original_cwd()
    mesh_dir = os.path.join(original_cwd, cfg.preprocessing.lidar.mesh_dir)
    obj_files = [f for f in os.listdir(mesh_dir) if f.endswith(".obj")]

    if not obj_files:
        raise FileNotFoundError(f"No .obj files found in {mesh_dir}")

    debug = cfg.get("debug", False)

    # Control logging levels
    if not debug:
        logging.getLogger("trimesh").setLevel(logging.WARNING)
    else:
        logging.getLogger("trimesh").setLevel(logging.INFO)
        print("[DEBUG] Debug mode enabled")
        print(f"Processing {len(obj_files)} mesh files from {mesh_dir}")

    # Determine number of processes
    n_processes = min(cpu_count() - 1, len(obj_files))  # Leave one CPU free
    if debug:
        n_processes = 1  # Serial processing for debugging

    print(f"Generating point clouds from meshes using {n_processes} processes...")

    # Prepare arguments for parallel processing
    mesh_paths = [os.path.join(mesh_dir, obj_file) for obj_file in obj_files]
    args_list = [
        (mesh_path, cfg, idx, len(obj_files))
        for idx, mesh_path in enumerate(mesh_paths)
    ]

    # Process in batches to manage memory
    batch_size = max(1, min(100, len(obj_files) // 10))  # Adaptive batch size
    processed_count = 0

    if n_processes == 1:
        # Serial processing (for debugging)
        for args in tqdm(args_list, desc="Processing meshes"):
            object_data, obj_idx, total = process_mesh_optimized(args)
            if object_data is not None:
                save_object_data_batch([object_data], cfg)
                if cfg.preprocessing.lidar.metadata_path:
                    update_metadata_batch([object_data], cfg)
                processed_count += 1
    else:
        # Parallel processing
        with Pool(processes=n_processes) as pool:
            for i in tqdm(
                range(0, len(args_list), batch_size), desc="Processing batches"
            ):
                batch_args = args_list[i : i + batch_size]

                # Process batch in parallel
                results = pool.map(process_mesh_optimized, batch_args)

                # Extract successful results
                object_data_list = [
                    result[0] for result in results if result[0] is not None
                ]

                if object_data_list:
                    # Batch save all objects from this batch
                    if cfg.preprocessing.lidar.save:
                        save_object_data_batch(object_data_list, cfg)

                    # Batch update metadata
                    if cfg.preprocessing.lidar.metadata_path:
                        update_metadata_batch(object_data_list, cfg)

                    processed_count += len(object_data_list)

    print(
        f"Point cloud generation complete. Processed {processed_count}/{len(obj_files)} meshes."
    )


# Maintain backward compatibility
def process_all_meshes(cfg):
    """
    Main entry point - uses optimized version by default
    """
    if cfg.get("use_optimized", True):
        process_all_meshes_optimized(cfg)
    else:
        # Fall back to original implementation if needed
        from .mesh_to_pointcloud import (
            process_all_meshes as original_process_all_meshes,
        )

        original_process_all_meshes(cfg)
