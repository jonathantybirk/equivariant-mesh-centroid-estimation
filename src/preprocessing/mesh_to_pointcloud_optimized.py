import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hydra.utils import get_original_cwd
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle

from src.utils import geometry, mesh, lidar


def visualize_pointcloud_with_mesh(
    mesh_obj, pointcloud, center_of_mass, obj_name, camera_positions=None
):
    """
    Visualize a single pointcloud alongside its original mesh with center of mass.

    Args:
        mesh_obj: Trimesh mesh object
        pointcloud: Generated pointcloud array (N, 3)
        center_of_mass: Center of mass coordinates (3,)
        obj_name: Name of the object for display
        camera_positions: Optional camera positions array
    """
    fig = plt.figure(figsize=(15, 5))

    # Plot 1: Original mesh with center of mass
    ax1 = fig.add_subplot(131, projection="3d")
    vertices = mesh_obj.vertices
    faces = mesh_obj.faces

    # Create mesh collection with transparency
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh_collection = Poly3DCollection(
        vertices[faces],
        alpha=0.3,
        facecolors="lightgray",
        edgecolors="gray",
        linewidths=0.5,
    )
    ax1.add_collection3d(mesh_collection)

    # Add center of mass
    ax1.scatter(
        center_of_mass[0],
        center_of_mass[1],
        center_of_mass[2],
        c="red",
        s=100,
        alpha=1.0,
        label="Center of Mass",
    )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"Original Mesh: {obj_name}")
    ax1.legend()

    # Plot 2: Generated pointcloud
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.scatter(
        pointcloud[:, 0],
        pointcloud[:, 1],
        pointcloud[:, 2],
        c="blue",
        s=1,
        alpha=0.6,
        label=f"Points ({len(pointcloud)})",
    )

    # Add center of mass to pointcloud view
    ax2.scatter(
        center_of_mass[0],
        center_of_mass[1],
        center_of_mass[2],
        c="red",
        s=100,
        alpha=1.0,
        label="Center of Mass",
    )

    # Add camera positions if available
    if camera_positions is not None:
        ax2.scatter(
            camera_positions[:, 0],
            camera_positions[:, 1],
            camera_positions[:, 2],
            c="green",
            s=50,
            marker="^",
            alpha=0.8,
            label=f"Cameras ({len(camera_positions)})",
        )

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title(f"Generated Pointcloud: {obj_name}")
    ax2.legend()

    # Plot 3: Overlay view
    ax3 = fig.add_subplot(133, projection="3d")

    # Add transparent mesh
    mesh_collection_overlay = Poly3DCollection(
        vertices[faces],
        alpha=0.15,
        facecolors="lightgray",
        edgecolors=None,
        linewidths=0,
    )
    ax3.add_collection3d(mesh_collection_overlay)

    # Add pointcloud
    ax3.scatter(
        pointcloud[:, 0],
        pointcloud[:, 1],
        pointcloud[:, 2],
        c="blue",
        s=2,
        alpha=0.7,
        label=f"Points ({len(pointcloud)})",
    )

    # Add center of mass
    ax3.scatter(
        center_of_mass[0],
        center_of_mass[1],
        center_of_mass[2],
        c="red",
        s=100,
        alpha=1.0,
        label="Center of Mass",
    )

    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title(f"Overlay: {obj_name}")
    ax3.legend()

    # Set equal aspect ratios for all plots
    for ax in [ax1, ax2, ax3]:
        max_range = np.array(
            [
                vertices[:, 0].max() - vertices[:, 0].min(),
                vertices[:, 1].max() - vertices[:, 1].min(),
                vertices[:, 2].max() - vertices[:, 2].min(),
            ]
        ).max() / 2.0

        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


def visualize_first_n_pointclouds(cfg, n_visualize=3):
    """
    Visualize the first N generated pointclouds during preprocessing.

    Args:
        cfg: Configuration object
        n_visualize: Number of pointclouds to visualize
    """
    print(f"\nVisualizing first {n_visualize} generated pointclouds...")

    original_cwd = get_original_cwd()
    mesh_dir = os.path.join(original_cwd, cfg.preprocessing.lidar.mesh_dir)
    output_dir = os.path.join(original_cwd, cfg.preprocessing.lidar.output_dir)

    # Get first N mesh files
    obj_files = [f for f in os.listdir(mesh_dir) if f.endswith(".obj")][:n_visualize]

    for i, obj_file in enumerate(obj_files):
        obj_name = os.path.splitext(obj_file)[0]
        mesh_path = os.path.join(mesh_dir, obj_file)
        obj_output_dir = os.path.join(output_dir, obj_name)

        try:
            # Load original mesh
            mesh_obj = mesh.load_mesh(mesh_path, debug=False)

            # Load generated data
            com_path = os.path.join(obj_output_dir, "center_of_mass.npy")
            pointcloud_path = os.path.join(obj_output_dir, "pointcloud_combined.npy")
            camera_path = os.path.join(obj_output_dir, "camera_positions.npy")

            if not all(os.path.exists(p) for p in [com_path, pointcloud_path]):
                print(f"Skipping {obj_name} - missing generated data files")
                continue

            # Load data
            center_of_mass = np.load(com_path)
            pointcloud = np.load(pointcloud_path)
            camera_positions = (
                np.load(camera_path) if os.path.exists(camera_path) else None
            )

            print(
                f"Visualizing {i+1}/{len(obj_files)}: {obj_name} ({len(pointcloud)} points)"
            )

            # Visualize
            visualize_pointcloud_with_mesh(
                mesh_obj, pointcloud, center_of_mass, obj_name, camera_positions
            )

        except Exception as e:
            print(f"Error visualizing {obj_name}: {e}")
            continue


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
    visualize_first_n = cfg.preprocessing.lidar.get("visualize_first_n", 0)

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
        # Serial processing (for debugging and real-time visualization)
        for args in tqdm(args_list, desc="Processing meshes"):
            object_data, obj_idx, total = process_mesh_optimized(args)
            if object_data is not None:
                # Real-time visualization during processing if requested
                if visualize_first_n > 0 and processed_count < visualize_first_n:
                    try:
                        # Load the original mesh for visualization
                        mesh_path = args[0]
                        mesh_obj = mesh.load_mesh(mesh_path, debug=False)
                        visualize_during_processing(
                            mesh_obj, object_data, object_data["center_of_mass"], 
                            object_data["obj_name"], processed_count, visualize_first_n
                        )
                    except Exception as e:
                        print(f"Visualization error for {object_data['obj_name']}: {e}")
                
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
    
    # Visualize first N pointclouds if requested
    if visualize_first_n > 0:
        if n_processes == 1:
            print("Real-time visualization was shown during processing.")
        else:
            print("Parallel processing was used - showing post-processing visualization...")
            visualize_first_n_pointclouds(cfg, visualize_first_n)


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


def visualize_during_processing(mesh_obj, pointcloud_data, center_of_mass, obj_name, obj_idx, total_objs):
    """
    Visualize pointcloud during processing (real-time visualization)
    
    Args:
        mesh_obj: Trimesh mesh object
        pointcloud_data: Dictionary containing pointcloud and camera data
        center_of_mass: Center of mass coordinates 
        obj_name: Name of the object
        obj_idx: Current object index (0-based)
        total_objs: Total number of objects being processed
    """
    # Extract data from the sample (use first sample if multiple)
    if pointcloud_data["samples"]:
        sample_data = pointcloud_data["samples"][0]  # Use first sample
        pointcloud = sample_data["combined_points"]
        camera_positions = sample_data["camera_positions"]
        
        print(f"Real-time visualization {obj_idx + 1}/{total_objs}: {obj_name} ({len(pointcloud)} points)")
        
        # Call the existing visualization function
        visualize_pointcloud_with_mesh(mesh_obj, pointcloud, center_of_mass, obj_name, camera_positions)
