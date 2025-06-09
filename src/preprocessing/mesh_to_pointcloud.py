import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle

from src.utils import geometry, mesh, lidar


def get_working_directory():
    """
    Get the working directory, handling both Hydra and non-Hydra contexts
    """
    try:
        from hydra.utils import get_original_cwd
        return get_original_cwd()
    except (ImportError, ValueError):
        # Hydra not initialized or not available, use current working directory
        return os.getcwd()


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


def preprocess_mesh_to_unit_sphere(mesh_obj):
    """
    Preprocess mesh according to specification:
    1. Translate mesh by -mesh_centroid (so mesh_centroid becomes 0)
    2. Scale both mesh and mesh_centroid to unit sphere  
    3. Apply random translation to both mesh and mesh_centroid
    
    Returns:
        tuple: (processed_mesh, original_centroid, scale_factor, final_mesh_centroid)
    """
    # Step 1: Calculate original mesh centroid
    original_mesh_centroid = mesh_obj.centroid.copy()
    
    # Step 2: Translate mesh by -mesh_centroid (centers mesh at origin)
    mesh_obj.vertices -= original_mesh_centroid
    mesh_centroid = np.array([0.0, 0.0, 0.0])  # Now mesh_centroid is at origin
    
    # Step 3: Scale to unit sphere - find max distance from origin
    max_dist = np.max(np.linalg.norm(mesh_obj.vertices, axis=1))
    
    # Step 4: Scale both mesh vertices and mesh_centroid
    mesh_obj.vertices /= max_dist
    mesh_centroid /= max_dist  # This remains [0,0,0] but for consistency
    
    # Step 5: Generate random translation vector
    direction = geometry.random_point_on_sphere(radius=1.0)
    magnitude = np.random.uniform(0, 0.5)
    translation = direction * magnitude
    
    # Step 6: Apply translation to both mesh and mesh_centroid
    mesh_obj.vertices += translation
    final_mesh_centroid = mesh_centroid + translation  # This equals translation since mesh_centroid was [0,0,0]
    
    return mesh_obj, original_mesh_centroid, max_dist, final_mesh_centroid


def generate_systematic_camera_positions(num_cameras, distance=2.0, angle_noise_deg=20.0):
    """
    Generate camera positions using systematic 6-position strategy with random selection.
    
    Args:
        num_cameras: Number of cameras (1-6, clamped automatically)
        distance: Distance from origin (fixed at 2.0)
        angle_noise_deg: Angular noise in degrees (±20°)
    
    Returns:
        numpy.ndarray: Camera positions (K, 3)
    """
    num_cameras = max(1, min(6, num_cameras))  # Clamp to [1, 6]
    
    # Step 1: Place first camera at random position
    first_direction = geometry.random_point_on_sphere(radius=1.0)
    camera_positions = [distance * first_direction]
    
    if num_cameras == 1:
        return np.array(camera_positions)
    
    # Step 2: Define 5 additional base positions relative to first camera
    # Generate two orthogonal vectors to first_direction
    arbitrary = np.array([0, 0, 1]) if abs(first_direction[2]) < 0.9 else np.array([0, 1, 0])
    ortho1 = np.cross(first_direction, arbitrary)
    ortho1 = ortho1 / np.linalg.norm(ortho1)
    ortho2 = np.cross(first_direction, ortho1)
    ortho2 = ortho2 / np.linalg.norm(ortho2)
    
    # Available base positions (5 positions around the first camera)
    available_positions = [
        -first_direction,  # opposite
        ortho1,           # ortho1
        ortho2,           # ortho2
        -ortho1,          # ortho3
        -ortho2           # ortho4
    ]
    
    # Step 3: Randomly select cameras from available positions
    max_angle_rad = np.radians(angle_noise_deg)
    
    # Randomly shuffle the available positions
    remaining_positions = available_positions.copy()
    np.random.shuffle(remaining_positions)
    
    for i in range(1, num_cameras):
        # Take the next position from the shuffled list
        base_direction = remaining_positions[i - 1]
        
        # Apply uniform noise
        perturbed_direction = geometry.perturb_direction(base_direction, max_angle=max_angle_rad)
        
        # Place camera
        camera_pos = distance * perturbed_direction
        camera_positions.append(camera_pos)
    
    return np.array(camera_positions)


def normalize_pointcloud_to_unit_sphere(pointcloud):
    """
    Normalize pointcloud to unit sphere centered at its centroid.
    
    Args:
        pointcloud: Raw pointcloud (N, 3)
    
    Returns:
        tuple: (normalized_pointcloud, pc_centroid, scale_factor)
            - normalized_pointcloud: Normalized pointcloud centered at origin
            - pc_centroid: Original pointcloud centroid
            - scale_factor: Scaling factor used for normalization
    """
    if len(pointcloud) == 0:
        return pointcloud, np.zeros(3), 1.0
    
    # Step a) Calculate pointcloud centroid
    pc_centroid = np.mean(pointcloud, axis=0)
    
    # Step b) Center pointcloud
    centered_pointcloud = pointcloud - pc_centroid
    
    # Step c) Calculate max distance
    distances = np.linalg.norm(centered_pointcloud, axis=1)
    max_dist = np.max(distances)
    
    # Step d) Normalize to unit sphere
    if max_dist > 0:
        normalized_pointcloud = centered_pointcloud / max_dist
        scale_factor = max_dist
    else:
        normalized_pointcloud = centered_pointcloud
        scale_factor = 1.0
    
    return normalized_pointcloud, pc_centroid, scale_factor


def apply_pointcloud_transformation(coordinates, pc_centroid, scale_factor):
    """
    Apply the same transformation used on pointcloud to other coordinates.
    
    Args:
        coordinates: Coordinates to transform (can be single point or array)
        pc_centroid: Pointcloud centroid used for centering
        scale_factor: Scale factor used for normalization
    
    Returns:
        numpy.ndarray: Transformed coordinates in pointcloud coordinate system
    """
    # Center around pointcloud centroid
    centered = coordinates - pc_centroid
    
    # Apply same scaling
    if scale_factor > 0:
        transformed = centered / scale_factor
    else:
        transformed = centered
    
    return transformed


def visualize_first_n_pointclouds(cfg, n_visualize=3):
    """
    Visualize the first N generated pointclouds with new file structure.

    Args:
        cfg: Configuration object
        n_visualize: Number of pointclouds to visualize
    """
    print(f"\nVisualizing first {n_visualize} generated pointclouds...")

    original_cwd = get_working_directory()
    mesh_dir = os.path.join(original_cwd, cfg.mesh_dir)
    output_dir = os.path.join(original_cwd, cfg.output_dir)

    # Get first N mesh files
    obj_files = [f for f in os.listdir(mesh_dir) if f.endswith(".obj")][:n_visualize]

    for i, obj_file in enumerate(obj_files):
        obj_name = os.path.splitext(obj_file)[0]
        mesh_path = os.path.join(mesh_dir, obj_file)
        obj_output_dir = os.path.join(output_dir, obj_name)

        try:
            # Load original mesh
            mesh_obj = mesh.load_mesh(mesh_path, debug=False)

            # Load generated data with new file structure
            mesh_centroid_path = os.path.join(obj_output_dir, "mesh_centroid.npy")
            pointcloud_path = os.path.join(obj_output_dir, "pointcloud.npy")
            camera_path = os.path.join(obj_output_dir, "camera_positions.npy")

            if not all(os.path.exists(p) for p in [mesh_centroid_path, pointcloud_path]):
                print(f"Skipping {obj_name} - missing generated data files")
                continue

            # Load data
            mesh_centroid = np.load(mesh_centroid_path)
            pointcloud = np.load(pointcloud_path)
            
            # Load camera data (first row is target, rest are positions)
            if os.path.exists(camera_path):
                camera_data = np.load(camera_path)
                if len(camera_data) > 1:
                    camera_positions = camera_data[1:]  # Skip first row (target)
                else:
                    camera_positions = None
            else:
                camera_positions = None

            print(
                f"Visualizing {i+1}/{len(obj_files)}: {obj_name} ({len(pointcloud)} points)"
            )

            # Visualize - note that mesh_centroid is now the reference point
            visualize_pointcloud_with_mesh(
                mesh_obj, pointcloud, mesh_centroid, obj_name, camera_positions
            )

        except Exception as e:
            print(f"Error visualizing {obj_name}: {e}")
            continue


def process_mesh_optimized(args):
    """
    Optimized mesh processing function implementing the new 4-stage pipeline:
    1. Preprocess mesh (center, scale, translate)
    2. Generate systematic camera positions
    3. Perform LIDAR scanning with 45° FOV
    4. Normalize pointcloud to unit sphere
    
    IMPORTANT: This function uses two different scale factors:
    - mesh_scale_factor: Used for initial mesh normalization to unit sphere
    - pc_scale_factor: Used for final pointcloud normalization (applied to all coordinates)
    
    The mesh centroid and camera positions are transformed using pc_scale_factor
    to ensure consistency with the normalized pointcloud coordinate system.
    """
    mesh_path, cfg, obj_idx, total_objs = args
    obj_name = os.path.splitext(os.path.basename(mesh_path))[0]

    try:
        # Load original mesh
        original_mesh = mesh.load_mesh(mesh_path, debug=False)
        
        # Stage 1: Preprocess mesh to unit sphere with random translation
        processed_mesh, original_centroid, mesh_scale_factor, final_mesh_centroid = preprocess_mesh_to_unit_sphere(
            original_mesh
        )
        
        # Stage 2: Generate systematic camera positions around origin (as per specification)
        camera_distance = cfg.get("camera_distance", 2.0)
        angle_noise = cfg.get("camera_angle_noise_deg", 20.0)
        num_cameras = cfg.num_cameras
        
        # Generate camera positions around origin - cameras point toward origin
        # This follows the specification: "Camera positions are stored relative to pre-translation coordinate system"
        # and "All cameras point toward origin with target_point = origin"
        camera_positions_pretranslation = generate_systematic_camera_positions(
            num_cameras, distance=camera_distance, angle_noise_deg=angle_noise
        )
        
        # For LIDAR scanning, we use the same positions (cameras point toward origin)
        camera_positions_posttranslation = camera_positions_pretranslation.copy()
        
        num_samples = cfg.get("num_samples", 1)

        # Collect all data for this object
        object_data = {
            "obj_name": obj_name,
            "original_centroid": original_centroid,
            "final_mesh_centroid": final_mesh_centroid,
            "mesh_scale_factor": mesh_scale_factor,
            "samples": [],
        }

        # Process all samples for this object
        for sample_idx in range(num_samples):
            sample_seed = cfg.seed + sample_idx
            np.random.seed(sample_seed)

            all_points_by_camera = []

            # Stage 3: LIDAR scanning from each camera position
            for cam_idx, camera_pos in enumerate(camera_positions_posttranslation):
                try:
                    # Target is the origin as per specification
                    target_point = np.array([0.0, 0.0, 0.0])
                    
                    # Perform LIDAR scanning with new FOV parameters
                    h_fov_deg = cfg.get("camera_fov_deg", 45.0)
                    v_fov_deg = cfg.get("camera_fov_deg", 45.0)
                    
                    # Use rays_step_x/y for scanning resolution
                    h_steps = cfg.get("rays_step_x", cfg.get("h_steps", 50))
                    v_steps = cfg.get("rays_step_y", cfg.get("v_steps", 50))
                    
                    pts = lidar.simulate_lidar(
                        processed_mesh,
                        camera_pos,
                        target_point,
                        h_fov_deg=h_fov_deg,
                        v_fov_deg=v_fov_deg,
                        h_steps=h_steps,
                        v_steps=v_steps,
                        max_distance=camera_distance * 2,  # Generous max distance
                        include_misses=cfg.get("include_missed_rays", False),
                    )
                    
                    if len(pts) > 0:
                        all_points_by_camera.append(pts)

                except RuntimeError as e:
                    print(f"LIDAR scan failed for camera {cam_idx} on {obj_name}: {e}")
                    continue

            if all_points_by_camera:
                # Combine all camera points
                raw_pointcloud = np.concatenate(all_points_by_camera, axis=0)
                
                # Stage 4: Normalize pointcloud to unit sphere centered on pointcloud centroid
                normalized_pointcloud, pc_centroid, pc_scale_factor = normalize_pointcloud_to_unit_sphere(raw_pointcloud)

                # Transform mesh centroid, camera positions, and camera target to pointcloud coordinate system
                # IMPORTANT: All coordinates must be transformed to pointcloud coordinate system
                # - final_mesh_centroid: the translated mesh center position
                # - camera_positions_posttranslation: the camera positions used for scanning
                # - camera_target_origin: the origin point [0,0,0] that cameras aimed at
                camera_target_origin = np.array([0.0, 0.0, 0.0])  # The origin where cameras aimed
                
                mesh_centroid_in_pc_coords = apply_pointcloud_transformation(
                    final_mesh_centroid, pc_centroid, pc_scale_factor
                )
                camera_positions_in_pc_coords = apply_pointcloud_transformation(
                    camera_positions_posttranslation, pc_centroid, pc_scale_factor
                )
                camera_target_in_pc_coords = apply_pointcloud_transformation(
                    camera_target_origin, pc_centroid, pc_scale_factor
                )

                # Prepare individual camera data if requested
                individual_cameras_data = None
                if cfg.get("save_individual_cameras", False):
                    individual_cameras_data = []
                    for cam_points in all_points_by_camera:
                        # Apply same transformation to individual camera pointclouds
                        if len(cam_points) > 0:
                            centered_cam = cam_points - pc_centroid
                            normalized_cam = centered_cam / pc_scale_factor if pc_scale_factor > 0 else centered_cam
                            individual_cameras_data.append(normalized_cam)
                        else:
                            individual_cameras_data.append(cam_points)

                sample_data = {
                    "sample_idx": sample_idx,
                    "normalized_points": normalized_pointcloud,
                    "camera_positions_pc_coords": camera_positions_in_pc_coords,
                    "camera_target_pc_coords": camera_target_in_pc_coords,
                    "mesh_centroid_pc_coords": mesh_centroid_in_pc_coords,
                    "individual_cameras": individual_cameras_data,
                    # Keep transformation info for reference
                    "pc_centroid": pc_centroid,
                    "pc_scale_factor": pc_scale_factor,
                }
                object_data["samples"].append(sample_data)

        return object_data, obj_idx, total_objs

    except Exception as e:
        print(f"Error processing {obj_name}: {e}")
        return None, obj_idx, total_objs


def save_object_data_batch(object_data_list, cfg):
    """
    Batch save multiple objects' data with new file structure:
    - pointcloud.npy: normalized pointcloud
    - camera_positions.npy: camera coordinates in pointcloud coordinate system
    - mesh_centroid.npy: mesh centroid position in pointcloud coordinate system
    """
    original_cwd = get_working_directory()

    for object_data in object_data_list:
        if object_data is None:
            continue

        obj_name = object_data["obj_name"]
        obj_output_dir = os.path.join(
            original_cwd, cfg.output_dir, obj_name
        )
        os.makedirs(obj_output_dir, exist_ok=True)

        # Save all samples for this object
        for sample_data in object_data["samples"]:
            sample_idx = sample_data["sample_idx"]
            num_samples = len(object_data["samples"])

            if num_samples > 1:
                # Multi-sample naming
                np.save(
                    os.path.join(
                        obj_output_dir, f"pointcloud_sample{sample_idx+1}.npy"
                    ),
                    sample_data["normalized_points"],
                )
                # Save camera data: first row is target, rest are camera positions
                camera_data = np.vstack([
                    sample_data["camera_target_pc_coords"].reshape(1, -1),
                    sample_data["camera_positions_pc_coords"]
                ])
                np.save(
                    os.path.join(
                        obj_output_dir, f"camera_positions_sample{sample_idx+1}.npy"
                    ),
                    camera_data,
                )
                np.save(
                    os.path.join(
                        obj_output_dir, f"mesh_centroid_sample{sample_idx+1}.npy"
                    ),
                    sample_data["mesh_centroid_pc_coords"],
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
                    os.path.join(obj_output_dir, "pointcloud.npy"),
                    sample_data["normalized_points"],
                )
                # Save camera data: first row is target, rest are camera positions
                camera_data = np.vstack([
                    sample_data["camera_target_pc_coords"].reshape(1, -1),
                    sample_data["camera_positions_pc_coords"]
                ])
                np.save(
                    os.path.join(obj_output_dir, "camera_positions.npy"),
                    camera_data,
                )
                np.save(
                    os.path.join(obj_output_dir, "mesh_centroid.npy"),
                    sample_data["mesh_centroid_pc_coords"],
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
    Batch update metadata for multiple objects with new data structure
    """
    original_cwd = get_working_directory()
    metadata_path = os.path.join(original_cwd, cfg.metadata_path)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Prepare all metadata lines
    lines = []
    header = "object,sample,mesh_centroid_pc_coords,camera_distance_pc_coords,num_cameras,scale_factor\n"

    for object_data in object_data_list:
        if object_data is None:
            continue

        obj_name = object_data["obj_name"]

        for sample_data in object_data["samples"]:
            sample_idx = sample_data["sample_idx"]
            mesh_centroid_pc = sample_data["mesh_centroid_pc_coords"]
            camera_positions_pc = sample_data["camera_positions_pc_coords"]
            pc_scale_factor = sample_data["pc_scale_factor"]
            
            num_cameras = len(camera_positions_pc)
            # Calculate average camera distance in pointcloud coordinate system
            avg_camera_distance_pc = np.mean([np.linalg.norm(cam_pos) for cam_pos in camera_positions_pc])
            
            line = f"{obj_name},{sample_idx+1},{mesh_centroid_pc.tolist()},{avg_camera_distance_pc},{num_cameras},{pc_scale_factor}\n"
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
    original_cwd = get_working_directory()
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
        (mesh_path, cfg.preprocessing.lidar, idx, len(obj_files))
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
                            mesh_obj, object_data, object_data["samples"][0]["mesh_centroid_pc_coords"], 
                            object_data["obj_name"], processed_count, visualize_first_n
                        )
                    except Exception as e:
                        print(f"Visualization error for {object_data['obj_name']}: {e}")
                
                save_object_data_batch([object_data], cfg.preprocessing.lidar)
                if cfg.preprocessing.lidar.get("metadata_path"):
                    update_metadata_batch([object_data], cfg.preprocessing.lidar)
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
                        save_object_data_batch(object_data_list, cfg.preprocessing.lidar)

                    # Batch update metadata
                    if cfg.preprocessing.lidar.get("metadata_path"):
                        update_metadata_batch(object_data_list, cfg.preprocessing.lidar)

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
            visualize_first_n_pointclouds(cfg.preprocessing.lidar, visualize_first_n)


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


def visualize_during_processing(mesh_obj, pointcloud_data, mesh_centroid, obj_name, obj_idx, total_objs):
    """
    Visualize pointcloud during processing (real-time visualization) with new data structure
    
    Args:
        mesh_obj: Trimesh mesh object
        pointcloud_data: Dictionary containing pointcloud and camera data
        mesh_centroid: Mesh centroid coordinates (in pointcloud coordinate system)
        obj_name: Name of the object
        obj_idx: Current object index (0-based)
        total_objs: Total number of objects being processed
    """
    # Extract data from the sample (use first sample if multiple)
    if pointcloud_data["samples"]:
        sample_data = pointcloud_data["samples"][0]  # Use first sample
        pointcloud = sample_data["normalized_points"]
        camera_positions = sample_data["camera_positions_pc_coords"]
        mesh_centroid_pc = sample_data["mesh_centroid_pc_coords"]
        
        print(f"Real-time visualization {obj_idx + 1}/{total_objs}: {obj_name} ({len(pointcloud)} points)")
        
        # Call the existing visualization function with pointcloud coordinate system data
        visualize_pointcloud_with_mesh(mesh_obj, pointcloud, mesh_centroid_pc, obj_name, camera_positions)
