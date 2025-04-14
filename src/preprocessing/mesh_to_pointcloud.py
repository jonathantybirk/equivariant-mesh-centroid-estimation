import os
import numpy as np
import matplotlib.pyplot as plt
from hydra.utils import get_original_cwd

from src.utils import geometry, mesh, lidar

def process_mesh(mesh_path, cfg):
    obj_name = os.path.splitext(os.path.basename(mesh_path))[0]
    print(f"Processing {obj_name}...")

    m = mesh.load_mesh(mesh_path)
    com = mesh.compute_center_of_mass(m)
    print(f"  Center of mass: {com}")

    # Determine camera distance using max_lidar_distance_factor
    radius = m.bounding_sphere.primitive.radius * cfg.preprocessing.lidar.max_lidar_distance_factor

    # Use Hydra's original working directory so paths are relative to your project root
    original_cwd = get_original_cwd()
    obj_output_dir = os.path.join(original_cwd, cfg.preprocessing.lidar.output_dir, obj_name)
    os.makedirs(obj_output_dir, exist_ok=True)

    output_dir = os.path.join(get_original_cwd(), cfg.preprocessing.lidar.output_dir, obj_name)
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(cfg.preprocessing.lidar.seed)

    camera_positions = []
    target_points = []
    all_points_by_camera = []

    for i in range(cfg.preprocessing.lidar.num_cameras):
        try:
            target_point, camera_pos = mesh.sample_visible_target_and_camera(m, radius)
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
                include_misses=cfg.preprocessing.lidar.include_missed_rays
            )
            all_points_by_camera.append(pts)

            if cfg.preprocessing.lidar.save:
                np.save(os.path.join(obj_output_dir, f"pointcloud_cam{i+1}.npy"), pts)

        except RuntimeError as e:
            print(f"  Warning: {e} Skipping this camera.")

    if not all_points_by_camera:
        print("  Skipped mesh — no cameras could see it.")
        return

    all_points = np.concatenate(all_points_by_camera, axis=0)

    if cfg.preprocessing.lidar.save:
        np.save(os.path.join(obj_output_dir, "pointcloud_combined.npy"), all_points)
        np.save(os.path.join(obj_output_dir, "camera_positions.npy"), np.array(camera_positions))
        np.save(os.path.join(obj_output_dir, "center_of_mass.npy"), np.array(com))

        metadata_path = os.path.join(original_cwd, cfg.preprocessing.lidar.metadata_path)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        header = "object,center_of_mass,camera_distance,num_cameras\n"
        line = f"{obj_name},{com.tolist()},{radius},{len(camera_positions)}\n"
        if not os.path.exists(metadata_path):
            with open(metadata_path, "w") as f:
                f.write(header)
        with open(metadata_path, "a") as f:
            f.write(line)

    if cfg.preprocessing.lidar.visualize:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        # Colors in binary order: 100, 010, 001, 011, 101, 110
        colors = [
            (1, 0, 0),    # red   (100)
            (0, 1, 0),    # green (010)
            (0, 0, 1),    # blue  (001)
            (0, 1, 1),    # cyan  (011)
            (1, 0, 1),    # magenta (101)
            (1, 1, 0)     # yellow (110)
        ]
        for idx, pts in enumerate(all_points_by_camera):
            color = colors[idx % len(colors)]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, color=color, alpha=0.6)
            cam = camera_positions[idx]
            ax.scatter(cam[0], cam[1], cam[2], marker='^', s=100, color=color)
            target = target_points[idx]
            ax.scatter(target[0], target[1], target[2], marker='x', s=50, color=color)
            if getattr(cfg, "show_target_lines", True):
                line = np.vstack((cam, target))
                ax.plot(line[:, 0], line[:, 1], line[:, 2], linestyle='--', linewidth=1, color=color)
        ax.scatter(com[0], com[1], com[2], marker='o', color='black', s=150)
        ax.set_title(f"{obj_name} - LiDAR Point Cloud View")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if cfg.preprocessing.lidar.export_images:
            vis_dir = os.path.join(obj_output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, "pointcloud_view.png")
            dpi = cfg.preprocessing.lidar.get("dpi", 300) if hasattr(cfg.preprocessing.lidar, "dpi") else 300
            plt.savefig(vis_path, dpi=dpi)
        plt.show()
        plt.close()

def process_all_meshes(cfg):
    original_cwd = get_original_cwd()
    mesh_dir = os.path.join(original_cwd, cfg.preprocessing.lidar.mesh_dir)
    obj_files = [f for f in os.listdir(mesh_dir) if f.endswith(".obj")]
    if not obj_files:
        raise FileNotFoundError(f"No .obj files found in {mesh_dir}")
    for obj_file in obj_files:
        mesh_path = os.path.join(mesh_dir, obj_file)
        process_mesh(mesh_path, cfg)
    print("Point cloud generation complete.")