# equivariant-center-of-mass-estimation/src/preprocessing/obj_to_pc.py
import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from src.utils import geometry, mesh, lidar

def process_mesh(mesh_path, output_base_dir, save=True, visualize=True, h_fov_deg=90, v_fov_deg=90, h_steps=40, v_steps=40):
    obj_name = os.path.splitext(os.path.basename(mesh_path))[0]
    print(f"Processing {obj_name}...")
    
    # Load mesh and compute center of mass
    m = mesh.load_mesh(mesh_path)
    com = mesh.compute_center_of_mass(m)
    print(f"  Center of mass: {com}")
    
    # Determine camera distance: 2x bounding sphere radius
    bounding_sphere = m.bounding_sphere
    radius = bounding_sphere.primitive.radius * 2.0
    
    # Generate camera positions (example: 3 cameras)
    num_cameras = 3
    cameras = geometry.generate_camera_positions(com, radius, num_cameras=num_cameras)
    
    all_points = []
    all_points_by_camera = []
    target_points = []
    for cam_idx, cam in enumerate(cameras):
        target = mesh.select_visible_target_point(m, cam)
        target_points.append(target)
        pts = lidar.simulate_lidar(m, cam, target, h_fov_deg=h_fov_deg, v_fov_deg=v_fov_deg, h_steps=h_steps, v_steps=v_steps)
        all_points_by_camera.append(pts)
        all_points.append(pts)
    
    all_points_concat = np.concatenate(all_points, axis=0)
    
    # Save outputs if requested
    if save:
        obj_output_dir = os.path.join(output_base_dir, obj_name)
        os.makedirs(obj_output_dir, exist_ok=True)
        for idx, pts in enumerate(all_points_by_camera):
            filename = os.path.join(obj_output_dir, f"pointcloud_cam{idx+1}.npy")
            np.save(filename, pts)
            print(f"Saved point cloud for Camera {idx+1} to {filename}")
        np.save(os.path.join(obj_output_dir, "pointcloud_combined.npy"), all_points_concat)
        np.save(os.path.join(obj_output_dir, "camera_positions.npy"), np.array(cameras))
        np.save(os.path.join(obj_output_dir, "center_of_mass.npy"), np.array(com))
        
        # Append metadata to a CSV file in data/processed
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        metadata_path = os.path.join(repo_root, "data", "processed", "metadata.csv")
        header = "object,center_of_mass,camera_distance,num_cameras\n"
        line = f"{obj_name},{com.tolist()},{radius},{num_cameras}\n"
        if not os.path.exists(metadata_path):
            with open(metadata_path, "w") as f:
                f.write(header)
        with open(metadata_path, "a") as f:
            f.write(line)
        print(f"Metadata updated at {metadata_path}")
    
    # Visualization
    if visualize:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for idx, pts in enumerate(all_points_by_camera):
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, color=colors[idx % 3], alpha=0.6)
            ax.scatter(cameras[idx][0], cameras[idx][1], cameras[idx][2], marker='^', s=100, color=colors[idx % 3])
            ax.scatter(target_points[idx][0], target_points[idx][1], target_points[idx][2], marker='x', s=50, color=colors[idx % 3])
            line = np.vstack((cameras[idx], target_points[idx]))
            ax.plot(line[:, 0], line[:, 1], line[:, 2], linestyle='--', linewidth=1, color=colors[idx % 3])
        ax.scatter(com[0], com[1], com[2], marker='o', color='black', s=150)
        ax.set_title(f"{obj_name} - LiDAR Point Cloud View")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
        plt.close()

def process_all_meshes(save=True, visualize=True):
    # Determine repository root (assuming this file is in src/preprocessing)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    meshes_dir = os.path.join(repo_root, "data", "meshes")
    obj_files = [f for f in os.listdir(meshes_dir) if f.endswith('.obj')]
    if not obj_files:
        raise FileNotFoundError(f"No .obj files found in {meshes_dir}")
    for obj_file in obj_files:
        mesh_path = os.path.join(meshes_dir, obj_file)
        process_mesh(mesh_path, os.path.join(repo_root, "data", "pointclouds"), save=save, visualize=visualize)
    print("Processing complete!")
