import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import argparse

# --- Utility functions ---

def load_mesh(filepath):
    """Load a mesh from an .obj file using trimesh."""
    mesh = trimesh.load(filepath, force='mesh')
    if not mesh.is_watertight:
        print(f"Warning: Mesh at {filepath} is not watertight; mass properties may be inaccurate.")
    return mesh

def compute_center_of_mass(mesh):
    """Compute the center of mass of the mesh assuming uniform density."""
    mass_props = mesh.mass_properties
    return mass_props.center_mass

def random_point_on_sphere(radius=1.0):
    """Return a random point on a sphere with given radius."""
    phi = np.random.uniform(0, 2 * np.pi)
    cos_theta = np.random.uniform(-1, 1)
    theta = np.arccos(cos_theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.array([x, y, z])

def perturb_direction(direction, max_angle=np.pi/6):
    """Perturb a direction by a random angle up to max_angle radians."""
    rand_vec = np.random.randn(3)
    axis = np.cross(direction, rand_vec)
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([1, 0, 0])
    axis = axis / np.linalg.norm(axis)
    angle = np.random.uniform(-max_angle, max_angle)
    return rotate_vector(direction, axis, angle)

def rotate_vector(vector, axis, angle):
    """Rotate a vector around a given axis by a given angle using Rodrigues' formula."""
    vector = np.array(vector)
    axis = np.array(axis) / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return (vector * cos_angle +
            np.cross(axis, vector) * sin_angle +
            axis * np.dot(axis, vector) * (1 - cos_angle))

def generate_camera_positions(center, radius, num_cameras=3):
    """
    Generate camera positions around the object center at a given distance.
    Places cameras in balanced positions around the object.
    """
    num_cameras = max(1, min(6, num_cameras))
    cameras = []
    
    # First camera: random direction
    dir1 = random_point_on_sphere(radius=1.0)
    cam1 = center + radius * dir1
    cameras.append(cam1)
    
    if num_cameras == 1:
        return cameras
    
    # Find orthogonal axes to dir1
    arbitrary = np.array([0, 0, 1]) if abs(dir1[2]) < 0.9 else np.array([0, 1, 0])
    ortho1 = np.cross(dir1, arbitrary)
    ortho1 = ortho1 / np.linalg.norm(ortho1)
    
    # Second orthogonal axis
    ortho2 = np.cross(dir1, ortho1)
    ortho2 = ortho2 / np.linalg.norm(ortho2)
    
    base_directions = [
        -dir1,
        ortho1,
        -ortho1,
        ortho2,
        -ortho2
    ]
    np.random.shuffle(base_directions)
    
    max_angle = np.pi / 9  # ~20 degrees
    while len(cameras) < num_cameras:
        if not base_directions:
            chosen_dir = random_point_on_sphere(radius=1.0)
        else:
            chosen_dir = base_directions.pop(0)
        
        perturbed_dir = perturb_direction(chosen_dir, max_angle=max_angle)
        
        too_similar = False
        for existing_cam in cameras:
            existing_dir = (existing_cam - center) / radius
            cos_angle = np.clip(np.dot(perturbed_dir, existing_dir), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < np.pi / 6:
                too_similar = True
                break
        
        if not too_similar:
            new_cam = center + radius * perturbed_dir
            cameras.append(new_cam)
        else:
            if base_directions and len(base_directions) < (num_cameras - len(cameras)):
                base_directions.append(chosen_dir)
    
    return cameras

def generate_lidar_rays(camera_pos, look_at, h_fov_deg=30, v_fov_deg=30, h_steps=20, v_steps=20):
    """Generate ray directions for a LiDAR sensor at camera_pos, pointing at look_at."""
    center_dir = look_at - camera_pos
    center_dir = center_dir / np.linalg.norm(center_dir)
    
    up = np.array([0, 0, 1])
    if np.allclose(center_dir, up) or np.allclose(center_dir, -up):
        up = np.array([0, 1, 0])
    right = np.cross(center_dir, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, center_dir)
    
    h_angles = np.linspace(-np.radians(h_fov_deg)/2, np.radians(h_fov_deg)/2, h_steps)
    v_angles = np.linspace(-np.radians(v_fov_deg)/2, np.radians(v_fov_deg)/2, v_steps)
    
    directions = []
    for v in v_angles:
        for h in h_angles:
            offset = np.tan(h) * right + np.tan(v) * up
            ray_dir = center_dir + offset
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            directions.append(ray_dir)
    directions = np.array(directions)
    
    origins = np.tile(camera_pos, (directions.shape[0], 1))
    return origins, directions

def simulate_lidar(mesh, camera_pos, target_point, max_distance=None, include_misses=False, **ray_params):
    """
    Simulate a LiDAR scan from a given camera position.
    """
    if max_distance is None:
        max_distance = 2.0 * np.linalg.norm(target_point - camera_pos)
    
    origins, directions = generate_lidar_rays(camera_pos, target_point, **ray_params)
    
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions
    )
    
    hit_ray_indices = np.unique(index_ray)
    
    unique_points = []
    for ray in hit_ray_indices:
        inds = np.where(index_ray == ray)[0]
        pts = locations[inds]
        distances = np.linalg.norm(pts - origins[ray], axis=1)
        closest = pts[np.argmin(distances)]
        unique_points.append(closest)
    
    if include_misses:
        all_ray_indices = np.arange(len(origins))
        missed_ray_indices = np.setdiff1d(all_ray_indices, hit_ray_indices)
        for ray in missed_ray_indices:
            miss_point = origins[ray] + directions[ray] * max_distance
            unique_points.append(miss_point)
    
    return np.array(unique_points)

def random_point_on_mesh(mesh):
    """Select a random point on the mesh surface, weighted by triangle area."""
    triangles = mesh.triangles
    areas = trimesh.triangles.area(triangles)
    probabilities = areas / np.sum(areas)
    tri_idx = np.random.choice(len(triangles), p=probabilities)
    triangle = triangles[tri_idx]
    
    u = np.random.random()
    v = np.random.random() * (1 - u)
    w = 1 - u - v
    point = u * triangle[0] + v * triangle[1] + w * triangle[2]
    return point

def select_visible_target_point(mesh, camera_pos, num_attempts=100):
    """
    Select a random point on the mesh that is visible from the camera position.
    """
    for _ in range(num_attempts):
        random_dir = random_point_on_sphere(radius=1.0)
        mesh_center = mesh.centroid
        center_dir = mesh_center - camera_pos
        center_dir = center_dir / np.linalg.norm(center_dir)
        dot_product = np.dot(random_dir, center_dir)
        
        if dot_product > 0.5:  # within ~60°
            origins = np.array([camera_pos])
            directions = np.array([random_dir])
            
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=origins,
                ray_directions=directions
            )
            
            if len(locations) > 0:
                distances = np.linalg.norm(locations - camera_pos, axis=1)
                return locations[np.argmin(distances)]
    
    print("  Warning: Could not find visible target point, using fallback method")
    return random_point_on_mesh(mesh)

# --- Main script ---

def main():
    parser = argparse.ArgumentParser(description='Process 3D objects and generate LiDAR point clouds.')
    parser.add_argument('--save', action='store_true', help='Save results to disk (default: False)')
    parser.add_argument('--no-show', action='store_true', help='Do not show interactive visualizations (default: Show)')
    parser.add_argument('--show-pointcloud', action='store_true', help='Show point cloud visualization')
    parser.add_argument('--show-mesh', action='store_true', help='Show mesh visualization (not shown by default)')
    parser.add_argument('--show-combined', action='store_true', help='Show combined mesh and point cloud visualization')
    args = parser.parse_args()
    
    # Determine default visualization if none specified
    if not (args.show_pointcloud or args.show_mesh or args.show_combined):
        show_pointcloud = True
        show_mesh = False
        show_combined = True
    else:
        show_pointcloud = args.show_pointcloud
        show_mesh = args.show_mesh
        show_combined = args.show_combined
    
    # Locate folders based on the new structure
    # We assume this script is in <repo_root>/scripts, so go up one level:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # INPUT: data/meshes/obj
    objects_dir = os.path.join(repo_root, "data", "meshes", "obj")
    # OUTPUT: data/lidar
    output_dir = os.path.join(repo_root, "data", "lidar")
    
    if args.save:
        os.makedirs(output_dir, exist_ok=True)
    
    # Gather all .obj files from data/meshes/obj
    obj_files = [f for f in os.listdir(objects_dir) if f.endswith('.obj')]
    if not obj_files:
        raise FileNotFoundError(f"No .obj files found in {objects_dir}")
    
    # Process each .obj
    for obj_file in obj_files:
        print(f"Processing {obj_file}...")
        mesh_path = os.path.join(objects_dir, obj_file)
        
        # Create output directory for this object
        obj_name = os.path.splitext(obj_file)[0]
        if args.save:
            obj_output_dir = os.path.join(output_dir, obj_name)
            os.makedirs(obj_output_dir, exist_ok=True)
        
        # Load mesh & compute COM
        mesh = load_mesh(mesh_path)
        center_of_mass = compute_center_of_mass(mesh)
        print(f"  Center of mass: {center_of_mass}")
        
        # Camera distance: 2x bounding sphere radius
        bounding_sphere = mesh.bounding_sphere
        radius = bounding_sphere.primitive.radius * 2.0
        
        # Generate camera positions (example: 1 camera)
        num_cameras = 1
        camera_positions = generate_camera_positions(center_of_mass, radius, num_cameras=num_cameras)
        
        # Simulate LiDAR from each camera
        all_points = []
        all_points_by_camera = []
        target_points = []
        for cam_idx, cam in enumerate(camera_positions):
            target_point = select_visible_target_point(mesh, cam)
            target_points.append(target_point)
            
            pts = simulate_lidar(
                mesh,
                cam,
                target_point,
                h_fov_deg=90,
                v_fov_deg=90,
                h_steps=40,
                v_steps=40
            )
            all_points.append(pts)
            all_points_by_camera.append(pts)
            
            if args.save:
                np.save(os.path.join(obj_output_dir, f"pointcloud_cam{cam_idx+1}.npy"), pts)
        
        all_points = np.concatenate(all_points, axis=0)
        
        if args.save:
            np.save(os.path.join(obj_output_dir, "pointcloud_combined.npy"), all_points)
            np.save(os.path.join(obj_output_dir, "camera_positions.npy"), camera_positions)
            np.save(os.path.join(obj_output_dir, "center_of_mass.npy"), center_of_mass)
            
            # Save metadata
            with open(os.path.join(obj_output_dir, "metadata.txt"), "w") as f:
                f.write(f"Object: {obj_name}\n")
                f.write(f"Center of mass: {center_of_mass}\n")
                f.write(f"Camera distance (radius): {radius}\n")
                for i, cam in enumerate(camera_positions):
                    f.write(f"Camera {i+1} position: {cam}\n")
            
            print(f"  Results saved to {obj_output_dir}")
        
        # Define colors for cameras
        camera_colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255]   # Cyan
        ]
        mpl_camera_colors = [(r/255, g/255, b/255) for r, g, b in camera_colors]
        
        # --- Plotting ---
        if (not args.no_show) and show_pointcloud:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot LiDAR points (colored by camera)
            for idx, pts in enumerate(all_points_by_camera):
                color = mpl_camera_colors[idx % len(mpl_camera_colors)]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, color=color, alpha=0.6)
            
            # Plot cameras & target points
            for idx, (cam, target) in enumerate(zip(camera_positions, target_points)):
                color = mpl_camera_colors[idx % len(mpl_camera_colors)]
                ax.scatter(cam[0], cam[1], cam[2], marker='^', s=100, color=color, label=f'Camera {idx+1}')
                ax.scatter(target[0], target[1], target[2], marker='x', s=50, color=color)
                line = np.vstack((cam, target))
                ax.plot(line[:, 0], line[:, 1], line[:, 2], linestyle='--', linewidth=1, color=color)
            
            # Plot center of mass in black
            ax.scatter(center_of_mass[0], center_of_mass[1], center_of_mass[2],
                       marker='o', color='black', s=150, label='Center of Mass')
            
            ax.set_title(f"{obj_name} - LiDAR Point Cloud View")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend()
            
            if args.save:
                plt.savefig(os.path.join(obj_output_dir, "pointcloud_visualization.png"), dpi=300)
            
            plt.show()
            plt.close()
        
        # Optional mesh-only visualization
        if (not args.no_show) and show_mesh:
            scene = trimesh.Scene()
            scene.add_geometry(mesh)
            
            com_sphere = trimesh.primitives.Sphere(radius=radius/15, center=center_of_mass)
            com_sphere.visual.face_colors = [255, 0, 0, 255]
            scene.add_geometry(com_sphere)
            
            for idx, (cam_pos, target) in enumerate(zip(camera_positions, target_points)):
                direction = target - cam_pos
                direction /= np.linalg.norm(direction)
                
                cam_marker = trimesh.creation.cone(radius=radius/20, height=radius/10)
                
                z_axis = np.array([0, 0, 1])
                rotation_axis = np.cross(z_axis, direction)
                if np.linalg.norm(rotation_axis) > 1e-6:
                    rotation_axis /= np.linalg.norm(rotation_axis)
                    angle = np.arccos(np.dot(z_axis, direction))
                    rotation = trimesh.transformations.rotation_matrix(angle, rotation_axis)
                else:
                    rotation = np.eye(4)
                
                translation = trimesh.transformations.translation_matrix(cam_pos)
                transform = trimesh.transformations.concatenate_matrices(translation, rotation)
                cam_marker.apply_transform(transform)
                
                color = camera_colors[idx % len(camera_colors)]
                cam_marker.visual.face_colors = color + [255]
                scene.add_geometry(cam_marker)
                
                target_marker = trimesh.primitives.Sphere(radius=radius/30, center=target)
                target_marker.visual.face_colors = color + [200]
                scene.add_geometry(target_marker)
            
            if args.save:
                png = scene.save_image(resolution=[720, 480])
                with open(os.path.join(obj_output_dir, "mesh_visualization.png"), 'wb') as f:
                    f.write(png)
            
            scene.show(resolution=(640, 480))
        
        # Combined visualization
        if (not args.no_show) and show_combined:
            scene = trimesh.Scene()
            
            mesh_copy = mesh.copy()
            mesh_copy.visual.face_colors = [200, 200, 200, 150]  # translucent gray
            scene.add_geometry(mesh_copy)
            
            com_sphere = trimesh.primitives.Sphere(radius=radius/15, center=center_of_mass)
            com_sphere.visual.face_colors = [0, 0, 0, 255]
            scene.add_geometry(com_sphere)
            
            for idx, pts in enumerate(all_points_by_camera):
                color = camera_colors[idx % len(camera_colors)]
                pc = trimesh.points.PointCloud(pts)
                pc.colors = np.tile(color + [200], (len(pts), 1))
                scene.add_geometry(pc)
            
            for idx, (cam_pos, target) in enumerate(zip(camera_positions, target_points)):
                direction = target - cam_pos
                direction /= np.linalg.norm(direction)
                
                cam_marker = trimesh.creation.cone(radius=radius/20, height=radius/10)
                
                z_axis = np.array([0, 0, 1])
                rotation_axis = np.cross(z_axis, direction)
                if np.linalg.norm(rotation_axis) > 1e-6:
                    rotation_axis /= np.linalg.norm(rotation_axis)
                    angle = np.arccos(np.dot(z_axis, direction))
                    rotation = trimesh.transformations.rotation_matrix(angle, rotation_axis)
                else:
                    rotation = np.eye(4)
                
                translation = trimesh.transformations.translation_matrix(cam_pos)
                transform = trimesh.transformations.concatenate_matrices(translation, rotation)
                cam_marker.apply_transform(transform)
                
                color = camera_colors[idx % len(camera_colors)]
                cam_marker.visual.face_colors = np.tile(color + [255], (len(cam_marker.faces), 1))
                scene.add_geometry(cam_marker)
                
                target_marker = trimesh.primitives.Sphere(radius=radius/30, center=target)
                target_marker.visual.face_colors = np.tile(color + [200], (len(target_marker.faces), 1))
                scene.add_geometry(target_marker)
            
            if args.save:
                png = scene.save_image(resolution=[720, 480])
                with open(os.path.join(obj_output_dir, "visualization.png"), 'wb') as f:
                    f.write(png)
            
            scene.show(resolution=(640, 480))
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
