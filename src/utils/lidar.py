# equivariant-center-of-mass-estimation/src/utils/lidar.py
import numpy as np

def generate_lidar_rays(camera_pos, look_at, h_fov_deg=30, v_fov_deg=30, h_steps=20, v_steps=20):
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

def simulate_lidar(mesh_obj, camera_pos, target_point, max_distance=None, include_misses=False, **ray_params):
    if max_distance is None:
        max_distance = 2.0 * np.linalg.norm(target_point - camera_pos)
    
    origins, directions = generate_lidar_rays(camera_pos, target_point, **ray_params)
    locations, index_ray, _ = mesh_obj.ray.intersects_location(
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
