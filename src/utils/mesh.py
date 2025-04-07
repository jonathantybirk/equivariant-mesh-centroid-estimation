# equivariant-center-of-mass-estimation/src/utils/mesh.py
import numpy as np
import trimesh

def load_mesh(filepath):
    mesh_obj = trimesh.load(filepath, force='mesh')
    if not mesh_obj.is_watertight:
        print(f"Warning: Mesh at {filepath} is not watertight; mass properties may be inaccurate.")
    return mesh_obj

def compute_center_of_mass(mesh_obj):
    mass_props = mesh_obj.mass_properties
    return mass_props.center_mass

def random_point_on_mesh(mesh_obj):
    triangles = mesh_obj.triangles
    areas = trimesh.triangles.area(triangles)
    probabilities = areas / areas.sum()
    tri_idx = np.random.choice(len(triangles), p=probabilities)
    triangle = triangles[tri_idx]
    u = np.random.random()
    v = np.random.random() * (1 - u)
    w = 1 - u - v
    return u * triangle[0] + v * triangle[1] + w * triangle[2]

def select_visible_target_point(mesh_obj, camera_pos, num_attempts=100):
    # Try to select a visible point by casting a random ray from the camera.
    for _ in range(num_attempts):
        random_dir = np.random.randn(3)
        random_dir = random_dir / np.linalg.norm(random_dir)
        origins = np.array([camera_pos])
        directions = np.array([random_dir])
        locations, index_ray, _ = mesh_obj.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions
        )
        if len(locations) > 0:
            distances = np.linalg.norm(locations - camera_pos, axis=1)
            return locations[np.argmin(distances)]
    print("Warning: Could not find a visible target point; using a random point on the mesh.")
    return random_point_on_mesh(mesh_obj)
