import numpy as np
import trimesh

def load_mesh(filepath):
    mesh_obj = trimesh.load(filepath, force='mesh')
    if not mesh_obj.is_watertight:
        print(f"Warning: Mesh at {filepath} is not watertight; mass properties may be inaccurate.")
    return mesh_obj

def compute_center_of_mass(mesh_obj):
    return mesh_obj.mass_properties.center_mass

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

def sample_surface_point(mesh_obj):
    areas = mesh_obj.area_faces
    tri_idx = np.random.choice(len(areas), p=areas / areas.sum())
    tri = mesh_obj.triangles[tri_idx]
    u = np.random.rand()
    v = np.random.rand() * (1 - u)
    w = 1 - u - v
    point = u * tri[0] + v * tri[1] + w * tri[2]
    return point, tri_idx

def is_visible_from_camera(mesh_obj, point, tri_idx, camera_pos):
    ray_dir = point - camera_pos
    ray_dir /= np.linalg.norm(ray_dir)

    origins = np.array([camera_pos])
    directions = np.array([ray_dir])

    locations, index_ray, index_tri = mesh_obj.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False
    )

    if len(locations) == 0:
        return False

    hit_point = locations[0]
    hit_tri = index_tri[0]

    if hit_tri != tri_idx:
        return False  # occluded

    normal = mesh_obj.face_normals[tri_idx]
    view_dir = camera_pos - hit_point
    view_dir /= np.linalg.norm(view_dir)

    return np.dot(normal, view_dir) > 0  # front-facing

def sample_visible_target_and_camera(mesh_obj, radius, max_attempts=200):
    """
    Sample a visible target point on a triangle and a camera position that can see it.
    """
    for _ in range(max_attempts):
        tri_idx = np.random.choice(
            len(mesh_obj.area_faces),
            p=mesh_obj.area_faces / mesh_obj.area_faces.sum()
        )
        tri = mesh_obj.triangles[tri_idx]

        # Sample a point inside the triangle using barycentric coordinates
        u = np.random.rand()
        v = np.random.rand() * (1 - u)
        w = 1 - u - v
        point = u * tri[0] + v * tri[1] + w * tri[2]

        cam_dir = trimesh.unitize(np.random.randn(3))
        camera_pos = mesh_obj.centroid + radius * cam_dir

        if is_visible_from_camera(mesh_obj, point, tri_idx, camera_pos):
            return point, camera_pos

    raise RuntimeError("Failed to find visible surface point from camera.")
