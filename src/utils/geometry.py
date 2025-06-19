import numpy as np


def random_point_on_sphere(radius=1.0):
    phi = np.random.uniform(0, 2 * np.pi)
    cos_theta = np.random.uniform(-1, 1)
    theta = np.arccos(cos_theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.array([x, y, z])


def rotate_vector(vector, axis, angle):
    vector = np.array(vector)
    axis = np.array(axis) / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return (
        vector * cos_angle
        + np.cross(axis, vector) * sin_angle
        + axis * np.dot(axis, vector) * (1 - cos_angle)
    )


def perturb_direction(direction, max_angle=np.pi / 6):
    rand_vec = np.random.randn(3)
    axis = np.cross(direction, rand_vec)
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([1, 0, 0])
    axis = axis / np.linalg.norm(axis)
    angle = np.random.uniform(-max_angle, max_angle)
    return rotate_vector(direction, axis, angle)


def generate_camera_positions(center, radius, num_cameras=3):
    num_cameras = max(1, min(6, num_cameras))
    cameras = []
    # First camera: random direction
    dir1 = random_point_on_sphere(radius=1.0)
    cam1 = center + radius * dir1
    cameras.append(cam1)

    if num_cameras == 1:
        return cameras

    # Determine two orthogonal axes for additional camera placements
    arbitrary = np.array([0, 0, 1]) if abs(dir1[2]) < 0.9 else np.array([0, 1, 0])
    ortho1 = np.cross(dir1, arbitrary)
    ortho1 = ortho1 / np.linalg.norm(ortho1)
    ortho2 = np.cross(dir1, ortho1)
    ortho2 = ortho2 / np.linalg.norm(ortho2)

    base_directions = [-dir1, ortho1, -ortho1, ortho2, -ortho2]
    np.random.shuffle(base_directions)

    max_angle = np.pi / 9  # about 20 degrees
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
