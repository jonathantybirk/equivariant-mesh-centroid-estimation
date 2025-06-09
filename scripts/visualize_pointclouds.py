#!/usr/bin/env python3
"""
3D Pointcloud Visualization Script with Centroids and Camera Positions

This script provides an interactive 3D visualization of preprocessed pointcloud data showing:
- Pointcloud points (colored by camera if individual files exist)
- Pointcloud centroid (red sphere) - center of normalized pointcloud
- Mesh centroid (blue sphere) - final position of original mesh center
- Camera positions (green spheres) - positions where LIDAR scans were taken
- Interactive controls for navigation and object selection

By default, the script shows pointclouds sequentially (one at a time) in random order.
Close a pointcloud window to see the next one automatically.

Usage:
    python visualize_pointclouds.py [--pointcloud_dir PATH] [--random] [--num_objects N]
    python visualize_pointclouds.py --multi [--num_objects N]  # Show multiple pointclouds in grid
    python visualize_pointclouds.py --single  # Show only one pointcloud
    python visualize_pointclouds.py --analyze  # Show statistical analysis
"""

import argparse
import os
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import mesh utilities for --show_mesh functionality
try:
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from src.utils.mesh import load_mesh, compute_center_of_mass
    MESH_SUPPORT = True
except ImportError:
    MESH_SUPPORT = False
    print("Warning: Mesh visualization dependencies not available. --show_mesh option will be disabled.")


def load_pointcloud_data(pointcloud_dir):
    """
    Load all pointcloud data from a directory.
    
    Returns:
        dict: Dictionary containing all loaded data:
            - pointcloud: Main combined pointcloud (N, 3)
            - mesh_centroid: Final mesh centroid position (3,)
            - camera_positions: Camera positions used for scanning (K, 3)
            - individual_clouds: List of individual camera pointclouds (optional)
    """
    data = {}
    
    # Load main pointcloud
    pointcloud_path = os.path.join(pointcloud_dir, "pointcloud.npy")
    if os.path.exists(pointcloud_path):
        data["pointcloud"] = np.load(pointcloud_path)
    else:
        raise FileNotFoundError(f"Main pointcloud not found: {pointcloud_path}")
    
    # Load mesh centroid
    mesh_centroid_path = os.path.join(pointcloud_dir, "mesh_centroid.npy")
    if os.path.exists(mesh_centroid_path):
        data["mesh_centroid"] = np.load(mesh_centroid_path)
    else:
        data["mesh_centroid"] = np.array([0.0, 0.0, 0.0])  # Default to origin
    
    # Load camera positions and target
    camera_positions_path = os.path.join(pointcloud_dir, "camera_positions.npy")
    if os.path.exists(camera_positions_path):
        camera_data = np.load(camera_positions_path)
        if len(camera_data) > 0:
            # First row is camera target, rest are camera positions
            data["camera_target"] = camera_data[0]
            data["camera_positions"] = camera_data[1:] if len(camera_data) > 1 else None
        else:
            data["camera_target"] = None
            data["camera_positions"] = None
    else:
        data["camera_target"] = None
        data["camera_positions"] = None
    
    # Load individual camera pointclouds (optional)
    individual_clouds = []
    for i in range(1, 7):  # Check for up to 6 cameras
        cam_path = os.path.join(pointcloud_dir, f"pointcloud_cam{i}.npy")
        if os.path.exists(cam_path):
            individual_clouds.append(np.load(cam_path))
    
    data["individual_clouds"] = individual_clouds if individual_clouds else None
    
    return data


def compute_pointcloud_centroid(pointcloud):
    """Compute the centroid (geometric center) of a pointcloud."""
    return np.mean(pointcloud, axis=0)


def visualize_single_pointcloud(data, object_name, show_individual_cameras=True):
    """
    Visualize a single pointcloud with all relevant information.
    
    Args:
        data: Dictionary containing pointcloud data
        object_name: Name of the object
        show_individual_cameras: Whether to color points by camera if available
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    pointcloud = data["pointcloud"]
    mesh_centroid = data["mesh_centroid"]
    camera_positions = data["camera_positions"]
    camera_target = data["camera_target"]
    individual_clouds = data["individual_clouds"]
    
    # Compute pointcloud centroid
    pointcloud_centroid = compute_pointcloud_centroid(pointcloud)
    
    # Plot pointcloud
    if show_individual_cameras and individual_clouds is not None:
        # Color points by camera
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        total_points = 0
        
        for i, cloud in enumerate(individual_clouds):
            if len(cloud) > 0:
                ax.scatter(
                    cloud[:, 0], cloud[:, 1], cloud[:, 2],
                    c=colors[i % len(colors)], s=2, alpha=0.6,
                    label=f'Camera {i+1} ({len(cloud)} pts)'
                )
                total_points += len(cloud)
    else:
        # Plot combined pointcloud
        ax.scatter(
            pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
            c='lightblue', s=2, alpha=0.6,
            label=f'Pointcloud ({len(pointcloud)} pts)'
        )
    
    # Plot pointcloud centroid (red sphere)
    ax.scatter(
        pointcloud_centroid[0], pointcloud_centroid[1], pointcloud_centroid[2],
        c='red', s=200, alpha=1.0, marker='o',
        label='Pointcloud Centroid', edgecolors='darkred', linewidth=2
    )
    
    # Plot mesh centroid (blue sphere) 
    ax.scatter(
        mesh_centroid[0], mesh_centroid[1], mesh_centroid[2],
        c='blue', s=200, alpha=1.0, marker='s',
        label='Mesh Centroid', edgecolors='darkblue', linewidth=2
    )
    
    # Plot camera target (orange sphere) - where cameras were aiming
    if camera_target is not None:
        ax.scatter(
            camera_target[0], camera_target[1], camera_target[2],
            c='orange', s=200, alpha=1.0, marker='*',
            label='Camera Target', edgecolors='darkorange', linewidth=2
        )
    
    # Plot camera positions (green spheres)
    if camera_positions is not None:
        ax.scatter(
            camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
            c='green', s=150, alpha=0.8, marker='^',
            label=f'Cameras ({len(camera_positions)})', edgecolors='darkgreen', linewidth=1
        )
        
        # Draw lines from cameras to camera target (where they were aiming)
        if camera_target is not None:
            for cam_pos in camera_positions:
                ax.plot(
                    [cam_pos[0], camera_target[0]],
                    [cam_pos[1], camera_target[1]], 
                    [cam_pos[2], camera_target[2]],
                    'g--', alpha=0.3, linewidth=1
                )
    
    # Set equal aspect ratio
    all_points = [pointcloud]
    if camera_positions is not None:
        all_points.append(camera_positions)
    if camera_target is not None:
        all_points.append(camera_target.reshape(1, -1))
    all_points.append(pointcloud_centroid.reshape(1, -1))
    all_points.append(mesh_centroid.reshape(1, -1))
    
    all_coords = np.vstack(all_points)
    max_range = np.array([
        all_coords[:, 0].max() - all_coords[:, 0].min(),
        all_coords[:, 1].max() - all_coords[:, 1].min(), 
        all_coords[:, 2].max() - all_coords[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
    mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
    mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Pointcloud: {object_name}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add distance information
    distance_pc_mesh = np.linalg.norm(pointcloud_centroid - mesh_centroid)
    
    info_text = f'Distance PC↔Mesh centroids: {distance_pc_mesh:.4f}\n'
    info_text += f'Pointcloud size: {len(pointcloud)} points\n'
    info_text += f'Pointcloud centroid: [{pointcloud_centroid[0]:.3f}, {pointcloud_centroid[1]:.3f}, {pointcloud_centroid[2]:.3f}]\n'
    info_text += f'Mesh centroid: [{mesh_centroid[0]:.3f}, {mesh_centroid[1]:.3f}, {mesh_centroid[2]:.3f}]\n'
    
    if camera_target is not None:
        distance_pc_target = np.linalg.norm(pointcloud_centroid - camera_target)
        info_text += f'Camera target: [{camera_target[0]:.3f}, {camera_target[1]:.3f}, {camera_target[2]:.3f}]\n'
    
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
              verticalalignment='top',
              bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    return fig, ax


def visualize_multiple_pointclouds(pointcloud_dirs, object_names, max_display=4, show_individual_cameras=True):
    """
    Visualize multiple pointclouds in a grid layout.
    
    Args:
        pointcloud_dirs: List of pointcloud directory paths
        object_names: List of object names
        max_display: Maximum number of pointclouds to display in grid
        show_individual_cameras: Whether to color points by camera if available
    """
    num_objects = min(len(pointcloud_dirs), max_display)
    cols = 2
    rows = (num_objects + 1) // 2
    
    fig = plt.figure(figsize=(20, 8 * rows))
    
    for i in range(num_objects):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        pointcloud_dir = pointcloud_dirs[i]
        object_name = object_names[i]
        
        try:
            data = load_pointcloud_data(pointcloud_dir)
            
            pointcloud = data["pointcloud"]
            mesh_centroid = data["mesh_centroid"]
            camera_positions = data["camera_positions"]
            individual_clouds = data["individual_clouds"]
            
            # Compute pointcloud centroid
            pointcloud_centroid = compute_pointcloud_centroid(pointcloud)
            
            # Plot pointcloud with individual camera coloring if requested
            if show_individual_cameras and individual_clouds is not None:
                # Color points by camera
                colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
                for j, cloud in enumerate(individual_clouds):
                    if len(cloud) > 0:
                        ax.scatter(
                            cloud[:, 0], cloud[:, 1], cloud[:, 2],
                            c=colors[j % len(colors)], s=1, alpha=0.6,
                            label=f'Cam{j+1}' if j < 3 else None  # Only label first 3 cameras for space
                        )
            else:
                # Plot combined pointcloud
                ax.scatter(
                    pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
                    c='lightblue', s=1, alpha=0.6
                )
            
            # Plot centroids
            ax.scatter(
                pointcloud_centroid[0], pointcloud_centroid[1], pointcloud_centroid[2],
                c='red', s=100, alpha=1.0, marker='o', label='PC Centroid'
            )
            
            ax.scatter(
                mesh_centroid[0], mesh_centroid[1], mesh_centroid[2],
                c='blue', s=100, alpha=1.0, marker='s', label='Mesh Centroid'
            )
            
            # Plot cameras
            if camera_positions is not None:
                ax.scatter(
                    camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                    c='green', s=80, alpha=0.8, marker='^', label='Cameras'
                )
            
            # Set equal aspect ratio (simplified)
            all_coords = pointcloud
            max_range = np.array([
                all_coords[:, 0].max() - all_coords[:, 0].min(),
                all_coords[:, 1].max() - all_coords[:, 1].min(),
                all_coords[:, 2].max() - all_coords[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
            mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
            mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{object_name}\n({len(pointcloud)} pts)')
            ax.legend(fontsize=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, 0.5, f'Error loading {object_name}:\n{str(e)}',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title(f'Error: {object_name}')
    
    plt.tight_layout()
    return fig


def analyze_pointcloud_statistics(pointcloud_dirs, object_names, sample_size=100):
    """
    Analyze and display statistics about the pointcloud dataset.
    """
    print("Analyzing pointcloud dataset...")
    
    stats = {
        'point_counts': [],
        'pc_centroid_distances': [],
        'mesh_centroid_distances': [],
        'pc_mesh_distances': [],
        'camera_counts': [],
        'camera_distances': []
    }
    
    successful_loads = 0
    errors = []
    
    # Sample random subset if dataset is large
    if len(pointcloud_dirs) > sample_size:
        indices = random.sample(range(len(pointcloud_dirs)), sample_size)
        sample_dirs = [pointcloud_dirs[i] for i in indices]
        sample_names = [object_names[i] for i in indices]
    else:
        sample_dirs = pointcloud_dirs
        sample_names = object_names
    
    for pointcloud_dir, object_name in zip(sample_dirs, sample_names):
        try:
            data = load_pointcloud_data(pointcloud_dir)
            
            pointcloud = data["pointcloud"]
            mesh_centroid = data["mesh_centroid"]
            camera_positions = data["camera_positions"]
            
            pointcloud_centroid = compute_pointcloud_centroid(pointcloud)
            
            # Collect statistics
            stats['point_counts'].append(len(pointcloud))
            stats['pc_centroid_distances'].append(np.linalg.norm(pointcloud_centroid))
            stats['mesh_centroid_distances'].append(np.linalg.norm(mesh_centroid))
            stats['pc_mesh_distances'].append(np.linalg.norm(pointcloud_centroid - mesh_centroid))
            
            if camera_positions is not None:
                stats['camera_counts'].append(len(camera_positions))
                avg_cam_dist = np.mean([np.linalg.norm(cam - mesh_centroid) for cam in camera_positions])
                stats['camera_distances'].append(avg_cam_dist)
            
            successful_loads += 1
            
        except Exception as e:
            errors.append(f"{object_name}: {str(e)}")
    
    # Display statistics
    print(f"\nPointcloud Dataset Analysis")
    print("=" * 50)
    print(f"Total objects found: {len(pointcloud_dirs)}")
    print(f"Successfully analyzed: {successful_loads}")
    print(f"Errors: {len(errors)}")
    
    if successful_loads > 0:
        point_counts = np.array(stats['point_counts'])
        pc_dists = np.array(stats['pc_centroid_distances'])
        mesh_dists = np.array(stats['mesh_centroid_distances'])
        pc_mesh_dists = np.array(stats['pc_mesh_distances'])
        
        print(f"\nPoint Count Statistics:")
        print(f"  Mean: {np.mean(point_counts):.1f}")
        print(f"  Std:  {np.std(point_counts):.1f}")
        print(f"  Min:  {np.min(point_counts)}")
        print(f"  Max:  {np.max(point_counts)}")
        
        print(f"\nPointcloud Centroid Distance from Origin:")
        print(f"  Mean: {np.mean(pc_dists):.4f}")
        print(f"  Std:  {np.std(pc_dists):.4f}")
        print(f"  Min:  {np.min(pc_dists):.4f}")
        print(f"  Max:  {np.max(pc_dists):.4f}")
        
        print(f"\nMesh Centroid Distance from Origin:")
        print(f"  Mean: {np.mean(mesh_dists):.4f}")
        print(f"  Std:  {np.std(mesh_dists):.4f}")
        print(f"  Min:  {np.min(mesh_dists):.4f}")
        print(f"  Max:  {np.max(mesh_dists):.4f}")
        
        print(f"\nDistance between PC and Mesh Centroids:")
        print(f"  Mean: {np.mean(pc_mesh_dists):.4f}")
        print(f"  Std:  {np.std(pc_mesh_dists):.4f}")
        print(f"  Min:  {np.min(pc_mesh_dists):.4f}")
        print(f"  Max:  {np.max(pc_mesh_dists):.4f}")
        
        if stats['camera_counts']:
            cam_counts = np.array(stats['camera_counts'])
            cam_dists = np.array(stats['camera_distances'])
            
            print(f"\nCamera Statistics:")
            print(f"  Cameras per object: {np.mean(cam_counts):.1f} ± {np.std(cam_counts):.1f}")
            print(f"  Average camera distance: {np.mean(cam_dists):.3f} ± {np.std(cam_dists):.3f}")
    
    if errors:
        print(f"\nErrors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D pointclouds with centroids and cameras')
    parser.add_argument('--pointcloud_dir', type=str, default='data/pointclouds',
                        help='Directory containing pointcloud subdirectories')
    parser.add_argument('--random', action='store_true',
                        help='Randomize the order of pointclouds')
    parser.add_argument('--num_objects', type=int, default=None,
                        help='Number of objects to visualize (default: all)')
    parser.add_argument('--multi', action='store_true',
                        help='Show multiple pointclouds in a grid layout')
    parser.add_argument('--single', action='store_true', 
                        help='Show only one pointcloud')
    parser.add_argument('--analyze', action='store_true',
                        help='Perform statistical analysis instead of visualization')
    parser.add_argument('--individual_cameras', action='store_true',
                        help='Color points by individual cameras (default: off)')
    parser.add_argument('--show_mesh', action='store_true',
                        help='Display original mesh alongside pointcloud')
    parser.add_argument('--mesh_dir', type=str, default='data/meshes',
                        help='Directory containing original mesh files (for --show_mesh option)')
    
    args = parser.parse_args()
    
    # Find all pointcloud directories
    if not os.path.exists(args.pointcloud_dir):
        print(f"Error: Pointcloud directory not found: {args.pointcloud_dir}")
        return
    
    pointcloud_dirs = []
    object_names = []
    
    for item in os.listdir(args.pointcloud_dir):
        item_path = os.path.join(args.pointcloud_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains pointcloud files
            pointcloud_file = os.path.join(item_path, "pointcloud.npy")
            if os.path.exists(pointcloud_file):
                pointcloud_dirs.append(item_path)
                object_names.append(item)
    
    if not pointcloud_dirs:
        print(f"No valid pointcloud directories found in {args.pointcloud_dir}")
        return
    
    print(f"Found {len(pointcloud_dirs)} pointcloud objects")
    
    # Shuffle if requested
    if args.random:
        combined = list(zip(pointcloud_dirs, object_names))
        random.shuffle(combined)
        pointcloud_dirs, object_names = zip(*combined)
        pointcloud_dirs, object_names = list(pointcloud_dirs), list(object_names)
    
    # Limit number if specified
    if args.num_objects is not None:
        pointcloud_dirs = pointcloud_dirs[:args.num_objects]
        object_names = object_names[:args.num_objects]
        print(f"Limited to {len(pointcloud_dirs)} objects")
    
    # Perform analysis if requested
    if args.analyze:
        analyze_pointcloud_statistics(pointcloud_dirs, object_names)
        return
    
    # Show visualizations
    if args.multi:
        print("Displaying multiple pointclouds in grid layout...")
        if args.show_mesh:
            print("Note: Mesh visualization is not supported in multi-grid mode. Use --single for mesh display.")
        fig = visualize_multiple_pointclouds(pointcloud_dirs, object_names, max_display=6, 
                                            show_individual_cameras=args.individual_cameras)
        plt.show()
        
    elif args.single:
        if pointcloud_dirs:
            print(f"Displaying single pointcloud: {object_names[0]}")
            data = load_pointcloud_data(pointcloud_dirs[0])
            if args.show_mesh:
                fig, ax = visualize_single_pointcloud_with_mesh(data, object_names[0], 
                                                               args.individual_cameras, args.mesh_dir)
            else:
                fig, ax = visualize_single_pointcloud(data, object_names[0], 
                                                     args.individual_cameras)
            plt.show()
        
    else:
        # Sequential visualization (default)
        print("Displaying pointclouds sequentially. Close window to see next.")
        for pointcloud_dir, object_name in zip(pointcloud_dirs, object_names):
            try:
                print(f"Loading: {object_name}")
                data = load_pointcloud_data(pointcloud_dir)
                if args.show_mesh:
                    fig, ax = visualize_single_pointcloud_with_mesh(data, object_name,
                                                                   args.individual_cameras, args.mesh_dir)
                else:
                    fig, ax = visualize_single_pointcloud(data, object_name,
                                                         args.individual_cameras)
                plt.show()
            except Exception as e:            print(f"Error loading {object_name}: {e}")
            continue


def visualize_single_pointcloud_with_mesh(data, object_name, show_individual_cameras=True, mesh_dir='data/meshes'):
    """
    Visualize a single pointcloud alongside its original mesh.
    
    Args:
        data: Dictionary containing pointcloud data
        object_name: Name of the object
        show_individual_cameras: Whether to color points by camera if available
        mesh_dir: Directory containing original mesh files
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    if not MESH_SUPPORT:
        print("Error: Mesh visualization not available. Please install trimesh.")
        return visualize_single_pointcloud(data, object_name, show_individual_cameras)
    
    # Try to load the corresponding mesh
    mesh_path = os.path.join(mesh_dir, f"{object_name}.obj")
    
    try:
        mesh_obj = load_mesh(mesh_path)
    except Exception as e:
        print(f"Warning: Could not load mesh for {object_name}: {e}")
        print("Falling back to pointcloud-only visualization")
        return visualize_single_pointcloud(data, object_name, show_individual_cameras)
    
    # Create figure with three subplots side by side
    fig = plt.figure(figsize=(18, 6))
    
    pointcloud = data["pointcloud"]
    mesh_centroid = data["mesh_centroid"]
    camera_positions = data["camera_positions"]
    individual_clouds = data["individual_clouds"]
    
    # Compute centers
    pointcloud_centroid = compute_pointcloud_centroid(pointcloud)
    mesh_center_of_mass = compute_center_of_mass(mesh_obj)
    
    # Plot 1: Original mesh with center of mass
    ax1 = fig.add_subplot(131, projection='3d')
    vertices = mesh_obj.vertices
    faces = mesh_obj.faces
    
    # Create mesh collection with transparency
    mesh_collection = Poly3DCollection(
        vertices[faces], 
        alpha=0.4,
        facecolors='lightgray',
        edgecolors='gray',
        linewidths=0.3
    )
    ax1.add_collection3d(mesh_collection)
    
    # Add center of mass
    ax1.scatter(
        mesh_center_of_mass[0], mesh_center_of_mass[1], mesh_center_of_mass[2],
        c='red', s=100, alpha=1.0, label='Center of Mass'
    )
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y') 
    ax1.set_zlabel('Z')
    ax1.set_title(f'Original Mesh: {object_name}')
    ax1.legend()
    
    # Plot 2: Generated pointcloud
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Plot pointcloud
    if show_individual_cameras and individual_clouds is not None:
        # Color points by camera
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        for i, cloud in enumerate(individual_clouds):
            if len(cloud) > 0:
                ax2.scatter(
                    cloud[:, 0], cloud[:, 1], cloud[:, 2],
                    c=colors[i % len(colors)], s=2, alpha=0.6,
                    label=f'Camera {i+1} ({len(cloud)} pts)'
                )
    else:
        # Plot combined pointcloud
        ax2.scatter(
            pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
            c='blue', s=2, alpha=0.6,
            label=f'Pointcloud ({len(pointcloud)} pts)'
        )
    
    # Add pointcloud centroid
    ax2.scatter(
        pointcloud_centroid[0], pointcloud_centroid[1], pointcloud_centroid[2],
        c='red', s=100, alpha=1.0, label='PC Centroid'
    )
    
    # Add mesh centroid
    ax2.scatter(
        mesh_centroid[0], mesh_centroid[1], mesh_centroid[2],
        c='blue', s=100, alpha=1.0, marker='s', label='Mesh Centroid'
    )
    
    # Add camera positions if available
    if camera_positions is not None:
        ax2.scatter(
            camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
            c='green', s=50, marker='^', alpha=0.8,
            label=f'Cameras ({len(camera_positions)})'
        )
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Generated Pointcloud: {object_name}')
    ax2.legend()
    
    # Plot 3: Overlay view
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Add transparent mesh
    mesh_collection_overlay = Poly3DCollection(
        vertices[faces],
        alpha=0.15,
        facecolors='lightgray',
        edgecolors=None,
        linewidths=0
    )
    ax3.add_collection3d(mesh_collection_overlay)
    
    # Add pointcloud
    ax3.scatter(
        pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
        c='blue', s=2, alpha=0.7,
        label=f'Points ({len(pointcloud)})'
    )
    
    # Add center of mass
    ax3.scatter(
        mesh_center_of_mass[0], mesh_center_of_mass[1], mesh_center_of_mass[2],
        c='red', s=100, alpha=1.0, label='Center of Mass'
    )
    
    # Add mesh centroid
    ax3.scatter(
        mesh_centroid[0], mesh_centroid[1], mesh_centroid[2],
        c='blue', s=100, alpha=1.0, marker='s', label='Mesh Centroid'
    )
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'Overlay: {object_name}')
    ax3.legend()
    
    # Set equal aspect ratios for all plots
    for ax in [ax1, ax2, ax3]:
        max_range = np.array([
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
            vertices[:, 2].max() - vertices[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    main()