#!/usr/bin/env python3
"""
3D Mesh Visualization Script with Center of Mass and Centroid Display

This script provides an interactive 3D visualization of mesh objects showing:
- Transparent mesh rendering
- Center of mass (red sphere)
- Centroid (blue sphere) 
- Interactive controls for navigation and mesh selection

By default, the script shows meshes sequentially (one at a time) in random order.
Close a mesh window to see the next one automatically.

Usage:
    python visualize_meshes.py [--mesh_dir PATH] [--random] [--num_meshes N]
    python visualize_meshes.py --multi [--num_meshes N]  # Show multiple meshes in grid
    python visualize_meshes.py --single  # Show only one mesh
    python visualize_meshes.py --analyze  # Show statistical analysis
"""

import argparse
import os
import random
from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches

# Import existing mesh utilities
import sys
sys.path.append('src')
from utils.mesh import load_mesh, compute_center_of_mass


def compute_centroid(mesh_obj):
    """Compute the centroid (geometric center) of a mesh."""
    return mesh_obj.centroid


def create_sphere_mesh(center, radius=0.05, color='red'):
    """Create a small sphere mesh for visualization markers."""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    sphere.vertices += center
    sphere.visual.face_colors = color
    return sphere


def visualize_single_mesh(mesh_obj, mesh_name, show_wireframe=False):
    """
    Visualize a single mesh with center of mass and centroid.
    
    Args:
        mesh_obj: Trimesh mesh object
        mesh_name: Name of the mesh for display
        show_wireframe: Whether to show wireframe overlay
    """
    # Compute centers
    center_of_mass = compute_center_of_mass(mesh_obj)
    centroid = compute_centroid(mesh_obj)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get mesh vertices and faces
    vertices = mesh_obj.vertices
    faces = mesh_obj.faces
    
    # Create mesh collection with transparency
    mesh_collection = Poly3DCollection(
        vertices[faces], 
        alpha=0.3,  # Transparency
        facecolors='lightgray',
        edgecolors='gray' if show_wireframe else None,
        linewidths=0.5 if show_wireframe else 0
    )
    ax.add_collection3d(mesh_collection)
    
    # Add center of mass (red sphere)
    ax.scatter(
        center_of_mass[0], center_of_mass[1], center_of_mass[2],
        c='red', s=100, alpha=1.0, label='Center of Mass'
    )
    
    # Add centroid (blue sphere)  
    ax.scatter(
        centroid[0], centroid[1], centroid[2],
        c='blue', s=100, alpha=1.0, label='Centroid'
    )
    
    # Set equal aspect ratio
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
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Mesh: {mesh_name}')
    ax.legend()
    
    # Add distance information
    distance = np.linalg.norm(center_of_mass - centroid)
    ax.text2D(0.05, 0.95, f'Distance between centers: {distance:.4f}', 
              transform=ax.transAxes, fontsize=10, 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig, ax


def visualize_multiple_meshes(mesh_objects, mesh_names, max_display=4):
    """
    Visualize multiple meshes in a grid layout.
    
    Args:
        mesh_objects: List of trimesh mesh objects
        mesh_names: List of mesh names
        max_display: Maximum number of meshes to display in grid
    """
    num_meshes = min(len(mesh_objects), max_display)
    cols = 2
    rows = (num_meshes + 1) // 2
    
    fig = plt.figure(figsize=(15, 7 * rows))
    
    for i in range(num_meshes):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        mesh_obj = mesh_objects[i]
        mesh_name = mesh_names[i]
        
        # Compute centers
        center_of_mass = compute_center_of_mass(mesh_obj)
        centroid = compute_centroid(mesh_obj)
        
        # Get mesh vertices and faces
        vertices = mesh_obj.vertices
        faces = mesh_obj.faces
        
        # Create mesh collection with transparency
        mesh_collection = Poly3DCollection(
            vertices[faces], 
            alpha=0.4,  # Transparency
            facecolors='lightgray',
            edgecolors='gray',
            linewidths=0.3
        )
        ax.add_collection3d(mesh_collection)
        
        # Add center points
        ax.scatter(
            center_of_mass[0], center_of_mass[1], center_of_mass[2],
            c='red', s=80, alpha=1.0, label='Center of Mass'
        )
        ax.scatter(
            centroid[0], centroid[1], centroid[2],
            c='blue', s=80, alpha=1.0, label='Centroid'
        )
        
        # Set equal aspect ratio
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
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{mesh_name}')
        if i == 0:  # Only show legend on first plot
            ax.legend()
        
        # Add distance information
        distance = np.linalg.norm(center_of_mass - centroid)
        ax.text2D(0.05, 0.95, f'Dist: {distance:.3f}', 
                  transform=ax.transAxes, fontsize=8, 
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig


def get_mesh_files(mesh_dir, num_meshes=None, random_selection=False):
    """
    Get list of mesh files from directory.
    
    Args:
        mesh_dir: Path to mesh directory
        num_meshes: Number of meshes to load (None for all)
        random_selection: Whether to randomly select meshes
    
    Returns:
        List of mesh file paths
    """
    mesh_dir = Path(mesh_dir)
    mesh_files = list(mesh_dir.glob('*.obj'))
    
    if not mesh_files:
        raise ValueError(f"No .obj files found in {mesh_dir}")
    
    if random_selection:
        random.shuffle(mesh_files)
    
    if num_meshes is not None:
        mesh_files = mesh_files[:num_meshes]
    
    return mesh_files


def analyze_center_differences(mesh_objects, mesh_names):
    """
    Analyze the differences between center of mass and centroid across meshes.
    
    Args:
        mesh_objects: List of mesh objects
        mesh_names: List of mesh names
    
    Returns:
        Dictionary with analysis results
    """
    distances = []
    volumes = []
    
    for mesh_obj in mesh_objects:
        center_of_mass = compute_center_of_mass(mesh_obj)
        centroid = compute_centroid(mesh_obj)
        distance = np.linalg.norm(center_of_mass - centroid)
        
        distances.append(distance)
        volumes.append(mesh_obj.volume)
    
    analysis = {
        'distances': distances,
        'volumes': volumes,
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'mean_volume': np.mean(volumes),
    }
    
    return analysis


def plot_analysis(analysis, mesh_names):
    """Plot analysis of center differences."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distance histogram
    ax1.hist(analysis['distances'], bins=20, alpha=0.7, color='purple')
    ax1.set_xlabel('Distance between Center of Mass and Centroid')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Center Distances')
    ax1.axvline(analysis['mean_distance'], color='red', linestyle='--', 
                label=f'Mean: {analysis["mean_distance"]:.4f}')
    ax1.legend()
    
    # Distance vs Volume scatter plot
    ax2.scatter(analysis['volumes'], analysis['distances'], alpha=0.7, color='green')
    ax2.set_xlabel('Mesh Volume')
    ax2.set_ylabel('Distance between Centers')
    ax2.set_title('Center Distance vs Mesh Volume')
    
    plt.tight_layout()
    return fig


def visualize_meshes_sequentially(mesh_files, wireframe=False):
    """
    Visualize meshes one at a time, automatically opening the next when current is closed.
    
    Args:
        mesh_files: List of mesh file paths
        wireframe: Whether to show wireframe overlay
    """
    print(f"Starting sequential visualization of {len(mesh_files)} meshes...")
    print("Close the current mesh window to see the next one.")
    print("Press Ctrl+C to stop the visualization.")
    
    try:
        for i, mesh_file in enumerate(mesh_files):
            print(f"\nShowing mesh {i+1}/{len(mesh_files)}: {mesh_file.name}")
            
            try:
                # Load the current mesh
                mesh_obj = load_mesh(str(mesh_file))
                mesh_name = mesh_file.stem
                
                # Visualize the mesh
                fig, ax = visualize_single_mesh(mesh_obj, mesh_name, wireframe)
                
                # Set up the window close event to continue to next mesh
                plt.show(block=True)  # This will block until window is closed
                
            except Exception as e:
                print(f"  Failed to load/display {mesh_file.name}: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    
    print("All meshes have been displayed.")


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D meshes with center of mass and centroid')
    parser.add_argument('--mesh_dir', type=str, default='data/meshes',
                       help='Path to mesh directory (default: data/meshes)')
    parser.add_argument('--num_meshes', type=int, default=4,
                       help='Number of meshes to visualize (default: 4)')
    parser.add_argument('--random', action='store_true', default=True,
                       help='Randomly select meshes (default: True)')
    parser.add_argument('--single', action='store_true',
                       help='Show only one mesh in detail')
    parser.add_argument('--sequential', action='store_true', default=True,
                       help='Show meshes one at a time, opening next when current is closed (default: True)')
    parser.add_argument('--multi', action='store_true',
                       help='Show multiple meshes in grid layout (overrides sequential)')
    parser.add_argument('--analyze', action='store_true',
                       help='Show analysis of center differences')
    parser.add_argument('--wireframe', action='store_true',
                       help='Show wireframe overlay')
    
    args = parser.parse_args()
    
    # Check if mesh directory exists
    if not os.path.exists(args.mesh_dir):
        print(f"Error: Mesh directory {args.mesh_dir} does not exist")
        return
    
    # Get mesh files
    try:
        if args.sequential and not args.multi and not args.analyze:
            # For sequential mode, load all available meshes and randomize if requested
            mesh_files = get_mesh_files(args.mesh_dir, None, args.random)
        else:
            mesh_files = get_mesh_files(
                args.mesh_dir, 
                args.num_meshes if not args.analyze else None,  # Load all for analysis
                args.random
            )
        print(f"Found {len(mesh_files)} mesh files")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Handle sequential visualization mode (default behavior)
    if args.sequential and not args.multi and not args.analyze and not args.single:
        visualize_meshes_sequentially(mesh_files, args.wireframe)
        return
    
    # Load meshes (for non-sequential modes)
    mesh_objects = []
    mesh_names = []
    
    print("Loading meshes...")
    for mesh_file in mesh_files:
        try:
            mesh_obj = load_mesh(str(mesh_file))
            mesh_objects.append(mesh_obj)
            mesh_names.append(mesh_file.stem)
            print(f"  Loaded: {mesh_file.name}")
        except Exception as e:
            print(f"  Failed to load {mesh_file.name}: {e}")
    
    if not mesh_objects:
        print("Error: No meshes could be loaded")
        return
    
    print(f"Successfully loaded {len(mesh_objects)} meshes")
    
    # Visualization
    if args.single:
        # Show single mesh in detail
        fig, ax = visualize_single_mesh(mesh_objects[0], mesh_names[0], args.wireframe)
        plt.show()
    elif args.analyze:
        # Show analysis
        analysis = analyze_center_differences(mesh_objects, mesh_names)
        print("\nAnalysis Results:")
        print(f"  Mean distance: {analysis['mean_distance']:.4f}")
        print(f"  Std distance: {analysis['std_distance']:.4f}")
        print(f"  Max distance: {analysis['max_distance']:.4f}")
        print(f"  Min distance: {analysis['min_distance']:.4f}")
        
        # Show analysis plots
        plot_analysis(analysis, mesh_names)
        plt.show()
        
        # Show sample meshes
        visualize_multiple_meshes(mesh_objects[:4], mesh_names[:4])
        plt.show()
    elif args.multi:
        # Show multiple meshes in grid
        fig = visualize_multiple_meshes(mesh_objects, mesh_names, args.num_meshes)
        plt.show()
    else:
        # Default: sequential mode (this shouldn't be reached due to early return above)
        print("Sequential mode should have been handled earlier.")


if __name__ == "__main__":
    main()
