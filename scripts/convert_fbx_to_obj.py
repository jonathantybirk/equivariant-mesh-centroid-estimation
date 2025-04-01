"""
FBX to OBJ Conversion Script

Usage (run from within Blender):
    blender --background --python scripts/convert_fbx_to_obj.py

This script:
  1. Finds all .fbx files in data/meshes/fbx/
  2. Loads each in a new Blender scene
  3. Exports as .obj to data/meshes/obj/
"""

import bpy
import os
import glob

def convert_fbx_to_obj(input_folder, output_folder):
    """Convert all FBX files in 'input_folder' to OBJ, saving them in 'output_folder'."""
    fbx_files = glob.glob(os.path.join(input_folder, "*.fbx"))
    if not fbx_files:
        print(f"No FBX files found in {input_folder}")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    for fbx_path in fbx_files:
        # Reset Blender to an empty scene
        bpy.ops.wm.read_homefile(use_empty=True)
        
        # Import the FBX file
        bpy.ops.import_scene.fbx(filepath=fbx_path)
        
        # Output OBJ path
        base_name = os.path.splitext(os.path.basename(fbx_path))[0]
        obj_path = os.path.join(output_folder, base_name + ".obj")
        
        # Export everything in the scene as OBJ
        bpy.ops.export_scene.obj(filepath=obj_path, use_selection=False)
        
        print(f"Converted {fbx_path} -> {obj_path}")

def main():
    # Derive paths relative to this script location
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder = os.path.join(repo_root, "data", "meshes", "fbx")
    output_folder = os.path.join(repo_root, "data", "meshes", "obj")
    
    convert_fbx_to_obj(input_folder, output_folder)

if __name__ == "__main__":
    main()
