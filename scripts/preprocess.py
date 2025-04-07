# equivariant-center-of-mass-estimation/scripts/preprocess.py
import argparse
from src.preprocessing.obj_to_pc import process_all_meshes

def main():
    parser = argparse.ArgumentParser(description="Preprocess .obj files into LiDAR point clouds and metadata.")
    parser.add_argument("--save", action="store_true", help="Save the processed outputs to disk")
    parser.add_argument("--no-visualize", action="store_true", help="Do not display visualizations")
    args = parser.parse_args()
    
    process_all_meshes(save=args.save, visualize=not args.no_visualize)

if __name__ == "__main__":
    main()
