import hydra
from omegaconf import DictConfig
from src.preprocessing.mesh_to_pointcloud import process_all_meshes
from src.preprocessing.pointcloud_to_graph import process_point_cloud_files


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main preprocessing pipeline"""
    debug = cfg.get("debug", False)

    if debug:
        print("[DEBUG] Debug mode enabled")
        print("Configuration:")
        print(f"   Mesh directory: {cfg.preprocessing.lidar.mesh_dir}")
        print(f"   Output directory: {cfg.preprocessing.lidar.output_dir}")
        print(f"   Processed directory: {cfg.preprocessing.processed_dir}")
        print(f"   Number of cameras: {cfg.preprocessing.lidar.num_cameras}")
        print(f"   Number of samples: {cfg.preprocessing.lidar.get('num_samples', 1)}")
        print(f"   k-NN neighbors: {cfg.preprocessing.graph.k_nn}")
        print(
            f"   Spherical harmonics: {cfg.preprocessing.graph.get('use_spherical_harmonics', False)}"
        )
        print("")

    print("Starting preprocessing pipeline...")

    # Step 1: Generate and save point clouds
    process_all_meshes(cfg)

    # Step 2: Convert saved point clouds to graph data
    process_point_cloud_files(cfg)

    print("Preprocessing pipeline complete!")


if __name__ == "__main__":
    main()
