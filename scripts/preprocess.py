import hydra
from omegaconf import DictConfig
from src.preprocessing.mesh_to_pointcloud import process_all_meshes
from src.preprocessing.pointcloud_to_graph import convert_all_pointclouds

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Step 1: Generate point clouds from meshes
    process_all_meshes(cfg)
    # Step 2: Convert saved point clouds into graph data
    convert_all_pointclouds(cfg)

if __name__ == "__main__":
    main()
