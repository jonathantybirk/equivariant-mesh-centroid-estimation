# scripts/preprocess.py
import hydra
from omegaconf import DictConfig
from src.preprocessing.obj_to_pc import process_all_meshes

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    process_all_meshes(cfg)

if __name__ == "__main__":
    main()
