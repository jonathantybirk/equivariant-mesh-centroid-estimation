import os
import sys
from pathlib import Path
import random
import shutil
import glob
import re

# Change working directory to the project root
def find_project_root(current: Path, markers=(".git", "pyproject.toml")):
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError("Project root not found.")

root = find_project_root(Path(__file__).resolve())
os.chdir(root)
sys.path.insert(0, str(root))

import hydra
from omegaconf import DictConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_mesh_name(filename):
    """Extract base mesh name from a filename, removing sample indicators."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    # Handle patterns like "Chair_sample1"
    if "_sample" in basename:
        return basename.split("_sample")[0]
    return basename


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Split processed data into train and test sets, ensuring meshes stay together."""
    
    # Get configuration values
    processed_dir = cfg.preprocessing.processed_dir
    train_ratio = cfg.preprocessing.split.train_ratio
    test_ratio = cfg.preprocessing.split.test_ratio
    random_seed = cfg.preprocessing.split.random_seed
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Check if ratios sum to 1
    total_ratio = train_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        logger.warning(f"Split ratios sum to {total_ratio}, not 1.0. Normalizing ratios.")
        train_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Create output directories
    train_dir = os.path.join(processed_dir, "train")
    test_dir = os.path.join(processed_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get list of processed data files
    data_files = glob.glob(os.path.join(processed_dir, "*.pt"))
    
    if not data_files:
        logger.error(f"No .pt files found in {processed_dir}. Make sure to run preprocessing first.")
        return
    
    # Group files by mesh name
    mesh_groups = {}
    for file_path in data_files:
        mesh_name = extract_mesh_name(file_path)
        if mesh_name not in mesh_groups:
            mesh_groups[mesh_name] = []
        mesh_groups[mesh_name].append(file_path)
    
    logger.info(f"Found {len(data_files)} files from {len(mesh_groups)} unique meshes")
    
    # Split at the mesh level, not file level
    mesh_names = list(mesh_groups.keys())
    random.shuffle(mesh_names)
    
    train_size = int(len(mesh_names) * train_ratio)
    train_meshes = mesh_names[:train_size]
    test_meshes = mesh_names[train_size:]
    
    logger.info(f"Splitting into {len(train_meshes)} training meshes and {len(test_meshes)} testing meshes")
    
    # Copy files to respective directories
    train_count = 0
    test_count = 0
    
    # Copy train files
    for mesh in train_meshes:
        for src_file in mesh_groups[mesh]:
            dst_file = os.path.join(train_dir, os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)
            train_count += 1
    
    # Copy test files
    for mesh in test_meshes:
        for src_file in mesh_groups[mesh]:
            dst_file = os.path.join(test_dir, os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)
            test_count += 1
    
    logger.info(f"Dataset split complete! Copied {train_count} files to {train_dir} and {test_count} files to {test_dir}.")
    logger.info(f"Train meshes: {', '.join(train_meshes)}")
    logger.info(f"Test meshes: {', '.join(test_meshes)}")


if __name__ == "__main__":
    main()