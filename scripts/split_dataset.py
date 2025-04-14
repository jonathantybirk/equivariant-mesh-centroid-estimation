import os
import sys
from pathlib import Path
import random
import shutil
import glob

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


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Split processed data into train and test sets"""
    
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
    
    # Shuffle the files
    random.shuffle(data_files)
    
    # Calculate split sizes
    total_files = len(data_files)
    train_size = int(total_files * train_ratio)
    test_size = total_files - train_size
    
    # Ensure at least one file in each split
    if train_size == 0:
        train_size = 1
        logger.warning("Train set would be empty. Assigning at least one file.")
    if test_size == 0:
        test_size = 1
        logger.warning("Test set would be empty. Assigning at least one file.")
    
    # Adjust sizes if necessary
    if train_size + test_size > total_files:
        excess = (train_size + test_size) - total_files
        if test_size > 1:
            test_size -= min(excess, test_size - 1)
        if train_size > 1 and excess > 0:
            train_size -= excess
    
    # Split the files
    train_files = data_files[:train_size]
    test_files = data_files[train_size:train_size + test_size]
    
    # Copy files to respective directories
    logger.info(f"Splitting {total_files} files into {train_size} train and {test_size} test files.")
    
    for src_file in train_files:
        dst_file = os.path.join(train_dir, os.path.basename(src_file))
        shutil.copy2(src_file, dst_file)
    
    for src_file in test_files:
        dst_file = os.path.join(test_dir, os.path.basename(src_file))
        shutil.copy2(src_file, dst_file)
    
    logger.info(f"Dataset split complete! Files in {train_dir} and {test_dir}.")


if __name__ == "__main__":
    main()