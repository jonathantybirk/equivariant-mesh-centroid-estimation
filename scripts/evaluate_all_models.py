#!/usr/bin/env python
"""
Comprehensive model evaluation and comparison script.
This script:
1. Resplits the dataset properly (grouping by mesh)
2. Evaluates all available model checkpoints
3. Generates comparison visualizations and reports

Usage:
    python scripts/evaluate_all_models.py [--resplit] [--models MODEL1 MODEL2...]
    
Options:
    --resplit: Force dataset resplitting even if already split
    --models: Space-separated list of model names to evaluate (default: all models in checkpoints/)
"""

import os
import sys
from pathlib import Path
import glob
import argparse
import subprocess
import re
import time
import logging
import shutil

# Change working directory to the project root
def find_project_root(current: Path, markers=(".git", "pyproject.toml")):
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError("Project root not found.")

root = find_project_root(Path(__file__).resolve())
os.chdir(root)
sys.path.insert(0, str(root))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def run_command(command):
    """Run a command with proper error handling"""
    logger.info(f"Running command: {command}")
    try:
        process = subprocess.run(command, shell=True, check=True, 
                               capture_output=True, text=True)
        return True, process.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with error code {e.returncode}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False, e.stderr

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate and compare all models')
    parser.add_argument('--resplit', action='store_true', help='Force dataset resplitting')
    parser.add_argument('--models', nargs='+', help='List of model names to evaluate (default: all)')
    args = parser.parse_args()
    
    # Step 1: Resplit the dataset if requested or if train/test directories don't exist
    processed_dir = 'data/processed'
    train_dir = os.path.join(processed_dir, 'train')
    test_dir = os.path.join(processed_dir, 'test')
    
    if args.resplit or not os.path.exists(train_dir) or not os.path.exists(test_dir):
        logger.info("Splitting dataset to ensure proper mesh grouping...")
        success, output = run_command("python scripts/split_dataset.py")
        if not success:
            logger.error("Dataset splitting failed. Exiting.")
            return
        logger.info("Dataset splitting completed successfully")
    else:
        logger.info("Using existing train/test split")
    
    # Step 2: Find unique models
    checkpoints_dir = 'checkpoints'
    
    if args.models:
        model_names = args.models
        logger.info(f"Using specified models: {', '.join(model_names)}")
    else:
        # Find all unique model names from checkpoint files
        checkpoint_files = glob.glob(os.path.join(checkpoints_dir, '*.ckpt'))
        if not checkpoint_files:
            logger.error(f"No checkpoint files found in {checkpoints_dir}")
            return
        
        # Extract model names from checkpoint filenames
        model_names = set()
        for checkpoint in checkpoint_files:
            basename = os.path.basename(checkpoint)
            model_name = basename.split('-')[0]
            if model_name.endswith('_final'):
                model_name = model_name[:-6]  # Remove '_final' suffix
            model_names.add(model_name)
        
        model_names = sorted(list(model_names))
        logger.info(f"Found {len(model_names)} unique model types: {', '.join(model_names)}")
    
    # Step 3: For each model, find its best checkpoint and evaluate
    for model_name in model_names:
        logger.info(f"Processing model: {model_name}")
        
        # Find the best checkpoint for this model
        checkpoints = glob.glob(os.path.join(checkpoints_dir, f"{model_name}*.ckpt"))
        if not checkpoints:
            logger.warning(f"No checkpoints found for {model_name}, skipping.")
            continue
        
        # Sort by validation loss (extract from filename)
        val_loss_pattern = r'val_loss=([0-9.]+)'
        
        def get_val_loss(filename):
            match = re.search(val_loss_pattern, filename)
            if match:
                # Clean up the value - remove any trailing periods
                val_loss_str = match.group(1).rstrip('.')
                try:
                    return float(val_loss_str)
                except ValueError:
                    logger.warning(f"Could not convert validation loss to float in {filename}, using high default value")
                    return float('inf')
            return float('inf')  # Default high value for files without val_loss
        
        # Find best checkpoint (lowest val_loss)
        best_checkpoint = min(checkpoints, key=get_val_loss)
        logger.info(f"Selected best checkpoint: {os.path.basename(best_checkpoint)}")
        
        # Determine model module path based on model name - FIXED to use the correct classes
        if model_name == 'gnn_baseline':
            model_module = 'src.model.lightning_module'  # Changed to use Lightning module wrapper
        elif model_name == 'se3_equivariant':
            model_module = 'src.model.SE3_equivariant'  # SE3 model already has Lightning integrated
        else:
            logger.warning(f"Unknown model type {model_name}, assuming default module")
            model_module = 'src.model.lightning_module'
        
        # Fix path for Hydra - convert backslashes to forward slashes
        checkpoint_path = best_checkpoint.replace('\\', '/')
        
        # Evaluate the model with correct Hydra syntax:
        # - Add + for checkpoint_path (appending to config)
        # - Do NOT add + for model.module_path (replacing default value)
        eval_cmd = f"python scripts/evaluate_gnn.py name={model_name} +checkpoint_path='\"{checkpoint_path}\"' model.module_path={model_module}"
        
        logger.info(f"Evaluating model {model_name} with checkpoint {os.path.basename(best_checkpoint)}")
        success, _ = run_command(eval_cmd)
        if not success:
            logger.error(f"Evaluation failed for model {model_name}")
            continue
        
        logger.info(f"Evaluation completed for {model_name}")
    
    # Step 4: Run model comparison to create combined visualizations
    logger.info("Comparing all evaluated models...")
    success, _ = run_command("python scripts/compare_models.py")
    if not success:
        logger.error("Model comparison failed")
        return
    
    logger.info("Model comparison completed successfully")
    logger.info("Results available in model_comparison/ directory")
    
    # Step 5: Print summary of results
    logger.info("\nEvaluation workflow completed! Summary:")
    logger.info("- Individual model results: evaluation_results/")
    logger.info("- Model comparison results: model_comparison/")
    logger.info("- Key metrics: model_comparison/model_comparison.csv")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Total evaluation time: {elapsed_time:.1f} seconds")