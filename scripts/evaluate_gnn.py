import os
import sys
from pathlib import Path

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
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from src.data.datamodule import PointCloudDataModule
from src.model.lightning_module import GNNLightningModule


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Evaluate a trained GNN model"""
    
    # Load best checkpoint - FIXED NESTING
    checkpoint_path = os.path.join(
        cfg.training.training.checkpoint_dir,
        f"{cfg.name}_final.ckpt"
    )
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Looking for best checkpoint...")
        # Try to find the best checkpoint based on naming pattern
        checkpoint_dir = cfg.training.training.checkpoint_dir
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(cfg.name) and f.endswith('.ckpt')]
            if checkpoints:
                # Sort by validation loss (assuming filename format includes val_loss)
                checkpoints.sort(key=lambda x: float(x.split('-')[-1].replace('.ckpt', '')))
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                print(f"Using checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")
    
    # Load model from checkpoint
    model = GNNLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Create data module - FIXED NESTING
    data_module = PointCloudDataModule(
        processed_dir=cfg.preprocessing.preprocessing.processed_dir,
        batch_size=1,  # Evaluate one sample at a time for detailed analysis
        num_workers=0,
        pin_memory=False,
        node_feature_dim=model.hparams.hidden_dim
    )
    data_module.setup(stage='test')
    test_dataset = data_module.test_dataset
    
    # Prepare for evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create directories for results
    results_dir = 'evaluation_results'
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Evaluate all test samples
    errors = []
    
    print(f"Evaluating {len(test_dataset)} test samples...")
    for idx in tqdm(range(len(test_dataset))):
        data = test_dataset[idx]
        data = data.to(device)
        
        # Add batch dimension for single sample
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
        
        # Get prediction
        with torch.no_grad():
            pred_com = model(data)
        
        # Calculate error
        true_com = data.y
        error = torch.norm(pred_com - true_com).item()
        errors.append(error)
        
        # Create visualization for this sample
        if idx < 10:  # Only visualize first 10 samples
            # Create 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get point cloud and predictions as numpy arrays
            point_cloud = data.pos.cpu().numpy()
            true_com_np = true_com.cpu().numpy()
            pred_com_np = pred_com[0].cpu().numpy()
            
            # Plot point cloud
            ax.scatter(
                point_cloud[:, 0], 
                point_cloud[:, 1], 
                point_cloud[:, 2], 
                c='blue', s=10, alpha=0.5,
                label='Point Cloud'
            )
            
            # Plot true center of mass
            ax.scatter(
                true_com_np[0], true_com_np[1], true_com_np[2],
                c='green', s=150, marker='*',
                label='True Center of Mass'
            )
            
            # Plot predicted center of mass
            ax.scatter(
                pred_com_np[0], pred_com_np[1], pred_com_np[2],
                c='red', s=150, marker='x',
                label='Predicted Center of Mass'
            )
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"Test Sample {idx} (Error: {error:.4f})")
            ax.legend()
            
            # Save figure
            plt.savefig(os.path.join(viz_dir, f'sample_{idx:03d}.png'), dpi=150)
            plt.close(fig)
    
    # Calculate statistics
    errors = np.array(errors)
    stats = {
        'mean_error': errors.mean(),
        'median_error': np.median(errors),
        'min_error': errors.min(),
        'max_error': errors.max(),
        'std_error': errors.std()
    }
    
    # Print statistics
    print("\nEvaluation Results:")
    print(f"Mean Error: {stats['mean_error']:.4f}")
    print(f"Median Error: {stats['median_error']:.4f}")
    print(f"Min Error: {stats['min_error']:.4f}")
    print(f"Max Error: {stats['max_error']:.4f}")
    print(f"Std Dev: {stats['std_error']:.4f}")
    
    # Save results
    pd.DataFrame(stats, index=[0]).to_csv(
        os.path.join(results_dir, 'error_statistics.csv'), 
        index=False
    )
    
    # Plot error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, color='blue', alpha=0.7)
    plt.axvline(stats['mean_error'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean_error']:.4f}")
    plt.axvline(stats['median_error'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median_error']:.4f}")
    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'error_histogram.png'), dpi=150)
    
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()