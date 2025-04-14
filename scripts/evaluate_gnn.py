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
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import pandas as pd
import glob

from src.data.datamodule import PointCloudDataModule
from src.model.lightning_module import GNNLightningModule


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Evaluate a trained GNN model with enhanced visualizations"""
    
    # Set up directories
    results_dir = 'evaluation_results'
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(os.path.join(viz_dir, 'best_cases'), exist_ok=True)
    os.makedirs(os.path.join(viz_dir, 'worst_cases'), exist_ok=True)
    os.makedirs(os.path.join(viz_dir, 'all_samples'), exist_ok=True)
    
    # Load checkpoint - allow custom checkpoint path via command line
    checkpoint_path = cfg.get('checkpoint_path', None)
    
    if not checkpoint_path:
        # Try to find the best checkpoint
        checkpoint_dir = cfg.training.checkpoint_dir
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, f"{cfg.name}*.ckpt")))
        
        if not checkpoints:
            # Fall back to final model
            checkpoint_path = os.path.join(checkpoint_dir, f"{cfg.name}_final.ckpt")
        else:
            # Use best checkpoint (assuming best is first when sorted by val_loss)
            checkpoint_path = checkpoints[0]
    
    print(f"Loading model from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    try:
        # Load model from checkpoint
        model = GNNLightningModule.load_from_checkpoint(checkpoint_path)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Create data module with CPU-friendly settings
    try:
        data_module = PointCloudDataModule(
            processed_dir=cfg.preprocessing.processed_dir,
            train_dir=os.path.join(cfg.preprocessing.processed_dir, "train"),
            test_dir=os.path.join(cfg.preprocessing.processed_dir, "test"),
            batch_size=1,  # Evaluate one sample at a time
            num_workers=0,  # No multiprocessing
            pin_memory=False,  # CPU-friendly
            node_feature_dim=model.hparams.hidden_dim
        )
        data_module.setup(stage='test')
        test_dataset = data_module.test_dataset
        print(f"Test dataset loaded: {len(test_dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Prepare for evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}")
    model = model.to(device)
    
    # Function to make predictions
    def predict(data):
        data = data.clone().to(device)
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
        with torch.no_grad():
            return model(data)
    
    # Evaluate all test samples
    errors = []
    predictions = []
    targets = []
    names = []
    
    print(f"Evaluating {len(test_dataset)} test samples...")
    try:
        for idx in tqdm(range(len(test_dataset))):
            data = test_dataset[idx]
            
            # Get prediction
            pred_com = predict(data)[0]
            
            # Calculate error
            true_com = data.y
            error = torch.norm(pred_com - true_com).item()
            errors.append(error)
            
            # Store predictions and targets
            predictions.append(pred_com.cpu().numpy())
            targets.append(true_com.cpu().numpy())
            
            # Store name if available
            name = f"Sample_{idx}" if not hasattr(data, 'name') else data.name
            names.append(name)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Convert to arrays
    errors = np.array(errors)
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate statistics
    stats = {
        'mean_error': errors.mean(),
        'median_error': np.median(errors),
        'min_error': errors.min(),
        'max_error': errors.max(),
        'std_error': errors.std(),
        'mse': np.mean(np.sum((predictions - targets)**2, axis=1)),
        'rmse': np.sqrt(np.mean(np.sum((predictions - targets)**2, axis=1)))
    }
    
    # Print statistics
    print("\n=== Evaluation Results ===")
    print(f"Mean Error: {stats['mean_error']:.4f}")
    print(f"Median Error: {stats['median_error']:.4f}")
    print(f"Min Error: {stats['min_error']:.4f} (Best)")
    print(f"Max Error: {stats['max_error']:.4f} (Worst)")
    print(f"Std Dev: {stats['std_error']:.4f}")
    print(f"MSE: {stats['mse']:.4f}")
    print(f"RMSE: {stats['rmse']:.4f}")
    
    # Save results to CSV
    pd.DataFrame(stats, index=[0]).to_csv(
        os.path.join(results_dir, 'error_statistics.csv'), 
        index=False
    )
    
    # Fix for the detailed results section
    
    # Print shapes for debugging
    print(f"Debug - Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
    
    # Normalize target data shape - simplify by flattening to (n_samples, 3)
    if len(targets.shape) == 3:  # If shape is (n_samples, 1, 3)
        targets = targets.reshape(targets.shape[0], -1)  # Reshape to (n_samples, 3)
        print(f"Normalized targets shape: {targets.shape}")
    
    # Save detailed results with simplified data handling
    detailed_results = pd.DataFrame({
        'sample': names,
        'error': errors,
        'pred_x': predictions[:, 0],
        'pred_y': predictions[:, 1], 
        'pred_z': predictions[:, 2],
        'true_x': targets[:, 0],
        'true_y': targets[:, 1],
        'true_z': targets[:, 2]
    })
    
    detailed_results.to_csv(os.path.join(results_dir, 'detailed_results.csv'), index=False)
    
    # Plot error histogram
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=min(20, len(errors)), color='blue', alpha=0.7)
        plt.axvline(stats['mean_error'], color='red', linestyle='--', linewidth=2, 
                    label=f"Mean: {stats['mean_error']:.4f}")
        plt.axvline(stats['median_error'], color='green', linestyle='--', linewidth=2, 
                    label=f"Median: {stats['median_error']:.4f}")
        plt.xlabel('Error (Euclidean Distance)')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Errors')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'error_histogram.png'), dpi=150)
        print("Error histogram saved")
    except Exception as e:
        print(f"Error generating histogram: {str(e)}")
    
    # Update the visualization function to properly handle target/prediction shapes

    def visualize_sample(idx, error, save_path, extra_title=""):
        try:
            data = test_dataset[idx].to(device)
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
            
            with torch.no_grad():
                pred_com = model(data)[0].cpu().numpy()
            
            # Get data as numpy arrays
            point_cloud = data.pos.cpu().numpy()
            true_com = data.y.cpu().numpy()
            
            # Ensure we have flattened arrays if needed
            if len(pred_com.shape) > 1:
                pred_com = pred_com.reshape(-1)
            if len(true_com.shape) > 1:
                true_com = true_com.reshape(-1)
            
            # Print shapes for debugging
            print(f"Sample {idx} - Point cloud: {point_cloud.shape}, True COM: {true_com.shape}, Pred COM: {pred_com.shape}")
            
            # Create 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot point cloud
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                      c='blue', s=10, alpha=0.5, label='Point Cloud')
            
            # Plot true center of mass
            ax.scatter(true_com[0], true_com[1], true_com[2],
                      c='green', s=150, marker='*', label='True Center of Mass')
            
            # Plot predicted center of mass
            ax.scatter(pred_com[0], pred_com[1], pred_com[2],
                      c='red', s=150, marker='x', label='Predicted COM')
            
            # Add error line connecting prediction to ground truth
            ax.plot([true_com[0], pred_com[0]],
                    [true_com[1], pred_com[1]],
                    [true_com[2], pred_com[2]],
                    'k--', alpha=0.7, label=f'Error: {error:.4f}')
            
            # Set labels and title
            obj_name = names[idx]
            ax.set_title(f"{extra_title}{obj_name} (Error: {error:.4f})")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            
            # Set equal aspect ratio
            max_range = np.array([
                point_cloud[:, 0].max() - point_cloud[:, 0].min(),
                point_cloud[:, 1].max() - point_cloud[:, 1].min(),
                point_cloud[:, 2].max() - point_cloud[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (point_cloud[:, 0].max() + point_cloud[:, 0].min()) * 0.5
            mid_y = (point_cloud[:, 1].max() + point_cloud[:, 1].min()) * 0.5
            mid_z = (point_cloud[:, 2].max() + point_cloud[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Save figure
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            return True
        except Exception as e:
            print(f"Error visualizing sample {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # Create visualizations
    try:
        # Create visualization for samples
        num_viz_samples = min(len(test_dataset), 10)  # Visualize up to 10 samples for speed
        print(f"Generating visualizations for {num_viz_samples} objects...")
        
        # Visualize some samples
        for idx in range(min(num_viz_samples, len(test_dataset))):
            visualize_sample(
                idx, 
                errors[idx],
                os.path.join(viz_dir, 'all_samples', f'sample_{idx:03d}.png')
            )
        
        # Find best and worst samples
        sample_indices = list(range(len(errors)))
        best_samples = sorted(zip(sample_indices, errors), key=lambda x: x[1])[:3]
        worst_samples = sorted(zip(sample_indices, errors), key=lambda x: x[1], reverse=True)[:3]
        
        # Visualize best cases
        for rank, (idx, error) in enumerate(best_samples):
            visualize_sample(
                idx, 
                error,
                os.path.join(viz_dir, 'best_cases', f'rank_{rank+1}_sample_{idx:03d}.png'),
                extra_title="BEST #" + str(rank+1) + ": "
            )
        
        # Visualize worst cases
        for rank, (idx, error) in enumerate(worst_samples):
            visualize_sample(
                idx, 
                error,
                os.path.join(viz_dir, 'worst_cases', f'rank_{rank+1}_sample_{idx:03d}.png'),
                extra_title="WORST #" + str(rank+1) + ": "
            )
            
        print(f"Visualizations saved to {viz_dir}")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
    
    print("\nEvaluation complete! Results saved to evaluation_results/")


if __name__ == "__main__":
    main()