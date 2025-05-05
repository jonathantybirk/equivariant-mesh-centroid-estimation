import os
import sys
from pathlib import Path
import glob
import importlib
import time
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
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import pandas as pd

from src.data.datamodule import PointCloudDataModule


def extract_mesh_name(filename):
    """Extract base mesh name from a filename, removing sample indicators."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    # Handle patterns like "Chair_sample1"
    if "_sample" in basename:
        return basename.split("_sample")[0]
    return basename


def get_class(class_path: str):
    """Helper function to dynamically import a class."""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Evaluate a trained GNN model with enhanced visualizations and metrics"""
    
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
    
    # Extract model name for results directory
    model_name = cfg.name
    checkpoint_basename = os.path.basename(checkpoint_path)
    if checkpoint_basename.endswith("_final.ckpt"):
        model_id = model_name + "_final"
    else:
        # Try to extract epoch and val_loss if available
        match = re.search(r'epoch=(\d+)-val_loss=([0-9.]+)', checkpoint_basename)
        if match:
            epoch = match.group(1)
            val_loss = match.group(2)
            model_id = f"{model_name}_e{epoch}_v{val_loss}"
        else:
            # Use timestamp as fallback
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_id = f"{model_name}_{timestamp}"
    
    # Set up model-specific results directory
    results_dir = os.path.join('evaluation_results', model_id)
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(os.path.join(viz_dir, 'best_cases'), exist_ok=True)
    os.makedirs(os.path.join(viz_dir, 'worst_cases'), exist_ok=True)
    os.makedirs(os.path.join(viz_dir, 'all_samples'), exist_ok=True)
    
    print(f"Loading model from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    try:
        # Dynamically get the model class
        ModelClass = get_class(cfg.model.module_path)
        print(f"Using model class: {ModelClass.__name__} from {cfg.model.module_path}")
        
        # Load model from checkpoint using the dynamically imported class
        model = ModelClass.load_from_checkpoint(checkpoint_path)
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
        data_module.setup(stage='fit')  # Load train data for generalization metrics 
        data_module.setup(stage='test')
        train_dataset = data_module.train_dataset
        test_dataset = data_module.test_dataset
        print(f"Datasets loaded: {len(train_dataset)} train samples, {len(test_dataset)} test samples")
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
    
    # Evaluate both train and test datasets for generalization metrics
    datasets = {
        'train': train_dataset,
        'test': test_dataset
    }
    
    all_results = {}
    
    for dataset_name, dataset in datasets.items():
        errors = []
        predictions = []
        targets = []
        names = []
        mesh_to_errors = {}  # Group errors by mesh for mesh-level metrics
        
        print(f"Evaluating {len(dataset)} {dataset_name} samples...")
        try:
            for idx in tqdm(range(len(dataset))):
                data = dataset[idx]
                
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
                if hasattr(data, 'name') and data.name is not None:
                    name = data.name
                else:
                    # Try to extract name from file path
                    if hasattr(dataset, 'file_list') and idx < len(dataset.file_list):
                        name = os.path.basename(dataset.file_list[idx])
                names.append(name)
                
                # Group by mesh name
                mesh_name = extract_mesh_name(name)
                if mesh_name not in mesh_to_errors:
                    mesh_to_errors[mesh_name] = []
                mesh_to_errors[mesh_name].append(error)
        except Exception as e:
            print(f"Error during {dataset_name} evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        # Convert to arrays
        errors = np.array(errors)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Always reshape targets to ensure consistent (n_samples, 3) shape
        targets = targets.reshape(targets.shape[0], -1)  # Will work for both (n,3) and (n,1,3)
        
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
        
        # Calculate mesh-level statistics
        mesh_stats = {}
        for mesh_name, mesh_errors in mesh_to_errors.items():
            mesh_errors_array = np.array(mesh_errors)
            mesh_stats[mesh_name] = {
                'mean': mesh_errors_array.mean(),
                'median': np.median(mesh_errors_array),
                'min': mesh_errors_array.min(),
                'max': mesh_errors_array.max(),
                'std': mesh_errors_array.std(),
                'count': len(mesh_errors_array)
            }
        
        # Store all results
        all_results[dataset_name] = {
            'errors': errors,
            'predictions': predictions,
            'targets': targets,
            'names': names,
            'stats': stats,
            'mesh_stats': mesh_stats
        }
    
    # Calculate generalization metrics
    if 'train' in all_results and 'test' in all_results:
        train_mean = all_results['train']['stats']['mean_error']
        test_mean = all_results['test']['stats']['mean_error']
        generalization_ratio = test_mean / train_mean if train_mean > 0 else float('inf')
        
        print("\n=== Generalization Metrics ===")
        print(f"Train Mean Error: {train_mean:.4f}")
        print(f"Test Mean Error: {test_mean:.4f}")
        print(f"Generalization Ratio (test/train): {generalization_ratio:.4f}")
        print(f"  (closer to 1.0 is better, indicates consistent performance across train and test sets)")
    
    # Print test statistics
    test_stats = all_results['test']['stats']
    print("\n=== Test Set Evaluation Results ===")
    print(f"Mean Error: {test_stats['mean_error']:.4f}")
    print(f"Median Error: {test_stats['median_error']:.4f}")
    print(f"Min Error: {test_stats['min_error']:.4f} (Best)")
    print(f"Max Error: {test_stats['max_error']:.4f} (Worst)")
    print(f"Std Dev: {test_stats['std_error']:.4f}")
    print(f"MSE: {test_stats['mse']:.4f}")
    print(f"RMSE: {test_stats['rmse']:.4f}")
    
    # Print shapes for debugging
    test_errors = all_results['test']['errors']
    test_predictions = all_results['test']['predictions'] 
    test_targets = all_results['test']['targets']
    print(f"Debug - Predictions shape: {test_predictions.shape}, Targets shape: {test_targets.shape}")
    
    # Save comprehensive results
    for dataset_name, results in all_results.items():
        # Save statistics for this dataset
        pd.DataFrame(results['stats'], index=[0]).to_csv(
            os.path.join(results_dir, f'{dataset_name}_error_statistics.csv'), 
            index=False
        )
        
        # Save mesh-level statistics 
        mesh_df_data = []
        for mesh_name, stats in results['mesh_stats'].items():
            row = {'mesh_name': mesh_name}
            row.update(stats)
            mesh_df_data.append(row)
        
        pd.DataFrame(mesh_df_data).to_csv(
            os.path.join(results_dir, f'{dataset_name}_mesh_statistics.csv'), 
            index=False
        )
        
        # Save detailed results
        detailed_results = pd.DataFrame({
            'sample': results['names'],
            'error': results['errors'],
            'pred_x': results['predictions'][:, 0],
            'pred_y': results['predictions'][:, 1], 
            'pred_z': results['predictions'][:, 2],
            'true_x': results['targets'][:, 0],
            'true_y': results['targets'][:, 1],
            'true_z': results['targets'][:, 2],
            'mesh': [extract_mesh_name(name) for name in results['names']]
        })
        
        detailed_results.to_csv(
            os.path.join(results_dir, f'{dataset_name}_detailed_results.csv'), 
            index=False
        )
    
    # Save generalization metrics
    if 'train' in all_results and 'test' in all_results:
        gen_stats = {
            'train_mean_error': train_mean,
            'test_mean_error': test_mean,
            'generalization_ratio': generalization_ratio,
            'model_name': model_id,
            'checkpoint_path': checkpoint_path
        }
        pd.DataFrame(gen_stats, index=[0]).to_csv(
            os.path.join(results_dir, 'generalization_metrics.csv'),
            index=False
        )
    
    # Plot error histogram
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(test_errors, bins=min(20, len(test_errors)), color='blue', alpha=0.7)
        plt.axvline(test_stats['mean_error'], color='red', linestyle='--', linewidth=2, 
                    label=f"Mean: {test_stats['mean_error']:.4f}")
        plt.axvline(test_stats['median_error'], color='green', linestyle='--', linewidth=2, 
                    label=f"Median: {test_stats['median_error']:.4f}")
        plt.xlabel('Error (Euclidean Distance)')
        plt.ylabel('Count')
        plt.title(f'Distribution of Prediction Errors - {model_id}')
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
            
            # Always reshape to ensure consistent (3,) shape
            pred_com = pred_com.reshape(-1)  # Reshape to (3,)
            true_com = true_com.reshape(-1)  # Reshape to (3,)
            
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
            obj_name = all_results['test']['names'][idx]
            mesh_name = extract_mesh_name(obj_name)
            ax.set_title(f"{extra_title}{obj_name} (Mesh: {mesh_name}, Error: {error:.4f})")
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
        
        test_errors = all_results['test']['errors']
        
        # Visualize some samples
        for idx in range(min(num_viz_samples, len(test_dataset))):
            visualize_sample(
                idx, 
                test_errors[idx],
                os.path.join(viz_dir, 'all_samples', f'sample_{idx:03d}.png')
            )
        
        # Find best and worst samples
        sample_indices = list(range(len(test_errors)))
        best_samples = sorted(zip(sample_indices, test_errors), key=lambda x: x[1])[:3]
        worst_samples = sorted(zip(sample_indices, test_errors), key=lambda x: x[1], reverse=True)[:3]
        
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
    
    print("\nEvaluation complete!")
    print(f"Results saved to {results_dir}/")
    
    # Write a summary file with key metrics
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"Checkpoint: {os.path.basename(checkpoint_path)}\n")
        f.write("\n=== Generalization Metrics ===\n")
        f.write(f"Train Mean Error: {train_mean:.4f}\n")
        f.write(f"Test Mean Error: {test_mean:.4f}\n")
        f.write(f"Generalization Ratio (test/train): {generalization_ratio:.4f}\n")
        f.write("  (closer to 1.0 is better, indicates consistent performance)\n\n")
        f.write("=== Test Set Metrics ===\n")
        f.write(f"Mean Error: {test_stats['mean_error']:.4f}\n")
        f.write(f"Median Error: {test_stats['median_error']:.4f}\n")
        f.write(f"Min Error: {test_stats['min_error']:.4f}\n")
        f.write(f"Max Error: {test_stats['max_error']:.4f}\n")
        f.write(f"RMSE: {test_stats['rmse']:.4f}\n")


if __name__ == "__main__":
    main()