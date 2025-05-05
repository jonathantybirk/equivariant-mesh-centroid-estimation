#!/usr/bin/env python
import os
import sys
from pathlib import Path
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Change working directory to the project root
def find_project_root(current: Path, markers=(".git", "pyproject.toml")):
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError("Project root not found.")

root = find_project_root(Path(__file__).resolve())
os.chdir(root)
sys.path.insert(0, str(root))


def load_model_results(results_dir):
    """Load model evaluation results from a directory"""
    model_name = os.path.basename(results_dir)
    
    # Load generalization metrics if available
    gen_metrics_path = os.path.join(results_dir, 'generalization_metrics.csv')
    if os.path.exists(gen_metrics_path):
        gen_metrics = pd.read_csv(gen_metrics_path)
        gen_ratio = gen_metrics['generalization_ratio'].values[0]
        train_error = gen_metrics['train_mean_error'].values[0]
        test_error = gen_metrics['test_mean_error'].values[0]
    else:
        # Fall back to just test metrics
        test_stats_path = os.path.join(results_dir, 'test_error_statistics.csv')
        if os.path.exists(test_stats_path):
            test_stats = pd.read_csv(test_stats_path)
            test_error = test_stats['mean_error'].values[0]
            train_error = None
            gen_ratio = None
        else:
            print(f"Warning: No metrics found for {model_name}")
            return None
    
    # Load mesh-level stats if available
    mesh_stats_path = os.path.join(results_dir, 'test_mesh_statistics.csv')
    if os.path.exists(mesh_stats_path):
        mesh_stats = pd.read_csv(mesh_stats_path)
        mesh_count = len(mesh_stats)
        mesh_error_mean = mesh_stats['mean'].mean()
        mesh_error_std = mesh_stats['mean'].std()
    else:
        mesh_count = None
        mesh_error_mean = None
        mesh_error_std = None
    
    return {
        'model_name': model_name,
        'train_error': train_error,
        'test_error': test_error,
        'generalization_ratio': gen_ratio,
        'mesh_count': mesh_count,
        'mesh_error_mean': mesh_error_mean,
        'mesh_error_std': mesh_error_std,
        'results_dir': results_dir
    }


def compare_models(output_dir='model_comparison'):
    """Compare evaluation results from multiple models"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all model result directories
    results_root = 'evaluation_results'
    model_dirs = [d for d in glob.glob(os.path.join(results_root, '*')) 
                 if os.path.isdir(d)]
    
    if not model_dirs:
        print("No model evaluation results found.")
        return
    
    print(f"Found {len(model_dirs)} model result directories")
    
    # Load results for each model
    model_results = []
    for model_dir in model_dirs:
        result = load_model_results(model_dir)
        if result:
            model_results.append(result)
    
    if not model_results:
        print("No valid model results found.")
        return
    
    # Convert to DataFrame for easy manipulation
    results_df = pd.DataFrame(model_results)
    
    # Sort by test error (lower is better)
    results_df = results_df.sort_values('test_error')
    
    # Display comparison table
    print("\n=== Model Comparison (sorted by test error) ===")
    display_cols = ['model_name', 'test_error', 'train_error', 'generalization_ratio']
    print(results_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Save comparison table
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Create comparison visualizations
    create_comparison_plots(results_df, output_dir)
    
    print(f"\nComparison results saved to {output_dir}/")
    return results_df


def create_comparison_plots(results_df, output_dir):
    """Create visualizations comparing model performance"""
    # Set up plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Plot test error for each model
    ax = sns.barplot(x='model_name', y='test_error', data=results_df, palette='viridis')
    plt.title('Test Error by Model')
    plt.xlabel('Model')
    plt.ylabel('Mean Error')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_error_comparison.png'), dpi=150)
    plt.close()
    
    # Plot generalization ratio if available
    if 'generalization_ratio' in results_df and not results_df['generalization_ratio'].isnull().all():
        plt.figure(figsize=(12, 6))
        valid_gen = results_df.dropna(subset=['generalization_ratio'])
        
        ax = sns.barplot(x='model_name', y='generalization_ratio', data=valid_gen, palette='viridis')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, 
                  label='Ideal generalization (ratio=1.0)')
        plt.title('Generalization Ratio by Model')
        plt.xlabel('Model')
        plt.ylabel('Generalization Ratio (test/train)')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'generalization_ratio_comparison.png'), dpi=150)
        plt.close()
    
    # If multiple models have mesh stats, create a comparison
    if 'mesh_error_mean' in results_df and not results_df['mesh_error_mean'].isnull().all():
        # Mesh-level error distribution
        plt.figure(figsize=(12, 6))
        valid_mesh = results_df.dropna(subset=['mesh_error_mean'])
        
        # Bar plot with error bars showing std dev across meshes
        bars = plt.bar(valid_mesh['model_name'], valid_mesh['mesh_error_mean'], 
                       yerr=valid_mesh['mesh_error_std'], capsize=10, alpha=0.7)
        plt.title('Average Mesh Error by Model (with standard deviation across meshes)')
        plt.xlabel('Model')
        plt.ylabel('Mean Error per Mesh')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mesh_error_comparison.png'), dpi=150)
        plt.close()
    
    # Create a radar chart if we have multiple metrics and models
    if len(results_df) > 1 and not results_df[['test_error', 'generalization_ratio']].isnull().any(axis=1).all():
        create_radar_chart(results_df, output_dir)


def create_radar_chart(results_df, output_dir, max_models=5):
    """Create a radar chart to visualize multiple metrics across models"""
    # Normalize metrics for radar chart
    metrics = ['test_error', 'generalization_ratio', 'mesh_error_std']
    available_metrics = [m for m in metrics if m in results_df.columns and not results_df[m].isnull().all()]
    
    if len(available_metrics) < 2:
        return  # Not enough metrics for radar chart
        
    # Select top models by test_error (at most max_models)
    top_models = results_df.sort_values('test_error').head(max_models)
    
    # Create a copy of the dataframe for normalization
    # Note: For test_error and mesh_error_std, lower is better, so we invert the normalization
    # For generalization_ratio, closer to 1.0 is better, so we compute distance from 1.0 and invert
    radar_df = top_models[['model_name'] + available_metrics].copy()
    
    for metric in available_metrics:
        if metric == 'generalization_ratio':
            # For gen ratio, distance from 1.0 is what matters (transform so lower is better)
            radar_df[metric] = abs(radar_df[metric] - 1.0)
            
        # Normalize all metrics to 0-1 scale (inverted so higher is better)
        if not radar_df[metric].isnull().all():
            min_val = radar_df[metric].min()
            max_val = radar_df[metric].max()
            if max_val > min_val:
                radar_df[metric] = 1 - ((radar_df[metric] - min_val) / (max_val - min_val))
            else:
                radar_df[metric] = 1  # All values are the same
    
    # Set up radar chart
    labels = [m.replace('_', ' ').title() for m in available_metrics]
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Add each model as a polygon on the radar chart
    for _, row in radar_df.iterrows():
        values = [row[m] for m in available_metrics]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, '-', linewidth=2, label=row['model_name'])
        ax.fill(angles, values, alpha=0.1)
    
    plt.title('Model Comparison (higher is better)', size=15)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'radar_chart_comparison.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    compare_models()
    print("Done!")