#!/usr/bin/env python3
"""
Performance Comparison Script for All Models

Tests all 4 models on the remaining test set (after 1000 train/val samples)
and performs statistical analysis with confidence intervals and paired t-tests.
"""

import sys
from pathlib import Path
import argparse
import pickle

sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from torch_geometric.data import Data, DataLoader
import warnings

warnings.filterwarnings("ignore")

# Import models and data loading from trainer
from src.scripts.trainer import EquivariantGNN, BasicGNN, Baseline

# Define professional color palette for academic publication (same as visualization script)
MODEL_COLORS = {
    "Baseline": "#696969",  # Dim Gray
    "Basic GNN": "#1f77b4",  # Professional Blue
    "Basic GNN + Augmentation": "#2ca02c",  # Professional Green
    "Equivariant GNN": "#d62728",  # Professional Red
}


def random_rotation_matrix(seed=None):
    """Generate a random 3D rotation matrix using Rodrigues' rotation formula"""
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random axis (normalized)
    axis = torch.randn(3)
    axis = axis / torch.norm(axis)

    # Random angle between 0 and 2π
    angle = torch.rand(1) * 2 * np.pi

    # Rodrigues' rotation formula
    K = torch.zeros(3, 3)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]

    # Compute rotation matrix
    I = torch.eye(3)
    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)

    return R


def apply_rotation_to_data(data, rotation_matrix):
    """Apply rotation to graph data"""
    rotated_data = data.clone()

    # Rotate node coordinates
    rotated_data.x = torch.matmul(data.x, rotation_matrix.T)

    # Rotate edge attributes (displacement vectors)
    if data.edge_attr is not None:
        rotated_data.edge_attr = torch.matmul(data.edge_attr, rotation_matrix.T)

    # Rotate target (center of mass)
    if data.y is not None:
        original_shape = data.y.shape
        y_flat = data.y.view(-1, 3)
        y_rotated = torch.matmul(y_flat, rotation_matrix.T)
        rotated_data.y = y_rotated.view(original_shape)

    return rotated_data


class TestDataLoader:
    """Create test dataset from remaining data (after 1000 train/val samples)"""

    def __init__(self, num_samples=None, data_dir="data/processed_dv"):
        self.data_dir = data_dir
        self.test_data = []
        self.num_samples = num_samples

    def setup(self):
        """Load test data - everything except the 1000 samples used for train/val"""
        print(f"Loading test dataset from {self.data_dir}...")

        # Load all .pt files
        file_list = list(Path(self.data_dir).glob("**/*.pt"))
        print(f"Found {len(file_list)} total files")

        # Get the same 1000 files used for train/val (with same random seed)
        torch.manual_seed(42)  # Same seed as trainer
        indices = torch.randperm(len(file_list))

        if len(file_list) > 1000:
            train_val_indices = indices[:1000]
            test_indices = indices[1000:]  # Use the REST for testing
            test_file_list = [file_list[i] for i in test_indices]
        else:
            # If less than 1000 files total, use all for testing
            test_file_list = file_list

        if self.num_samples is not None:
            test_file_list = test_file_list[: self.num_samples]

        print(f"Using {len(test_file_list)} files for testing")

        # Load test files
        for file_path in tqdm(test_file_list, desc="Loading test data"):
            try:
                data = torch.load(file_path, weights_only=False)
                x = data["node_features"].float().contiguous()
                edge_index = data["edge_index"].long().contiguous()
                edge_attr = data.get("edge_attr")
                if edge_attr is not None:
                    edge_attr = edge_attr.float().contiguous()
                target = data["target"].squeeze().float().contiguous()

                self.test_data.append(
                    Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=target.unsqueeze(0),
                    )
                )
            except Exception as e:
                continue

        print(f"Successfully loaded {len(self.test_data)} test samples")


def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    """Load a trained model from checkpoint"""
    print(f"🔄 Loading model from checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Determine model type from checkpoint filename
    checkpoint_name = Path(checkpoint_path).stem.lower()

    if "eq_gnn" in checkpoint_name:
        model_class = EquivariantGNN
        model_name = "EquivariantGNN"
    elif "basic_gnn" in checkpoint_name:
        model_class = BasicGNN
        model_name = "BasicGNN"
    elif "baseline" in checkpoint_name:
        model_class = Baseline
        model_name = "Baseline"
    else:
        model_class = BasicGNN
        model_name = "BasicGNN (default)"

    print(f"   Detected model type: {model_name}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters if available
    if "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]

        # Filter hyperparameters
        model_hparams = {}
        valid_params = {
            "input_dim",
            "hidden_dim",
            "message_passing_steps",
            "final_mlp_dims",
            "edge_sh_degree",
            "node_l_values",
            "node_multiplicity",
            "message_mlp_dims",
            "init_method",
            "seed",
            "debug",
            "lr",
            "weight_decay",
            "dropout",
        }

        for key, value in hparams.items():
            if key in valid_params:
                model_hparams[key] = value

        # Create model with filtered hyperparameters
        try:
            model = model_class(**model_hparams)
        except Exception as e:
            print(
                f"   Warning: Failed to create {model_name} with hyperparameters: {e}"
            )
            model = model_class()
    else:
        model = model_class()

    # Load state dict
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        print(f"   ✅ Model weights loaded successfully")
    else:
        raise ValueError("No 'state_dict' found in checkpoint")

    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")

    return model, model_name, total_params


def evaluate_model_performance_and_equivariance(
    model, test_data, model_name, device="cpu", batch_size=64
):
    """Evaluate both model performance and equivariance on test set using batches"""
    import time

    print(f"\n📊 Evaluating {model_name} performance and equivariance...")

    model.to(device)
    model.eval()

    all_performance_errors = []
    all_equivariance_errors = []
    all_inference_times = []

    # Create DataLoader for batching
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(
            tqdm(test_loader, desc=f"Testing {model_name}")
        ):
            batch_data = batch_data.to(device)

            # Generate single rotation matrix for entire batch
            rotation_seed = 42 + batch_idx  # Deterministic but different per batch
            R = random_rotation_matrix(seed=rotation_seed).to(device)

            # Time the original prediction on entire batch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            pred_original = model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.batch,  # Use batch information
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Store inference time per batch (parallel processing)
            batch_inference_time = end_time - start_time
            all_inference_times.append(batch_inference_time)

            # Performance error (unrotated) - Use same L2 calculation as training
            displacement = pred_original - batch_data.y  # [B, 3]
            per_sample_distances = torch.norm(
                displacement, dim=1
            )  # [B] - L2 norm per sample

            # Store individual sample errors (not batch mean)
            all_performance_errors.extend(per_sample_distances.cpu().numpy())

            # Apply rotation to entire batch
            rotated_batch_data = apply_rotation_to_batch(batch_data, R)

            # Prediction on rotated batch (don't time this as it's just for equivariance test)
            pred_rotated = model(
                rotated_batch_data.x,
                rotated_batch_data.edge_index,
                rotated_batch_data.edge_attr,
                rotated_batch_data.batch,  # Use batch information
            )

            # Expected prediction (original prediction rotated)
            pred_expected = torch.matmul(pred_original, R.T)

            # Equivariance error per sample in batch
            equivariance_errors_batch = torch.norm(pred_rotated - pred_expected, dim=1)
            all_equivariance_errors.extend(equivariance_errors_batch.cpu().numpy())

    # Convert to numpy arrays
    all_performance_errors = np.array(all_performance_errors)
    all_equivariance_errors = np.array(all_equivariance_errors)
    all_inference_times = np.array(all_inference_times)

    # Calculate statistics
    mean_performance = np.mean(all_performance_errors)
    std_performance = np.std(all_performance_errors)
    mean_equivariance = np.mean(all_equivariance_errors)
    std_equivariance = np.std(all_equivariance_errors)
    mean_inference_time = np.mean(all_inference_times)
    std_inference_time = np.std(all_inference_times)

    print(f"   Mean Performance L2 Distance: {mean_performance:.6f}")
    print(f"   Mean Equivariance Error: {mean_equivariance:.6f}")
    print(f"   Mean Inference Time per Batch: {mean_inference_time*1000:.1f} ms")

    return {
        "model_name": model_name,
        "performance_errors": all_performance_errors,
        "equivariance_errors": all_equivariance_errors,
        "inference_times": all_inference_times,
        "mean_performance": mean_performance,
        "std_performance": std_performance,
        "mean_equivariance": mean_equivariance,
        "std_equivariance": std_equivariance,
        "mean_inference_time": mean_inference_time,
        "std_inference_time": std_inference_time,
        "n_samples": len(all_performance_errors),
    }


def apply_rotation_to_batch(batch_data, rotation_matrix):
    """Apply rotation to batched graph data"""
    rotated_data = batch_data.clone()

    # Rotate node coordinates
    rotated_data.x = torch.matmul(batch_data.x, rotation_matrix.T)

    # Rotate edge attributes (displacement vectors)
    if batch_data.edge_attr is not None:
        rotated_data.edge_attr = torch.matmul(batch_data.edge_attr, rotation_matrix.T)

    # Rotate target (center of mass) - handle batch dimension
    if batch_data.y is not None:
        original_shape = batch_data.y.shape
        y_flat = batch_data.y.view(-1, 3)
        y_rotated = torch.matmul(y_flat, rotation_matrix.T)
        rotated_data.y = y_rotated.view(original_shape)

    return rotated_data


def perform_statistical_analysis(results_dict, model_params):
    """Perform statistical analysis with t-tests and confidence intervals"""
    print("\n" + "=" * 80)
    print("📈 STATISTICAL ANALYSIS")
    print("=" * 80)

    models = list(results_dict.keys())
    n_models = len(models)

    # 1. Performance statistics with confidence intervals
    print("\n1. PERFORMANCE STATISTICS (95% Confidence Intervals)")
    print("-" * 80)
    print(f"{'Model':<20} {'Mean L2 Dist':<12} {'95% CI':<25} {'p-value':<12} {'n'}")
    print("-" * 80)

    for model in models:
        data = results_dict[model]
        errors = data["performance_errors"]
        n = len(errors)
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # T-test against null hypothesis (mean = 0, perfect predictions)
        t_statistic, p_value = stats.ttest_1samp(errors, 0.0)

        # 95% confidence interval
        confidence_level = 0.95
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin_of_error = t_critical * (std_error / np.sqrt(n))
        ci_lower = mean_error - margin_of_error
        ci_upper = mean_error + margin_of_error

        ci_str = f"[{ci_lower:.6f}, {ci_upper:.6f}]"
        print(f"{model:<20} {mean_error:<12.6f} {ci_str:<25} {p_value:<12.2e} {n}")

    # 2. Equivariance statistics with confidence intervals
    print("\n2. EQUIVARIANCE STATISTICS (95% Confidence Intervals)")
    print("-" * 80)
    print(f"{'Model':<20} {'Mean Eq Error':<12} {'95% CI':<25} {'p-value':<12} {'n'}")
    print("-" * 80)

    for model in models:
        data = results_dict[model]
        errors = data["equivariance_errors"]

        # Skip models without equivariance data (e.g., Baseline)
        if errors is None:
            print(f"{model:<20} {'N/A':<12} {'N/A':<25} {'N/A':<12} {'N/A'}")
            continue

        n = len(errors)
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # Check if errors are essentially zero (numerical precision)
        if mean_error < 1e-6:  # Less than 1 micro-unit
            print(
                f"{model:<20} {'~0 (perfect)':<12} {'[~0, ~0]':<25} {'Perfect Eq':<12} {n}"
            )
            continue

        # T-test against null hypothesis (mean = 0, perfect equivariance)
        # Only perform if standard deviation is not essentially zero
        if std_error < 1e-10:
            print(
                f"{model:<20} {mean_error:<12.6f} {'[const, const]':<25} {'Constant':<12} {n}"
            )
            continue

        t_statistic, p_value = stats.ttest_1samp(errors, 0.0)

        # 95% confidence interval
        confidence_level = 0.95
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin_of_error = t_critical * (std_error / np.sqrt(n))
        ci_lower = mean_error - margin_of_error
        ci_upper = mean_error + margin_of_error

        ci_str = f"[{ci_lower:.6f}, {ci_upper:.6f}]"
        print(f"{model:<20} {mean_error:<12.6f} {ci_str:<25} {p_value:<12.2e} {n}")

    # 3. Inference time statistics
    print("\n3. INFERENCE TIME STATISTICS (95% Confidence Intervals)")
    print("-" * 80)
    print(
        f"{'Model':<20} {'Mean Batch Time (ms)':<20} {'95% CI (ms)':<25} {'Std (ms)':<12} {'n'}"
    )
    print("-" * 80)

    for model in models:
        data = results_dict[model]
        times = data.get("inference_times")

        if times is not None:
            n = len(times)
            mean_time_ms = np.mean(times) * 1000  # Convert to milliseconds
            std_time_ms = np.std(times) * 1000

            # 95% confidence interval
            confidence_level = 0.95
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
            margin_of_error = t_critical * (std_time_ms / np.sqrt(n))
            ci_lower = mean_time_ms - margin_of_error
            ci_upper = mean_time_ms + margin_of_error

            ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            print(
                f"{model:<20} {mean_time_ms:<20.1f} {ci_str:<25} {std_time_ms:<12.1f} {n}"
            )
        else:
            print(f"{model:<20} {'N/A':<20} {'N/A':<25} {'N/A':<12} {'N/A'}")

    # 4. Pairwise paired t-tests for performance
    print(f"\n4. PAIRWISE PAIRED T-TESTS (Performance)")
    print("-" * 80)
    print("Comparing models pairwise (paired t-test on same test samples)")
    print(
        f"{'Model 1':<20} {'Model 2':<20} {'Mean Diff':<12} {'p-value':<12} {'Significant'}"
    )
    print("-" * 80)

    pairwise_results = {}

    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1, model2 = models[i], models[j]
            errors1 = results_dict[model1]["performance_errors"]
            errors2 = results_dict[model2]["performance_errors"]

            # Ensure same length (should be, but just in case)
            min_len = min(len(errors1), len(errors2))
            errors1 = errors1[:min_len]
            errors2 = errors2[:min_len]

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(errors1, errors2)
            mean_diff = np.mean(errors1 - errors2)
            is_significant = p_val < 0.05

            pairwise_results[(model1, model2)] = {
                "mean_diff": mean_diff,
                "p_value": p_val,
                "significant": is_significant,
            }

            sig_str = "✅ Yes" if is_significant else "❌ No"
            print(
                f"{model1:<20} {model2:<20} {mean_diff:<12.6f} {p_val:<12.2e} {sig_str}"
            )

    # 5. Pairwise paired t-tests for equivariance
    print(f"\n5. PAIRWISE PAIRED T-TESTS (Equivariance)")
    print("-" * 80)
    print("Comparing models pairwise (paired t-test on same test samples)")
    print(
        f"{'Model 1':<20} {'Model 2':<20} {'Mean Diff':<12} {'p-value':<12} {'Significant'}"
    )
    print("-" * 80)

    equivariance_pairwise_results = {}

    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1, model2 = models[i], models[j]
            eq_errors1 = results_dict[model1]["equivariance_errors"]
            eq_errors2 = results_dict[model2]["equivariance_errors"]

            # Skip comparisons if either model has no equivariance data
            if eq_errors1 is None or eq_errors2 is None:
                print(f"{model1:<20} {model2:<20} {'N/A':<12} {'N/A':<12} {'N/A'}")
                continue

            # Ensure same length (should be, but just in case)
            min_len = min(len(eq_errors1), len(eq_errors2))
            eq_errors1 = eq_errors1[:min_len]
            eq_errors2 = eq_errors2[:min_len]

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(eq_errors1, eq_errors2)
            mean_diff = np.mean(eq_errors1 - eq_errors2)
            is_significant = p_val < 0.05

            equivariance_pairwise_results[(model1, model2)] = {
                "mean_diff": mean_diff,
                "p_value": p_val,
                "significant": is_significant,
            }

            sig_str = "✅ Yes" if is_significant else "❌ No"
            print(
                f"{model1:<20} {model2:<20} {mean_diff:<12.6f} {p_val:<12.2e} {sig_str}"
            )

    # 6. Model ranking
    print(f"\n6. MODEL RANKING")
    print("-" * 60)
    print("By Performance (L2 Distance):")
    sorted_models = sorted(models, key=lambda x: results_dict[x]["mean_performance"])
    for rank, model in enumerate(sorted_models, 1):
        l2_dist = results_dict[model]["mean_performance"]
        print(f"  {rank}. {model:<20} L2 Dist: {l2_dist:.6f}")

    print("\nBy Equivariance (Error):")
    # Filter out models without equivariance data
    models_with_eq = [
        model
        for model in models
        if results_dict[model]["equivariance_errors"] is not None
    ]
    sorted_models = sorted(
        models_with_eq, key=lambda x: results_dict[x]["mean_equivariance"]
    )
    for rank, model in enumerate(sorted_models, 1):
        eq_error = results_dict[model]["mean_equivariance"]
        print(f"  {rank}. {model:<20} Eq Error: {eq_error:.6f}")

    # Show models without equivariance data separately
    models_without_eq = [
        model for model in models if results_dict[model]["equivariance_errors"] is None
    ]
    if models_without_eq:
        print("  Models without equivariance data:")
        for model in models_without_eq:
            print(f"     - {model:<20} (N/A - not equivariant by design)")

    print("\nBy Parameter Count:")
    sorted_models = sorted(models, key=lambda x: model_params[x])
    for rank, model in enumerate(sorted_models, 1):
        params = model_params[model]
        print(f"  {rank}. {model:<20} Parameters: {params:,}")

    return results_dict, pairwise_results, equivariance_pairwise_results


def create_combined_visualization(results_dict):
    """Create 1x2 figure with performance/equivariance bars and inference time bars"""

    # Set up professional plotting style
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.shadow": False,
            "legend.edgecolor": "black",
            "legend.facecolor": "white",
            "legend.borderpad": 0.6,
        }
    )

    # Ensure correct model order
    model_order = [
        "Baseline",
        "Basic GNN",
        "Basic GNN + Augmentation",
        "Equivariant GNN",
    ]
    models = [model for model in model_order if model in results_dict.keys()]
    n_models = len(models)

    # Create 1x2 figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ============ LEFT SUBPLOT: Performance/Equivariance Bars ============

    # Extract data in correct order
    performance_means = [results_dict[model]["mean_performance"] for model in models]
    equivariance_means = [results_dict[model]["mean_equivariance"] for model in models]

    # Calculate confidence intervals
    performance_cis = []
    equivariance_cis = []

    for model in models:
        # Performance CI
        perf_errors = results_dict[model]["performance_errors"]
        n = len(perf_errors)
        std_error = np.std(perf_errors)
        t_critical = stats.t.ppf(0.975, df=n - 1)
        margin = t_critical * (std_error / np.sqrt(n))
        performance_cis.append(margin)

        # Equivariance CI (handle None case for baseline)
        eq_errors = results_dict[model]["equivariance_errors"]
        if eq_errors is not None:
            n = len(eq_errors)
            std_error = np.std(eq_errors)
            t_critical = stats.t.ppf(0.975, df=n - 1)
            margin = t_critical * (std_error / np.sqrt(n))
            equivariance_cis.append(margin)
        else:
            # For baseline with no equivariance, set CI to 0
            equivariance_cis.append(0.0)

    # Set up positions for bars
    x = np.arange(n_models)
    width = 0.35

    # Performance bars with model colors
    bars1 = ax1.bar(
        x - width / 2,
        performance_means,
        width,
        yerr=performance_cis,
        capsize=5,
        color=[MODEL_COLORS[model] for model in models],
        alpha=0.8,
        label="Prediction L2 Error",
        edgecolor="white",
        linewidth=1.2,
    )

    # Equivariance bars with model colors and stripes/hatching
    bars2 = ax1.bar(
        x + width / 2,
        equivariance_means,
        width,
        yerr=equivariance_cis,
        capsize=5,
        color=[MODEL_COLORS[model] for model in models],
        alpha=0.8,
        label="Equivariance L2 Error",
        edgecolor="white",
        linewidth=1.2,
        hatch="///",  # Add diagonal stripes
    )

    # Labels and formatting for left subplot
    ax1.set_ylabel("L2 Error", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, rotation=45, ha="right")
    ax1.set_title(
        "(a) Performance vs Equivariance",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax1.legend(loc="upper right", fontsize=10)

    # Add value labels on bars
    for i, (bar1, bar2, perf, eq, perf_ci, eq_ci) in enumerate(
        zip(
            bars1,
            bars2,
            performance_means,
            equivariance_means,
            performance_cis,
            equivariance_cis,
        )
    ):
        # Performance labels
        height1 = bar1.get_height()
        ax1.text(
            bar1.get_x() + bar1.get_width() / 2.0,
            height1 + perf_ci * 1.1,
            f"{perf:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

        # Equivariance labels
        height2 = bar2.get_height()
        ax1.text(
            bar2.get_x() + bar2.get_width() / 2.0,
            height2 + eq_ci * 1.1,
            f"{eq:.5f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax1.grid(True, alpha=0.3, axis="y", linewidth=0.8)

    # ============ RIGHT SUBPLOT: Inference Time Bars ============

    # Extract inference time data
    inference_times_ms = []
    inference_time_cis = []

    for model in models:
        times = results_dict[model].get("inference_times")
        if times is not None:
            times_ms = times * 1000  # Convert to milliseconds
            mean_time = np.mean(times_ms)
            std_time = np.std(times_ms)
            n = len(times_ms)

            # 95% confidence interval
            t_critical = stats.t.ppf(0.975, df=n - 1)
            margin = t_critical * (std_time / np.sqrt(n))

            inference_times_ms.append(mean_time)
            inference_time_cis.append(margin)
        else:
            inference_times_ms.append(0)
            inference_time_cis.append(0)

    # Create bars for inference time
    bars3 = ax2.bar(
        models,
        inference_times_ms,
        yerr=inference_time_cis,
        capsize=5,
        color=[MODEL_COLORS[model] for model in models],
        alpha=0.8,
        edgecolor="white",
        linewidth=1.2,
    )

    # Labels and formatting for right subplot
    ax2.set_ylabel("Inference Time per Batch (ms)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "(b) Inference Time Comparison", fontsize=12, fontweight="bold", pad=15
    )
    ax2.set_xticklabels(models, fontsize=10, rotation=45, ha="right")

    # Add value labels on bars
    for bar, time_ms, ci in zip(bars3, inference_times_ms, inference_time_cis):
        height = bar.get_height()
        if height > 0:  # Only add label if there's actual data
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + ci * 1.1,
                f"{time_ms:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax2.grid(True, alpha=0.3, axis="y", linewidth=0.8)

    # Professional layout
    plt.tight_layout(pad=2.0)

    # Save both PNG and PDF for publications
    plt.savefig(
        "performance_equivariance_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        "performance_equivariance_comparison.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.show()

    return fig


def save_results_data(
    results_dict, model_params=None, filename="model_results_raw.pkl"
):
    """Save raw results data to pickle file"""
    print(f"\n💾 Saving raw results data to {filename}...")

    # Create a clean data structure for saving
    save_data = {"results": {}, "model_params": model_params or {}}

    for model_name, data in results_dict.items():
        save_data["results"][model_name] = {
            "performance_errors": (
                data["performance_errors"].tolist()
                if isinstance(data["performance_errors"], np.ndarray)
                else data["performance_errors"]
            ),
            "equivariance_errors": (
                data["equivariance_errors"].tolist()
                if data["equivariance_errors"] is not None
                and isinstance(data["equivariance_errors"], np.ndarray)
                else data["equivariance_errors"]
            ),
            "inference_times": (
                data["inference_times"].tolist()
                if data.get("inference_times") is not None
                and isinstance(data["inference_times"], np.ndarray)
                else data.get("inference_times")
            ),
            "mean_performance": float(data["mean_performance"]),
            "mean_equivariance": float(data["mean_equivariance"]),
            "mean_inference_time": float(data.get("mean_inference_time", 0.0)),
        }

    with open(filename, "wb") as f:
        pickle.dump(save_data, f)

    print(f"✅ Raw results saved successfully!")


def load_results_data(filename="model_results_raw.pkl"):
    """Load raw results data from pickle file"""
    print(f"\n📂 Loading raw results data from {filename}...")

    if not Path(filename).exists():
        raise FileNotFoundError(f"Results file not found: {filename}")

    with open(filename, "rb") as f:
        save_data = pickle.load(f)

    # Handle both old and new file formats
    if "results" in save_data:
        # New format with model_params
        results_data = save_data["results"]
        model_params = save_data.get("model_params", {})
    else:
        # Old format - just results
        results_data = save_data
        model_params = {
            "Baseline": 0,  # Baseline has 0 parameters
            "Basic GNN": 7041,  # Updated parameter count
            "Basic GNN + Augmentation": 7041,
            "Equivariant GNN": 8531,
        }

    # Convert back to numpy arrays
    results_dict = {}
    for model_name, data in results_data.items():
        results_dict[model_name] = {
            "performance_errors": (
                np.array(data["performance_errors"])
                if data["performance_errors"] is not None
                else None
            ),
            "equivariance_errors": (
                np.array(data["equivariance_errors"])
                if data["equivariance_errors"] is not None
                else None
            ),
            "inference_times": (
                np.array(data["inference_times"])
                if data.get("inference_times") is not None
                else None
            ),
            "mean_performance": data["mean_performance"],
            "mean_equivariance": data["mean_equivariance"],
            "mean_inference_time": data.get("mean_inference_time", 0.0),
        }

    print(f"✅ Raw results loaded successfully!")
    print(f"   Loaded data for {len(results_dict)} models")

    return results_dict, model_params


def load_model_results(num_samples=None):
    """Load all model results for comparison"""
    # Define model checkpoints with correct model order
    model_checkpoints = {
        "Basic GNN": "weigths/basic_gnn.ckpt",
        "Basic GNN + Augmentation": "weigths/basic_gnn_aug.ckpt",
        "Equivariant GNN": "weigths/eq_gnn.ckpt",
    }

    # Set up test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    test_loader = TestDataLoader(num_samples=num_samples)
    test_loader.setup()

    if not test_loader.test_data:
        raise ValueError("No test data available")

    print(f"Loaded {len(test_loader.test_data)} test samples")

    results_dict = {}
    model_params = {}

    # Handle baseline model separately (no checkpoint needed)
    try:
        print(f"\n🔄 Testing Baseline...")
        model = Baseline()  # Create baseline model directly
        model.eval()

        # Count parameters for baseline
        total_params = sum(p.numel() for p in model.parameters())
        model_params["Baseline"] = total_params

        # Evaluate model - this returns a dictionary
        result = evaluate_model_performance_and_equivariance(
            model, test_loader.test_data, "Baseline", device
        )

        # Store results (baseline has no equivariance)
        results_dict["Baseline"] = {
            "performance_errors": result["performance_errors"],
            "equivariance_errors": None,  # Baseline doesn't have equivariance
            "inference_times": result["inference_times"],
            "mean_performance": result["mean_performance"],
            "mean_equivariance": 0.0,  # Set to 0 for baseline
            "mean_inference_time": result["mean_inference_time"],
        }

        print(f"✅ Baseline completed")
        print(
            f"   Performance: {result['mean_performance']:.6f} ± {result['std_performance']:.6f}"
        )
        print(f"   Parameters: {total_params:,}")

    except Exception as e:
        print(f"❌ Failed to evaluate Baseline: {e}")

    # Test models with checkpoints
    for model_name, checkpoint_path in model_checkpoints.items():
        try:
            print(f"\n🔄 Testing {model_name}...")

            # Check if checkpoint exists
            if not Path(checkpoint_path).exists():
                print(f"⚠️ Checkpoint not found: {checkpoint_path}")
                continue

            # Load model - this returns a tuple (model, model_name, total_params)
            model, loaded_model_name, total_params = load_model_from_checkpoint(
                checkpoint_path, device
            )
            model.eval()

            # Store parameter count
            model_params[model_name] = total_params

            # Evaluate model - this returns a dictionary
            result = evaluate_model_performance_and_equivariance(
                model, test_loader.test_data, model_name, device
            )

            # Store results
            results_dict[model_name] = {
                "performance_errors": result["performance_errors"],
                "equivariance_errors": result["equivariance_errors"],
                "inference_times": result["inference_times"],
                "mean_performance": result["mean_performance"],
                "mean_equivariance": result["mean_equivariance"],
                "mean_inference_time": result["mean_inference_time"],
            }

            print(f"✅ {model_name} completed")
            print(
                f"   Performance: {result['mean_performance']:.6f} ± {result['std_performance']:.6f}"
            )
            print(
                f"   Equivariance: {result['mean_equivariance']:.6f} ± {result['std_equivariance']:.6f}"
            )

        except Exception as e:
            print(f"❌ Failed to evaluate {model_name}: {e}")
            continue

    if not results_dict:
        raise ValueError("No models were successfully evaluated")

    return results_dict, model_params


def print_summary_statistics(results_dict):
    """Print summary statistics for all models"""
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)

    # Extract model names in correct order
    model_order = [
        "Baseline",
        "Basic GNN",
        "Basic GNN + Augmentation",
        "Equivariant GNN",
    ]
    models = [model for model in model_order if model in results_dict.keys()]

    # Performance ranking
    print("\n🏆 Performance Ranking (Lower is Better):")
    sorted_models = sorted(models, key=lambda x: results_dict[x]["mean_performance"])
    for rank, model in enumerate(sorted_models, 1):
        perf = results_dict[model]["mean_performance"]
        std = np.std(results_dict[model]["performance_errors"])
        print(f"  {rank}. {model:<25} {perf:.6f} ± {std:.6f}")

    # Equivariance ranking
    print("\n🔄 Equivariance Ranking (Lower is Better):")
    models_with_eq = [
        model
        for model in models
        if results_dict[model]["equivariance_errors"] is not None
    ]
    sorted_models = sorted(
        models_with_eq, key=lambda x: results_dict[x]["mean_equivariance"]
    )
    for rank, model in enumerate(sorted_models, 1):
        eq = results_dict[model]["mean_equivariance"]
        std = np.std(results_dict[model]["equivariance_errors"])
        print(f"  {rank}. {model:<25} {eq:.6f} ± {std:.6f}")

    # Inference time ranking
    print("\n⚡ Inference Time Ranking (Lower is Better):")
    models_with_timing = [
        model
        for model in models
        if results_dict[model].get("inference_times") is not None
    ]
    sorted_models = sorted(
        models_with_timing, key=lambda x: results_dict[x]["mean_inference_time"]
    )
    for rank, model in enumerate(sorted_models, 1):
        time_ms = results_dict[model]["mean_inference_time"] * 1000
        std_ms = np.std(results_dict[model]["inference_times"]) * 1000
        print(f"  {rank}. {model:<25} {time_ms:.1f} ± {std_ms:.1f} ms per batch")


def main():
    """Main function to run the performance comparison"""
    parser = argparse.ArgumentParser(
        description="Performance comparison for GNN models"
    )
    parser.add_argument(
        "--load_data",
        action="store_true",
        help="Load previously saved raw results instead of computing them",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="model_results_raw.pkl",
        help="Filename for saving/loading raw results data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of test samples to use (only for computation, not loading)",
    )

    args = parser.parse_args()

    print("🔍 Starting performance comparison analysis...")

    if args.load_data:
        # Load previously saved data
        try:
            results_dict, model_params = load_results_data(args.data_file)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("   Run without --load_data flag first to generate the data.")
            return
    else:
        # Compute new results
        results_dict, model_params = load_model_results(num_samples=args.num_samples)

        # Save raw results for future use
        save_results_data(results_dict, model_params, args.data_file)

    # Print summary statistics
    print_summary_statistics(results_dict)

    # Perform statistical analysis with confidence intervals and t-tests
    print("\n📊 Performing statistical analysis...")
    perform_statistical_analysis(results_dict, model_params)

    # Create combined 1x2 visualization with both performance/equivariance and inference time
    print("\n📊 Creating combined visualization...")
    create_combined_visualization(results_dict)

    print("\n✅ Analysis complete! Check the generated PNG and PDF files.")


if __name__ == "__main__":
    main()
