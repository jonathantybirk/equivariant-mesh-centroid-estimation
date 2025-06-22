#!/usr/bin/env python3
"""
Performance Comparison Script for All Models

Tests all 4 models on the remaining test set (after 1000 train/val samples)
and performs statistical analysis with confidence intervals and paired t-tests.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from torch_geometric.data import Data, DataLoader
import warnings

warnings.filterwarnings("ignore")

# Import models and data loading from trainer
from trainer import PointCloudData, EquivariantGNN, BasicGNN, ImprovedBasicGNN, Baseline


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
    elif "improved_basic" in checkpoint_name:
        model_class = ImprovedBasicGNN
        model_name = "ImprovedBasicGNN"
    elif "simple_gnn" in checkpoint_name or "basic" in checkpoint_name:
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
    print(f"\n📊 Evaluating {model_name} performance and equivariance...")

    model.to(device)
    model.eval()

    all_performance_errors = []
    all_equivariance_errors = []

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

            # Original prediction on entire batch
            pred_original = model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.batch,  # Use batch information
            )

            # Performance error (unrotated) - Use same L2 calculation as training
            displacement = pred_original - batch_data.y  # [B, 3]
            per_sample_distances = torch.norm(
                displacement, dim=1
            )  # [B] - L2 norm per sample

            # Store individual sample errors (not batch mean)
            all_performance_errors.extend(per_sample_distances.cpu().numpy())

            # Apply rotation to entire batch
            rotated_batch_data = apply_rotation_to_batch(batch_data, R)

            # Prediction on rotated batch
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

    # Calculate statistics
    mean_performance = np.mean(all_performance_errors)
    std_performance = np.std(all_performance_errors)
    mean_equivariance = np.mean(all_equivariance_errors)
    std_equivariance = np.std(all_equivariance_errors)

    print(f"   Mean Performance L2 Distance: {mean_performance:.6f}")
    print(f"   Mean Equivariance Error: {mean_equivariance:.6f}")

    return {
        "model_name": model_name,
        "performance_errors": all_performance_errors,
        "equivariance_errors": all_equivariance_errors,
        "mean_performance": mean_performance,
        "std_performance": std_performance,
        "mean_equivariance": mean_equivariance,
        "std_equivariance": std_equivariance,
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
        n = len(errors)
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # T-test against null hypothesis (mean = 0, perfect equivariance)
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

    # 3. Pairwise paired t-tests for performance
    print(f"\n3. PAIRWISE PAIRED T-TESTS (Performance)")
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

    # 4. Pairwise paired t-tests for equivariance
    print(f"\n4. PAIRWISE PAIRED T-TESTS (Equivariance)")
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

    # 5. Model ranking
    print(f"\n5. MODEL RANKING")
    print("-" * 60)
    print("By Performance (L2 Distance):")
    sorted_models = sorted(models, key=lambda x: results_dict[x]["mean_performance"])
    for rank, model in enumerate(sorted_models, 1):
        l2_dist = results_dict[model]["mean_performance"]
        print(f"  {rank}. {model:<20} L2 Dist: {l2_dist:.6f}")

    print("\nBy Equivariance (Error):")
    sorted_models = sorted(models, key=lambda x: results_dict[x]["mean_equivariance"])
    for rank, model in enumerate(sorted_models, 1):
        eq_error = results_dict[model]["mean_equivariance"]
        print(f"  {rank}. {model:<20} Eq Error: {eq_error:.6f}")

    print("\nBy Parameter Count:")
    sorted_models = sorted(models, key=lambda x: model_params[x])
    for rank, model in enumerate(sorted_models, 1):
        params = model_params[model]
        print(f"  {rank}. {model:<20} Parameters: {params:,}")

    return results_dict, pairwise_results, equivariance_pairwise_results


def create_combined_visualization(results_dict):
    """Create combined bar chart showing performance and equivariance"""
    models = list(results_dict.keys())
    n_models = len(models)

    # Extract data
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

        # Equivariance CI
        eq_errors = results_dict[model]["equivariance_errors"]
        n = len(eq_errors)
        std_error = np.std(eq_errors)
        t_critical = stats.t.ppf(0.975, df=n - 1)
        margin = t_critical * (std_error / np.sqrt(n))
        equivariance_cis.append(margin)

    # Create figure with single y-axis (same scale for both metrics)
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set up positions for bars
    x = np.arange(n_models)
    width = 0.35

    # Colors
    perf_color = "#3498db"  # Blue
    eq_color = "#e74c3c"  # Red

    # Performance bars
    bars1 = ax.bar(
        x - width / 2,
        performance_means,
        width,
        yerr=performance_cis,
        capsize=5,
        color=perf_color,
        alpha=0.8,
        label="Prediction Error",
        edgecolor="white",
        linewidth=1.5,
    )

    # Equivariance bars (same y-axis)
    bars2 = ax.bar(
        x + width / 2,
        equivariance_means,
        width,
        yerr=equivariance_cis,
        capsize=5,
        color=eq_color,
        alpha=0.8,
        label="Equivariance Error",
        edgecolor="white",
        linewidth=1.5,
    )

    # Labels and formatting
    ax.set_xlabel("Models", fontsize=14, fontweight="bold")
    ax.set_ylabel("Error Magnitude (Same Scale)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight="bold", rotation=45)

    # Add title and legend
    plt.title(
        "Model Performance vs Equivariance Error\n(Same Scale Comparison with 95% Confidence Intervals)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.legend(loc="upper left", fontsize=12)

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
        ax.text(
            bar1.get_x() + bar1.get_width() / 2.0,
            height1 + perf_ci * 0.1,
            f"{perf:.4f}\n±{perf_ci:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=perf_color,
                alpha=0.7,
                edgecolor="white",
            ),
        )

        # Equivariance labels
        height2 = bar2.get_height()
        ax.text(
            bar2.get_x() + bar2.get_width() / 2.0,
            height2 + eq_ci * 0.1,
            f"{eq:.6f}\n±{eq_ci:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=eq_color,
                alpha=0.7,
                edgecolor="white",
            ),
        )

    # Improve layout
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    return fig


def main():
    """Main function to run performance comparison"""
    print("🚀 MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    # Model checkpoints
    checkpoints = {
        "basic_gnn.ckpt": "BasicGNN",
        "improved_basic_gnn.ckpt": "ImprovedBasicGNN",
        "eq_gnn.ckpt": "EquivariantGNN",
        "simple_gnn.ckpt": "SimpleGNN",
    }

    # Add baseline model (no checkpoint needed)
    baseline_models = {"baseline": "Baseline"}

    # Check which checkpoints exist
    available_checkpoints = {}
    for ckpt_path, model_name in checkpoints.items():
        if Path(ckpt_path).exists():
            available_checkpoints[ckpt_path] = model_name
        else:
            print(f"⚠️  Checkpoint not found: {ckpt_path}")

    if not available_checkpoints:
        print("❌ No checkpoints found!")
        return

    print(f"Found {len(available_checkpoints)} model checkpoints")

    # Create test dataset (only once)
    print("\n📂 Setting up test dataset...")
    test_data_loader = TestDataLoader()
    test_data_loader.setup()
    test_data = (
        test_data_loader.test_data
    )  # Get individual data samples for equivariance testing

    # Evaluate all models
    results = {}
    model_params = {}  # Store parameter counts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Evaluate checkpoint-based models
    for ckpt_path, expected_name in available_checkpoints.items():
        try:
            model, model_name, total_params = load_model_from_checkpoint(
                ckpt_path, device
            )
            result = evaluate_model_performance_and_equivariance(
                model, test_data, model_name, device
            )
            results[model_name] = result
            model_params[model_name] = total_params

            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"❌ Error with {ckpt_path}: {e}")
            continue

    # Evaluate baseline model (no checkpoint needed)
    for baseline_key, baseline_name in baseline_models.items():
        try:
            print(f"🔄 Creating {baseline_name} model (no checkpoint needed)")
            model = Baseline()

            # Count parameters for baseline
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(
                f"   Parameters: {total_params:,} total, {trainable_params:,} trainable"
            )

            result = evaluate_model_performance_and_equivariance(
                model, test_data, baseline_name, device
            )
            results[baseline_name] = result
            model_params[baseline_name] = total_params

            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"❌ Error with {baseline_name}: {e}")
            continue

    if not results:
        print("❌ No models successfully evaluated!")
        return

    # Perform statistical analysis
    analysis_results, pairwise_results, equivariance_pairwise_results = (
        perform_statistical_analysis(results, model_params)
    )

    # Create visualization
    print(f"\n📊 Creating combined visualization...")
    fig = create_combined_visualization(results)

    # Save the plot
    plt.savefig("performance_equivariance_comparison.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ Plot saved as 'performance_equivariance_comparison.png'")

    # Summary
    print(f"\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)

    best_performance_model = min(
        results.keys(), key=lambda x: results[x]["mean_performance"]
    )
    best_performance = results[best_performance_model]["mean_performance"]

    best_equivariance_model = min(
        results.keys(), key=lambda x: results[x]["mean_equivariance"]
    )
    best_equivariance = results[best_equivariance_model]["mean_equivariance"]

    print(
        f"🥇 Best performing model (L2 Distance): {best_performance_model} ({best_performance:.6f})"
    )
    print(
        f"🔄 Most equivariant model: {best_equivariance_model} ({best_equivariance:.6f})"
    )

    # Show all models with both metrics and parameters
    print(f"\n📊 All Models Summary:")
    print(
        f"{'Model':<20} {'Performance (L2)':<18} {'Equivariance Error':<18} {'Parameters':<12}"
    )
    print("-" * 75)
    for model in results.keys():
        perf = results[model]["mean_performance"]
        eq = results[model]["mean_equivariance"]
        params = model_params[model]
        print(f"{model:<20} {perf:<18.6f} {eq:<18.6f} {params:<12,}")

    # Check if best performer is also most equivariant
    if best_performance_model == best_equivariance_model:
        print(
            f"\n🎯 {best_performance_model} is both the best performer AND most equivariant!"
        )
    else:
        print(
            f"\n⚖️  Trade-off observed: Best performer ({best_performance_model}) differs from most equivariant ({best_equivariance_model})"
        )

        # Show significant pairwise differences for performance
    print(f"\n🔍 Significant Pairwise Differences (Performance):")
    significant_found = False
    for (model1, model2), result in pairwise_results.items():
        if result["significant"]:
            better_model = model1 if result["mean_diff"] < 0 else model2
            worse_model = model2 if result["mean_diff"] < 0 else model1
            print(
                f"   {better_model} significantly outperforms {worse_model} (p={result['p_value']:.2e})"
            )
            significant_found = True

    if not significant_found:
        print(
            "   No statistically significant performance differences found between models"
        )

    # Show significant pairwise differences for equivariance
    print(f"\n🔄 Significant Pairwise Differences (Equivariance):")
    significant_found = False
    for (model1, model2), result in equivariance_pairwise_results.items():
        if result["significant"]:
            better_model = model1 if result["mean_diff"] < 0 else model2
            worse_model = model2 if result["mean_diff"] < 0 else model1
            print(
                f"   {better_model} significantly more equivariant than {worse_model} (p={result['p_value']:.2e})"
            )
            significant_found = True

    if not significant_found:
        print(
            "   No statistically significant equivariance differences found between models"
        )


if __name__ == "__main__":
    main()
