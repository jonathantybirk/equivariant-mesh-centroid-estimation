#!/usr/bin/env python3
"""
Paper Visualization Script

Creates publication-quality visualizations comparing EquivariantGNN and ImprovedBasicGNN
predictions on point clouds before and after rotation, suitable for research papers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Import necessary functions from the performance comparison script
from performance_comparison import (
    load_model_from_checkpoint,
    random_rotation_matrix,
    apply_rotation_to_batch,
    TestDataLoader,
)
from trainer import EquivariantGNN, ImprovedBasicGNN
from torch_geometric.data import DataLoader


def create_paper_visualization(
    sample_data, eq_model, improved_model, rotation_matrix, device="cpu", sample_idx=0
):
    """
    Create a comprehensive visualization for the paper showing:
    - Original point cloud with both model predictions
    - Rotated point cloud with both model predictions
    - Comparison of prediction movements
    """

    # Set up the figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 12))

    # Convert single sample to batch format for model inference
    batch_data = sample_data.clone()
    batch_data.batch = torch.zeros(
        batch_data.x.shape[0], dtype=torch.long, device=device
    )

    # Ensure all data is on the correct device
    batch_data.x = batch_data.x.to(device)
    batch_data.edge_index = batch_data.edge_index.to(device)
    if batch_data.edge_attr is not None:
        batch_data.edge_attr = batch_data.edge_attr.to(device)
    batch_data.y = batch_data.y.to(device)
    batch_data.batch = batch_data.batch.to(device)

    # Get original predictions
    with torch.no_grad():
        eq_pred_original = (
            eq_model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.batch,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        improved_pred_original = (
            improved_model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.batch,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    # Apply rotation to data
    rotated_data = apply_rotation_to_batch(batch_data, rotation_matrix)

    # Get rotated predictions
    with torch.no_grad():
        eq_pred_rotated = (
            eq_model(
                rotated_data.x,
                rotated_data.edge_index,
                rotated_data.edge_attr,
                rotated_data.batch,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        improved_pred_rotated = (
            improved_model(
                rotated_data.x,
                rotated_data.edge_index,
                rotated_data.edge_attr,
                rotated_data.batch,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    # Expected predictions (original predictions rotated)
    eq_pred_expected = (
        rotation_matrix.cpu() @ torch.tensor(eq_pred_original).float()
    ).numpy()
    improved_pred_expected = (
        rotation_matrix.cpu() @ torch.tensor(improved_pred_original).float()
    ).numpy()

    # Convert data to numpy for plotting
    original_points = batch_data.x.cpu().numpy()
    rotated_points = rotated_data.x.cpu().numpy()
    true_centroid_original = batch_data.y.squeeze().cpu().numpy()
    true_centroid_rotated = rotated_data.y.squeeze().cpu().numpy()

    # Subplot 1: Original Point Cloud with Predictions
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.scatter(
        original_points[:, 0],
        original_points[:, 1],
        original_points[:, 2],
        c="lightblue",
        alpha=0.6,
        s=20,
        label="Point Cloud",
    )
    ax1.scatter(
        *true_centroid_original,
        c="black",
        s=200,
        marker="*",
        label="True Centroid",
        edgecolors="white",
        linewidth=2,
    )
    ax1.scatter(
        *eq_pred_original,
        c="red",
        s=150,
        marker="o",
        label="EquivariantGNN",
        edgecolors="white",
        linewidth=2,
    )
    ax1.scatter(
        *improved_pred_original,
        c="blue",
        s=150,
        marker="s",
        label="ImprovedBasicGNN",
        edgecolors="white",
        linewidth=2,
    )

    ax1.set_title(
        "Original Point Cloud\nwith Model Predictions", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("X", fontsize=12)
    ax1.set_ylabel("Y", fontsize=12)
    ax1.set_zlabel("Z", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Rotated Point Cloud with Predictions
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.scatter(
        rotated_points[:, 0],
        rotated_points[:, 1],
        rotated_points[:, 2],
        c="lightcoral",
        alpha=0.6,
        s=20,
        label="Rotated Point Cloud",
    )
    ax2.scatter(
        *true_centroid_rotated,
        c="black",
        s=200,
        marker="*",
        label="True Centroid",
        edgecolors="white",
        linewidth=2,
    )
    ax2.scatter(
        *eq_pred_rotated,
        c="red",
        s=150,
        marker="o",
        label="EquivariantGNN",
        edgecolors="white",
        linewidth=2,
    )
    ax2.scatter(
        *improved_pred_rotated,
        c="blue",
        s=150,
        marker="s",
        label="ImprovedBasicGNN",
        edgecolors="white",
        linewidth=2,
    )

    ax2.set_title(
        "Rotated Point Cloud\nwith Model Predictions", fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("X", fontsize=12)
    ax2.set_ylabel("Y", fontsize=12)
    ax2.set_zlabel("Z", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Equivariance Comparison for EquivariantGNN
    ax3 = fig.add_subplot(223, projection="3d")
    ax3.scatter(
        rotated_points[:, 0],
        rotated_points[:, 1],
        rotated_points[:, 2],
        c="lightcoral",
        alpha=0.3,
        s=15,
        label="Rotated Point Cloud",
    )
    ax3.scatter(
        *eq_pred_rotated,
        c="red",
        s=150,
        marker="o",
        label="Actual Prediction",
        edgecolors="white",
        linewidth=2,
    )
    ax3.scatter(
        *eq_pred_expected,
        c="darkred",
        s=150,
        marker="^",
        label="Expected (Rotated Original)",
        edgecolors="white",
        linewidth=2,
    )

    # Draw arrow showing equivariance error
    if np.linalg.norm(eq_pred_rotated - eq_pred_expected) > 1e-6:
        ax3.quiver(
            eq_pred_expected[0],
            eq_pred_expected[1],
            eq_pred_expected[2],
            eq_pred_rotated[0] - eq_pred_expected[0],
            eq_pred_rotated[1] - eq_pred_expected[1],
            eq_pred_rotated[2] - eq_pred_expected[2],
            color="orange",
            arrow_length_ratio=0.1,
            linewidth=3,
            label=f"Equivariance Error: {np.linalg.norm(eq_pred_rotated - eq_pred_expected):.6f}",
        )

    ax3.set_title(
        "EquivariantGNN\nEquivariance Analysis", fontsize=14, fontweight="bold"
    )
    ax3.set_xlabel("X", fontsize=12)
    ax3.set_ylabel("Y", fontsize=12)
    ax3.set_zlabel("Z", fontsize=12)
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Equivariance Comparison for ImprovedBasicGNN
    ax4 = fig.add_subplot(224, projection="3d")
    ax4.scatter(
        rotated_points[:, 0],
        rotated_points[:, 1],
        rotated_points[:, 2],
        c="lightcoral",
        alpha=0.3,
        s=15,
        label="Rotated Point Cloud",
    )
    ax4.scatter(
        *improved_pred_rotated,
        c="blue",
        s=150,
        marker="s",
        label="Actual Prediction",
        edgecolors="white",
        linewidth=2,
    )
    ax4.scatter(
        *improved_pred_expected,
        c="darkblue",
        s=150,
        marker="^",
        label="Expected (Rotated Original)",
        edgecolors="white",
        linewidth=2,
    )

    # Draw arrow showing equivariance error
    equivariance_error = np.linalg.norm(improved_pred_rotated - improved_pred_expected)
    ax4.quiver(
        improved_pred_expected[0],
        improved_pred_expected[1],
        improved_pred_expected[2],
        improved_pred_rotated[0] - improved_pred_expected[0],
        improved_pred_rotated[1] - improved_pred_expected[1],
        improved_pred_rotated[2] - improved_pred_expected[2],
        color="orange",
        arrow_length_ratio=0.1,
        linewidth=3,
        label=f"Equivariance Error: {equivariance_error:.6f}",
    )

    ax4.set_title(
        "ImprovedBasicGNN\nEquivariance Analysis", fontsize=14, fontweight="bold"
    )
    ax4.set_xlabel("X", fontsize=12)
    ax4.set_ylabel("Y", fontsize=12)
    ax4.set_zlabel("Z", fontsize=12)
    ax4.legend(loc="upper right", fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle(
        f"Model Comparison: Prediction Accuracy and Equivariance\nSample {sample_idx}",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # Calculate and print metrics
    eq_performance_error = np.linalg.norm(eq_pred_original - true_centroid_original)
    improved_performance_error = np.linalg.norm(
        improved_pred_original - true_centroid_original
    )
    eq_equivariance_error = np.linalg.norm(eq_pred_rotated - eq_pred_expected)
    improved_equivariance_error = np.linalg.norm(
        improved_pred_rotated - improved_pred_expected
    )

    print(f"\n📊 Sample {sample_idx} Metrics:")
    print(f"Performance Errors (L2 Distance):")
    print(f"  EquivariantGNN:     {eq_performance_error:.6f}")
    print(f"  ImprovedBasicGNN:   {improved_performance_error:.6f}")
    print(f"Equivariance Errors:")
    print(f"  EquivariantGNN:     {eq_equivariance_error:.6f}")
    print(f"  ImprovedBasicGNN:   {improved_equivariance_error:.6f}")

    return fig, {
        "eq_performance_error": eq_performance_error,
        "improved_performance_error": improved_performance_error,
        "eq_equivariance_error": eq_equivariance_error,
        "improved_equivariance_error": improved_equivariance_error,
    }


def create_side_by_side_comparison(
    sample_data, eq_model, improved_model, rotation_matrix, device="cpu", sample_idx=0
):
    """
    Create a cleaner side-by-side comparison suitable for paper figures
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(14, 10), subplot_kw={"projection": "3d"}
    )

    # Convert single sample to batch format
    batch_data = sample_data.clone()
    batch_data.batch = torch.zeros(
        batch_data.x.shape[0], dtype=torch.long, device=device
    )

    # Ensure all data is on the correct device
    batch_data.x = batch_data.x.to(device)
    batch_data.edge_index = batch_data.edge_index.to(device)
    if batch_data.edge_attr is not None:
        batch_data.edge_attr = batch_data.edge_attr.to(device)
    batch_data.y = batch_data.y.to(device)
    batch_data.batch = batch_data.batch.to(device)

    # Get predictions
    with torch.no_grad():
        eq_pred_original = (
            eq_model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.batch,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        improved_pred_original = (
            improved_model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.batch,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    # Apply rotation
    rotated_data = apply_rotation_to_batch(batch_data, rotation_matrix)

    with torch.no_grad():
        eq_pred_rotated = (
            eq_model(
                rotated_data.x,
                rotated_data.edge_index,
                rotated_data.edge_attr,
                rotated_data.batch,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        improved_pred_rotated = (
            improved_model(
                rotated_data.x,
                rotated_data.edge_index,
                rotated_data.edge_attr,
                rotated_data.batch,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    # Expected predictions
    eq_pred_expected = (
        rotation_matrix.cpu() @ torch.tensor(eq_pred_original).float()
    ).numpy()
    improved_pred_expected = (
        rotation_matrix.cpu() @ torch.tensor(improved_pred_original).float()
    ).numpy()

    # Data for plotting
    original_points = batch_data.x.cpu().numpy()
    rotated_points = rotated_data.x.cpu().numpy()
    true_centroid_original = batch_data.y.squeeze().cpu().numpy()
    true_centroid_rotated = rotated_data.y.squeeze().cpu().numpy()

    # Plot settings
    point_alpha = 0.4
    point_size = 15
    pred_size = 100

    # Original - EquivariantGNN
    ax1.scatter(
        original_points[:, 0],
        original_points[:, 1],
        original_points[:, 2],
        c="lightblue",
        alpha=point_alpha,
        s=point_size,
    )
    ax1.scatter(
        *true_centroid_original,
        c="black",
        s=pred_size * 1.5,
        marker="*",
        edgecolors="white",
        linewidth=2,
    )
    ax1.scatter(
        *eq_pred_original,
        c="red",
        s=pred_size,
        marker="o",
        edgecolors="white",
        linewidth=2,
    )
    ax1.set_title("EquivariantGNN\n(Original)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Rotated - EquivariantGNN
    ax2.scatter(
        rotated_points[:, 0],
        rotated_points[:, 1],
        rotated_points[:, 2],
        c="lightcoral",
        alpha=point_alpha,
        s=point_size,
    )
    ax2.scatter(
        *true_centroid_rotated,
        c="black",
        s=pred_size * 1.5,
        marker="*",
        edgecolors="white",
        linewidth=2,
    )
    ax2.scatter(
        *eq_pred_rotated,
        c="red",
        s=pred_size,
        marker="o",
        edgecolors="white",
        linewidth=2,
    )
    ax2.scatter(
        *eq_pred_expected,
        c="darkred",
        s=pred_size,
        marker="^",
        edgecolors="white",
        linewidth=2,
        alpha=0.7,
    )
    ax2.set_title("EquivariantGNN\n(Rotated)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Original - ImprovedBasicGNN
    ax3.scatter(
        original_points[:, 0],
        original_points[:, 1],
        original_points[:, 2],
        c="lightblue",
        alpha=point_alpha,
        s=point_size,
    )
    ax3.scatter(
        *true_centroid_original,
        c="black",
        s=pred_size * 1.5,
        marker="*",
        edgecolors="white",
        linewidth=2,
    )
    ax3.scatter(
        *improved_pred_original,
        c="blue",
        s=pred_size,
        marker="s",
        edgecolors="white",
        linewidth=2,
    )
    ax3.set_title("ImprovedBasicGNN\n(Original)", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Rotated - ImprovedBasicGNN
    ax4.scatter(
        rotated_points[:, 0],
        rotated_points[:, 1],
        rotated_points[:, 2],
        c="lightcoral",
        alpha=point_alpha,
        s=point_size,
    )
    ax4.scatter(
        *true_centroid_rotated,
        c="black",
        s=pred_size * 1.5,
        marker="*",
        edgecolors="white",
        linewidth=2,
    )
    ax4.scatter(
        *improved_pred_rotated,
        c="blue",
        s=pred_size,
        marker="s",
        edgecolors="white",
        linewidth=2,
    )
    ax4.scatter(
        *improved_pred_expected,
        c="darkblue",
        s=pred_size,
        marker="^",
        edgecolors="white",
        linewidth=2,
        alpha=0.7,
    )
    ax4.set_title("ImprovedBasicGNN\n(Rotated)", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # Set equal aspect ratios and remove axis labels for cleaner look
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="black",
            markersize=12,
            label="True Centroid",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="EquivariantGNN",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="ImprovedBasicGNN",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="Expected (Rotated)",
            alpha=0.7,
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=4,
        fontsize=11,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    return fig


def main():
    """Main function to create paper visualizations"""
    print("🎨 Creating Paper Visualizations")
    print("=" * 50)

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load EquivariantGNN
    eq_model, eq_name, eq_params = load_model_from_checkpoint("eq_gnn.ckpt", device)
    eq_model.to(device)  # Ensure all buffers are moved to device
    print(f"✅ Loaded {eq_name} with {eq_params:,} parameters")

    # Load ImprovedBasicGNN
    improved_model, improved_name, improved_params = load_model_from_checkpoint(
        "improved_basic_gnn.ckpt", device
    )
    improved_model.to(device)  # Ensure all buffers are moved to device
    print(f"✅ Loaded {improved_name} with {improved_params:,} parameters")

    # Load test data (only first 10 samples for debugging)
    print(f"\n📂 Loading test data...")
    test_data_loader = TestDataLoader(num_samples=10)
    test_data_loader.setup()

    # Select interesting samples for visualization
    sample_indices = [0, 1, 2]  # Just first 3 samples

    for i, sample_idx in enumerate(range(len(test_data_loader.test_data))):
        print(f"\n🎯 Creating visualization for sample {sample_idx}...")

        # Get sample
        sample_data = test_data_loader.test_data[sample_idx].to(device)

        # Generate rotation matrix
        rotation_seed = 42 + sample_idx
        R = random_rotation_matrix(seed=rotation_seed).to(device)

        # Create comprehensive visualization
        fig1, metrics = create_paper_visualization(
            sample_data, eq_model, improved_model, R, device, sample_idx
        )

        # Save comprehensive version
        filename1 = f"paper_visualization_comprehensive_sample_{sample_idx}.png"
        plt.figure(fig1.number)
        plt.savefig(filename1, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"   ✅ Saved comprehensive visualization: {filename1}")

        # Create clean side-by-side version
        fig2 = create_side_by_side_comparison(
            sample_data, eq_model, improved_model, R, device, sample_idx
        )

        # Save clean version
        filename2 = f"paper_visualization_clean_sample_{sample_idx}.png"
        plt.figure(fig2.number)
        plt.savefig(filename2, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"   ✅ Saved clean visualization: {filename2}")

        plt.close("all")  # Close figures to save memory

    print(f"\n🎉 Paper visualizations created successfully!")
    print(f"Files saved:")
    print(f"  - Comprehensive versions: paper_visualization_comprehensive_sample_*.png")
    print(f"  - Clean versions: paper_visualization_clean_sample_*.png")
    print(f"\nRecommendation: Use the clean versions for paper figures")


if __name__ == "__main__":
    main()
