import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define professional color palette for academic publication
MODEL_COLORS = {
    "Baseline": "#696969",  # Dim Gray
    "Basic GNN": "#1f77b4",  # Professional Blue
    "Basic GNN + Augmentation": "#2ca02c",  # Professional Green
    "Equivariant GNN": "#d62728",  # Professional Red
}

# Create results directory if it doesn't exist
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_and_clean_data():
    """Load and clean the training and validation data"""
    # Load the data from results directory
    train_df = pd.read_csv(RESULTS_DIR / "train.csv")
    val_df = pd.read_csv(RESULTS_DIR / "val.csv")

    # Extract model performance data (in desired display order)
    models = {
        "Baseline": {
            "train": None,  # No training data for baseline
            "val": val_df["baseline - val_displacement_distance_epoch"].dropna(),
        },
        "Basic GNN": {
            "train": train_df["basic_gnn - train_displacement_distance_epoch"].dropna(),
            "val": val_df["basic_gnn - val_displacement_distance_epoch"].dropna(),
        },
        "Basic GNN + Augmentation": {
            "train": train_df[
                "basic_gnn_aug - train_displacement_distance_epoch"
            ].dropna(),
            "val": val_df["basic_gnn_aug - val_displacement_distance_epoch"].dropna(),
        },
        "Equivariant GNN": {
            "train": train_df["eq_gnn - train_displacement_distance_epoch"].dropna(),
            "val": val_df["eq_gnn - val_displacement_distance_epoch"].dropna(),
        },
    }

    return models


def create_focused_visualization():
    """Create professional visualization for academic publication"""
    models = load_and_clean_data()

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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Store handles and labels for shared legend
    handles, labels = [], []

    # Plot 1: Training curves
    ax1 = axes[0]
    for model_name, data in models.items():
        if data["train"] is not None:
            epochs = range(len(data["train"]))
            line = ax1.plot(
                epochs,
                data["train"],
                label=model_name,
                color=MODEL_COLORS[model_name],
                linewidth=2.5,
                alpha=0.8,
            )[0]
            # Collect handles and labels for shared legend
            if model_name not in labels:
                handles.append(line)
                labels.append(model_name)

    ax1.set_title("(a) Training $L^2$ Distance", fontsize=12, fontweight="bold", pad=15)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("$L^2$ Distance", fontsize=11)
    ax1.grid(True, alpha=0.3, linewidth=0.8)

    # Plot 2: Validation curves
    ax2 = axes[1]
    for model_name, data in models.items():
        if data["val"] is not None:
            epochs = range(len(data["val"]))
            ax2.plot(
                epochs,
                data["val"],
                label=model_name,
                color=MODEL_COLORS[model_name],
                linewidth=2.5,
                alpha=0.8,
            )

    ax2.set_title(
        "(b) Validation $L^2$ Distance", fontsize=12, fontweight="bold", pad=15
    )
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("$L^2$ Distance", fontsize=11)
    ax2.grid(True, alpha=0.3, linewidth=0.8)

    # Plot 3: Overfitting analysis (train-val gap)
    ax3 = axes[2]
    for model_name, data in models.items():
        if data["train"] is not None and data["val"] is not None:
            min_len = min(len(data["train"]), len(data["val"]))
            train_subset = data["train"].iloc[:min_len]
            val_subset = data["val"].iloc[:min_len]
            gap = val_subset.values - train_subset.values
            epochs = range(len(gap))
            ax3.plot(
                epochs,
                gap,
                label=model_name,
                color=MODEL_COLORS[model_name],
                linewidth=2.5,
                alpha=0.8,
            )

    ax3.set_title("(c) Generalization Gap", fontsize=12, fontweight="bold", pad=15)
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Validation - Training $L^2$ Distance", fontsize=11)
    ax3.grid(True, alpha=0.3, linewidth=0.8)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.6, linewidth=1)

    # Add shared legend at the bottom
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        fontsize=10,
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor="black",
        facecolor="white",
    )

    # Adjust layout for publication with extra space for legend
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.15)  # Make room for legend

    plt.savefig(
        RESULTS_DIR / "training_validation_analysis.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        RESULTS_DIR / "training_validation_analysis.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )  # Also save as PDF for publications
    plt.show()


def print_performance_summary():
    """Print a summary of model performances"""
    models = load_and_clean_data()

    print("=" * 70)
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 70)

    for model_name, data in models.items():
        print(f"\n{model_name}:")
        print("-" * (len(model_name) + 1))

        if data["train"] is not None:
            train_final = data["train"].iloc[-1]
            train_best = data["train"].min()
            print(f"  Training   - Final: {train_final:.6f}, Best: {train_best:.6f}")

        if data["val"] is not None:
            val_final = data["val"].iloc[-1]
            val_best = data["val"].min()
            val_avg_last10 = (
                data["val"].iloc[-10:].mean() if len(data["val"]) >= 10 else val_final
            )
            print(f"  Validation - Final: {val_final:.6f}, Best: {val_best:.6f}")
            print(f"  Validation - Last 10 Avg: {val_avg_last10:.6f}")

        # Calculate overfitting gap if both train and val data exist
        if data["train"] is not None and data["val"] is not None:
            min_len = min(len(data["train"]), len(data["val"]))
            train_subset = data["train"].iloc[:min_len]
            val_subset = data["val"].iloc[:min_len]
            final_gap = val_subset.iloc[-1] - train_subset.iloc[-1]
            avg_gap = np.mean(val_subset.values - train_subset.values)
            print(f"  Gen. Gap   - Final: {final_gap:.6f}, Average: {avg_gap:.6f}")

    print("\n" + "=" * 70)
    print("PERFORMANCE RANKING (Validation, Last 10 Epochs Average)")
    print("=" * 70)

    rankings = []
    for model_name, data in models.items():
        if data["val"] is not None and len(data["val"]) >= 10:
            avg_last10 = data["val"].iloc[-10:].mean()
            rankings.append((model_name, avg_last10))

    rankings.sort(key=lambda x: x[1])  # Sort by performance (lower is better)

    for i, (model_name, performance) in enumerate(rankings, 1):
        print(f"{i}. {model_name:<25}: {performance:.6f}")


if __name__ == "__main__":
    print("Generating publication-ready visualization...")
    create_focused_visualization()

    print("\nPerformance Analysis:")
    print_performance_summary()

    print("\nFiles generated:")
    print(f"- {RESULTS_DIR / 'training_validation_analysis.png'} (High-res PNG)")
    print(
        f"- {RESULTS_DIR / 'training_validation_analysis.pdf'} (Vector PDF for publications)"
    )
