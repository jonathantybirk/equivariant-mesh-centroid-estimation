#!/usr/bin/env python3
# val_displacement_distance=0.0588, train_displacement_distance=0.0329
"""
Debug script to analyze validation behavior
"""
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data, DataLoader
from src.model.eq_gnn import EquivariantGNN


def load_data(data_dir="data/processed_sh2", max_samples=20):
    """Load a small subset of data for debugging"""
    file_list = list(Path(data_dir).glob("**/*.pt"))[:max_samples]
    data_cache = []

    for file_path in file_list:
        try:
            data = torch.load(file_path, weights_only=False)
            x = data["node_features"].float().contiguous()
            edge_index = data["edge_index"].long().contiguous()
            edge_attr = data.get("edge_attr")
            if edge_attr is not None:
                edge_attr = edge_attr.float().contiguous()
            target = data["target"].squeeze().float().contiguous()

            data_cache.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=target.unsqueeze(0),
                )
            )
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    return data_cache


def debug_model_predictions():
    """Debug the model predictions on validation data"""

    # Load data
    print("Loading data...")
    data_cache = load_data(max_samples=20)
    print(f"Loaded {len(data_cache)} samples")

    if len(data_cache) < 4:
        print("Not enough data samples to debug")
        return

    # Split into train/val (same as trainer)
    train_size = int(0.8 * len(data_cache))
    train_data, val_data = torch.utils.data.random_split(
        data_cache,
        [train_size, len(data_cache) - train_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # Create model
    model = EquivariantGNN(debug=False)  # Turn off debug to reduce noise

    print("\n=== ANALYZING DISPLACEMENT CALCULATION ===")

    # Test with one validation sample
    if len(val_data) > 0:
        sample = val_data[0]

        # Test both train and eval mode
        for mode in ["train", "eval"]:
            if mode == "train":
                model.train()
            else:
                model.eval()

            print(f"\n--- {mode.upper()} MODE ---")

            with torch.no_grad():
                # Single sample
                pred = model(sample.x, sample.edge_index, sample.edge_attr, None)
                target = sample.y

                print(f"Single sample:")
                print(f"  Prediction: {pred.detach().numpy()}")
                print(f"  Target: {target.detach().numpy()}")
                print(
                    f"  Individual displacement: {torch.norm(pred - target).item():.6f}"
                )

                # Current method (problematic)
                displacement = pred - target
                mean_displacement_vector = displacement.mean(dim=0)
                mean_displacement_distance = torch.norm(mean_displacement_vector)
                print(
                    f"  Current method (mean then norm): {mean_displacement_distance.item():.6f}"
                )

                # Better method (per-sample then mean)
                per_sample_distances = torch.norm(pred - target, dim=1)
                mean_per_sample_distance = per_sample_distances.mean()
                print(
                    f"  Better method (norm then mean): {mean_per_sample_distance.item():.6f}"
                )

    # Test with multiple samples to show the problem
    if len(val_data) >= 2:
        print(f"\n=== BATCH EFFECT DEMONSTRATION ===")

        # Create a small batch
        batch_samples = [val_data[i] for i in range(min(2, len(val_data)))]
        batch_loader = DataLoader(batch_samples, batch_size=2, shuffle=False)

        model.eval()
        batch = next(iter(batch_loader))

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y

            print(f"Batch of {pred.shape[0]} samples:")
            print(f"  Predictions: {pred.detach().numpy()}")
            print(f"  Targets: {target.detach().numpy()}")

            # Current method (problematic)
            displacement = pred - target  # [B, 3]
            mean_displacement_vector = displacement.mean(
                dim=0
            )  # [3] - cancels out opposing errors!
            mean_displacement_distance = torch.norm(mean_displacement_vector)  # scalar
            print(
                f"  Current method (mean then norm): {mean_displacement_distance.item():.6f}"
            )

            # Better method
            per_sample_distances = torch.norm(pred - target, dim=1)  # [B]
            mean_per_sample_distance = per_sample_distances.mean()  # scalar
            print(
                f"  Better method (norm then mean): {mean_per_sample_distance.item():.6f}"
            )
            print(f"  Per-sample distances: {per_sample_distances.detach().numpy()}")

            print("\n  PROBLEM: Current method can cancel out errors!")
            print(
                "  If one sample has error [+1, +1, +1] and another has [-1, -1, -1],"
            )
            print("  the mean displacement vector becomes [0, 0, 0] with norm=0!")


if __name__ == "__main__":
    debug_model_predictions()
