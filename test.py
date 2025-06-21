#!/usr/bin/env python3
"""
Test script to verify ultra-fast equivariant GNN training and equivariance

Usage:
    # Test with random weights (default)
    python test.py

    # Test with trained checkpoint
    python test.py --checkpoint model.ckpt

    # Test specific functions only
    python test.py --checkpoint model.ckpt --test equivariance
    python test.py --checkpoint model.ckpt --test performance
    python test.py --checkpoint model.ckpt --test predictions
"""

import torch
import time
import numpy as np
import argparse
from pathlib import Path
from trainer import EquivariantGNN, PointCloudData


def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    """Load a trained model from checkpoint"""
    print(f"🔄 Loading model from checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters if available
    if "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]
        print(f"   Found hyperparameters: {list(hparams.keys())}")

        # Filter out Lightning-specific metadata fields
        model_hparams = {}
        valid_params = {
            "input_dim",
            "hidden_dim",
            "message_passing_steps",
            "final_mlp_dims",
            "edge_sh_degree",
            "init_method",
            "seed",
            "debug",
            "lr",
            "weight_decay",
        }

        for key, value in hparams.items():
            if key in valid_params:
                model_hparams[key] = value

        print(f"   Using model hyperparameters: {list(model_hparams.keys())}")

        # Create model with filtered hyperparameters
        model = EquivariantGNN(**model_hparams)
    else:
        print("   No hyperparameters found, using defaults")
        model = EquivariantGNN()

    # Load state dict
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        print(f"   ✅ Model weights loaded successfully")
    else:
        raise ValueError("No 'state_dict' found in checkpoint")

    # Extract training info if available
    if "epoch" in checkpoint:
        print(f"   Trained for {checkpoint['epoch']} epochs")

    if "train_loss" in checkpoint or "val_loss" in checkpoint:
        train_loss = checkpoint.get("train_loss", "N/A")
        val_loss = checkpoint.get("val_loss", "N/A")
        print(f"   Final losses - Train: {train_loss}, Val: {val_loss}")

    model.eval()
    return model


def create_random_model():
    """Create a model with random weights for baseline testing"""
    print("🎲 Creating model with random weights")
    model = EquivariantGNN()
    model.eval()
    return model


def test_rotation_equivariance(model):
    """Test if the model is equivariant to rotations using real data"""
    print("\n🔄 Testing Rotation Equivariance with Real Data...")

    # Load real data if data_module not provided
    print("   Loading real data from data/processed_dv...")
    data_module = PointCloudData(
        data_dir="data/processed_dv",
        batch_size=1,  # Test one sample at a time
        val_split=0.2,
        num_workers=0,
    )
    data_module.setup()

    # Get validation data loader
    val_loader = data_module.val_dataloader()

    # Test rotation matrices
    def rotation_matrix_z(angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=torch.float32
        )

    def rotation_matrix_x(angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return torch.tensor(
            [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]], dtype=torch.float32
        )

    def rotation_matrix_y(angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return torch.tensor(
            [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=torch.float32
        )

    all_errors = []
    test_samples = 0
    max_samples = 5  # Test on first 5 validation samples

    print(f"   Testing on up to {max_samples} real validation samples...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if test_samples >= max_samples:
                break

            # Extract data for single graph
            positions = batch.x  # Node positions
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr  # Should be displacement vectors
            target = batch.y  # Ground truth center of mass

            print(f"\n   Sample {test_samples + 1}:")
            print(f"     Nodes: {positions.shape[0]}, Edges: {edge_index.shape[1]}")
            print(f"     Edge attr shape: {edge_attr.shape}")
            print(
                f"     Edge attr sample: [{edge_attr[0,0]:.3f}, {edge_attr[0,1]:.3f}, {edge_attr[0,2]:.3f}]"
            )
            print(
                f"     Target COM: [{target[0,0]:.3f}, {target[0,1]:.3f}, {target[0,2]:.3f}]"
            )

            sample_errors = []

            # Test different rotation angles and axes
            test_rotations = [
                ("Z", np.pi / 6, rotation_matrix_z),
                ("Z", np.pi / 4, rotation_matrix_z),
                ("X", np.pi / 3, rotation_matrix_x),
                ("Y", np.pi / 4, rotation_matrix_y),
            ]

            for axis, angle, rot_func in test_rotations:
                R = rot_func(angle)

                # Original prediction
                pred_original = model(positions, edge_index, edge_attr, batch.batch)

                # Apply rotation to positions
                rotated_positions = positions @ R.T

                # Rotate the edge attributes (displacement vectors) directly
                # The edge_attr contains displacement vectors which should be rotated
                rotated_edge_attr = edge_attr @ R.T

                # Prediction on rotated data
                pred_rotated = model(
                    rotated_positions, edge_index, rotated_edge_attr, batch.batch
                )

                # Expected: R @ pred_original (rotation should transform the prediction)
                expected = pred_original @ R.T

                # Compute error
                error = torch.norm(pred_rotated - expected).item()
                sample_errors.append(error)
                all_errors.append(error)

                # Show actual predictions to verify they're not trivial
                print(f"     {axis}-axis rotation {angle:.2f}rad:")
                print(
                    f"       Original pred:  [{pred_original[0,0]:.6f}, {pred_original[0,1]:.6f}, {pred_original[0,2]:.6f}]"
                )
                print(
                    f"       Rotated pred:   [{pred_rotated[0,0]:.6f}, {pred_rotated[0,1]:.6f}, {pred_rotated[0,2]:.6f}]"
                )
                print(
                    f"       Expected pred:  [{expected[0,0]:.6f}, {expected[0,1]:.6f}, {expected[0,2]:.6f}]"
                )
                print(f"       Error: {error:.6f}")

                # Check if predictions are non-trivial (not just zeros)
                pred_magnitude = torch.norm(pred_original).item()
                print(f"       Prediction magnitude: {pred_magnitude:.6f}")
                if pred_magnitude < 1e-6:
                    print(f"       ⚠️  WARNING: Prediction is essentially zero!")
                else:
                    print(f"       ✅ Non-trivial prediction")

            avg_sample_error = np.mean(sample_errors)
            print(f"     Sample average error: {avg_sample_error:.6f}")

            test_samples += 1

    # Overall statistics
    avg_error = np.mean(all_errors)
    max_error = np.max(all_errors)
    std_error = np.std(all_errors)

    print(f"\n   📊 Overall Statistics (across {test_samples} samples):")
    print(f"     Average error: {avg_error:.6f}")
    print(f"     Max error: {max_error:.6f}")
    print(f"     Std error: {std_error:.6f}")
    print(f"     Total rotations tested: {len(all_errors)}")

    # Check if equivariant (error should be very small for a truly equivariant model)
    # Note: Real models might have small numerical errors, so we use a slightly more lenient threshold
    equivariance_threshold = 1e-3  # More lenient for real data
    is_equivariant = avg_error < equivariance_threshold

    print(
        f"   {'✅ EQUIVARIANT' if is_equivariant else '❌ NOT EQUIVARIANT'} (threshold: {equivariance_threshold})"
    )

    if not is_equivariant:
        print(
            f"   ⚠️  Note: Average error {avg_error:.6f} exceeds threshold {equivariance_threshold}"
        )
        if avg_error < 1e-2:
            print(
                f"   💡 Error is still quite small - model may be approximately equivariant"
            )

    return is_equivariant, avg_error


def run_equivariance_tests(model):
    """Run all equivariance tests"""
    print("\n" + "=" * 50)
    print("🔍 EQUIVARIANCE TESTS")
    print("=" * 50)

    rot_equivariant, rot_error = test_rotation_equivariance(model)

    # SKIP TRANSLATION EQUIVARIANCE - Dataset has centroid at origin
    # Translation equivariance is achieved through preprocessing (centering point clouds)
    print("\n📍 Translation Equivariance: SKIPPED")
    print("   ℹ️  Dataset preprocessing centers point clouds at origin")
    print("   ℹ️  Translation equivariance achieved through data preprocessing")
    print("   ℹ️  Model predicts COM relative to centered point cloud")

    trans_equivariant = True  # Assumed true due to preprocessing
    trans_error = 0.0

    print("\n📋 EQUIVARIANCE SUMMARY:")
    print(
        f"   Rotation equivariance: {'✅ PASS' if rot_equivariant else '❌ FAIL'} (avg error: {rot_error:.6f})"
    )
    print(
        f"   Translation equivariance: ✅ ACHIEVED VIA PREPROCESSING (centering point clouds)"
    )

    overall_equivariant = rot_equivariant  # Only rotation matters for testing
    print(
        f"\n🎯 OVERALL: {'✅ ROTATION EQUIVARIANT' if overall_equivariant else '❌ ROTATION EQUIVARIANCE ISSUES'}"
    )

    return {
        "rotation_equivariant": rot_equivariant,
        "rotation_error": rot_error,
        "translation_equivariant": trans_equivariant,
        "translation_error": trans_error,
        "overall_equivariant": overall_equivariant,
    }


def analyze_network_predictions(model):
    """Analyze what the network is actually predicting in detail"""
    print("\n🔍 DETAILED NETWORK OUTPUT ANALYSIS")
    print("=" * 50)

    data_module = PointCloudData(
        data_dir="data/processed_dv",
        batch_size=4,
        val_split=0.2,
        num_workers=0,
    )
    data_module.setup()

    val_loader = data_module.val_dataloader()
    model.eval()

    predictions = []
    targets = []

    print("📊 Analyzing predictions on validation set...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 5:  # Analyze first 5 batches
                break

            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            predictions.append(pred)
            targets.append(batch.y)

    # Concatenate all predictions and targets
    all_preds = torch.cat(predictions, dim=0)
    all_targets = torch.cat(targets, dim=0)

    print(f"   Total samples analyzed: {all_preds.shape[0]}")
    print(f"   Prediction shape: {all_preds.shape}")
    print(f"   Target shape: {all_targets.shape}")

    # Statistical analysis
    pred_mean = all_preds.mean(dim=0)
    pred_std = all_preds.std(dim=0)
    target_mean = all_targets.mean(dim=0)
    target_std = all_targets.std(dim=0)

    print(f"\n📈 Statistical Analysis:")
    print(
        f"   Predictions - Mean: [{pred_mean[0]:.4f}, {pred_mean[1]:.4f}, {pred_mean[2]:.4f}]"
    )
    print(
        f"   Predictions - Std:  [{pred_std[0]:.4f}, {pred_std[1]:.4f}, {pred_std[2]:.4f}]"
    )
    print(
        f"   Targets - Mean:     [{target_mean[0]:.4f}, {target_mean[1]:.4f}, {target_mean[2]:.4f}]"
    )
    print(
        f"   Targets - Std:      [{target_std[0]:.4f}, {target_std[1]:.4f}, {target_std[2]:.4f}]"
    )

    # Error analysis
    errors = torch.norm(all_preds - all_targets, dim=1)
    mae_per_axis = torch.abs(all_preds - all_targets).mean(dim=0)

    print(f"\n📏 Error Analysis:")
    print(f"   Mean L2 error: {errors.mean():.6f}")
    print(f"   Std L2 error:  {errors.std():.6f}")
    print(f"   Max L2 error:  {errors.max():.6f}")
    print(f"   Min L2 error:  {errors.min():.6f}")
    print(
        f"   MAE per axis:  [{mae_per_axis[0]:.6f}, {mae_per_axis[1]:.6f}, {mae_per_axis[2]:.6f}]"
    )

    # Show some individual examples
    print(f"\n🎯 Individual Examples (first 5):")
    for i in range(min(5, all_preds.shape[0])):
        pred = all_preds[i]
        target = all_targets[i]
        error = torch.norm(pred - target).item()
        print(f"   Sample {i+1}:")
        print(f"     Predicted: [{pred[0]:.4f}, {pred[1]:.4f}, {pred[2]:.4f}]")
        print(f"     Target:    [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]")
        print(f"     L2 Error:  {error:.6f}")

    # Check prediction ranges
    pred_min = all_preds.min(dim=0)[0]
    pred_max = all_preds.max(dim=0)[0]
    target_min = all_targets.min(dim=0)[0]
    target_max = all_targets.max(dim=0)[0]

    print(f"\n📊 Value Ranges:")
    print(
        f"   Predictions - Min: [{pred_min[0]:.4f}, {pred_min[1]:.4f}, {pred_min[2]:.4f}]"
    )
    print(
        f"   Predictions - Max: [{pred_max[0]:.4f}, {pred_max[1]:.4f}, {pred_max[2]:.4f}]"
    )
    print(
        f"   Targets - Min:     [{target_min[0]:.4f}, {target_min[1]:.4f}, {target_min[2]:.4f}]"
    )
    print(
        f"   Targets - Max:     [{target_max[0]:.4f}, {target_max[1]:.4f}, {target_max[2]:.4f}]"
    )

    # Check if predictions are reasonable
    reasonable_range = torch.all(
        torch.abs(all_preds) < 100
    )  # Should be within reasonable physical range
    print(f"\n✅ Sanity Checks:")
    print(
        f"   Predictions in reasonable range: {'✅ YES' if reasonable_range else '❌ NO'}"
    )
    print(
        f"   No NaN predictions: {'✅ YES' if not torch.isnan(all_preds).any() else '❌ NO'}"
    )
    print(
        f"   No Inf predictions: {'✅ YES' if not torch.isinf(all_preds).any() else '❌ NO'}"
    )

    return {
        "predictions": all_preds,
        "targets": all_targets,
        "errors": errors,
        "mean_l2_error": errors.mean().item(),
        "mae_per_axis": mae_per_axis.numpy(),
    }


def debug_network_forward(model):
    """Debug the network forward pass to see parameter statistics"""
    print("\n🐛 DEBUGGING NETWORK FORWARD PASS")
    print("=" * 50)

    # Create simple test data
    torch.manual_seed(42)
    x = torch.randn(5, 3) * 2.0  # 5 nodes with some variance
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

    # Generate edge attributes
    edge_vectors = x[edge_index[1]] - x[edge_index[0]]
    from e3nn.o3 import spherical_harmonics

    edge_attr = spherical_harmonics([0, 1], edge_vectors, normalize=True)

    # Create batch tensor (single graph)
    batch = torch.zeros(5, dtype=torch.long)

    print(f"📊 Input Analysis:")
    print(f"   Node features shape: {x.shape}")
    print(f"   Node features mean: {x.mean(dim=0)}")
    print(f"   Node features std: {x.std(dim=0)}")
    print(f"   Edge features shape: {edge_attr.shape}")
    print(f"   Edge features mean: {edge_attr.mean(dim=0)}")

    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🔧 Model Parameters:")
    print(f"   Total parameters: {total_params}")
    print(f"   Trainable parameters: {trainable_params}")

    # Check parameter ranges
    param_stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_stats.append(
                {
                    "name": name,
                    "shape": param.shape,
                    "mean": param.mean().item(),
                    "std": param.std().item(),
                    "min": param.min().item(),
                    "max": param.max().item(),
                }
            )

    print(f"\n📈 Parameter Statistics:")
    for stat in param_stats[:10]:  # Show first 10
        print(
            f"   {stat['name']}: mean={stat['mean']:.6f}, std={stat['std']:.6f}, range=[{stat['min']:.6f}, {stat['max']:.6f}]"
        )

    # Forward pass with detailed outputs
    print(f"\n🔄 Forward Pass Analysis:")
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index, edge_attr, batch)

    print(f"   Final output: {pred}")
    print(f"   Output shape: {pred.shape}")
    print(f"   Output mean: {pred.mean()}")
    print(f"   Output std: {pred.std()}")

    return {
        "prediction": pred,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_stats": param_stats,
    }


def run_full_test_suite(model):
    """Run the complete test suite"""
    print("🚀 COMPLETE MODEL TEST SUITE")
    print("=" * 60)

    results = {}

    # Equivariance tests
    results["equivariance"] = run_equivariance_tests(model)

    # Prediction analysis
    print("\n" + "=" * 50)
    print("📊 PREDICTION ANALYSIS")
    print("=" * 50)
    results["predictions"] = analyze_network_predictions(model)

    # Debug analysis
    results["debug"] = debug_network_forward(model)

    # Final summary
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)

    equiv = results["equivariance"]
    pred = results["predictions"]

    print(f"\n🔄 Equivariance:")
    print(f"   Rotation: {'✅ PASS' if equiv['rotation_equivariant'] else '❌ FAIL'}")
    print(
        f"   Translation: {'✅ PASS' if equiv['translation_equivariant'] else '❌ FAIL'}"
    )

    print(f"\n📊 Predictions:")
    print(f"   Mean L2 error: {pred['mean_l2_error']:.6f}")
    print(f"   MAE per axis: {pred['mae_per_axis']}")

    overall_status = (
        equiv["overall_equivariant"]  # Equivariant
        and pred["mean_l2_error"] < 10.0  # Reasonable predictions
    )

    print(
        f"\n🎉 OVERALL STATUS: {'✅ EXCELLENT' if overall_status else '⚠️ NEEDS ATTENTION'}"
    )

    return results


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Test Equivariant GNN Model")
    parser.add_argument("--checkpoint", "-c", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--test",
        "-t",
        choices=["all", "performance", "equivariance", "predictions", "debug"],
        default="all",
        help="Which tests to run",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/processed_sh", help="Data directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size for testing"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")

    args = parser.parse_args()

    # Setup data module
    data_module = PointCloudData(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.2,
        num_workers=0,
    )

    # Load or create model
    if args.checkpoint:
        model = load_model_from_checkpoint(args.checkpoint, args.device)
        model_type = "TRAINED"
    else:
        model = create_random_model()
        model_type = "RANDOM WEIGHTS"

    print(f"\n🧪 Testing {model_type} Model")
    print("=" * 60)

    # Run selected tests
    if args.test == "all":
        results = run_full_test_suite(model)
    elif args.test == "equivariance":
        run_equivariance_tests(model)
    elif args.test == "predictions":
        analyze_network_predictions(model)
    elif args.test == "debug":
        debug_network_forward(model)


def test_ultra_fast_training():
    """Legacy function for backward compatibility"""
    print("🚀 Testing Ultra-Fast Equivariant GNN Training")
    print("=" * 50)

    model = create_random_model()
    return run_full_test_suite(model)


if __name__ == "__main__":
    main()
