#!/usr/bin/env python3
"""
Test script to verify ultra-fast equivariant GNN training and equivariance

Usage:
    # Test with random weights (default)
    python test_ultra_fast_training.py

    # Test with trained checkpoint
    python test_ultra_fast_training.py --checkpoint model.ckpt

    # Test specific functions only
    python test_ultra_fast_training.py --checkpoint model.ckpt --test equivariance
    python test_ultra_fast_training.py --checkpoint model.ckpt --test performance
    python test_ultra_fast_training.py --checkpoint model.ckpt --test predictions
"""

import torch
import time
import numpy as np
import argparse
from pathlib import Path
from train_gnn_optimized import EquivariantGNN, PointCloudData


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
            "max_sh_degree",
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
    """Test if the model is equivariant to rotations"""
    print("\n🔄 Testing Rotation Equivariance...")

    # Create simple test graph
    torch.manual_seed(42)
    positions = torch.randn(5, 3) * 2  # 5 nodes
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3, 4], [1, 2, 0, 3, 1, 4, 1, 4, 3]], dtype=torch.long
    )

    # Generate edge attributes (spherical harmonics)
    edge_vectors = positions[edge_index[1]] - positions[edge_index[0]]
    from e3nn.o3 import spherical_harmonics

    edge_attr = spherical_harmonics([0, 1], edge_vectors, normalize=True)

    # Test rotation matrices
    def rotation_matrix_z(angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=torch.float32
        )

    errors = []
    for angle in [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]:
        R = rotation_matrix_z(angle)

        # Original prediction
        with torch.no_grad():
            pred_original = model(positions, edge_index, edge_attr)

        # Rotated positions
        rotated_positions = positions @ R.T
        rotated_edge_vectors = (
            rotated_positions[edge_index[1]] - rotated_positions[edge_index[0]]
        )
        rotated_edge_attr = spherical_harmonics(
            [0, 1], rotated_edge_vectors, normalize=True
        )

        with torch.no_grad():
            pred_rotated = model(rotated_positions, edge_index, rotated_edge_attr)

        # Expected: R @ pred_original
        expected = pred_original @ R.T

        error = torch.norm(pred_rotated - expected).item()
        errors.append(error)

        print(f"   Rotation {angle:.2f}rad: error = {error:.6f}")

    avg_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"   Average error: {avg_error:.6f}")
    print(f"   Max error: {max_error:.6f}")

    # Check if equivariant (error should be very small)
    is_equivariant = avg_error < 1e-4
    print(f"   {'✅ EQUIVARIANT' if is_equivariant else '❌ NOT EQUIVARIANT'}")

    return is_equivariant, avg_error


def test_translation_equivariance(model):
    """Test if the model is equivariant to translations"""
    print("\n📍 Testing Translation Equivariance...")

    # Create simple test graph
    torch.manual_seed(42)
    positions = torch.randn(4, 3)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )

    # Generate edge attributes
    edge_vectors = positions[edge_index[1]] - positions[edge_index[0]]
    from e3nn.o3 import spherical_harmonics

    edge_attr = spherical_harmonics([0, 1], edge_vectors, normalize=True)

    # Test translation
    translation = torch.tensor([2.0, -1.0, 3.0])

    # Original prediction
    with torch.no_grad():
        pred_original = model(positions, edge_index, edge_attr)

    # Translated positions (edge attributes stay the same for translation!)
    translated_positions = positions + translation

    with torch.no_grad():
        pred_translated = model(translated_positions, edge_index, edge_attr)

    # Expected: pred_original + translation
    expected = pred_original + translation

    error = torch.norm(pred_translated - expected).item()
    print(f"   Translation error: {error:.6f}")

    is_equivariant = error < 1e-4
    print(
        f"   {'✅ TRANSLATION EQUIVARIANT' if is_equivariant else '❌ NOT TRANSLATION EQUIVARIANT'}"
    )

    return is_equivariant, error


def test_model_performance(model, data_module=None):
    """Test model performance on real data"""
    print("\n⚡ Testing Model Performance...")

    if data_module is None:
        data_module = PointCloudData(
            data_dir="data/processed_sh",
            batch_size=2,
            val_split=0.2,
            num_workers=0,
        )
        data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Test single batch
    print("📊 Testing single batch...")

    # Get a batch
    batch = next(iter(train_loader))
    print(f"   Batch keys: {batch.keys}")
    print(f"   Batch size: {batch.x.shape[0]} nodes")

    # Forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        pred_com = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        forward_time = time.time() - start_time

    print(f"   Forward pass: {forward_time*1000:.1f}ms")
    print(f"   Prediction shape: {pred_com.shape}")

    # Test training step
    print("\n🏋️ Testing training step...")
    model.train()

    start_time = time.time()
    loss = model.training_step(batch, 0)
    training_time = time.time() - start_time

    print(f"   Training step: {training_time*1000:.1f}ms")
    print(f"   Loss: {loss:.6f}")

    # Test validation step
    print("\n✅ Testing validation step...")
    model.eval()

    val_batch = next(iter(val_loader))
    start_time = time.time()
    val_loss = model.validation_step(val_batch, 0)
    val_time = time.time() - start_time

    print(f"   Validation step: {val_time*1000:.1f}ms")
    print(f"   Val loss: {val_loss:.6f}")

    # Performance summary
    print(f"\n📈 Performance Summary:")
    print(f"   Forward pass: {forward_time*1000:.1f}ms")
    print(f"   Training step: {training_time*1000:.1f}ms")
    print(f"   Validation step: {val_time*1000:.1f}ms")

    if forward_time <= 0.050:
        print("   ✅ Forward pass meets 50ms target!")
    else:
        print(f"   ⚠️  Forward pass {forward_time/0.050:.1f}x slower than target")

    return {
        "forward_time": forward_time,
        "training_time": training_time,
        "val_time": val_time,
        "train_loss": loss.item(),
        "val_loss": val_loss.item(),
    }


def run_equivariance_tests(model):
    """Run all equivariance tests"""
    print("\n" + "=" * 50)
    print("🔍 EQUIVARIANCE TESTS")
    print("=" * 50)

    rot_equivariant, rot_error = test_rotation_equivariance(model)
    trans_equivariant, trans_error = test_translation_equivariance(model)

    print("\n📋 EQUIVARIANCE SUMMARY:")
    print(
        f"   Rotation equivariance: {'✅ PASS' if rot_equivariant else '❌ FAIL'} (avg error: {rot_error:.6f})"
    )
    print(
        f"   Translation equivariance: {'✅ PASS' if trans_equivariant else '❌ FAIL'} (error: {trans_error:.6f})"
    )

    overall_equivariant = rot_equivariant and trans_equivariant
    print(
        f"\n🎯 OVERALL: {'✅ TRULY EQUIVARIANT' if overall_equivariant else '❌ EQUIVARIANCE ISSUES'}"
    )

    return {
        "rotation_equivariant": rot_equivariant,
        "rotation_error": rot_error,
        "translation_equivariant": trans_equivariant,
        "translation_error": trans_error,
        "overall_equivariant": overall_equivariant,
    }


def analyze_network_predictions(model, data_module=None):
    """Analyze what the network is actually predicting in detail"""
    print("\n🔍 DETAILED NETWORK OUTPUT ANALYSIS")
    print("=" * 50)

    if data_module is None:
        data_module = PointCloudData(
            data_dir="data/processed_sh",
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


def run_full_test_suite(model, data_module=None):
    """Run the complete test suite"""
    print("🚀 COMPLETE MODEL TEST SUITE")
    print("=" * 60)

    results = {}

    # Performance tests
    print("\n" + "=" * 50)
    print("⚡ PERFORMANCE TESTS")
    print("=" * 50)
    results["performance"] = test_model_performance(model, data_module)

    # Equivariance tests
    results["equivariance"] = run_equivariance_tests(model)

    # Prediction analysis
    print("\n" + "=" * 50)
    print("📊 PREDICTION ANALYSIS")
    print("=" * 50)
    results["predictions"] = analyze_network_predictions(model, data_module)

    # Debug analysis
    results["debug"] = debug_network_forward(model)

    # Final summary
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)

    perf = results["performance"]
    equiv = results["equivariance"]
    pred = results["predictions"]

    print(f"🎯 Performance:")
    print(f"   Forward time: {perf['forward_time']*1000:.1f}ms")
    print(f"   Train loss: {perf['train_loss']:.6f}")
    print(f"   Val loss: {perf['val_loss']:.6f}")

    print(f"\n🔄 Equivariance:")
    print(f"   Rotation: {'✅ PASS' if equiv['rotation_equivariant'] else '❌ FAIL'}")
    print(
        f"   Translation: {'✅ PASS' if equiv['translation_equivariant'] else '❌ FAIL'}"
    )

    print(f"\n📊 Predictions:")
    print(f"   Mean L2 error: {pred['mean_l2_error']:.6f}")
    print(f"   MAE per axis: {pred['mae_per_axis']}")

    overall_status = (
        perf["forward_time"] < 0.050  # Performance OK
        and equiv["overall_equivariant"]  # Equivariant
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
        results = run_full_test_suite(model, data_module)
    elif args.test == "performance":
        test_model_performance(model, data_module)
    elif args.test == "equivariance":
        run_equivariance_tests(model)
    elif args.test == "predictions":
        analyze_network_predictions(model, data_module)
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
