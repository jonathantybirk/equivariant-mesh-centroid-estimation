#!/usr/bin/env python3
"""
Test script to verify equivariant GNN model and equivariance properties

Usage:
    # Run basic equivariance tests (default)
    python test_eq_gnn.py

    # Test with trained checkpoint
    python test_eq_gnn.py --checkpoint model.ckpt

    # Test specific functions only
    python test_eq_gnn.py --test equivariance
    python test_eq_gnn.py --test debug
    python test_eq_gnn.py --test basic_gnn
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# Add src to path so we can import the model
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from model.eq_gnn import EquivariantGNN
    from model.gnn import BasicGNN

    print("✅ Successfully imported both EquivariantGNN and BasicGNN")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def create_synthetic_data(num_nodes=12, num_edges=50, device="cpu"):
    """Create synthetic graph data for testing"""
    torch.manual_seed(42)

    # Node positions (3D coordinates)
    node_pos = torch.randn(num_nodes, 3, device=device)

    # Edge indices (random connections)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)

    # Edge attributes (spherical harmonics up to l=2: 1+3+5=9 features)
    edge_attr = torch.randn(num_edges, 9, device=device)

    # Batch (single graph)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    return node_pos, edge_index, edge_attr, batch


def test_basic_gnn():
    """Test BasicGNN model with dropout functionality"""
    print("\n🔬 Testing BasicGNN with Dropout")
    print("=" * 50)

    device = torch.device("cpu")

    # Test with different dropout values
    dropout_values = [0.0, 0.1, 0.3]

    for dropout in dropout_values:
        print(f"\n📋 Testing with dropout={dropout}")

        # Create model
        model = BasicGNN(
            hidden_dim=32,
            message_passing_steps=2,
            message_mlp_dims=[64, 32],
            update_mlp_dims=[32],
            final_mlp_dims=[32, 16],
            dropout=dropout,
            lr=1e-3,
            weight_decay=1e-5,
        ).to(device)

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   📊 Trainable parameters: {trainable_params:,}")
        print(f"   📊 Dropout rate: {model.dropout}")

        # Create test data
        node_pos, edge_index, edge_attr_full, batch = create_synthetic_data(
            device=device
        )

        # BasicGNN uses 3D positions as node features and edge features
        node_features = node_pos  # [N, 3]
        edge_attr = edge_attr_full[:, :3]  # Use first 3 features [E, 3]

        # Test forward pass
        print(f"   🔄 Testing forward pass...")
        model.eval()
        with torch.no_grad():
            pred_eval = model(node_features, edge_index, edge_attr, batch)

        model.train()
        pred_train = model(node_features, edge_index, edge_attr, batch)

        print(f"   ✅ Forward pass successful")
        print(f"   📊 Input shape: {node_features.shape}")
        print(f"   📊 Output shape: {pred_eval.shape}")
        print(f"   📊 Prediction (eval): {pred_eval.squeeze().tolist()}")
        print(f"   📊 Prediction (train): {pred_train.squeeze().tolist()}")

        # Check if dropout makes a difference (should be different in train vs eval mode)
        if dropout > 0:
            diff = torch.abs(pred_eval - pred_train).max().item()
            print(f"   📊 Max difference (train vs eval): {diff:.6f}")
            if diff > 1e-6:
                print(
                    f"   ✅ Dropout is working (predictions differ between train/eval)"
                )
            else:
                print(f"   ⚠️  Dropout effect is minimal (might be due to random seed)")
        else:
            print(f"   ✅ No dropout - predictions should be identical")

    print("\n🎉 BasicGNN test completed successfully!")


def test_equivariance():
    """Test equivariance properties of EquivariantGNN"""
    print("\n🔬 Testing Equivariance Properties")
    print("=" * 50)

    device = torch.device("cpu")

    # Create model
    model = EquivariantGNN(
        max_sh_degree=1,
        base_l_values=[0, 1],
        multiplicity=2,
        num_cg_layers=2,
        message_passing_steps=2,
        debug=False,
    ).to(device)

    print(
        f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Create test data
    node_pos, edge_index, edge_attr, batch = create_synthetic_data(device=device)

    # Test rotational equivariance
    print("\n🔄 Testing Rotation Equivariance...")

    angles_to_test = [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    rotation_errors = []

    model.eval()
    with torch.no_grad():
        # Original prediction
        pred_original = model(node_pos, edge_index, edge_attr, batch, node_pos=node_pos)

        for angle in angles_to_test:
            # Rotation around Z-axis
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = torch.tensor(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]],
                dtype=torch.float32,
                device=device,
            )

            # Rotate positions
            node_pos_rot = torch.matmul(node_pos, R.T)

            # Predict on rotated input
            pred_rotated_input = model(
                node_pos_rot, edge_index, edge_attr, batch, node_pos=node_pos_rot
            )

            # Manually rotate original prediction
            pred_rotated_manual = torch.matmul(pred_original, R.T)

            # Compare
            error = torch.norm(pred_rotated_input - pred_rotated_manual).item()
            rotation_errors.append(error)

            print(f"   📐 Angle {angle:.2f} rad: Error = {error:.8f}")

    avg_rotation_error = np.mean(rotation_errors)
    print(f"\n📊 Average Rotation Error: {avg_rotation_error:.8f}")

    # Test translational equivariance
    print("\n🔄 Testing Translation Equivariance...")

    translations_to_test = [
        torch.tensor([1.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 2.0, 0.0], device=device),
        torch.tensor([0.0, 0.0, 1.5], device=device),
        torch.tensor([1.0, 1.0, 1.0], device=device),
    ]

    translation_errors = []

    with torch.no_grad():
        for translation in translations_to_test:
            # Translate positions
            node_pos_trans = node_pos + translation.unsqueeze(0)

            # Predict on translated input
            pred_translated_input = model(
                node_pos_trans, edge_index, edge_attr, batch, node_pos=node_pos_trans
            )

            # Manually translate original prediction
            pred_translated_manual = pred_original + translation.unsqueeze(0)

            # Compare
            error = torch.norm(pred_translated_input - pred_translated_manual).item()
            translation_errors.append(error)

            print(f"   📐 Translation {translation.tolist()}: Error = {error:.8f}")

    avg_translation_error = np.mean(translation_errors)
    print(f"\n📊 Average Translation Error: {avg_translation_error:.8f}")

    # Check if errors are within acceptable tolerance
    tolerance = 1e-5
    if avg_rotation_error < tolerance and avg_translation_error < tolerance:
        print(f"\n🎉 ✅ EQUIVARIANCE TEST PASSED!")
        print(f"   Rotation error: {avg_rotation_error:.8f} < {tolerance}")
        print(f"   Translation error: {avg_translation_error:.8f} < {tolerance}")
    else:
        print(f"\n❌ EQUIVARIANCE TEST FAILED!")
        print(f"   Rotation error: {avg_rotation_error:.8f}")
        print(f"   Translation error: {avg_translation_error:.8f}")
        print(f"   Tolerance: {tolerance}")


def test_debug_model():
    """Test model creation and basic operations with debug output"""
    print("\n🔧 Debug Model Test")
    print("=" * 50)

    device = torch.device("cpu")

    # Create model with debug=True
    model = EquivariantGNN(
        max_sh_degree=1,
        base_l_values=[0, 1],
        multiplicity=1,
        num_cg_layers=2,
        message_passing_steps=1,
        debug=True,  # Enable debug output
    ).to(device)

    print(f"✅ Model created successfully")

    # Create test data
    node_pos, edge_index, edge_attr, batch = create_synthetic_data(device=device)

    print(f"\n🔄 Running forward pass with debug output...")
    model.eval()
    with torch.no_grad():
        pred = model(node_pos, edge_index, edge_attr, batch, node_pos=node_pos)

    print(f"✅ Debug test completed")
    print(f"📊 Prediction: {pred.squeeze().tolist()}")


def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    """Load a trained model from checkpoint"""
    print(f"🔄 Loading model from checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters
    hparams = checkpoint.get("hyper_parameters", {})
    print(f"📋 Found hyperparameters: {hparams}")

    # Create model with saved hyperparameters
    model = EquivariantGNN(**hparams)

    # Load state dict
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    return model


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Test EquivariantGNN model")
    parser.add_argument(
        "--test",
        type=str,
        default="equivariance",
        choices=["equivariance", "debug", "basic_gnn"],
        help="Which test to run",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for testing",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run tests on"
    )

    args = parser.parse_args()

    print("🧪 EquivariantGNN Test Suite")
    print("=" * 50)
    print(f"🔧 Test: {args.test}")
    print(f"💻 Device: {args.device}")
    if args.checkpoint:
        print(f"📁 Checkpoint: {args.checkpoint}")

    try:
        if args.test == "equivariance":
            test_equivariance()
        elif args.test == "debug":
            test_debug_model()
        elif args.test == "basic_gnn":
            test_basic_gnn()

        print(f"\n🎉 All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
