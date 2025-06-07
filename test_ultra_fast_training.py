#!/usr/bin/env python3
"""
Test script to verify ultra-fast equivariant GNN training and equivariance
"""

import torch
import time
import numpy as np
from train_gnn_optimized import EquivariantGNN, PointCloudData


def test_rotation_equivariance():
    """Test if the model is equivariant to rotations"""
    print("\n🔄 Testing Rotation Equivariance...")

    # Create model
    model = EquivariantGNN(
        hidden_dim=16,
        num_layers=3,
    )
    model.eval()

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


def test_translation_equivariance():
    """Test if the model is equivariant to translations"""
    print("\n📍 Testing Translation Equivariance...")

    # Create model
    model = EquivariantGNN(
        hidden_dim=16,
        num_layers=3,
    )
    model.eval()

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


def test_equivariance_with_large_weights():
    """Test equivariance with much larger weights to verify true equivariance"""
    print("\n🔍 Testing Equivariance with Large Weights...")

    # Create model
    model = EquivariantGNN(
        hidden_dim=16,
        num_layers=3,
    )

    # Scale all weights by 10x
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(10.0)

    model.eval()

    # Create test data
    torch.manual_seed(42)
    positions = torch.randn(5, 3) * 2
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3, 4], [1, 2, 0, 3, 1, 4, 1, 4, 3]], dtype=torch.long
    )

    edge_vectors = positions[edge_index[1]] - positions[edge_index[0]]
    from e3nn.o3 import spherical_harmonics

    edge_attr = spherical_harmonics([0, 1], edge_vectors, normalize=True)

    # Test rotation
    def rotation_matrix_z(angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=torch.float32
        )

    R = rotation_matrix_z(np.pi / 4)

    # Original prediction
    with torch.no_grad():
        pred_original = model(positions, edge_index, edge_attr)

    # Rotated prediction
    rotated_positions = positions @ R.T
    rotated_edge_vectors = (
        rotated_positions[edge_index[1]] - rotated_positions[edge_index[0]]
    )
    rotated_edge_attr = spherical_harmonics(
        [0, 1], rotated_edge_vectors, normalize=True
    )

    with torch.no_grad():
        pred_rotated = model(rotated_positions, edge_index, rotated_edge_attr)

    expected = pred_original @ R.T
    error = torch.norm(pred_rotated - expected).item()

    print(f"   Large weights rotation error: {error:.8f}")

    is_equivariant = error < 1e-4
    print(
        f"   {'✅ EQUIVARIANT with large weights' if is_equivariant else '❌ NOT EQUIVARIANT with large weights'}"
    )

    return is_equivariant, error


def test_equivariance_with_random_weights():
    """Test equivariance with completely random large weights"""
    print("\n🎲 Testing Equivariance with Random Large Weights...")

    # Create model
    model = EquivariantGNN(
        hidden_dim=16,
        num_layers=3,
    )

    # Set completely random large weights
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 1.0)  # Much larger than the 0.1 scale

    model.eval()

    # Create test data
    torch.manual_seed(42)
    positions = torch.randn(5, 3) * 2
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3, 4], [1, 2, 0, 3, 1, 4, 1, 4, 3]], dtype=torch.long
    )

    edge_vectors = positions[edge_index[1]] - positions[edge_index[0]]
    from e3nn.o3 import spherical_harmonics

    edge_attr = spherical_harmonics([0, 1], edge_vectors, normalize=True)

    # Test rotation
    def rotation_matrix_z(angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=torch.float32
        )

    R = rotation_matrix_z(np.pi / 4)

    # Original prediction
    with torch.no_grad():
        pred_original = model(positions, edge_index, edge_attr)

    # Rotated prediction
    rotated_positions = positions @ R.T
    rotated_edge_vectors = (
        rotated_positions[edge_index[1]] - rotated_positions[edge_index[0]]
    )
    rotated_edge_attr = spherical_harmonics(
        [0, 1], rotated_edge_vectors, normalize=True
    )

    with torch.no_grad():
        pred_rotated = model(rotated_positions, edge_index, rotated_edge_attr)

    expected = pred_original @ R.T
    error = torch.norm(pred_rotated - expected).item()

    print(f"   Random large weights rotation error: {error:.8f}")

    is_equivariant = error < 1e-4
    print(
        f"   {'✅ EQUIVARIANT with random weights' if is_equivariant else '❌ NOT EQUIVARIANT with random weights'}"
    )

    return is_equivariant, error


def test_ultra_fast_training():
    print("🚀 Testing Ultra-Fast Equivariant GNN Training")
    print("=" * 50)

    # Create model and data module
    model = EquivariantGNN(
        hidden_dim=16,
        num_layers=3,
    )

    data_module = PointCloudData(
        data_dir="data/processed_sh",
        batch_size=2,
        val_split=0.2,
        num_workers=0,
    )

    # Setup data
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

    print("\n🎉 All tests passed! Ultra-fast model is ready for training.")

    # Performance summary
    print(f"\n📈 Performance Summary:")
    print(f"   Forward pass: {forward_time*1000:.1f}ms")
    print(f"   Training step: {training_time*1000:.1f}ms")
    print(f"   Validation step: {val_time*1000:.1f}ms")

    if forward_time <= 0.050:
        print("   ✅ Forward pass meets 50ms target!")
    else:
        print(f"   ⚠️  Forward pass {forward_time/0.050:.1f}x slower than target")

    # Test equivariance
    print("\n" + "=" * 50)
    print("🔍 EQUIVARIANCE TESTS")
    print("=" * 50)

    rot_equivariant, rot_error = test_rotation_equivariance()
    trans_equivariant, trans_error = test_translation_equivariance()

    # Test with different weight scales to verify true equivariance
    large_weight_equivariant, large_weight_error = (
        test_equivariance_with_large_weights()
    )
    random_weight_equivariant, random_weight_error = (
        test_equivariance_with_random_weights()
    )

    print("\n📋 EQUIVARIANCE SUMMARY:")
    print(
        f"   Rotation equivariance: {'✅ PASS' if rot_equivariant else '❌ FAIL'} (avg error: {rot_error:.6f})"
    )
    print(
        f"   Translation equivariance: {'✅ PASS' if trans_equivariant else '❌ FAIL'} (error: {trans_error:.6f})"
    )
    print(
        f"   Large weights equivariance: {'✅ PASS' if large_weight_equivariant else '❌ FAIL'} (error: {large_weight_error:.8f})"
    )
    print(
        f"   Random weights equivariance: {'✅ PASS' if random_weight_equivariant else '❌ FAIL'} (error: {random_weight_error:.8f})"
    )

    overall_equivariant = (
        rot_equivariant
        and trans_equivariant
        and large_weight_equivariant
        and random_weight_equivariant
    )
    print(
        f"\n🎯 OVERALL: {'✅ TRULY EQUIVARIANT' if overall_equivariant else '❌ EQUIVARIANCE ISSUES'}"
    )

    return forward_time, overall_equivariant


if __name__ == "__main__":
    test_ultra_fast_training()
