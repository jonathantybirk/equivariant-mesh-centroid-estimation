#!/usr/bin/env python3
"""
Diagnostic script to test why validation MAE < training MAE
Tests the hypothesis that it's due to dropout/layernorm noise during training
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.model.eq_gnn import EquivariantGNN
from train_gnn_optimized import PointCloudData
from torch_geometric.data import DataLoader


def test_train_val_gap():
    """Test if train/val gap is due to dropout/layernorm during training"""

    # Create model with dropout
    model = EquivariantGNN(
        lr=1e-3,
        weight_decay=1e-5,
        dropout=0.1,  # This should cause the issue
        multiplicity=2,
    )

    # Load a small dataset
    data_module = PointCloudData(
        data_dir="data/processed_sh", batch_size=8, val_split=0.1
    )
    data_module.setup()

    # Get small samples for testing
    train_loader = DataLoader(
        list(data_module.train_data)[:20], batch_size=8, shuffle=False
    )
    val_loader = DataLoader(
        list(data_module.val_data)[:20], batch_size=8, shuffle=False
    )

    print("=== DIAGNOSTIC TEST: Train vs Val MAE ===")

    # Test 1: Training mode MAE (with dropout noise)
    model.train()
    train_mae_noisy = 0.0
    train_samples = 0

    with torch.no_grad():
        for batch in train_loader:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            mae = F.l1_loss(pred, batch.y, reduction="sum")
            train_mae_noisy += mae.item()
            train_samples += batch.y.size(0)

    train_mae_noisy /= train_samples

    # Test 2: Training data in EVAL mode (no dropout noise)
    model.eval()
    train_mae_clean = 0.0

    with torch.no_grad():
        for batch in train_loader:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            mae = F.l1_loss(pred, batch.y, reduction="sum")
            train_mae_clean += mae.item()

    train_mae_clean /= train_samples

    # Test 3: Validation MAE (always in eval mode)
    val_mae = 0.0
    val_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            mae = F.l1_loss(pred, batch.y, reduction="sum")
            val_mae += mae.item()
            val_samples += batch.y.size(0)

    val_mae /= val_samples

    # Results
    print(f"1. Training MAE (with dropout/noise): {train_mae_noisy:.6f}")
    print(f"2. Training MAE (clean, eval mode):   {train_mae_clean:.6f}")
    print(f"3. Validation MAE (eval mode):       {val_mae:.6f}")
    print()

    # Analysis
    gap_noisy_vs_clean = train_mae_noisy - train_mae_clean
    gap_clean_vs_val = train_mae_clean - val_mae

    print(f"Gap due to train/eval mode: {gap_noisy_vs_clean:.6f}")
    print(f"True generalization gap:    {gap_clean_vs_val:.6f}")
    print()

    if abs(gap_noisy_vs_clean) > abs(gap_clean_vs_val):
        print(
            "✅ CONFIRMED: The train/val gap is mostly due to dropout/layernorm noise!"
        )
        print("   The 'clean' training MAE is much closer to validation MAE.")
    else:
        print("❌ UNEXPECTED: The gap persists even in eval mode.")

    return train_mae_noisy, train_mae_clean, val_mae


if __name__ == "__main__":
    test_train_val_gap()
