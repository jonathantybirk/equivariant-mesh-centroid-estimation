#!/usr/bin/env python3
"""
Test script to diagnose local minima issue with real data
"""
import torch
from src.model.eq_gnn import EquivariantGNN
from trainer import PointCloudData
import numpy as np


def test_real_data_learning():
    print("TESTING WITH REAL DATA...")

    # Load actual data
    data_module = PointCloudData(
        data_dir="data/processed_dv", batch_size=4, num_workers=0
    )
    data_module.setup()

    # Get a real batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    print(f"Real data batch:")
    print(f"  Nodes: {batch.x.shape}")
    print(f"  Edges: {batch.edge_index.shape}")
    print(f"  Edge attrs: {batch.edge_attr.shape}")
    print(f"  Targets: {batch.y.shape}")
    print(f"  Batch sizes: {batch.batch.bincount()}")
    print(f"  True COMs sample: {batch.y[:3]}")

    # Create model with current config
    model = EquivariantGNN(
        lr=1e-4,
        weight_decay=0,
        node_multiplicity=2,
        message_passing_steps=3,
        num_cg_layers=1,
        final_mlp_dims=[64, 32],
        dropout=0,
        init_method="xavier",
        debug=False,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test initial prediction
    model.eval()
    with torch.no_grad():
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        initial_loss = torch.nn.MSELoss()(pred, batch.y)
        print(f"Initial predictions sample: {pred[:3]}")
        print(f"Initial loss: {initial_loss.item():.6f}")

    # Test training for several steps with real data
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # Higher LR for testing
    losses = []

    print("\nTRAINING ON REAL DATA:")
    for step in range(25):
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = torch.nn.MSELoss()(pred, batch.y)
        loss.backward()

        # Check gradient norms for different parts
        cg_grad_norm = (
            sum(
                p.grad.norm().item() ** 2
                for name, p in model.named_parameters()
                if p.grad is not None and "message_layers" in name
            )
            ** 0.5
        )
        mlp_grad_norm = (
            sum(
                p.grad.norm().item() ** 2
                for name, p in model.named_parameters()
                if p.grad is not None and "final_mlp" in name
            )
            ** 0.5
        )
        total_grad_norm = (
            sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

        optimizer.step()
        losses.append(loss.item())

        if step % 5 == 0:
            print(
                f"Step {step:2d}: loss={loss.item():.6f}, total_grad={total_grad_norm:.8f}, cg_grad={cg_grad_norm:.8f}, mlp_grad={mlp_grad_norm:.8f}"
            )

    print(f"\nRESULTS:")
    initial_loss_val = losses[0]
    final_loss_val = losses[-1]
    improvement = (initial_loss_val - final_loss_val) / initial_loss_val * 100
    print(f"  Initial loss: {initial_loss_val:.6f}")
    print(f"  Final loss: {final_loss_val:.6f}")
    print(f"  Improvement: {improvement:.3f}%")

    if improvement < 0.1:
        print("CRITICAL: Model is not learning (<0.1% improvement)")
    elif improvement < 1:
        print("WARNING: Very slow learning (<1% improvement)")
    else:
        print("Model is learning")

    # Check if predictions are just averaging
    pred_std = pred.std().item()
    target_std = batch.y.std().item()
    print(f"\nPREDICTION ANALYSIS:")
    print(f"  Prediction std: {pred_std:.6f}")
    print(f"  Target std: {target_std:.6f}")
    if pred_std < target_std * 0.1:
        print("PROBLEM: Predictions have very low variance - model is just averaging!")

    # Test different graphs in the batch
    print(f"\nPER-GRAPH ANALYSIS:")
    for i in range(min(3, len(batch.y))):
        print(
            f"  Graph {i}: pred={pred[i].detach().numpy()}, target={batch.y[i].numpy()}"
        )

    return improvement > 1


if __name__ == "__main__":
    test_real_data_learning()
