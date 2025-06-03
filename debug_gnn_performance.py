#!/usr/bin/env python3
"""
Debug script to identify GNN performance bottlenecks
"""

import torch
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.model.eq_gnn import GNN


def test_gnn_performance():
    print("🔍 Testing GNN Performance Bottlenecks")
    print("=" * 50)

    # Load a real data sample
    print("📊 Loading real data sample...")
    data = torch.load("data/processed_new/train/Toaster_.pt", weights_only=False)

    print(f"   Nodes: {data['num_nodes']}")
    print(f"   Edges: {data['num_edges']}")
    print(f"   Edge attr: {data['edge_attr'].shape}")

    # Create model
    print("\n🤖 Creating model...")
    model = GNN(
        input_dim=3,
        hidden_dim=4,
        max_sh_degree=1,
        message_passing_steps=1,  # Just 1 step to isolate
        final_mlp_dims=[8, 4],
    )

    # Prepare inputs
    x = data["node_features"].float()
    edge_index = data["edge_index"]
    edge_attr = data["edge_attr"]

    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(
        f"   Input: x={x.shape}, edge_index={edge_index.shape}, edge_attr={edge_attr.shape}"
    )

    # Test forward pass with timing
    print("\n⏱️  Testing forward pass...")

    model.eval()
    with torch.no_grad():
        start_time = time.time()

        try:
            pred = model(x, edge_index, edge_attr, batch=None)

            total_time = time.time() - start_time
            print(f"✅ Forward pass completed in {total_time:.3f}s")
            print(f"   Prediction shape: {pred.shape}")
            print(f"   Performance: {total_time/data['num_edges']*1000:.2f}ms per edge")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_gnn_performance()
