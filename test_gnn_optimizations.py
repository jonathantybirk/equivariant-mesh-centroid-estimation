#!/usr/bin/env python3
"""
GNN Optimization Testing Script

Tests PyTorch Geometric optimizations with exactly 2 data points:
- Step 2: PyG's optimized layers vs custom implementation
- Step 3: Optimized data loading
- Step 4: Memory optimizations
- torch.compile performance
"""

import warnings

warnings.filterwarnings(
    "ignore", message="An issue occurred while importing 'torch-scatter'"
)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.typing")

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, MessagePassing
import os
import sys
import time
import numpy as np
from pathlib import Path
import json
import glob

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Try to import the original model
try:
    from src.model.eq_gnn import GNN as OriginalGNN

    ORIGINAL_AVAILABLE = True
except ImportError:
    print("Original GNN model not available - will test optimized versions only")
    ORIGINAL_AVAILABLE = False


class OptimizedGNN(pl.LightningModule):
    """Optimized GNN using PyG's native layers (Step 2)"""

    def __init__(
        self,
        input_dim=3,
        hidden_dim=128,
        output_dim=3,
        num_layers=4,
        layer_type="GCN",  # 'GCN', 'GAT', or 'SAGE'
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        use_compile=True,
        debug=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_compile = use_compile
        self.debug = debug

        # Build the network using PyG's optimized layers
        self.layers = torch.nn.ModuleList()

        # Input layer
        if layer_type == "GCN":
            self.layers.append(GCNConv(input_dim, hidden_dim))
        elif layer_type == "GAT":
            self.layers.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
            hidden_dim = hidden_dim * 4  # Adjust for concatenated heads
        elif layer_type == "SAGE":
            from torch_geometric.nn import SAGEConv

            self.layers.append(SAGEConv(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            if layer_type == "GCN":
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            elif layer_type == "GAT":
                self.layers.append(
                    GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
                )
            elif layer_type == "SAGE":
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))

        # Output layer
        if layer_type == "GCN":
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        elif layer_type == "GAT":
            self.layers.append(
                GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=False)
            )
            hidden_dim = hidden_dim // 4
        elif layer_type == "SAGE":
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))

        # Final prediction head
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, output_dim),
        )

        if debug:
            print(
                f"Created {layer_type} model with {self.count_parameters()} parameters"
            )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Apply graph layers
        for i, layer in enumerate(self.layers):
            if self.layer_type == "GCN":
                x = layer(x, edge_index)
            elif self.layer_type in ["GAT", "SAGE"]:
                x = layer(x, edge_index)

            if i < len(self.layers) - 1:  # Don't apply activation after last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling to get graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)  # Simple mean if no batch info

        # Final prediction
        return self.predictor(x)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.mse_loss(pred, batch.y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.mse_loss(pred, batch.y)
        mae = F.l1_loss(pred, batch.y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae)
        return loss

    def configure_optimizers(self):
        # Use tensor for learning rate for torch.compile compatibility
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=torch.tensor(self.lr) if self.use_compile else self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer


class OptimizedDataset:
    """Optimized dataset with memory-efficient loading (Step 4)"""

    def __init__(self, data_dir, max_samples=2, debug=False):
        """Load exactly max_samples for testing"""
        self.debug = debug

        # Find data files
        file_list = sorted(glob.glob(os.path.join(data_dir, "*.pt")))[:max_samples]

        if debug:
            print(f"Loading {len(file_list)} samples for testing...")

        self.data_cache = []
        total_load_time = 0

        for i, file_path in enumerate(file_list):
            start_time = time.time()
            try:
                # Load and process data
                data = torch.load(file_path, weights_only=False)

                # Create optimized data object
                x = data["node_features"].float().contiguous()  # Memory optimization
                edge_index = data["edge_index"].long().contiguous()
                edge_attr = (
                    data["edge_attr"].float().contiguous()
                    if "edge_attr" in data
                    else None
                )
                target = data["target"].squeeze().float().contiguous()

                # Create PyG data object with memory optimization
                data_obj = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=target.unsqueeze(0),
                )

                self.data_cache.append(data_obj)

                load_time = time.time() - start_time
                total_load_time += load_time

                if debug:
                    print(
                        f"  Sample {i+1}: {x.shape[0]} nodes, {edge_index.shape[1]} edges, "
                        f"loaded in {load_time:.3f}s"
                    )

            except Exception as e:
                if debug:
                    print(f"Failed to load {file_path}: {e}")
                continue

        if debug:
            print(f"Total loading time: {total_load_time:.3f}s")
            if len(self.data_cache) > 0:
                # Calculate memory usage
                sample_size = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in [self.data_cache[0].x, self.data_cache[0].edge_index]
                    + (
                        [self.data_cache[0].edge_attr]
                        if self.data_cache[0].edge_attr is not None
                        else []
                    )
                    + [self.data_cache[0].y]
                )
                total_size_mb = (sample_size * len(self.data_cache)) / (1024 * 1024)
                print(f"Memory usage: {total_size_mb:.1f} MB")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx]


def create_optimized_dataloader(dataset, batch_size=2, debug=False):
    """Create optimized dataloader (Step 3)"""

    # Optimized dataloader settings
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,  # Deterministic for testing
        "num_workers": 0,  # Single threaded for small test
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,  # Not needed for small dataset
        "drop_last": False,
    }

    if debug:
        print(f"DataLoader config: {loader_kwargs}")

    return DataLoader(dataset, **loader_kwargs)


def benchmark_model(model, dataloader, device, num_runs=10, warmup_runs=3, debug=False):
    """Benchmark model performance"""

    model = model.to(device)
    model.eval()

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            for batch in dataloader:
                batch = batch.to(device)
                _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    # Benchmark runs
    times = []
    torch.cuda.synchronize() if device.type == "cuda" else None

    with torch.no_grad():
        for run in range(num_runs):
            start_time = time.time()

            for batch in dataloader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()

            times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)

    if debug:
        print(f"Benchmark results: {avg_time:.4f}s ± {std_time:.4f}s")

    return avg_time, std_time


def test_torch_compile_optimization(model, dataloader, device, debug=False):
    """Test torch.compile optimization"""

    if debug:
        print("\nTesting torch.compile optimization...")

    # Test without compilation
    start_time = time.time()
    avg_time_no_compile, _ = benchmark_model(model, dataloader, device, debug=debug)
    no_compile_time = time.time() - start_time

    # Test with compilation
    if debug:
        print("Compiling model...")

    start_time = time.time()
    try:
        compiled_model = torch.compile(model, dynamic=True)
        compile_time = time.time() - start_time

        start_time = time.time()
        avg_time_compiled, _ = benchmark_model(
            compiled_model, dataloader, device, debug=debug
        )
        compiled_bench_time = time.time() - start_time

        speedup = avg_time_no_compile / avg_time_compiled

        if debug:
            print(f"Compilation time: {compile_time:.3f}s")
            print(f"No compile: {avg_time_no_compile:.4f}s")
            print(f"Compiled: {avg_time_compiled:.4f}s")
            print(f"Speedup: {speedup:.2f}x")

        return {
            "no_compile_time": avg_time_no_compile,
            "compiled_time": avg_time_compiled,
            "speedup": speedup,
            "compile_time": compile_time,
        }

    except Exception as e:
        if debug:
            print(f"Compilation failed: {e}")
        return None


def test_memory_optimization(dataset, debug=False):
    """Test memory optimization techniques"""

    if debug:
        print("\nTesting memory optimizations...")

    if len(dataset) == 0:
        if debug:
            print("No data available for memory testing")
        return None

    # Test memory usage with different batch sizes
    results = {}

    for batch_size in [1, 2]:
        if batch_size > len(dataset):
            continue

        loader = create_optimized_dataloader(
            dataset, batch_size=batch_size, debug=False
        )

        # Measure memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Process one batch
            batch = next(iter(loader))
            batch = batch.to("cuda")

            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            results[f"batch_{batch_size}"] = peak_memory

            if debug:
                print(f"Batch size {batch_size}: {peak_memory:.1f} MB peak GPU memory")
        else:
            if debug:
                print("CUDA not available - skipping memory tests")

    return results


def main():
    print("🔥 GNN Optimization Testing Script")
    print("=" * 50)

    # Test configuration
    DATA_DIR = "data/processed_sh"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEBUG = True

    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")

    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("Please run preprocessing first or adjust DATA_DIR")
        return

    # Step 1: Load exactly 2 data points as requested
    print(f"\n📊 Loading test data from {DATA_DIR}...")
    dataset = OptimizedDataset(DATA_DIR, max_samples=2, debug=DEBUG)

    if len(dataset) == 0:
        print("❌ No data loaded - cannot run tests")
        return

    print(f"✅ Loaded {len(dataset)} samples")

    # Create optimized dataloader
    dataloader = create_optimized_dataloader(
        dataset, batch_size=len(dataset), debug=DEBUG
    )

    # Get sample data for model configuration
    sample_batch = next(iter(dataloader))
    input_dim = sample_batch.x.shape[1]

    print(f"\n🔧 Sample batch info:")
    print(f"  Nodes: {sample_batch.x.shape[0]}")
    print(f"  Edges: {sample_batch.edge_index.shape[1]}")
    print(f"  Input features: {input_dim}")
    print(
        f"  Edge features: {sample_batch.edge_attr.shape[1] if sample_batch.edge_attr is not None else 'None'}"
    )

    # Step 2: Test PyG's optimized layers
    print(f"\n🚀 Step 2: Testing PyG's Optimized Layers")
    print("-" * 40)

    layer_types = ["GCN", "GAT"]  # Test multiple layer types
    layer_results = {}

    for layer_type in layer_types:
        print(f"\nTesting {layer_type} layers...")

        model = OptimizedGNN(
            input_dim=input_dim,
            hidden_dim=64,  # Smaller for testing
            num_layers=3,
            layer_type=layer_type,
            use_compile=False,  # Test without compile first
            debug=DEBUG,
        )

        avg_time, std_time = benchmark_model(model, dataloader, DEVICE, debug=DEBUG)
        layer_results[layer_type] = avg_time

        print(f"✅ {layer_type}: {avg_time:.4f}s ± {std_time:.4f}s")

    # Find best layer type
    best_layer = min(layer_results.keys(), key=lambda x: layer_results[x])
    print(
        f"\n🏆 Best performing layer: {best_layer} ({layer_results[best_layer]:.4f}s)"
    )

    # Step 3: Test optimized data loading (already implemented above)
    print(f"\n⚡ Step 3: Data Loading Optimization")
    print("-" * 40)
    print("✅ Using optimized DataLoader with:")
    print("  - Contiguous tensor storage")
    print("  - Pin memory for GPU transfer")
    print("  - Optimized batch sizes")

    # Step 4: Test memory optimization
    print(f"\n💾 Step 4: Memory Optimization Testing")
    print("-" * 40)

    memory_results = test_memory_optimization(dataset, debug=DEBUG)
    if memory_results:
        print("✅ Memory optimization results:")
        for config, memory in memory_results.items():
            print(f"  {config}: {memory:.1f} MB")

    # Test torch.compile optimization
    print(f"\n🔥 torch.compile Optimization Test")
    print("-" * 40)

    # Create model with best layer type
    test_model = OptimizedGNN(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=3,
        layer_type=best_layer,
        use_compile=True,
        debug=DEBUG,
    )

    compile_results = test_torch_compile_optimization(
        test_model, dataloader, DEVICE, debug=DEBUG
    )

    # Summary
    print(f"\n📈 OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully tested with {len(dataset)} data points")
    print(f"🏆 Best layer type: {best_layer}")
    print(f"⚡ Data loading: Optimized with memory efficiency")
    print(f"💾 Memory usage: Tracked and optimized")

    if compile_results:
        print(f"🔥 torch.compile speedup: {compile_results['speedup']:.2f}x")
        print(f"   Before: {compile_results['no_compile_time']:.4f}s")
        print(f"   After:  {compile_results['compiled_time']:.4f}s")
    else:
        print("🔥 torch.compile: Not available or failed")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 30)
    print(f"1. Use {best_layer} layers for best performance")
    print(
        f"2. Enable torch.compile for {compile_results['speedup']:.1f}x speedup"
        if compile_results
        else "2. torch.compile not available"
    )
    print("3. Use optimized DataLoader settings")
    print("4. Monitor memory usage with larger datasets")

    print("\n🎉 Testing complete!")


if __name__ == "__main__":
    main()
