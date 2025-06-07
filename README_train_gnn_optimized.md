# 🚀 Optimized GNN Training Infrastructure

Documentation for `train_gnn_optimized.py` - A high-performance training script for Equivariant Graph Neural Networks with comprehensive timing, optimization, and monitoring.

## 📊 Performance Characteristics

### Startup Phases & Timing

The training script provides detailed timing for 8 distinct startup phases:

1. **Directory Setup** (~0.01s) - Creates checkpoint directories
2. **Dataset Loading** (~0.3s) - Loads and validates all data files
3. **Dataset Splitting** (~0.01s) - Creates train/validation splits
4. **DataLoader Creation** (~0.01s) - Configures optimized data loaders
5. **Model Initialization** (~0.15s) - Creates and moves model to GPU
6. **Model Compilation** (optional) - PyTorch compilation if enabled
7. **Logging Setup** (~0.01s for local, ~2s for wandb)
8. **Training Configuration** (~0.03s) - Sets up trainer and callbacks

**Total startup time**: ~0.5s (without wandb) to ~3s (with wandb)

### CUDA Kernel Warmup

When using CUDA, an optional warmup phase pre-compiles GPU kernels:

- **Duration**: ~30 seconds (one-time cost)
- **Components**: Forward pass, backward pass, optimizer kernels
- **Benefit**: Ensures consistent epoch timing from the start
- **Skip condition**: Automatically disabled for `--fast_dev_run`

## 🎯 Precision Settings & Performance Impact

### Available Precision Options

```bash
--precision 32        # FP32 (default) - Consistent performance
--precision 16        # FP16 - Memory efficient, potential speed gains
--precision bf16      # BFloat16 - Better numerical stability than FP16
--precision 16-mixed  # Automatic Mixed Precision - Best memory/speed balance
--precision bf16-mixed # BFloat16 Mixed Precision
```

### Performance Characteristics by Precision

| Precision    | First Epoch            | Memory Usage | Numerical Stability | Recommended Use        |
| ------------ | ---------------------- | ------------ | ------------------- | ---------------------- |
| `32`         | Fast (~0.5s)           | High         | Excellent           | Development, debugging |
| `16-mixed`   | Slow (~30s first time) | Low          | Good                | Production training    |
| `bf16-mixed` | Slow (~30s first time) | Low          | Better              | Modern hardware        |

### First Epoch Timing Explanation

- **FP32**: No compilation overhead, consistent from start
- **Mixed Precision**: PyTorch AMP compiles optimizations during first epoch
- **Subsequent epochs**: All precisions perform similarly after compilation

## 💾 Checkpoint Configuration

### Default Settings

```python
Checkpoint Callback:
├── Directory: checkpoints/ (configurable with --checkpoint_dir)
├── Filename: best-model-{epoch:02d}-{val_loss:.3f}.ckpt
├── Monitor: val_loss (lower is better)
├── Save Strategy: Top 1 best + last checkpoint
└── Timing: Save after validation (not during training)
```

### Checkpoint Optimization

- **save_weights_only=False**: Includes optimizer states for resuming
- **save_on_train_epoch_end=False**: Only saves after validation
- **auto_insert_metric_name=False**: Faster filename generation

## 📈 Weights & Biases Integration

### Automatic Logging

When `--use_wandb` is enabled, the following metrics are automatically logged:

**Training Metrics:**

- `train_loss`, `train_mae`, `train_mse` (per step and epoch)
- `train_mae_x`, `train_mae_y`, `train_mae_z` (per component)
- `train_mse_x`, `train_mse_y`, `train_mse_z` (per component)

**Validation Metrics:**

- `val_loss`, `val_mae`, `val_mse`
- `val_mae_x`, `val_mae_y`, `val_mae_z`
- `val_mse_x`, `val_mse_y`, `val_mse_z`

**System Metrics:**

- `learning_rate` (tracks scheduler changes)
- `epoch` (current epoch number)

**Dataset Information:**

- Complete dataset statistics (nodes, edges, features)
- Train/validation split information
- Data directory and utilization metrics

### Configuration

```bash
--use_wandb                    # Enable wandb logging
--wandb_project "my-project"   # Project name (default: equivariant-center-of-mass)
--wandb_name "experiment-1"    # Run name (auto-generated if not provided)
--wandb_tags tag1 tag2         # Add tags to organize experiments
```

## ⚡ DataLoader Optimization

### Optimized Configuration

```python
DataLoader Settings:
├── batch_size: 16 (configurable)
├── num_workers: 4 (configurable, set to 0 for Windows compatibility)
├── pin_memory: True (when CUDA available)
├── persistent_workers: True (when num_workers > 0)
├── prefetch_factor: 4 (when num_workers > 0)
└── drop_last: True (consistent batch sizes for training)
```

### Windows Compatibility

The script automatically handles Windows-specific issues:

- Sets `num_workers=0` if needed for stability
- Proper path handling for checkpoint directories
- Compatible progress bar rendering

## 🔧 Command Line Interface

### Essential Arguments

```bash
# Data and Model
--data_dir "data/processed_sh"     # Path to processed dataset
--batch_size 16                    # Training batch size
--hidden_dim 128                   # Model hidden dimension
--num_layers 4                     # Number of GNN layers

# Training Configuration
--max_epochs 100                   # Maximum training epochs
--lr 1e-3                         # Learning rate
--weight_decay 1e-5               # L2 regularization
--precision 32                    # Training precision

# Infrastructure
--checkpoint_dir "checkpoints"     # Checkpoint save directory
--num_workers 4                   # DataLoader workers
--debug                           # Enable detailed logging

# Development
--fast_dev_run 5                  # Quick test with N batches
--use_compile                     # Enable PyTorch compilation
```

### Example Usage

```bash
# Development (fast startup, detailed logs)
python train_gnn_optimized.py --fast_dev_run 10 --debug

# Production (mixed precision, wandb logging)
python train_gnn_optimized.py --precision 16-mixed --use_wandb --max_epochs 200

# Custom configuration
python train_gnn_optimized.py \
    --data_dir "data/my_dataset" \
    --checkpoint_dir "experiments/run_1" \
    --batch_size 32 \
    --hidden_dim 256 \
    --wandb_name "large_model_experiment"
```

## 🔍 Debug Mode Features

When `--debug` is enabled:

- **Detailed timing**: Shows time for each startup phase
- **Dataset validation**: Reports successful/failed file loads
- **Model architecture**: Displays parameter count and layer structure
- **DataLoader config**: Shows optimization settings
- **First epoch tracking**: Special timing for initial epoch
- **Batch-level timing**: Detailed timing for first few batches

## 🎛️ Model Architecture

### GCN-Based Design

```python
Model Architecture:
├── Input Layer: GCNConv(input_dim → hidden_dim)
├── Hidden Layers: N × GCNConv(hidden_dim → hidden_dim)
├── Output Layer: GCNConv(hidden_dim → hidden_dim)
└── Prediction Head:
    ├── Linear(hidden_dim → hidden_dim//2)
    ├── ReLU + Dropout
    ├── Linear(hidden_dim//2 → hidden_dim//4)
    ├── ReLU + Dropout
    └── Linear(hidden_dim//4 → 3)  # 3D center of mass
```

### Performance Optimization

- **Native PyG layers**: Uses PyTorch Geometric's optimized GCN implementation
- **Global pooling**: Efficient graph-to-vector aggregation
- **Gradient clipping**: Prevents training instability
- **Learning rate scheduling**: ReduceLROnPlateau for adaptive learning

## 🚨 Troubleshooting

### Common Issues & Solutions

**Slow First Epoch (30+ seconds)**

- **Cause**: Mixed precision compilation or CUDA kernel warmup
- **Solution**: Use `--precision 32` for development, or accept one-time cost

**Memory Errors**

- **Solution**: Reduce `--batch_size` or use mixed precision
- **Windows**: Set `--num_workers 0`

**Checkpoint Directory Issues**

- **Cause**: Permissions or existing files
- **Solution**: Script auto-creates directories, check write permissions

**DataLoader Hangs**

- **Windows**: Use `--num_workers 0`
- **General**: Reduce `--num_workers` or disable `persistent_workers`

**Wandb Connection Issues**

- **Solution**: Run `wandb login` or use `--use_wandb` only when needed
- **Offline**: Script gracefully falls back to local logging

### Performance Expectations

**Typical Training Speed (per epoch):**

- Small graphs (< 100 nodes): ~0.3s
- Medium graphs (100-500 nodes): ~0.5s
- Large graphs (500+ nodes): ~1.0s

**Memory Usage:**

- FP32: ~2-4GB VRAM for typical datasets
- Mixed Precision: ~1-2GB VRAM for same datasets

## 📝 Loss Function & Metrics

### Primary Loss

- **MAE (Mean Absolute Error)**: Primary optimization target
- **Per-component tracking**: Separate MAE for X, Y, Z coordinates

### Secondary Metrics

- **MSE (Mean Squared Error)**: For comparison and analysis
- **Per-component MSE**: Detailed coordinate-wise performance

### Learning Rate Scheduling

- **ReduceLROnPlateau**: Automatically reduces LR when validation loss plateaus
- **Monitoring**: `val_loss` (MAE-based)
- **Patience**: 10 epochs before reduction
- **Factor**: 0.5 (halves learning rate)
- **Minimum LR**: 1e-6

## 🔄 Resuming Training

To resume from a checkpoint:

```python
# Manual resuming (modify script or use PyTorch Lightning's resume functionality)
trainer.fit(model, train_loader, val_loader, ckpt_path="checkpoints/best-model-XX-Y.YYY.ckpt")
```

The script saves both model weights and optimizer states, enabling exact training resumption.

---

## 📊 Quick Reference

| Feature       | Default     | Purpose              | Performance Impact                        |
| ------------- | ----------- | -------------------- | ----------------------------------------- |
| Precision     | `32`        | Numerical precision  | FP32: Fast start, Mixed: Slow first epoch |
| Batch Size    | `16`        | Training batch size  | Higher: More memory, potentially faster   |
| Hidden Dim    | `128`       | Model capacity       | Higher: More parameters, better capacity  |
| Num Workers   | `4`         | Data loading threads | Higher: Faster loading (if not Windows)   |
| CUDA Warmup   | Auto        | Kernel compilation   | 30s cost, consistent epochs after         |
| Checkpointing | Every epoch | Model saving         | Minimal impact with optimizations         |
| Debug Mode    | Off         | Detailed logging     | No performance impact                     |

---

_For questions or issues, check the troubleshooting section or examine the detailed startup logs with `--debug` enabled._
