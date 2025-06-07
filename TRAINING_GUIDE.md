# GNN Training Guide - Optimized Logging & W&B Integration

## Overview

This guide covers the optimized training setup with clean logging and Weights & Biases integration for center of mass estimation using Graph Neural Networks.

## Key Optimizations Made

### 1. **Logging Strategy**

- **Per-epoch logging only**: Since epochs are now fast (~15ms), we log only per epoch to reduce overhead
- **MSE as loss function**: Better for optimization due to smooth gradients
- **MAE as primary metric**: More interpretable since it's in the same units as the target (center of mass coordinates)
- **Simplified metrics**: Focus on `train_mae`, `val_mae` as the key metrics to monitor

### 2. **Loss Function Choice**

Based on research, we use:

- **MSE for training loss**: Better differentiability and faster convergence
- **MAE for monitoring**: Easier to interpret and understand magnitude of errors
- **Scheduler monitors MAE**: More intuitive for learning rate adjustments

## Training Commands

### Basic Training (TensorBoard)

```bash
python train_gnn_optimized.py fit --model.class_path=EquivariantGNN --data.class_path=PointCloudData
```

### Training with Weights & Biases

#### Option 1: Command Line

```bash
python train_gnn_optimized.py fit \
    --model.class_path=EquivariantGNN \
    --data.class_path=PointCloudData \
    --trainer.logger.class_path=lightning.pytorch.loggers.WandbLogger \
    --trainer.logger.init_args.project="gnn-optimization" \
    --trainer.logger.init_args.name="my-experiment"
```

#### Option 2: Config File (Recommended)

```bash
python train_gnn_optimized.py fit --config config_wandb.yaml
```

#### Option 3: Quick Testing

```bash
python train_gnn_optimized.py fit \
    --model.class_path=EquivariantGNN \
    --data.class_path=PointCloudData \
    --trainer.fast_dev_run=true
```

## Available Models

1. **EquivariantGNN**: The optimized equivariant model (recommended)
2. **BasicGNN**: Standard GNN without equivariance
3. **ZeroBaseline**: Always predicts zero (for comparison)

## Metrics Explanation

- `train_loss` / `val_loss`: MSE loss used for optimization
- `train_mae` / `val_mae`: **Primary metric** - Mean Absolute Error in same units as center of mass
- `train_mae_x`, `train_mae_y`, `train_mae_z`: Per-component MAE for detailed analysis
- `val_mae_x`, `val_mae_y`, `val_mae_z`: Validation per-component MAE for each coordinate
- Learning rate scheduler monitors `val_mae` for interpretability

## W&B Features

When using Weights & Biases, you automatically get:

- **Hyperparameter tracking**: All model parameters saved
- **Real-time metrics**: Live training/validation curves
- **Model checkpoints**: Best models saved as W&B artifacts
- **System monitoring**: GPU/CPU usage, memory, etc.
- **Code versioning**: Git commits and diffs tracked
- **Experiment comparison**: Easy to compare different runs

## Example W&B Configuration

See `config_wandb.yaml` for a complete example with:

- Model checkpointing (save top 3 models by `val_mae`)
- Early stopping (patience=20 epochs on `val_mae`)
- Learning rate monitoring
- W&B project organization with tags

## Performance Expectations

With the optimizations:

- **Speed**: ~15ms per batch (ultra-fast training)
- **Accuracy**: Maintain equivariance properties
- **Interpretability**: MAE gives direct error in coordinate units
- **Monitoring**: Clean, focused metrics without clutter

## Tips

1. **Focus on MAE**: This is your main metric - errors are in the same units as your target
2. **Use config files**: More maintainable than long command lines
3. **Monitor val_mae**: This drives learning rate scheduling and early stopping
4. **Tag experiments**: Use W&B tags to organize different experiment types
5. **Save top models**: The config saves top 3 models automatically
6. **Test quickly**: Use `--trainer.fast_dev_run=true` for rapid testing
