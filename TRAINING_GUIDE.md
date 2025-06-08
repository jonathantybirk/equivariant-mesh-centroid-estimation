# GNN Training Guide - Optimized Logging & W&B Integration

## Overview

This guide covers the optimized training setup with clean logging and Weights & Biases integration for center of mass estimation using Graph Neural Networks.

## Training Commands

### Basic Training (TensorBoard)

```bash
python trainer.py fit --model.class_path=src.model.eq_gnn.EquivariantGNN
```

### Training with Weights & Biases

```bash
python trainer.py fit --config config_wandb.yaml
```

## Available Models

1. **EquivariantGNN**: The optimized equivariant model (recommended)
2. **BasicGNN**: Standard GNN without equivariance
3. **ZeroBaseline**: Always predicts zero (for comparison)
