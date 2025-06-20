# Model Configuration Structure

This directory contains organized configuration files for all GNN models in the project.

## Directory Structure

```
configs/models/
├── config_base.yaml          # Base configuration shared by all models
├── multirun.yaml             # Multi-run configuration for training all models
├── README.md                 # This file
├── baseline/                 # Baseline model (predicts zeros)
│   ├── config.yaml          # Model-specific configuration
│   └── wandb.yaml           # WandB logging configuration
├── basic_gnn/               # Basic GNN (~7K parameters)
│   ├── config.yaml          # Model-specific configuration
│   └── wandb.yaml           # WandB logging configuration
├── eq_gnn/                  # Equivariant GNN (~7K parameters)
│   ├── config.yaml          # Model-specific configuration
│   └── wandb.yaml           # WandB logging configuration
└── large_gnn/               # Large GNN (~295K parameters)
    ├── config.yaml          # Model-specific configuration
    └── wandb.yaml           # WandB logging configuration
```

## Usage

### Training Individual Models

```bash
# Baseline model
python trainer.py fit --config configs/models/config_base.yaml --config configs/models/baseline/config.yaml --config configs/models/baseline/wandb.yaml

# Basic GNN
python trainer.py fit --config configs/models/config_base.yaml --config configs/models/basic_gnn/config.yaml --config configs/models/basic_gnn/wandb.yaml

# Equivariant GNN
python trainer.py fit --config configs/models/config_base.yaml --config configs/models/eq_gnn/config.yaml --config configs/models/eq_gnn/wandb.yaml

# Large GNN
python trainer.py fit --config configs/models/config_base.yaml --config configs/models/large_gnn/config.yaml --config configs/models/large_gnn/wandb.yaml
```

### Training All Models Sequentially

```bash
# Using bash script
bash train_all_models.sh

# Using Python script
python train_sequential.py

# Using Hydra multirun
python trainer.py fit --config configs/models/multirun.yaml -m
```

### Custom Naming

You can override experiment names from the command line:

```bash
python trainer.py fit --config configs/models/config_base.yaml --config configs/models/eq_gnn/config.yaml --config configs/models/eq_gnn/wandb.yaml trainer.logger.init_args.name="my-custom-experiment-name"
```

## Model Details

| Model           | Parameters | Architecture             | Purpose                         |
| --------------- | ---------- | ------------------------ | ------------------------------- |
| Baseline        | 0          | Zero prediction          | Baseline comparison             |
| Basic GNN       | ~7K        | Simple message passing   | Fair comparison with EQ-GNN     |
| Equivariant GNN | ~7K        | E(3)-equivariant         | Exploiting geometric symmetries |
| Large GNN       | ~295K      | Advanced message passing | High-capacity model             |

## Checkpoint Naming

Each model uses descriptive checkpoint naming:

- `baseline-{epoch:02d}-{val_displacement_distance_epoch:.4f}.ckpt`
- `basic-gnn-{epoch:02d}-{val_displacement_distance_epoch:.4f}.ckpt`
- `eq-gnn-{epoch:02d}-{val_displacement_distance_epoch:.4f}.ckpt`
- `large-gnn-{epoch:02d}-{val_displacement_distance_epoch:.4f}.ckpt`

## WandB Organization

Each model has specific WandB configurations:

- **Project**: `gnn-center-of-mass`
- **Tags**: Model-specific tags for easy filtering
- **Names**: Descriptive experiment names for identification
