# Recovering Mesh Centroids from LiDAR Point Clouds using SE(3)-Equivariant Graph Networks

_By Benjamin Banks(s234802), Jonathan Tybirk(s216136) og Lucas Rieneck Gottfried Pedersen(s234842)_

This was made for `02466 Project work - Bachelor of Artificial Intelligence and Data, Spring 2025`

## Abstract

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Training](#training)
  - [Analysis and Visualization](#analysis-and-visualization)
- [Model Configuration](#model-configuration)
- [Weights & Biases Integration](#weights--biases-integration)

## Setup Instructions

### Prerequisites

- Python 3.11.0
- pip
- Git
- uv (install with `pip install uv`)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/jonathantybirk/equivariant-center-of-mass-estimation
   cd equivariant-center-of-mass-estimation
   ```

2. Create and activate a virtual environment with uv:

   ```bash
   # Create virtual environment
   uv venv

   # On Windows
   .\venv\Scripts\activate

   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies with uv:

   ```bash
   uv pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing

Generate point clouds from meshes and convert them to graph data:

```bash
python src/scripts/preprocess.py
```

Configuration options available in `/configs/preprocessing/`

### Training

#### Training Individual Models

```bash
# Baseline model
python src/scripts/trainer.py fit --config configs/models/config_base.yaml --config configs/models/baseline/config.yaml --config configs/models/baseline/wandb.yaml

# Basic GNN
python src/scripts/trainer.py fit --config configs/models/config_base.yaml --config configs/models/basic_gnn/config.yaml --config configs/models/basic_gnn/wandb.yaml

# Basic GNN with Augmentation
python src/scripts/trainer.py fit --config configs/models/config_base.yaml --config configs/models/basic_gnn_aug/config.yaml --config configs/models/basic_gnn_aug/wandb.yaml

# Equivariant GNN
python src/scripts/trainer.py fit --config configs/models/config_base.yaml --config configs/models/eq_gnn/config.yaml --config configs/models/eq_gnn/wandb.yaml
```

#### Training All Models Sequentially

```bash
# Using bash script
bash src/scripts/train_all_models.sh
```

### Analysis and Visualization

After training, analyze and visualize the results:

```bash
# Training visualization (automatically saves to results/)
python src/scripts/visualize_training.py

# Performance comparison (automatically saves to results/)
python src/scripts/performance_comparison.py --load_data

# Pointcloud visualization with saving
python src/scripts/visualize_pointclouds.py --save --single
python src/scripts/visualize_pointclouds.py --save --multi
```

## Model Configuration

### Data Augmentation

Data augmentation parameters in model configs:

- `use_augmentation`: Enable/disable data augmentation (default: false)
- `rotation_prob`: Probability of applying random rotation (default: 0.5)

**Note**: The augmentation applies only rotation transformations, which is recommended for graph-based models since the important information is in the **edge relationships** (3D vectors between connected points), not absolute positions. Rotation preserves all geometric relationships while making the model rotation-invariant.

**Note**: Data augmentation is applied only to training data, not validation data. This is particularly useful for comparing base GNN models with equivariant models that have inherent rotation invariance.

### Checkpoint Naming

Each model uses descriptive checkpoint naming:

- `baseline-{epoch:02d}-{val_displacement_distance_epoch:.4f}.ckpt`
- `basic-gnn-{epoch:02d}-{val_displacement_distance_epoch:.4f}.ckpt`
- `basic-gnn-aug-{epoch:02d}-{val_displacement_distance_epoch:.4f}.ckpt`
- `eq-gnn-{epoch:02d}-{val_displacement_distance_epoch:.4f}.ckpt`

## Weights & Biases Integration

This project uses Weights & Biases for experiment tracking:

1. Install W&B: `pip install wandb`
2. Login (one-time): `wandb login`

After logging in once, W&B will remember your credentials.

### WandB Organization

Each model has specific WandB configurations:

- **Project**: `gnn-center-of-mass`
- **Tags**: Model-specific tags for easy filtering
- **Names**: Descriptive experiment names for identification

Configuration files available in `/configs/models/*/wandb.yaml`
