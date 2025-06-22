# Equivariant Center of Mass Estimation

This project implements a Graph Neural Network (GNN) approach for equivariant center of mass estimation from point cloud data.

## Setup Instructions

### Prerequisites

- Python 3.8+
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
python scripts/preprocess.py
```

Find all the options in /configs/preprocess

### Training

Find all the options in /configs/models/config_base.yaml

Data augmentation parameters:

- `use_augmentation`: Enable/disable data augmentation (default: false)
- `rotation_prob`: Probability of applying random rotation (default: 0.5)

**Note**: The augmentation applies only rotation transformations, which is recommended for graph-based models since the important information is in the **edge relationships** (3D vectors between connected points), not absolute positions. Rotation preserves all geometric relationships while making the model rotation-invariant.

**Note**: Data augmentation is applied only to training data, not validation data. This is particularly useful for comparing base GNN models with equivariant models that have inherent rotation invariance.

## Weights & Biases Integration

This project uses Weights & Biases for experiment tracking:

1. Install W&B: `pip install wandb`
2. Login (one-time): `wandb login`

After logging in once, W&B will remember your credentials.

### Training Individual Models

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

### Training All Models Sequentially

```bash
# Using bash script
bash src/scripts/train_all_models.sh
```

## Checkpoint Naming

Each model uses descriptive checkpoint naming:

- `baseline-{epoch:02d}-{val_displacement_distance_epoch:.4f}.ckpt`
- `basic-gnn-{epoch:02d}-{val_displacement_distance_epoch:.4f}.ckpt`
- `basic-gnn-aug-{epoch:02d}-{val_displacement_distance_epoch:.4f}.ckpt`
- `eq-gnn-{epoch:02d}-{val_displacement_distance_epoch:.4f}.ckpt`

## WandB Organization

Each model has specific WandB configurations:

- **Project**: `gnn-center-of-mass`
- **Tags**: Model-specific tags for easy filtering
- **Names**: Descriptive experiment names for identification
