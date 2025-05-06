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
invoke preprocess --save --no-visualize
```

Options:

- `--save`: Whether to save point clouds to disk
- `--no-visualize`: Disable point cloud visualization
- `--num-cameras=N`: Number of LiDAR cameras per point cloud (1-6)
- `--h-steps=N`: Horizontal resolution of LiDAR scan
- `--v-steps=N`: Vertical resolution of LiDAR scan
- `--num-samples=N`: Number of different point clouds to generate per mesh (with different camera initializations)

### Training

Train the GNN model:

```bash
invoke train
```

Options:

- `--batch-size=N`: Batch size for training (default: 32)
- `--lr=N`: Learning rate (default: 0.001)
- `--epochs=N`: Maximum number of training epochs (default: 100)
- `--gpus=N`: Number of GPUs to use (0 for CPU, default: 1)
- `--name=NAME`: Experiment name (default: "gnn_baseline")
- `--patience=N`: Early stopping patience (default: 3)
- `--workers=N`: Number of data loading workers (default: 4)
- `--test`: Run test evaluation after training
- `--fast`: Use faster training mode (less validation)
- `--model-module=NAME`: Python path to the LightningModule class (e.g., src.model.SE3_equivariant.GNNLightningModule)
- `--sample-balanced`: Balance training by sampling one instance of each mesh instead of using all samples

### Evaluation

Evaluate a trained model:

```bash
invoke evaluate --name=gnn_baseline
```

Options:
- `--name=NAME`: Experiment name (should match a trained model)
- `--checkpoint-path=PATH`: Specific checkpoint to evaluate (optional)
- `--model-module=NAME`: Python path to the LightningModule class

### Visualization

Launch interactive visualization to explore equivariance properties of the model:

```bash
invoke visualize
```

Options:
- `--checkpoint-path=PATH`: Path to the model checkpoint to visualize
- `--model-module=NAME`: Python path to the LightningModule class
- `--sample-index=N`: Index of the initial sample to visualize (0-based)
- `--gpus=N`: Number of GPUs to use (0 for CPU, default: 1)

## Weights & Biases Integration

This project uses Weights & Biases for experiment tracking:

1. Install W&B: `pip install wandb`
2. Login (one-time): `wandb login`

After logging in once, W&B will remember your credentials.
