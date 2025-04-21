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

4. Create necessary project directories (not currently implemented): 
   ```bash
   invoke setup
   ```

## Usage

### Data Preprocessing

Generate point clouds from meshes and convert them to graph data:

```bash
invoke preprocess --save --no-visualize
```

Options:

- `--save`: Save generated point clouds to disk
- `--no-visualize`: Disable visualization of point clouds
- `--num-cameras=N`: Number of LiDAR cameras (1-6)
- `--h-steps=N`: Horizontal resolution of LiDAR scan
- `--v-steps=N`: Vertical resolution of LiDAR scan

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
- `--patience=N`: Early stopping patience (default: 10)
- `--workers=N`: Number of data loading workers (default: 4)

### Evaluation

Evaluate a trained model:

```bash
invoke evaluate --name=gnn_baseline
```

### Cleaning

Clean generated files:

```bash
invoke clean
```

Options:

- `--directory=DIRNAME`: Specific directory to clean (logs, checkpoints, visualizations, or evaluation)

## Weights & Biases Integration

This project uses Weights & Biases for experiment tracking:

1. Install W&B: `pip install wandb`
2. Login (one-time): `wandb login`
3. Run training: `invoke train`

After logging in once, W&B will remember your credentials.
