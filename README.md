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

Train the GNN model:

```bash
python trainer.py fit --config config.yaml
```

Find all the options in config.yaml

## Geometric Baselines

Two geometric baselines are available for comparison:

- Centroid Baseline: Predicts the center of mass as the mean of all point positions in the point cloud.
- Convex Hull Centroid Baseline: Predicts the center of mass as the centroid of the convex hull fitted around the point cloud.

### Training a Baseline

Train the centroid baseline:

```bash
invoke train --model-module=src.model.centroid_baseline.CentroidBaseline --name=centroid_baseline
```

Train the convex hull centroid baseline:

```bash
invoke train --model-module=src.model.centroid_baseline.ConvexHullCentroidBaseline --name=convex_hull_baseline
```

### Evaluating a Baseline

Evaluate a trained centroid baseline:

```bash
invoke evaluate --name=centroid_baseline --model-module=src.model.centroid_baseline.CentroidBaseline
```

Evaluate a trained convex hull centroid baseline:

```bash
invoke evaluate --name=convex_hull_baseline --model-module=src.model.centroid_baseline.ConvexHullCentroidBaseline
```

These baselines do not require hyperparameters. The convex hull baseline requires scipy (install with `pip install scipy`).

## Weights & Biases Integration

This project uses Weights & Biases for experiment tracking:

1. Install W&B: `pip install wandb`
2. Login (one-time): `wandb login`

After logging in once, W&B will remember your credentials.
