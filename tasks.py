from invoke import task
import os
import platform

@task(
    help={
        "save": "Whether to save point clouds to disk",
        "no_visualize": "Disable point cloud visualization",
        "num_cameras": "Number of LiDAR cameras (1–6)",
        "h_steps": "Horizontal resolution of LiDAR scan",
        "v_steps": "Vertical resolution of LiDAR scan",
    }
)
def preprocess(ctx, save=False, no_visualize=False, num_cameras=3, h_steps=40, v_steps=40):
    """
    Run full preprocessing: generate point clouds from meshes and then convert them to graph data.
    """
    cmd = (
        f"python scripts/preprocess.py "
        f"preprocessing.lidar.save={str(save).lower()} "
        f"preprocessing.lidar.visualize={str(not no_visualize).lower()} "
        f"preprocessing.lidar.num_cameras={num_cameras} "
        f"preprocessing.lidar.h_steps={h_steps} "
        f"preprocessing.lidar.v_steps={v_steps}"
    )
    ctx.run(cmd)
    ctx.run("python scripts/split_dataset.py")

# Add fast option to the train task

@task(
    help={
        "batch_size": "Batch size for training",
        "lr": "Learning rate",
        "epochs": "Maximum number of training epochs",
        "gpus": "Number of GPUs to use (0 for CPU)",
        "name": "Experiment name",
        "patience": "Early stopping patience",
        "workers": "Number of data loading workers",
        "test": "Run test evaluation after training",
        "fast": "Use faster training mode (less validation)",
        "model_module": "Python path to the LightningModule class (e.g., src.model.SE3_equivariant.GNNLightningModule)"
    }
)
def train(ctx, batch_size=32, lr=0.001, epochs=100, gpus=1, name="gnn_baseline", 
          patience=10, workers=4, test=False, fast=False, model_module=None):
    """
    Train the GNN model for center of mass estimation.
    """
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Command with proper configuration paths
    cmd = (
        f"python scripts/train_gnn.py "
        f"training.batch_size={batch_size} "
        f"training.lr={lr} "
        f"training.max_epochs={epochs} "
        f"training.gpus={gpus} "
        f"training.patience={patience} "
        f"training.num_workers={workers} "
        f"name={name} "
        f"+do_test={str(test).lower()} "
        f"+fast_train={str(fast).lower()}"
    )
    # Add model override if provided (no special quoting/escaping needed now)
    if model_module:
        cmd += f" model.module_path={model_module}"
        
    ctx.run(cmd)

@task(
    help={
        "name": "Experiment name (should match a trained model)",
        "checkpoint_path": "Specific checkpoint to evaluate (optional)",
        "model_module": "Python path to the LightningModule class (e.g., src.model.SE3_equivariant.GNNLightningModule)"
    }
)
def evaluate(ctx, name="gnn_baseline", checkpoint_path=None, model_module=None):
    """
    Evaluate a trained GNN model.
    """
    os.makedirs("evaluation_results", exist_ok=True)
    
    cmd = f"python scripts/evaluate_gnn.py name={name}"
    if checkpoint_path:
        cmd += f" +checkpoint_path={checkpoint_path}"
    # Add model override if provided (no special quoting/escaping needed now)
    if model_module:
        cmd += f" model.module_path={model_module}"
        
    ctx.run(cmd)
