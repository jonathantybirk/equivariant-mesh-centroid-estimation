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

@task(
    help={
        "batch_size": "Batch size for training",
        "lr": "Learning rate",
        "epochs": "Maximum number of training epochs",
        "gpus": "Number of GPUs to use (0 for CPU)",
        "name": "Experiment name",
        "patience": "Early stopping patience",
        "workers": "Number of data loading workers",
    }
)
def train(ctx, batch_size=32, lr=0.001, epochs=100, gpus=1, name="gnn_baseline", patience=10, workers=4):
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
        f"name={name}"
    )
    ctx.run(cmd)

@task(
    help={
        "name": "Experiment name (should match a trained model)",
    }
)
def evaluate(ctx, name="gnn_baseline"):
    """
    Evaluate a trained GNN model.
    """
    os.makedirs("evaluation_results", exist_ok=True)
    
    cmd = f"python scripts/evaluate_gnn.py name={name}"
    ctx.run(cmd)

@task(
    help={
        "directory": "Directory to clean (logs, checkpoints, visualizations, or evaluation)",
    }
)
def clean(ctx, directory="all"):
    """
    Clean generated files (logs, checkpoints, visualizations).
    """
    # Use appropriate command based on OS
    rm_cmd = "rm -rf" if platform.system() != "Windows" else "rmdir /s /q"
    
    if directory == "all" or directory == "logs":
        ctx.run(f"{rm_cmd} logs", warn=True)
    if directory == "all" or directory == "checkpoints":
        ctx.run(f"{rm_cmd} checkpoints", warn=True)
    if directory == "all" or directory == "visualizations":
        ctx.run(f"{rm_cmd} visualizations", warn=True)
    if directory == "all" or directory == "evaluation":
        ctx.run(f"{rm_cmd} evaluation_results", warn=True)
    
    print(f"Cleaned {directory} directories")

@task
def setup(ctx):
    """
    Create necessary directories for the project.
    """
    directories = [
        "src/model",
        "src/data",
        "scripts",
        "logs",
        "checkpoints",
        "visualizations",
        "evaluation_results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print("Project directories created successfully")