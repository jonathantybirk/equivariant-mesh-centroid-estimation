from invoke import task
import os
import platform
import glob
import re


@task(
    help={
        "save": "Whether to save point clouds to disk",
        "no_visualize": "Disable point cloud visualization",
        "num_cameras": "Number of LiDAR cameras per point cloud (1–6)",
        "h_steps": "Horizontal resolution of LiDAR scan",
        "v_steps": "Vertical resolution of LiDAR scan",
        "num_samples": "Number of different point clouds to generate per mesh (with different camera initializations)",
        "debug": "Enable detailed debug logging",
    }
)
def preprocess(
    ctx,
    save=False,
    no_visualize=True,
    num_cameras=3,
    h_steps=40,
    v_steps=40,
    num_samples=1,
    debug=False,
):
    """
    Run full preprocessing: generate point clouds from meshes and then convert them to graph data.
    """
    cmd = (
        f"python scripts/preprocess.py "
        f"preprocessing.lidar.save={str(save).lower()} "
        f"preprocessing.lidar.visualize={str(not no_visualize).lower()} "
        f"preprocessing.lidar.num_cameras={num_cameras} "
        f"preprocessing.lidar.h_steps={h_steps} "
        f"preprocessing.lidar.v_steps={v_steps} "
        f"preprocessing.lidar.num_samples={num_samples} "
        f"preprocessing.debug={str(debug).lower()}"
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
        "model_module": "Python path to the LightningModule class (e.g., src.model.SE3_equivariant.GNNLightningModule)",
        "sample_balanced": "Balance training by sampling one instance of each mesh instead of using all samples",
    }
)
def train(
    ctx,
    batch_size=32,
    lr=0.001,
    epochs=100,
    gpus=1,
    name="gnn_baseline",
    patience=3,
    workers=4,
    test=False,
    fast=False,
    model_module=None,
    sample_balanced=False,
):
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
        f"+fast_train={str(fast).lower()} "
        f"+sample_balanced={str(sample_balanced).lower()}"
    )
    # Add model override if provided (no special quoting/escaping needed now)
    if model_module:
        cmd += f" model.module_path={model_module}"

    ctx.run(cmd)


@task(
    help={
        "name": "Experiment name (should match a trained model)",
        "checkpoint_path": "Specific checkpoint to evaluate (optional). If not provided, tries to find the best checkpoint based on val_loss.",
        "model_module": "Python path to the LightningModule class (e.g., src.model.SE3_equivariant.GNNLightningModule)",
    }
)
def evaluate(ctx, name="gnn_baseline", checkpoint_path=None, model_module=None):
    """
    Evaluate a trained GNN model.
    If checkpoint_path is not provided, it attempts to find the best checkpoint
    for the given 'name' in the 'checkpoints/' directory based on 'val_loss' in the filename.
    Falls back to '{name}_final.ckpt' if no such best checkpoint is found.
    """
    os.makedirs("evaluation_results", exist_ok=True)

    selected_checkpoint_path = checkpoint_path

    if not selected_checkpoint_path and name:
        checkpoint_dir = "checkpoints"
        best_checkpoint = None
        min_val_loss = float("inf")

        # Try to find checkpoints like name-epoch=XX-val_loss=YY.ckpt
        # Example: gnn_baseline-epoch=02-val_loss=0.0104.ckpt
        # Note: Using os.path.join for robust path construction
        pattern_str = os.path.join(checkpoint_dir, f"{name}-epoch=*-val_loss=*.ckpt")
        potential_checkpoints = glob.glob(pattern_str)

        if potential_checkpoints:
            for cp_path in potential_checkpoints:
                # Extract val_loss using regex
                match = re.search(
                    r"val_loss=([\\d\\.]+)\\.ckpt$", cp_path
                )  # Escaped . and \\d for regex in string
                if match:
                    try:
                        val_loss = float(match.group(1))
                        if val_loss < min_val_loss:
                            min_val_loss = val_loss
                            best_checkpoint = cp_path
                    except ValueError:
                        pass  # Ignore if parsing fails

        if best_checkpoint:
            selected_checkpoint_path = best_checkpoint
            print(
                f"No checkpoint_path provided. Found best checkpoint: {selected_checkpoint_path} for name '{name}'"
            )
        else:
            # Fallback to _final.ckpt
            fallback_checkpoint = os.path.join(checkpoint_dir, f"{name}_final.ckpt")
            if os.path.exists(fallback_checkpoint):
                selected_checkpoint_path = fallback_checkpoint
                print(
                    f"No checkpoint_path provided and no best checkpoint found. Using fallback: {selected_checkpoint_path}"
                )
            else:
                print(
                    f"Warning: No checkpoint_path provided and could not find a best or final checkpoint for name '{name}'. Evaluation might fail or use a default model from the script."
                )

    cmd = f"python scripts/evaluate_gnn.py name={name}"
    if selected_checkpoint_path:
        # Add with + to ensure it's a Hydra override. Quote path for safety.
        cmd += f" +checkpoint_path='{selected_checkpoint_path}'"

    if model_module:
        cmd += f" model.module_path={model_module}"

    ctx.run(cmd)


@task(
    help={
        "checkpoint_path": "Path to the model checkpoint to visualize",
        "model_module": "Python path to the LightningModule class (e.g., src.model.SE3_equivariant.GNNLightningModule)",
        "sample_index": "Index of the initial sample to visualize (0-based)",
        "gpus": "Number of GPUs to use (0 for CPU)",
    }
)
def visualize(ctx, checkpoint_path=None, model_module=None, sample_index=0, gpus=1):
    """
    Launch interactive visualization to explore equivariance properties of the model.
    """
    os.makedirs("outputs", exist_ok=True)

    cmd = f"python scripts/visualize_equivariance.py"

    if checkpoint_path:
        cmd += f" checkpoint_path={checkpoint_path}"

    if model_module:
        cmd += f" model.module_path={model_module}"

    cmd += f" sample_index={sample_index} gpus={gpus}"

    print("Launching visualization tool...")
    ctx.run(cmd)
    print("Visualization closed.")
