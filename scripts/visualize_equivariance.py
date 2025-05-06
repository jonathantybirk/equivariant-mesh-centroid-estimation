import os
import sys
import importlib
from pathlib import Path
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib
matplotlib.use('TkAgg') # Try setting the backend explicitly
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import pytorch_lightning as pl # For LightningModule loading if needed
from torch_geometric.data import Data # Import Data class
from torch import nn
import torch.nn.functional as F

# --- Boilerplate for project root and imports ---
def find_project_root(current: Path, markers=(".git", "pyproject.toml")):
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError("Project root not found.")

root = find_project_root(Path(__file__).resolve())
os.chdir(root)
sys.path.insert(0, str(root))

# Dynamically import model class helper
def get_class(class_path: str):
    """Helper function to dynamically import a class."""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

# Import data module AFTER setting sys.path
from src.data.datamodule import PointCloudDataModule

# --- Global-like storage for visualization state (simpler for a script) ---
app_state = {
    "train_dataset": None,
    "test_dataset": None,
    "active_dataset_name": "test", # Start with test set
    "current_sample_index": 0,
    "original_data": None,
    "original_pred_com": None,
    "model": None,
    "device": None,
    "checkpoint_path": None,
    # Plot elements
    "fig": None,
    "ax": None,
    "scatter": None,
    "pred_marker": None,
    "rotated_orig_marker": None,
    # Sliders
    "s_alpha": None,
    "s_beta": None,
    "s_gamma": None,
}

# --- Rotation Helper --- # (Adding back the function definition)
def euler_to_rotation_matrix(alpha, beta, gamma):
    """Convert Euler angles (ZYX convention) to a rotation matrix."""
    # Convert degrees to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    # Rotation matrices around Z, Y, X axes
    Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                   [np.sin(alpha),  np.cos(alpha), 0],
                   [0,              0,             1]])

    Ry = np.array([[np.cos(beta),  0, np.sin(beta)],
                   [0,             1, 0           ],
                   [-np.sin(beta), 0, np.cos(beta)]])

    Rx = np.array([[1, 0,            0           ],
                   [0, np.cos(gamma), -np.sin(gamma)],
                   [0, np.sin(gamma),  np.cos(gamma)]])

    # Combined rotation matrix (ZYX)
    R = Rz @ Ry @ Rx
    return torch.tensor(R, dtype=torch.float32)

# --- Define an ULTRA-SIMPLE model for debugging --- # (Removing/Commenting out)
# class MinimalLinearModel(nn.Module):
#    ...

# --- Define a completely different random architecture --- # (Removing/Commenting out)
# class RandomModel(nn.Module):
#    ...

# --- Main Visualization Function ---
@hydra.main(config_path="../configs", config_name="visualize", version_base=None)
def visualize(cfg: DictConfig):
    print("Starting visualization...")
    print(OmegaConf.to_yaml(cfg))
    
    # --- Load Model From Checkpoint ---
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    ModelClass = get_class(cfg.model.module_path)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.gpus > 0 else "cpu")
    try:
        # Load the trained model weights and hyperparameters
        model = ModelClass.load_from_checkpoint(cfg.checkpoint_path, map_location=device)
        model.eval()
        model.to(device)
        print(f"Model loaded successfully onto {device}.")
        # Print number of parameters for confirmation
        params = list(model.parameters())
        print(f"Loaded model has {sum(p.numel() for p in params)} trainable parameters.")
        
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {cfg.checkpoint_path}")
        print("Please ensure the path is correct and the file exists.")
        # Try renaming the file in the config if it has '=' signs
        if '=' in cfg.checkpoint_path:
             renamed_path = cfg.checkpoint_path.replace('=', '_').replace(':', '_')
             print(f"Suggestion: Does the file exist as '{renamed_path}'?")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Store model, device, and checkpoint path in state
    app_state["model"] = model
    app_state["device"] = device
    app_state["checkpoint_path"] = cfg.checkpoint_path

    # --- Load Data ---
    print("Loading data...")
    try:
        # Load both train and test datasets
        # Determine node_feature_dim for DataModule
        node_feature_dim = 16 # Default GNN hidden dim
        if hasattr(model, 'hparams') and hasattr(model.hparams, 'hidden_dim'):
            node_feature_dim = model.hparams.hidden_dim
            print(f"Using hidden_dim from model.hparams: {node_feature_dim}")
        elif hasattr(model, 'hidden_dim'): # For models that might store it directly
             node_feature_dim = model.hidden_dim
             print(f"Using hidden_dim from model.hidden_dim: {node_feature_dim}")
        else:
            model_class_name = model.__class__.__name__
            if "Baseline" in model_class_name:
                node_feature_dim = 3 # Raw (x,y,z) coordinates for baselines
                print(f"Warning: hidden_dim not found for model '{model_class_name}'. Defaulting node_feature_dim to {node_feature_dim} (for raw coordinates).")
            else:
                # Try to get from visualize.yaml's model config, else use default_gnn_hidden_dim
                default_gnn_hidden_dim = 16 
                # cfg.model is DictConfig for the 'model' section in visualize.yaml
                node_feature_dim = cfg.model.get('hidden_dim', default_gnn_hidden_dim) 
                if node_feature_dim == default_gnn_hidden_dim and not cfg.model.get('hidden_dim'): # only print warning if not explicitly set in yaml
                    print(f"Warning: hidden_dim not found on model or in visualize.yaml model config. Defaulting node_feature_dim to {node_feature_dim}.")
                elif cfg.model.get('hidden_dim'):
                     print(f"Using hidden_dim from visualize.yaml model config: {node_feature_dim}")
        
        data_module = PointCloudDataModule(
            processed_dir=cfg.data.processed_dir,
            train_dir=os.path.join(cfg.data.processed_dir, "train"),
            test_dir=os.path.join(cfg.data.processed_dir, "test"),
            batch_size=1, # Load one at a time
            num_workers=0,
            pin_memory=False,
            node_feature_dim=node_feature_dim
        )
        data_module.setup(stage='fit') # Load train/val
        data_module.setup(stage='test') # Load test
        
        app_state["train_dataset"] = data_module.train_dataset # Changed from train_val_dataset to train_dataset
        app_state["test_dataset"] = data_module.test_dataset
        
        if not app_state["test_dataset"]:
            print("Error: Test dataset is empty.")
            return
        if not app_state["train_dataset"]:
            print("Error: Train dataset is empty.")
            return
        
        print(f"Loaded {len(app_state['train_dataset'])} train samples and {len(app_state['test_dataset'])} test samples.")
        app_state["active_dataset_name"] = "test"
        app_state["current_sample_index"] = cfg.sample_index if cfg.sample_index < len(app_state["test_dataset"]) else 0
        print("DEBUG: Data loading successful.")

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    print("DEBUG: Proceeding to plot setup.")

    # --- Plot Setup ---
    app_state["fig"] = plt.figure(figsize=(10, 9))
    app_state["ax"] = app_state["fig"].add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.35)
    print("DEBUG: Plot setup complete (figure and axes created).")

    # --- Initial Plot Call --- 
    print("DEBUG: Calling load_and_plot_new_sample() for the first time...")
    load_and_plot_new_sample()
    print("DEBUG: Returned from initial load_and_plot_new_sample().")

    # --- Create Plot Elements using initial data ---
    ax = app_state["ax"]
    original_data = app_state["original_data"]
    original_pred_com = app_state["original_pred_com"]
    
    if original_data is None or original_pred_com is None: # Check if initial load failed
        print("ERROR: Initial data/prediction failed to load. Cannot create plot elements.")
        return
        
    points_np = original_data.pos.cpu().numpy()
    pred_com_np = original_pred_com.cpu().numpy()
    
    # Handle true COM data carefully
    true_com_np = None
    if hasattr(original_data, 'y') and original_data.y is not None:
        y_data = original_data.y.cpu().numpy()
        print(f"DEBUG: True COM data shape: {y_data.shape}")
        
        # Make sure we have a properly shaped array for plotting
        if len(y_data.shape) == 1 and y_data.shape[0] == 3:
            # Already in the right shape [x, y, z]
            true_com_np = y_data
        elif len(y_data.shape) == 1 and y_data.shape[0] == 1:
            # Only one value, can't use as 3D point
            print("Warning: True COM has only one value, can't plot in 3D")
            true_com_np = None
        elif len(y_data.shape) == 2 and y_data.shape[0] == 1 and y_data.shape[1] == 3:
            # Shape [1, 3], extract the first row
            true_com_np = y_data[0]
        elif len(y_data.shape) == 2 and y_data.shape[0] == 3 and y_data.shape[1] == 1:
            # Shape [3, 1], flatten
            true_com_np = y_data.flatten()
        else:
            print(f"Warning: Unexpected true COM shape {y_data.shape}, can't plot")
            true_com_np = None
    
    # Create and store plot elements in state
    app_state["scatter"] = ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=5, label='Point Cloud', alpha=0.6)
    app_state["pred_marker"], = ax.plot(pred_com_np[0:1], pred_com_np[1:2], pred_com_np[2:3], 'rx', markersize=10, label='Predicted COM (from rotated input)')
    app_state["rotated_orig_marker"], = ax.plot(pred_com_np[0:1], pred_com_np[1:2], pred_com_np[2:3], 'g*', markersize=10, label='Original Prediction Rotated')
    
    # Add true center of mass marker
    if true_com_np is not None:
        try:
            app_state["true_marker"], = ax.plot(true_com_np[0:1], true_com_np[1:2], true_com_np[2:3], 'bo', markersize=10, label='True COM')
        except IndexError:
            print("Warning: Error creating true COM marker, skipping")
            app_state["true_marker"] = None
    else:
        app_state["true_marker"] = None

    ax.legend()
    print("DEBUG: Plot elements created.")

    # --- Sliders --- 
    axcolor = 'lightgoldenrodyellow'
    # Slider positions adjusted
    ax_alpha = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_beta  = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_gamma = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    app_state["s_alpha"] = Slider(ax_alpha, 'Yaw (Z)', -180.0, 180.0, valinit=0.0)
    app_state["s_beta"]  = Slider(ax_beta, 'Pitch (Y)', -180.0, 180.0, valinit=0.0)
    app_state["s_gamma"] = Slider(ax_gamma, 'Roll (X)', -180.0, 180.0, valinit=0.0)

    # Connect sliders to update function
    app_state["s_alpha"].on_changed(update_rotation)
    app_state["s_beta"].on_changed(update_rotation)
    app_state["s_gamma"].on_changed(update_rotation)
    
    # --- Buttons --- 
    ax_prev = plt.axes([0.25, 0.25, 0.1, 0.04])
    ax_next = plt.axes([0.36, 0.25, 0.1, 0.04])
    ax_switch = plt.axes([0.50, 0.25, 0.2, 0.04])
    
    button_prev = Button(ax_prev, 'Prev Sample')
    button_next = Button(ax_next, 'Next Sample')
    button_switch = Button(ax_switch, 'Switch Dataset')
    
    button_prev.on_clicked(prev_sample)
    button_next.on_clicked(next_sample)
    button_switch.on_clicked(switch_dataset)
    print("DEBUG: Buttons created.")

    print("Showing interactive plot...")
    plt.show()
    print("Visualization finished.")

# --- Callback Functions and Update Logic ---

def plot_initial_sample():
    """Calculates initial prediction and updates plot elements for the current sample."""
    print("DEBUG: plot_initial_sample() called") # DEBUG
    global app_state
    
    original_data = app_state["original_data"]
    model = app_state["model"]
    
    if None in [original_data, model]:
        print("Error: Data or model not initialized for plotting.")
        return
        
    # Calculate the initial prediction for the new sample
    with torch.no_grad():
        app_state["original_pred_com"] = model(original_data.clone())[0]
    
    # Get data for plotting
    points_np = original_data.pos.cpu().numpy()
    pred_com_np = app_state["original_pred_com"].cpu().numpy()
    
    # Handle true COM data safely
    true_com_np = None
    if hasattr(original_data, 'y') and original_data.y is not None:
        y_data = original_data.y.cpu().numpy()
        print(f"DEBUG: True COM data shape: {y_data.shape}")
        
        # Make sure we have a properly shaped array for plotting
        if len(y_data.shape) == 1 and y_data.shape[0] == 3:
            # Already in the right shape [x, y, z]
            true_com_np = y_data
        elif len(y_data.shape) == 1 and y_data.shape[0] == 1:
            # Only one value, can't use as 3D point
            print("Warning: True COM has only one value, can't plot in 3D")
            true_com_np = None
        elif len(y_data.shape) == 2 and y_data.shape[0] == 1 and y_data.shape[1] == 3:
            # Shape [1, 3], extract the first row
            true_com_np = y_data[0]
        elif len(y_data.shape) == 2 and y_data.shape[0] == 3 and y_data.shape[1] == 1:
            # Shape [3, 1], flatten
            true_com_np = y_data.flatten()
        else:
            print(f"Warning: Unexpected true COM shape {y_data.shape}, can't plot")
            true_com_np = None
    
    # Only update existing plot elements if they exist
    ax = app_state["ax"]
    scatter = app_state["scatter"]
    pred_marker = app_state["pred_marker"]
    rotated_orig_marker = app_state["rotated_orig_marker"]
    true_marker = app_state.get("true_marker")
    
    if None not in [ax, scatter, pred_marker, rotated_orig_marker]:
        # Update scatter plot data (DO NOT RECREATE)
        scatter._offsets3d = (points_np[:, 0], points_np[:, 1], points_np[:, 2])
        pred_marker.set_data_3d([pred_com_np[0]], [pred_com_np[1]], [pred_com_np[2]])
        rotated_orig_marker.set_data_3d([pred_com_np[0]], [pred_com_np[1]], [pred_com_np[2]])
        
        # Update true COM marker if it exists and true_com_np is valid
        if true_marker is not None and true_com_np is not None:
            try:
                true_marker.set_data_3d([true_com_np[0]], [true_com_np[1]], [true_com_np[2]])
                true_marker.set_visible(True)
            except IndexError:
                print("Warning: Index error when plotting true COM, hiding marker")
                true_marker.set_visible(False)
        elif true_marker is not None:
            true_marker.set_visible(False)
        
        # Set plot limits based on new data
        max_range = np.array([points_np[:, 0].max()-points_np[:, 0].min(),
                            points_np[:, 1].max()-points_np[:, 1].min(),
                            points_np[:, 2].max()-points_np[:, 2].min()]).max() / 2.0
        max_range = max(max_range, 0.1) * 1.1 
        mid_x = (points_np[:, 0].max()+points_np[:, 0].min()) * 0.5
        mid_y = (points_np[:, 1].max()+points_np[:, 1].min()) * 0.5
        mid_z = (points_np[:, 2].max()+points_np[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Get model name for display
        model_name = type(model).__name__
        checkpoint_name = os.path.basename(app_state.get("checkpoint_path", "unknown_checkpoint"))
        
        # Update title with model info
        active_dataset = app_state[f"{app_state['active_dataset_name']}_dataset"]
        title = (f'Model: {model_name} ({checkpoint_name})\n'
                f'{app_state["active_dataset_name"]} sample '
                f'{app_state["current_sample_index"]+1}/{len(active_dataset)}')
        ax.set_title(title)
    
    print("DEBUG: plot_initial_sample() finished") # DEBUG

def load_and_plot_new_sample():
    """Loads the current sample based on state, ensures edge_attr, and calls plot_initial_sample."""
    print("DEBUG: load_and_plot_new_sample() called") # DEBUG
    global app_state
    device = app_state["device"]
    active_dataset = app_state[f"{app_state['active_dataset_name']}_dataset"]
    
    # Load data
    app_state["original_data"] = active_dataset[app_state["current_sample_index"]].to(device)
    
    # Ensure edge_attr exists (needed for some models/datasets)
    odata = app_state["original_data"]
    if not hasattr(odata, 'edge_attr') or odata.edge_attr is None or odata.edge_attr.shape[1] != 3:
         print("Warning: edge_attr not found/valid. Creating from pos difference.")
         row, col = odata.edge_index
         odata.edge_attr = odata.pos[row] - odata.pos[col]

    # Plot the newly loaded sample
    plot_initial_sample()
    
    # Reset sliders
    if app_state["s_alpha"]: app_state["s_alpha"].reset()
    if app_state["s_beta"]: app_state["s_beta"].reset()
    if app_state["s_gamma"]: app_state["s_gamma"].reset()
    
    # Redraw
    if app_state["fig"]: app_state["fig"].canvas.draw_idle()
    print("DEBUG: load_and_plot_new_sample() finished") # DEBUG

def next_sample(event):
    """Callback for Next Sample button."""
    global app_state
    active_dataset = app_state[f"{app_state['active_dataset_name']}_dataset"]
    app_state["current_sample_index"] = (app_state["current_sample_index"] + 1) % len(active_dataset)
    load_and_plot_new_sample()

def prev_sample(event):
    """Callback for Previous Sample button."""
    global app_state
    active_dataset = app_state[f"{app_state['active_dataset_name']}_dataset"]
    app_state["current_sample_index"] = (app_state["current_sample_index"] - 1 + len(active_dataset)) % len(active_dataset)
    load_and_plot_new_sample()

def switch_dataset(event):
    """Callback for Switch Dataset button."""
    global app_state
    if app_state["active_dataset_name"] == "test":
        app_state["active_dataset_name"] = "train"
    else:
        app_state["active_dataset_name"] = "test"
    app_state["current_sample_index"] = 0 # Reset index when switching
    load_and_plot_new_sample()

def update_rotation(val):
    """Callback for slider changes. Renamed from update to avoid name clash."""
    global app_state
    # Access state variables needed for rotation update
    original_data = app_state["original_data"]
    original_pred_com = app_state["original_pred_com"]
    model = app_state["model"]
    device = app_state["device"]
    ax = app_state["ax"]
    fig = app_state["fig"]
    scatter = app_state["scatter"]
    pred_marker = app_state["pred_marker"]
    rotated_orig_marker = app_state["rotated_orig_marker"]
    true_marker = app_state.get("true_marker")
    s_alpha = app_state["s_alpha"]
    s_beta = app_state["s_beta"]
    s_gamma = app_state["s_gamma"]
    
    if None in [original_data, original_pred_com, model, device, fig, scatter, pred_marker, rotated_orig_marker, s_alpha, s_beta, s_gamma]:
        return # Not fully initialized
        
    alpha = s_alpha.val
    beta = s_beta.val
    gamma = s_gamma.val

    # 1. Get rotation matrix
    R = euler_to_rotation_matrix(alpha, beta, gamma).to(device)

    # 2. Rotate the absolute original tensors
    # Need original tensors (assuming they are stored correctly in original_data)
    abs_original_pos = original_data.pos
    abs_original_edge_attr = original_data.edge_attr
    rotated_pos = torch.matmul(abs_original_pos, R.T)
    rotated_edge_attr = torch.matmul(abs_original_edge_attr, R.T) if abs_original_edge_attr is not None else None
    
    # Rotate true COM if it exists
    rotated_true_com = None
    if hasattr(original_data, 'y') and original_data.y is not None:
        y_data = original_data.y
        
        # Handle different tensor shapes
        if len(y_data.shape) == 1 and y_data.shape[0] == 3:
            # Already in the right shape [x, y, z]
            true_com = y_data
            rotated_true_com = torch.matmul(true_com.unsqueeze(0), R.T).squeeze(0)
        elif len(y_data.shape) == 2 and y_data.shape[0] == 1 and y_data.shape[1] == 3:
            # Shape [1, 3]
            true_com = y_data.squeeze(0)
            rotated_true_com = torch.matmul(true_com.unsqueeze(0), R.T).squeeze(0)
        elif len(y_data.shape) == 2 and y_data.shape[0] == 3 and y_data.shape[1] == 1:
            # Shape [3, 1]
            true_com = y_data.squeeze(1)
            rotated_true_com = torch.matmul(true_com.unsqueeze(0), R.T).squeeze(0)
    
    # 3. Create a NEW Data object with rotated attributes
    input_data_for_model = Data(
        x=original_data.x,
        edge_index=original_data.edge_index,
        pos=rotated_pos,
        edge_attr=rotated_edge_attr,
        batch=original_data.batch
    )

    # 4. Run inference on the NEW Data object using the loaded model
    with torch.no_grad():
        # GNNLightningModule expects the Data object directly
        new_pred_com = model(input_data_for_model)[0]

    # 5. Rotate original prediction
    rotated_original_pred = torch.matmul(original_pred_com.unsqueeze(0), R.T).squeeze(0)

    # 6. Update plot using the calculated rotated positions and predictions
    rotated_points_np = rotated_pos.cpu().numpy()
    scatter._offsets3d = (rotated_points_np[:, 0], rotated_points_np[:, 1], rotated_points_np[:, 2])

    new_pred_com_np = new_pred_com.cpu().numpy()
    pred_marker.set_data_3d([new_pred_com_np[0]], [new_pred_com_np[1]], [new_pred_com_np[2]])

    rotated_original_pred_np = rotated_original_pred.cpu().numpy()
    rotated_orig_marker.set_data_3d([rotated_original_pred_np[0]], [rotated_original_pred_np[1]], [rotated_original_pred_np[2]])

    # Update true COM marker if it exists
    if true_marker is not None and rotated_true_com is not None:
        try:
            rotated_true_com_np = rotated_true_com.cpu().numpy()
            true_marker.set_data_3d([rotated_true_com_np[0]], [rotated_true_com_np[1]], [rotated_true_com_np[2]])
            true_marker.set_visible(True)
        except IndexError:
            print("Warning: Index error when plotting rotated true COM, hiding marker")
            true_marker.set_visible(False)
    elif true_marker is not None:
        true_marker.set_visible(False)

    fig.canvas.draw_idle()


if __name__ == "__main__":
    visualize()