import os
import sys
from pathlib import Path
import importlib

# Change working directory to the project root
def find_project_root(current: Path, markers=(".git", "pyproject.toml")):
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError("Project root not found.")

root = find_project_root(Path(__file__).resolve())
os.chdir(root)
sys.path.insert(0, str(root))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from src.data.datamodule import PointCloudDataModule


def get_class(class_path: str):
    """Helper function to dynamically import a class."""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training function using Hydra for configuration"""
    print(OmegaConf.to_yaml(cfg))
    
    # Set seeds for reproducibility using the root-level seed
    pl.seed_everything(cfg.seed)
    
    # Create directories if they don't exist
    os.makedirs(cfg.training.log_dir, exist_ok=True)
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    
    # Create data module (with train-test split)
    data_module = PointCloudDataModule(
        processed_dir=cfg.preprocessing.processed_dir,
        train_dir=os.path.join(cfg.preprocessing.processed_dir, "train"),
        test_dir=os.path.join(cfg.preprocessing.processed_dir, "test"),
        batch_size=cfg.training.batch_size,
        num_workers=0,  # ⚠️ IMPORTANT: Set to 0 to avoid multiprocessing issues
        pin_memory=False,  # Turn this off for CPU training
        node_feature_dim=cfg.models.gnn.hidden_dim,
        val_split=0.1  # Use 10% of training data for validation
    )
    
    # Set up the data module to access dataset sizes
    data_module.setup(stage='fit')
    
    # Dynamically get the model class
    ModelClass = get_class(cfg.model.module_path)
    print(f"Using model class: {ModelClass.__name__} from {cfg.model.module_path}")

    # Create model instance using the dynamically imported class
    model = ModelClass(
        hidden_dim=cfg.models.gnn.hidden_dim,
        message_passing_steps=cfg.models.gnn.message_passing_steps,
        message_mlp_dims=cfg.models.gnn.message_mlp_dims,
        lr=cfg.training.lr
    )
    
    # Set up Weights & Biases logger (user-agnostic)
    wandb_logger = WandbLogger(
        # No entity specified - uses whoever is logged in
        project="equivariant-center-of-mass",
        name=cfg.name,
        save_dir=cfg.training.log_dir,
        log_model="all",  # Log model checkpoints
        config=OmegaConf.to_container(cfg, resolve=True)  # Log configuration
    )
    
    # Set up callbacks
    callbacks = [
        # Save best models
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=cfg.training.checkpoint_dir,
            filename=f'{cfg.name}-{{epoch:02d}}-{{val_loss:.4f}}',
            save_top_k=3,
            mode='min'
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=cfg.training.patience,
            mode='min'
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Check if GPU is available
    gpu_available = torch.cuda.is_available()
    
    # Configure trainer settings based on hardware
    if gpu_available and cfg.training.gpus > 0:
        # Use GPU
        accelerator = "gpu"
        devices = min(cfg.training.gpus, torch.cuda.device_count())
    else:
        # Use CPU
        if cfg.training.gpus > 0:
            print("WARNING: GPU requested but not available. Using CPU instead.")
        accelerator = "cpu"
        devices = 1  # Use 1 CPU core
    
    # Create trainer with optimized settings for CPU training
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        check_val_every_n_epoch=5,  # Validate less frequently
        num_sanity_val_steps=0,     # Skip sanity validation 
        # Additional settings to fix validation hanging
        detect_anomaly=False,       # Turn off anomaly detection for speed
        enable_progress_bar=True,    # Keep progress bar for monitoring
        enable_model_summary=True,  # Show model summary
        precision='32-true',        # Use 32-bit precision but avoid unnecessary checks
        profiler=None,              # Disable profiler for speed
        # This is critical - manually set max steps to break long validation run
        val_check_interval=1.0,     # Set validation check interval to once per epoch
    )
    
    # Print training info
    print(f"\n{'='*50}")
    print(f"Training on {accelerator.upper()} with {devices} device(s)")
    print(f"Validating every 5 epochs")
    print(f"Training set size: {len(data_module.train_dataset)} samples")
    print(f"Validation set size: {len(data_module.val_dataset)} samples")
    print(f"{'='*50}\n")
    
    # Force garbage collection before training
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Train model with error handling
    try:
        trainer.fit(model, data_module)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test model only if requested and training succeeded
    if hasattr(trainer, 'callback_metrics') and cfg.get('do_test', False):
        print("Running test evaluation...")
        trainer.test(model, data_module)
    else:
        print("Skipping test evaluation")
    
    # Save final model
    try:
        final_path = os.path.join(cfg.training.checkpoint_dir, f"{cfg.name}_final.ckpt")
        trainer.save_checkpoint(final_path)
        print(f"Final model saved to {final_path}")
    except Exception as e:
        print(f"Error saving final model: {str(e)}")
    
    # Finish the W&B run
    try:
        wandb_url = wandb_logger.experiment.url
        wandb_logger.experiment.finish()
        print(f"View your training results at: {wandb_url}")
    except Exception as e:
        print(f"Error finishing W&B run: {str(e)}")
    
    print("\nTraining process complete!")


if __name__ == "__main__":
    main()