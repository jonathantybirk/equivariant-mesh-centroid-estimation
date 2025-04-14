import os
import sys
from pathlib import Path

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
from src.model.lightning_module import GNNLightningModule


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training function using Hydra for configuration"""
    print(OmegaConf.to_yaml(cfg))
    
    # Set seeds for reproducibility
    pl.seed_everything(cfg.models.models.seed)
    
    # Create directories if they don't exist
    os.makedirs(cfg.training.log_dir, exist_ok=True)
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    
    # Create data module
    data_module = PointCloudDataModule(
        processed_dir=cfg.preprocessing.processed_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        node_feature_dim=cfg.models.gnn.hidden_dim
    )
    
    # Create model
    model = GNNLightningModule(
        hidden_dim=cfg.models.gnn.hidden_dim,
        message_passing_steps=cfg.models.gnn.message_passing_steps,
        message_mlp_dims=cfg.models.gnn.message_mlp_dims,
        update_mlp_dims=cfg.models.gnn.update_mlp_dims,
        final_mlp_dims=cfg.models.gnn.final_mlp_dims,
        lr=cfg.training.lr,
        seed=cfg.models.models.seed
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
        devices = 1  # Use 1 CPU core (you can adjust this if needed)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)
    
    # Save final model
    trainer.save_checkpoint(
        os.path.join(cfg.training.checkpoint_dir, f"{cfg.name}_final.ckpt")
    )
    
    # Finish the W&B run
    wandb_logger.experiment.finish()
    
    print(f"Training completed! Final model saved to {cfg.training.checkpoint_dir}/{cfg.name}_final.ckpt")
    print(f"View your training at: {wandb_logger.experiment.url}")


if __name__ == "__main__":
    main()