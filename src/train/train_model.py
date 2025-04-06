import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Import your model and DataModule.
# If your model is really an EGNN, consider renaming CNNModel to EGNNModel.
from src.models.egnn import CNNModel  
from src.data.dataset import LiDARDataModule

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)
    
    # Initialize your DataModule.
    data_module = LiDARDataModule(
        data_path=cfg.data.path,
        batch_size=cfg.model.batch_size,
        num_points=cfg.model.num_points
    )
    
    # Initialize your model.
    model = CNNModel(
        learning_rate=cfg.model.learning_rate,
        num_points=cfg.model.num_points
    )
    
    # Setup a ModelCheckpoint callback (optional).
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, verbose=True
    )
    
    # Initialize the Trainer using the new API.
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        callbacks=[checkpoint_callback]
    )
    
    # Train the model.
    trainer.fit(model, datamodule=data_module)
    
    # Optionally run testing.
    trainer.test(model, datamodule=data_module)
    
if __name__ == "__main__":
    main()
