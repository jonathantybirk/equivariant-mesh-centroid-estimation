import torch
import torch.nn.functional as F
import lightning.pytorch as pl


class BaseModel(pl.LightningModule):
    """
    Base class with all common training/validation/logging logic

    Key design decisions based on research:
    - MSE as loss function for better optimization (smooth gradients)
    - MAE as primary interpretable metric (same units as target)
    - Per-epoch logging only (since epochs are fast now)
    - Per-component MAE for detailed analysis (x, y, z coordinates)
    """

    def __init__(self, lr=1e-3, weight_decay=1e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Use displacement distance as both loss and metric
        displacement = pred - batch.y  # Displacement vectors
        mean_displacement_vector = displacement.mean(
            dim=0
        )  # Mean displacement vector [x, y, z]
        mean_displacement_distance = torch.norm(
            mean_displacement_vector
        )  # L2 norm of mean displacement vector

        # Log metrics per epoch only (since epochs are fast now)
        batch_size = batch.y.size(0)
        self.log(
            "train_displacement_distance",
            mean_displacement_distance,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # Log per-component displacement
        for i, component in enumerate(["x", "y", "z"]):
            self.log(
                f"train_displacement_{component}",
                mean_displacement_vector[i],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        return mean_displacement_distance

    def validation_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Use displacement distance for validation consistency
        displacement = pred - batch.y  # Displacement vectors
        mean_displacement_vector = displacement.mean(
            dim=0
        )  # Mean displacement vector [x, y, z]
        mean_displacement_distance = torch.norm(
            mean_displacement_vector
        )  # L2 norm of mean displacement vector

        # Log metrics per epoch only
        batch_size = batch.y.size(0)
        self.log(
            "val_displacement_distance",
            mean_displacement_distance,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Log per-component displacement
        for i, component in enumerate(["x", "y", "z"]):
            self.log(
                f"val_displacement_{component}",
                mean_displacement_vector[i],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        return mean_displacement_distance  # Return displacement distance for monitoring (more interpretable)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_displacement_distance",  # Monitor displacement distance instead of loss for interpretability
                "frequency": 1,
            },
        }
