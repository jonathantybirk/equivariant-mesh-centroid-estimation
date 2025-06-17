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

    def _calculate_and_log_displacement(
        self,
        pred,
        target,
        prefix,
    ):
        """
        Calculate and log displacement metrics consistently.

        Args:
            pred: Predicted coordinates [B, 3]
            target: Target coordinates [B, 3]
            prefix: Prefix for logging ('train' or 'val')
            on_step: Whether to log on step level
            sync_dist: Whether to sync across distributed training
        """
        # Compute distance per sample, then mean (prevents error cancellation)
        per_sample_distances = torch.norm(pred - target, dim=1)  # [B]
        mean_displacement_distance = per_sample_distances.mean()  # scalar

        # Also compute individual displacement components for analysis
        displacement = pred - target  # [B, 3]
        mean_displacement_vector = displacement.mean(dim=0)  # [3]

        # Log metrics
        batch_size = target.size(0)
        self.log(
            f"{prefix}_displacement_distance",
            mean_displacement_distance,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # Log per-component displacement (for analysis only)
        for i, component in enumerate(["x", "y", "z"]):
            self.log(
                f"{prefix}_displacement_{component}",
                mean_displacement_vector[i],
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )

        return mean_displacement_distance

    def training_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return self._calculate_and_log_displacement(pred, batch.y, "train")

    def validation_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return self._calculate_and_log_displacement(pred, batch.y, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_displacement_distance",
                "frequency": 1,
            },
        }
