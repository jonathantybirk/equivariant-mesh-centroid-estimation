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

        # Use MSE as loss function for better optimization (smooth gradients)
        loss = F.mse_loss(pred, batch.y)

        # Track MAE as the primary interpretable metric
        mae = F.l1_loss(pred, batch.y)

        # Per-component MAE for detailed analysis
        mae_per_component = F.l1_loss(pred, batch.y, reduction="none").mean(dim=0)

        # Log metrics per epoch only (since epochs are fast now)
        batch_size = batch.y.size(0)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train_mae",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # Log per-component MAE
        for i, component in enumerate(["x", "y", "z"]):
            self.log(
                f"train_mae_{component}",
                mae_per_component[i],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Use MSE as validation loss for consistency with training
        loss = F.mse_loss(pred, batch.y)

        # Track MAE as the primary interpretable metric
        mae = F.l1_loss(pred, batch.y)

        # Per-component MAE for detailed analysis
        mae_per_component = F.l1_loss(pred, batch.y, reduction="none").mean(dim=0)

        # Log metrics per epoch only
        batch_size = batch.y.size(0)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val_mae",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Log per-component MAE
        for i, component in enumerate(["x", "y", "z"]):
            self.log(
                f"val_mae_{component}",
                mae_per_component[i],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        return mae  # Return MAE for monitoring (more interpretable)

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
                "monitor": "val_mae",  # Monitor MAE instead of loss for interpretability
                "frequency": 1,
            },
        }

    def compute_clean_train_mae(self, train_loader):
        """
        Diagnostic method: compute training MAE in eval() mode (no dropout/noise)
        to check if train/val gap is due to regularization during training
        """
        self.eval()
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in train_loader:
                pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                mae = F.l1_loss(pred, batch.y, reduction="sum")
                total_mae += mae.item()
                total_samples += batch.y.size(0)

        return total_mae / total_samples
