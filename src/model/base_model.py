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
        displacement = pred - target  # [B, 3]

        per_sample_distances = torch.norm(displacement, dim=1)  # [B]
        mean_displacement_distance = per_sample_distances.mean()  # scalar

        # FIXED: Add loss scaling to help with magnitude issues
        # Scale loss to make gradients more meaningful
        loss_scale_factor = 10.0  # FIXED: Much more conservative scaling (was 1000x)
        scaled_loss = mean_displacement_distance * loss_scale_factor

        # Also compute individual displacement components for analysis
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

        # FIXED: Return scaled loss for better gradients
        return scaled_loss

    def training_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return self._calculate_and_log_displacement(pred, batch.y, "train")

    def validation_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return self._calculate_and_log_displacement(pred, batch.y, "val")

    def configure_optimizers(self):
        print("lr:", self.hparams.lr)
        print("weight_decay:", self.hparams.weight_decay)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # FIXED: Much more conservative learning rate schedule for complex models
        # Cap max_lr at 0.02 regardless of base lr, and make schedule gentler
        base_lr = self.hparams.lr
        max_lr = min(
            0.02, base_lr * 2.0
        )  # Cap at 0.02, or 2x base (whichever is lower)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_displacement_distance_epoch",
                "interval": "epoch",
                "frequency": 1,
            },
        }

        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=max_lr,  # Conservative peak - max 0.02
        #     epochs=100,
        #     steps_per_epoch=200,
        #     pct_start=0.3,  # Longer warmup (30% vs 10%)
        #     anneal_strategy="cos",
        #     div_factor=5.0,  # Start at max_lr/5 (gentler start)
        #     final_div_factor=20.0,  # End at max_lr/20 (gentler end)
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",  # Update every step, not epoch
        #         "frequency": 1,
        #     },
        # }
