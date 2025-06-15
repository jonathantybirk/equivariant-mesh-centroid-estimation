import torch
from .base_model import BaseModel


class BaselineZero(BaseModel):
    """
    Baseline model that predicts the center of mass of the graph.
    This is a simple model that predicts the center of mass of the graph.
    It is a baseline model that is used to compare the performance of the other models.
    It is a simple model that is used to compare the performance of the other models.
    It is a simple model that is used to compare the performance of the other models.
    """

    def __init__(
        self,
        lr=1e-3,
        weight_decay=1e-5,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        # Disable automatic optimization since this model has no parameters
        self.automatic_optimization = False

    def forward(self, x, edge_index, edge_attr, batch):
        # Determine the number of graphs in the batch
        if batch is not None:
            batch_size = batch.max().item() + 1
        else:
            batch_size = 1

        # Return zeros for each graph in the batch
        return torch.zeros(batch_size, 3, device=x.device)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # FIXED: Compute distance per sample, then mean (prevents error cancellation)
        per_sample_distances = torch.norm(pred - batch.y, dim=1)  # [B]
        mean_displacement_distance = per_sample_distances.mean()  # scalar

        # Also compute individual displacement components for analysis
        displacement = pred - batch.y  # [B, 3]
        mean_displacement_vector = displacement.mean(dim=0)  # [3]

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

        # Log per-component displacement (for analysis only)
        for i, component in enumerate(["x", "y", "z"]):
            self.log(
                f"train_displacement_{component}",
                mean_displacement_vector[i],
                on_step=False,  # Changed to False to match validation
                on_epoch=True,
                batch_size=batch_size,
            )

        return mean_displacement_distance

    def configure_optimizers(self):
        # BaselineZero has no parameters to optimize, so return None
        return None
