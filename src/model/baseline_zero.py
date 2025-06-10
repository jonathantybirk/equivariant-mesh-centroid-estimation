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

    def forward(self, x, edge_index, edge_attr, batch):
        return torch.tensor([0, 0, 0], device=x.device)
