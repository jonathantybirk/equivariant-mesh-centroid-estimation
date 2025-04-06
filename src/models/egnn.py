import torch
import torch.nn as nn
import pytorch_lightning as pl

class CNNModel(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3, num_points: int = 1024):
        super().__init__()
        self.save_hyperparameters()
        # Dummy architecture: replace with your EGNN layers and equivariant logic
        self.net = nn.Sequential(
            nn.Linear(num_points * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Predict 3D center of mass
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x shape: [B, num_points, 3]
        x = x.view(x.size(0), -1)
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        point_clouds, targets = batch  # targets: exact center of mass
        preds = self(point_clouds)
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        point_clouds, targets = batch
        preds = self(point_clouds)
        loss = self.loss_fn(preds, targets)
        self.log("val_loss", loss)
    
    def test_step(self, batch, batch_idx):
        point_clouds, targets = batch
        preds = self(point_clouds)
        loss = self.loss_fn(preds, targets)
        self.log("test_loss", loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
