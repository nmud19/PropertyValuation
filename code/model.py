import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule,LightningModule 
from typing import List

class QuantModel(LightningModule):
    def __init__(self, num_features:int=4, quantiles:List[float]=[0.1,0.5,0.9])->None:
        """Init a model"""
        super(QuantModel, self).__init__()
        self._quantile_list = quantiles
        self._num_quantiles = len(self._quantile_list)
        self.fc1 = nn.Linear(num_features, 256)
        self.fc21 = nn.Linear(256, 128)
        self.fc22 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, self._num_quantiles)

    def forward(self, x):
        """Prediction; pass the input through all layers"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc21(x))
        x = F.relu(self.fc22(x))
        x = self.fc3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        """Invoked while training the model. Quantile loss is used here."""
        x, y = batch
        y_hat = self(x)
        loss = quantile_loss(y_hat, y,self._quantile_list)
        return loss

    def validation_step(self, batch, batch_idx):
        """Invoked while training the model. Quantile loss is used here."""
        x, y = batch
        y_hat = self(x)
        loss = quantile_loss(y_hat, y,self._quantile_list)
        self.log_dict({"quant_loss" : loss} )
        return loss
    
    def test_step(self, batch, batch_idx):
        """Invoked while training the model. Quantile loss is used here."""
        x, y = batch
        y_hat = self(x)
        loss = quantile_loss(y_hat, y,self._quantile_list)
        self.log_dict({"quant_loss" : loss} )
        return loss

    def predict_step(self, batch, batch_idx):
        """Invoked while training the model. Quantile loss is used here."""
        x, _ = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def quantile_loss(preds, target, quantiles):
    """Quantile loss function"""
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss