import os
import torch
from torch.optim import Adam, SGD
from torch import nn
import torch.nn.functional as F
from torch import optim, nn
from torchvision import transforms
import torch.utils.data as data
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
import lightning as L
from omegaconf import DictConfig, OmegaConf
import torchvision.models as models
import rootutils
import hydra
import lightning as L
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import pandas as pd
from model.metric import MAE
from model.CNNmodel import CNNModel

## import from src after this line
root_path = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "conf")

class CNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        self.train_loss = []
        self.metric = MAE()
        self.best_mae, self.best_epoch = float('inf'), 0

        self.model = CNNModel()

        
        
    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y, y_hat):
        return torch.sqrt(self.loss(y, y_hat))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.compute_loss(y, y_hat)
        self.train_loss.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self.metric.update(y, y_hat)
    
    def on_validation_epoch_end(self):
        mae = self.metric.compute()
        self.metric.reset()
        avg_train_loss = 0 if len(self.train_loss) == 0 else round(sum(self.train_loss) / len(self.train_loss), 4)
        self.log("train_loss", avg_train_loss, sync_dist=False)
        self.log("mae", mae, sync_dist=False)
        if mae <= self.best_mae:
            self.best_mae = mae
            self.best_epoch = self.current_epoch
        print(f"Epoch {self.current_epoch}: train_loss={avg_train_loss}, mae={mae}, min_mae={self.best_mae}, best_epoch={self.best_epoch}")
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self.metric.update(y, y_hat)
    
    def on_test_epoch_end(self):
        # At the end of the test epoch, compute and log the final MAE for the test set
        mae = self.metric.compute()
        self.metric.reset()
        
        # Log the test MAE
        self.log("test_mae", mae, sync_dist=False)
        
        # Optionally, print the result for better visibility
        print(f"Test MAE: {mae}")
    
    def predict_step(self, batch, batch_idx):
        x = batch
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0003, weight_decay=0.0001)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3000, eta_min=8e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

                        