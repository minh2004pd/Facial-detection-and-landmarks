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

## import from src after this line
root_path = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "conf")

class ResNetv2(L.LightningModule):
    def __init__(self, num_classes: int = 68*2, num_boxes = 4):
        super().__init__()
        self.loss = nn.MSELoss()
        self.loss_box = nn.SmoothL1Loss()
        self.train_loss = []
        self.metric = MAE()
        self.best_mae, self.best_epoch = float('inf'), 0

        self.model_name='resnet50'
        self.model=models.resnet50()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        in_features = self.model.fc_infeatures
        self.model.fc = nn.Identity()

        self.fc1=nn.Linear(in_features, num_classes) # for landmarks
        self.fc2=nn.Linear(in_features, num_boxes) # for detection
        
    def forward(self, x):
        x = self.model(x)

        landmarks = self.fc1(x)
        boxes = self.fc2(x)
        return landmarks, boxes

    def compute_loss(self, y, pred_landmarks, pred_boxes):
        return torch.sqrt(self.loss(y, pred_landmarks)) + self.loss_box(pred_boxes + y)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y, y_hat)
        self.train_loss.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
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
        y_hat = self(x)
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
        x, _ = batch
        return self(x)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0003, weight_decay=0.0001)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3000, eta_min=8e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    from torchinfo import summary
    import torch

    model = ResNet()

    # show model
    batch_size = 16
    # summary(model=model,
    #         input_size=(batch_size, 96, 96, 1),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])

    random_input = torch.randn([16, 96, 96, 1])
    output = model(random_input)

    print(f"\n\nINPUT SHAPE: {random_input.shape}")
    print(f"OUTPUT SHAPE: {output.shape}")

                        