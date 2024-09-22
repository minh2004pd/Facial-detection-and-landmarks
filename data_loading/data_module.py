import os
import torch
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
from data_loading.dataset import FacialKeypointsDataset
from data_loading.dataset2 import FaceLandmarksDataset
from data_loading.transform import Transforms

## import from src after this line
root_path = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "conf")

class DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data"):
        super().__init__()
        self.train_dir = f'{data_dir}/training.csv'
        self.test_dir = f'{data_dir}/test.csv'
    
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_full_data = FacialKeypointsDataset(csv_file=self.train_dir)

            # Calculate the sizes for train and validation sets
            total_size = train_full_data.length_train_data()
            train_size = int(total_size*0.8)
            val_size = total_size - train_size  # Ensure the sum of train_size and val_size equals total_size

            self.train_data, self.val_data = random_split(
                train_full_data, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
        
        if stage == "test" or stage == "predict":
            self.test_data = FacialKeypointsDataset(csv_file=self.test_dir)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32, num_workers=30)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=32, num_workers=30)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=32, num_workers=30)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=32, num_workers=30)

class DataModule2(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data"):
        super().__init__()
        train_path = f"{data_dir}/labels_ibug_300W_train.xml"
        test_path = f"{data_dir}/labels_ibug_300W_test.xml"

        train_transform = Transforms(training=True)
        test_transform = Transforms(training=False)
        
        self.dataset = FaceLandmarksDataset(data_path=train_path, transform=train_transform)
        self.test_dataset = FaceLandmarksDataset(data_path=test_path, transform=test_transform)
    
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_size = int(0.8*len(self.dataset))
            val_size = len(self.dataset) - train_size

            self.train_data, self.val_data = random_split(
                self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
        
        elif stage == "test" or stage == "predict" :
            self.test_data = self.test_dataset
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32, num_workers=30)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=32, num_workers=30)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=32, num_workers=30)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=30)





