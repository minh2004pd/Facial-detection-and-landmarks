import os
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torch import optim, nn
from torchvision import transforms
import torch.utils.data as data
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset, Dataset
import lightning as L
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torchvision.models as models
import rootutils
import hydra
import lightning as L
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import pandas as pd

## import from src after this line
root_path = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "conf/basic")

# Custom Dataset class
class FacialKeypointsDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data.fillna(method='ffill', inplace=True)
        self.transform = transform

        imag = []
        for i in range(len(self.data)):
            img = self.data['Image'][i].split(' ')
            img = ['0' if x == '' else x for x in img]
            imag.append(img)
        
        # reshape and convert it into float value
        self.images = np.array(imag, dtype='float16').reshape(-1, 96, 96, 1)
        self.keypoints = self.data.drop('Image', axis=1).values.astype('float16')

    def __len__(self):
        return len(self.data)
    
    def length_train_data(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        keypoints = self.keypoints[idx]

        if self.transform:
            image = self.transform(image)

        return image, keypoints


