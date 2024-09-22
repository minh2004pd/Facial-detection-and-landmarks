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
import json
from PIL import Image, ImageDraw

## import from src after this line
root_path = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "conf")

# Custom Dataset class
class PngDataset(Dataset):
    def __init__(self, json_file, transform=None):
        # Read the JSON file
        with open(json_file, 'r') as json_file:
            data = json.load(json_file)
        
        # Access the data
        self.imgs = data.get('imgs', [])  # Access the 'imgs' key or return an empty list if 'imgs' is not found
        test_data = []
        # Print the contents
        for item in self.imgs:
            filename = item.get('filename', None)
            if filename:
                # Load the image
                image = Image.open(filename)
                # Convert the image to grayscale
                gray_image = image.convert('L')

                # Resize the grayscale image to 96x96
                resized_image = gray_image.resize((96, 96))

                # Convert the image to a NumPy array
                image_array = np.array(resized_image, dtype='float16')
                
                # Expand dimensions to match shape (-1, 96, 96, 1)
                image_array = np.expand_dims(image_array, axis=-1)
                
                # Append to test_data
                test_data.append(image_array)
                
        self.transform = transform
        
        # reshape and convert it into float value
        self.images = np.array(test_data, dtype='float16').reshape(-1, 96, 96, 1)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        return image


