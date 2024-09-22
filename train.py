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
from model.ResNetModel import ResNet
from model.cnn import CNN
from data_loading.data_module import DataModule, DataModule2

## import from src after this line
root_path = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "conf")

@hydra.main(version_base=None, config_path=config_path, config_name="train")
def main(cfg: DictConfig) -> None:
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)
    data_module = DataModule2()
    model = ResNet()
    trainer.fit(model, data_module)
    
    trainer.test(model, data_module)

if __name__ == "__main__":
    main()

