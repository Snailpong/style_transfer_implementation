from torch.utils.data import DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class CycleGANDataset(DataLoader):
    def __init__(self, dataset_dir):
        self.transform = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.toTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __getitem__(self, index):
        pass