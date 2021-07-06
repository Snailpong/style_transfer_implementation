from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import random
import torch
from PIL import Image
import numpy as np


class TypesDataset(DataLoader):
    def __init__(self, dataset_dir, dirs, transform):
        self.dataset_dir = dataset_dir
        self.dirs = dirs
        self.transform = transform
        self.train_lists_a = os.listdir(f'{dataset_dir}/{dirs[0]}')
        self.train_lists_b = os.listdir(f'{dataset_dir}/{dirs[1]}')

    def __getitem__(self, index):
        items = []
        items.append(self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[0]}/{self.train_lists_a[index]}')))
        indexb = random.randint(0, len(self.train_lists_b) - 1)
        items.append(self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[1]}/{self.train_lists_b[indexb]}')))
        return items

    def __len__(self):
        return len(self.train_lists_a)


class CartoonGANDataset(TypesDataset):
    def __init__(self, dataset_dir, dirs, transform):
        super().__init__(dataset_dir, dirs, transform)
        self.blur = transforms.GaussianBlur(5, 1)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

    def __getitem__(self, index):
        photo = self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[0]}/{self.train_lists_a[index]}'))
        photo = self.to_tensor(photo)

        indexb = random.randint(0, len(self.train_lists_b) - 1)
        _cartoon = self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[1]}/{self.train_lists_b[indexb]}'))
        cartoon = self.to_tensor(_cartoon)
        cartoon_blur = self.to_tensor(self.blur(_cartoon))

        return photo, [cartoon, cartoon_blur]