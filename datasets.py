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
        self.train_lists_a = os.listdir(f'{dataset_dir}/{dirs[0]}')
        self.train_lists_b = os.listdir(f'{dataset_dir}/{dirs[1]}')
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

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
        self.train_lists_c = os.listdir(f'{dataset_dir}/{dirs[2]}')
        self.transform = transforms.Compose([
            transforms.RandomCrop((768, 768), pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

    def __getitem__(self, index):
        photo = self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[0]}/{self.train_lists_a[index]}'))

        indexb = random.randint(0, len(self.train_lists_b) - 1)
        cartoon = self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[1]}/{self.train_lists_b[indexb]}'))

        # indexc = random.randint(0, len(self.train_lists_c) - 1)
        cartoon_blur = self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[2]}/{self.train_lists_c[indexb]}'))

        return photo, [cartoon, cartoon_blur]