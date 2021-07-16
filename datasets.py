from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import random
import torch
from PIL import Image
import numpy as np


class TypesDataset(DataLoader):
    def __init__(self, dataset_dir, dirs):
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
    def __init__(self, dataset_dir, dirs):
        super().__init__(dataset_dir, dirs)
        self.train_lists_c = os.listdir(f'{dataset_dir}/{dirs[2]}')
        self.resize = transforms.Compose([
            transforms.RandomCrop((768, 768), pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((256, 256))
            ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.transform = transforms.Compose([
            self.resize, self.to_tensor
        ])

    def __getitem__(self, index):
        photo = self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[0]}/{self.train_lists_a[index]}'))

        indexb = random.randint(0, len(self.train_lists_b) - 1)
        cartoon = self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[1]}/{self.train_lists_b[indexb]}'))

        indexc = random.randint(0, len(self.train_lists_c) - 1)
        cartoon_blur = self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[2]}/{self.train_lists_c[indexc]}'))

        return photo, [cartoon, cartoon_blur]


class AnimeGANDataset(CartoonGANDataset):
    def __init__(self, dataset_dir, dirs):
        super().__init__(dataset_dir, dirs)
        self.gray_to_tensor = transforms.Compose([
            self.resize, 
            transforms.Grayscale(num_output_channels=3),
            self.to_tensor
        ])

    def __getitem__(self, index):
        photo = self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[0]}/{self.train_lists_a[index]}'))

        indexb = random.randint(0, len(self.train_lists_b) - 1)
        indexc = random.randint(0, len(self.train_lists_c) - 1)

        cartoon_raw = self.resize(Image.open(f'{self.dataset_dir}/{self.dirs[1]}/{self.train_lists_b[indexb]}'))
        cartoon_blur_raw = self.resize(Image.open(f'{self.dataset_dir}/{self.dirs[2]}/{self.train_lists_c[indexc]}'))

        cartoon = self.to_tensor(cartoon_raw)
        cartoon_blur = self.to_tensor(cartoon_blur_raw)
        cartoon_gray = self.gray_to_tensor(cartoon_raw)
        cartoon_blur_gray = self.gray_to_tensor(cartoon_blur_raw)

        return photo, [cartoon, cartoon_blur, cartoon_gray, cartoon_blur_gray]