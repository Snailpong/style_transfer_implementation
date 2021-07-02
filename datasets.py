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
        self.toTensor = transforms.ToTensor()

    def __getitem___(self, index):
        items = []
        photo = self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[0]}/{self.train_lists_a[index]}'))
        items.append(self.toTensor(photo))

        indexb = random.randint(0, len(self.train_lists_b) - 1)
        cartoon = self.transform(Image.open(f'{self.dataset_dir}/{self.dirs[1]}/{self.train_lists_b[indexb]}'))
        # items.append(self.toTensor(cartoon))
        # items.append(self.toTensor(self.blur(cartoon)))
        # cartoon_blur = self.transform(Image.open(os.path.join(self.dataset_dirs[1], self.train_lists[i][indexb])))
        items.append(torch.stack([self.toTensor(cartoon), self.toTensor(self.blur(cartoon))], axis=1))
        # print(items[1].shape)
        # print('aaaaaaa')

        return items[0], items[1]