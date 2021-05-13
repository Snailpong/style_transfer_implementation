from torch.utils.data import DataLoader
import os
import random
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class TypesDataset(DataLoader):
    def __init__(self, dataset_dir, dirs):
        self.dirs = dirs
        self.dataset_dirs = []
        self.train_lists = []
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])

        for idx, dr in enumerate(dirs):
            self.dataset_dirs.append(os.path.join(dataset_dir, dr))
            self.train_lists.append(os.listdir(self.dataset_dirs[-1]))


    def __getitem__(self, index):
        items = []
        items.append(self.transform(Image.open(os.path.join(self.dataset_dirs[0], self.train_lists[0][index]))))

        for i in range(1, len(self.dirs)):
            indexb = random.randint(0, len(self.train_lists[i]) - 1)
            items.append(self.transform(Image.open(os.path.join(self.dataset_dirs[1], self.train_lists[i][indexb]))))

        return items


    def __len__(self):
        return len(self.train_lists[0])