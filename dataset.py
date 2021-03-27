from torch.utils.data import DataLoader
import os
import random
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class CycleGANDataset(DataLoader):
    def __init__(self, dataset_dir):
        self.dataset_dir_a = os.path.join(dataset_dir, 'trainA')
        self.dataset_dir_b = os.path.join(dataset_dir, 'trainB')
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.train_a_list = os.listdir(self.dataset_dir_a)
        self.train_b_list = os.listdir(self.dataset_dir_b)


    def __getitem__(self, index):
        indexb = random.randint(0, len(self.train_b_list) - 1)
        image_a = self.transform(Image.open(os.path.join(self.dataset_dir_a, self.train_a_list[index])))
        image_b = self.transform(Image.open(os.path.join(self.dataset_dir_b, self.train_b_list[indexb])))

        return image_a, image_b


    def __len__(self):
        return len(self.train_a_list)