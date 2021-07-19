import click
import os
import torch
import random
from datetime import datetime
from PIL import Image

from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import transforms

import numpy as np

from utils import init_device_seed
from model_cartoongan import CartoonGANGenerator
from model_animegan import AnimeGANGenerator


@click.command()
@click.option('--image_path', default='./result/pic')
@click.option('--model_name', default='cartoongan')
def test(image_path, model_name):
    device = init_device_seed(1234)
    os.makedirs('./result', exist_ok=True)

    if model_name == 'cartoongan':
        checkpoint = torch.load('./model/cartoongan', map_location=device)
        generator = AnimeGANGenerator().to(device)
    else:
        checkpoint = torch.load('./model/animegan', map_location=device)
        generator = AnimeGANGenerator().to(device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    to_pil = transforms.Compose([
        transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
        transforms.ToPILImage()
    ])

    if os.path.isdir(image_path):
        files_list = []
        file_names_list = os.listdir(image_path)
        for file_name in file_names_list:
            files_list.append(os.path.join(image_path, file_name))
        output_dir = './result/{}'.format(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
        os.makedirs(output_dir, exist_ok=True)
    else:
        files_list = [image_path]

    for idx, file_path in enumerate(files_list):
        file_name = '.'.join(os.path.basename(file_path).split('.')[:-1])
        print('\r{}/{} {}'.format(idx, len(files_list), file_name), end=' ')
    
        image = Image.open(file_path)
        size_min = min(image.size)

        transform = transforms.Compose([
            transforms.CenterCrop((size_min, size_min)),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((256, 256))
        ])

        image = transform(image)
        image.save('{}/{}_orig.jpg'.format(output_dir,file_name))

        image = to_tensor(image)
        image = torch.unsqueeze(image, 0).to(device)

        output = generator(image).detach().cpu()[0]
        output = to_pil(output)

        output.save('{}/{}.jpg'.format(output_dir,file_name))


if __name__ == '__main__':
    test()