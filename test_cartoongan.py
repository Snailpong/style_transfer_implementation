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
from model_cartoongan import CartoonGANGenerator, CartoonGANDiscriminator


@click.command()
@click.option('--image_path', default='./result/pic')
def test(image_path):
    device = init_device_seed(1234)
    os.makedirs('./result', exist_ok=True)

    checkpoint = torch.load('./model/cartoongan', map_location=device)
    generator = CartoonGANGenerator().to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

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

        image = transforms.ToTensor()(image)
        image = torch.unsqueeze(image, 0)

        output = generator(image)
        output = torch.clip(output.detach().cpu()[0], 0, 1)
        output = transforms.ToPILImage()(output)

        output.save('{}/{}.jpg'.format(output_dir,file_name))


if __name__ == '__main__':
    test()