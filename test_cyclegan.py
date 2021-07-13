import click
import os
import torch
import random
from datetime import datetime
from PIL import Image

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchvision import transforms

import numpy as np

from utils import init_device_seed
from datasets import TypesDataset
from model_cyclegan import CycleGANGenerator, CycleGANDiscriminator


@click.command()
@click.option('--dataset_type', default='summer2winter_yosemite')
@click.option('--image_path', default='./data/summer2winter_yosemite/testA')
@click.option('--model_type', default='x2y')
def test(dataset_type, image_path, model_type):
    device = init_device_seed(1234)

    os.makedirs('./result', exist_ok=True)

    checkpoint = torch.load('./model/cyclegan_' + dataset_type, map_location=device)
    generator = CycleGANGenerator().to(device)
    generator.load_state_dict(checkpoint[model_type + '_state_dict'])
    generator.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    to_pil = transforms.Compose([
        transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
        transforms.ToPILImage()
    ])

    # generator_inv = CycleGANGenerator()
    # generator_inv.load_state_dict(checkpoint[model_type[::-1] + '_state_dict'])
    # generator_inv.to(device)
    # generator_inv.eval()

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
        image = to_tensor(image)
        image = torch.unsqueeze(image, 0).to(device)

        output = generator(image)

        # cycle = torch.clip(generator_inv(output).detach().cpu()[0], 0, 1)
        # cycle = transforms.ToPILImage()(cycle)
        # identity = torch.clip(generator_inv(image).detach().cpu()[0], 0, 1)
        # identity = transforms.ToPILImage()(identity)
        output = output.detach().cpu()[0]
        output = to_pil(output)

        output.save('{}/{}.jpg'.format(output_dir,file_name))
        # cycle.save('./result/{}_cycle.jpg'.format(image_path_base))
        # identity.save('./result/{}_identity.jpg'.format(image_path_base))


if __name__ == '__main__':
    test()