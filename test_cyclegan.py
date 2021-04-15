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

from dataset import CycleGANDataset
from cyclegan import CycleGANGenerator, CycleGANDiscriminator


@click.command()
@click.option('--dataset_type', default='summer2winter_yosemite')
@click.option('--image_path', default='./data/summer2winter_yosemite/testA')
@click.option('--model_type', default='x2y')
def test(dataset_type, image_path, model_type):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    os.makedirs('./result', exist_ok=True)

    checkpoint = torch.load('./model/cyclegan_' + dataset_type, map_location=device)
    generator = CycleGANGenerator()
    generator.load_state_dict(checkpoint[model_type + '_state_dict'])
    generator.to(device)
    generator.eval()

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
        image = ToTensor()(image)
        image = torch.unsqueeze(image, 0)

        output = generator(image)

        # cycle = torch.clip(generator_inv(output).detach().cpu()[0], 0, 1)
        # cycle = transforms.ToPILImage()(cycle)
        # identity = torch.clip(generator_inv(image).detach().cpu()[0], 0, 1)
        # identity = transforms.ToPILImage()(identity)
        output = torch.clip(output.detach().cpu()[0], 0, 1)
        output = transforms.ToPILImage()(output)

        output.save('{}/{}.jpg'.format(output_dir,file_name))
        # cycle.save('./result/{}_cycle.jpg'.format(image_path_base))
        # identity.save('./result/{}_identity.jpg'.format(image_path_base))


if __name__ == '__main__':
    test()