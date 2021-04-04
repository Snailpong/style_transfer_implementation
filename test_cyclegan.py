import click
import os
import torch
import random
from PIL import Image

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.autograd import Variable

import numpy as np

from dataset import CycleGANDataset
from cyclegan import CycleGANGenerator, CycleGANDiscriminator


@click.command()
@click.option('--dataset_type', default='summer2winter_yosemite')
@click.option('--image_path', default='./data/summer2winter_yosemite/testA/2011-09-13 23_03_31.jpg')
@click.option('--model_type', default='x2y')
def test(dataset_type, image_path, model_type):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_path_base = '.'.join(os.path.basename(image_path).split('.')[:-1])
    print('Device: {}'.format(device))

    os.makedirs('./result', exist_ok=True)

    checkpoint = torch.load('./model/cyclegan_' + dataset_type, map_location=device)
    generator = CycleGANGenerator()
    generator.load_state_dict(checkpoint[model_type + '_state_dict'])
    generator.to(device)
    generator.eval()
    
    image = Image.open(image_path)
    image = ToTensor()(image)
    image = torch.unsqueeze(image, 0)

    output = generator(image)

    generator_inv = CycleGANGenerator()
    generator_inv.load_state_dict(checkpoint[model_type[::-1] + '_state_dict'])
    generator_inv.to(device)
    generator_inv.eval()

    cycle = generator_inv(output).detach().cpu()[0].permute(1, 2, 0) * 255.
    identity = generator_inv(image).detach().cpu()[0].permute(1, 2, 0) * 255.
    output = output.detach().cpu()[0].permute(1, 2, 0) * 255.

    Image.fromarray(np.array(image[0].permute(1, 2, 0) * 255., dtype=np.uint8), 'RGB').save('./result/{}_input.jpg'.format(image_path_base))
    Image.fromarray(np.array(output, dtype=np.uint8), 'RGB').save('./result/{}_result.jpg'.format(image_path_base))
    Image.fromarray(np.array(cycle, dtype=np.uint8), 'RGB').save('./result/{}_cycle.jpg'.format(image_path_base))
    Image.fromarray(np.array(identity, dtype=np.uint8), 'RGB').save('./result/{}_identity.jpg'.format(image_path_base))


if __name__ == '__main__':
    test()