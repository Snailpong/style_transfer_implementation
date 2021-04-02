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
@click.option('--image_path', default='./data/summer2winter_yosemite/testA/2011-07-01 00_00_00.jpg')
@click.option('--fg', default='f')
def test(dataset_type, image_path, fg):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    os.makedirs('./result', exist_ok=True)

    checkpoint = torch.load('./model/cyclegan_' + dataset_type, map_location=device)
    generator = CycleGANGenerator()
    generator.load_state_dict(checkpoint[fg + '_state_dict'])
    generator.to(device)
    generator.eval()
    
    image = Image.open(image_path)
    image = ToTensor()(image)
    image = torch.unsqueeze(image, 0)

    output = generator(image)
    print(output.shape)
    output = output.detach().cpu()[0].permute(1, 2, 0)

    output = Image.fromarray(np.array(output, dtype=np.uint8), 'RGB')
    output.save('./result/{}_result.jpg'.format('.'.join(os.path.basename(image_path).split('.')[:-1])))


if __name__ == '__main__':
    test()