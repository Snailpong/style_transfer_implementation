import click
import os
import torch
import random

from torch import nn, optim
from torch.uitls.data import DataLoader

from tqdm import tqdm
import numpy as np

from dataset import CycleGANDataset
from cyclegan import CycleGANGenerator, CycleGANDiscriminator


BATCH_SIZE = 1

@click.command()
@click.option('--dataset_dir', default='./data/summer2winter_yosemite')
def train(dataset_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    dataset = CycleGANDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    os.makedirs('./model', exist_ok=True)

    f = CycleGANGenerator().to(device)
    dx = CycleGANDiscriminator().to(device)
    g = CycleGANGenerator().to(device)
    dy = CycleGANDiscriminator().to(device)

    optimizer = optim.Adam(list(f) + list(g) + list(dx) + list(dy), lr=0.0002)
    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()

    for epoch in range(3000):
        f.train()
        g.train()
        dx.train()
        dy.train()

        pbar = tqdm(range(len(dataloader)))
        pbar.set_description('Epoch {}'.format(epoch+1))

        for idx, (x_images, y_images) in enumerate(dataloader):
            x_images = x_images.to(device, dtype=torch.float32)
            y_images = y_images.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            fgx = f(g(x_images))
            gfy = g(f(y_images))
            dfgx = dx(fgx)
            dgfy = dy(gfy)

            loss = ...

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            pbar.set_postfix_str('loss: ' + str(np.around(total_loss / (idx + 1), 4)))
            pbar.update()

        torch.save(model.state_dict(), os.path.join(os.getcwd(), 'model', '{}.pth'.format(model.__class__.__name__)))


if __name__ == '__main__':
    train()

    


