import click
import os
import torch
import random

from torch import nn, optim
from torch.utils.data import DataLoader

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
    torch.autograd.set_detect_anomaly(True)
    print('Device: {}'.format(device))

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    dataset = CycleGANDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    os.makedirs('./model', exist_ok=True)

    f = CycleGANGenerator().to(device)
    g = CycleGANGenerator().to(device)
    dx = CycleGANDiscriminator().to(device)
    dy = CycleGANDiscriminator().to(device)

    fg_optimizer = optim.Adam(list(f.parameters()) + list(g.parameters()), lr=0.0002)
    dx_optimizer = optim.Adam(dx.parameters(), lr=0.0002)
    dy_optimizer = optim.Adam(dy.parameters(), lr=0.0002)
    
    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()

    for epoch in range(3000):
        f.train()
        g.train()
        dx.train()
        dy.train()

        pbar = tqdm(range(len(dataloader)))
        pbar.set_description('Epoch {}'.format(epoch+1))
        total_loss = .0

        for idx, (x_images, y_images) in enumerate(dataloader):
            real_x = x_images.to(device, dtype=torch.float32)
            real_y = y_images.to(device, dtype=torch.float32)

            fake_y = g(x_images)
            fake_x = f(y_images)
            cycle_x = f(fake_y)
            cycle_y = g(fake_x)
            identity_x = f(x_images)
            identity_y = g(y_images)

            disc_fake_x = dx(fake_x)
            disc_fake_y = dy(fake_y)

            loss_cyc = mae_criterion(cycle_x, real_x) + mae_criterion(cycle_y, real_y)
            loss_identity = mae_criterion(identity_x, real_x) + mae_criterion(identity_y, real_y)
            loss_gan_g = mse_criterion(disc_fake_x, torch.ones_like(disc_fake_x))
            loss_gan_f = mse_criterion(disc_fake_y, torch.ones_like(disc_fake_y))
            fg_loss = loss_cyc + loss_gan_f + loss_gan_g

            fg_optimizer.zero_grad()
            fg_loss.backward()
            fg_optimizer.step()

            disc_real_x = dx(real_x)
            disc_fake_x = dx(fake_x.detach())
            
            dx_loss = mse_criterion(disc_real_x, torch.ones_like(disc_real_x)) + mse_criterion(disc_fake_x, torch.zeros_like(disc_fake_x))

            dx_optimizer.zero_grad()
            dx_loss.backward()
            dx_optimizer.step()

            disc_real_y = dy(real_y)
            disc_fake_y = dy(fake_y.detach())

            dy_loss = mse_criterion(disc_real_y, torch.ones_like(disc_real_y)) + mse_criterion(disc_fake_y, torch.zeros_like(disc_fake_y))

            dy_optimizer.zero_grad()
            dy_loss.backward()
            dy_optimizer.step()

            total_loss += fg_loss.detach().cpu().numpy() + dx_loss.detach().cpu().numpy() + dy_loss.detach().cpu().numpy()
            pbar.set_postfix_str('loss: ' + str(np.around(total_loss / (idx + 1), 4)))
            pbar.update()

        torch.save(model.state_dict(), os.path.join(os.getcwd(), 'model', '{}.pth'.format(model.__class__.__name__)))


if __name__ == '__main__':
    train()

    


