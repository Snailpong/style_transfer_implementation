import click
import os
import torch
import time
import random

from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np

from utils import init_device_seed
from datasets import AnimeGANDataset
from model_animegan import AnimeGANGenerator, VGG19
from model_cartoongan import CartoonGANDiscriminator


BATCH_SIZE = 4
W_ADV = 300
W_CON = 1.5
W_GRA = 3
W_COL = 10

@click.command()
@click.option('--load_model', type=click.BOOL, default=False)
def train(load_model):
    device = init_device_seed(1234)

    dataset = AnimeGANDataset('./data/cartoon_dataset', ['photo', 'cartoon', 'cartoon_smoothed'])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    os.makedirs('./model', exist_ok=True)

    generator = AnimeGANGenerator().to(device)
    discriminator = CartoonGANDiscriminator().to(device)
    feature_extractor = VGG19().to(device)

    epoch = 0

    if load_model:
        checkpoint = torch.load('./model/animegan', map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        epoch = checkpoint['epoch']

    optimizer_init = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_gen = optim.Adam(generator.parameters(), lr=8e-5, betas=(0.5, 0.999))
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=16e-5, betas=(0.5, 0.999))
    
    criterion_mae = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    criterion_huber = nn.SmoothL1Loss()

    while epoch <= 100:
        epoch += 1

        generator.train()
        discriminator.train()

        pbar = tqdm(range(len(dataloader)))
        pbar.set_description('Epoch {}'.format(epoch))
        total_loss_gen = .0
        total_loss_con = .0
        total_loss_disc = .0

        for idx, images in enumerate(dataloader):
            img_photo = images[0].to(device, dtype=torch.float32)
            img_cartoon = images[1].to(device, dtype=torch.float32)
            img_cartoon_blur = images[2].to(device, dtype=torch.float32)
            img_cartoon_gray = images[3].to(device, dtype=torch.float32)
            img_cartoon_blur_gray = images[4].to(device, dtype=torch.float32)

            # Initializaiton phase
            if epoch <= 3:
                optimizer_gen.zero_grad()

                gen_photo = generator(img_photo)
                x_features = feature_extractor((img_photo + 1) / 2).detach()
                Gx_features = feature_extractor((gen_photo + 1) / 2)

                loss_con = W_CON * criterion_mae(Gx_features, x_features)
                loss_con.backward()
                optimizer_gen.step()

                total_loss_con += loss_con.item()
                pbar.set_postfix_str('CLoss: ' + str(np.around(total_loss_con / (idx + 1), 4)))
                pbar.update()
                continue

            # Discriminator loss and update
            optimizer_disc.zero_grad()

            gen_photo = generator(img_photo).detach()
            label_gen = discriminator(gen_photo)
            label_cartoon = discriminator(img_cartoon)
            label_cartoon_blur = discriminator(img_cartoon_blur)
            
            loss_generated_disc = criterion_disc(label_gen, torch.zeros_like(label_gen))
            loss_cartoon_disc = criterion_disc(label_cartoon, torch.ones_like(label_cartoon))
            loss_blur_disc = criterion_disc(label_cartoon_blur, torch.zeros_like(label_cartoon_blur))
            loss_disc = loss_generated_disc + loss_cartoon_disc + loss_blur_disc

            loss_disc.backward()
            optimizer_disc.step()

            # Generator loss and update
            optimizer_gen.zero_grad()
            gen_photo = generator(img_photo)

            x_features = feature_extractor((img_photo + 1) / 2).detach()
            Gx_features = feature_extractor((gen_photo + 1) / 2)

            loss_con = criterion_gen(Gx_features, x_features) * 10
            label_gen = discriminator(gen_photo)
            loss_generated_gen = criterion_disc(label_gen, torch.ones_like(label_gen))
            loss_gen = loss_generated_gen + loss_con

            loss_gen.backward()
            optimizer_gen.step()
            optimizer_gen.zero_grad()

            # Loss display
            total_loss_gen += loss_generated_gen.item()
            total_loss_con += loss_con.item()
            total_loss_disc += loss_disc.item()
            pbar.set_postfix_str('G_GAN: {}, G_Content: {}, D: {}'.format(
                np.around(total_loss_gen / (idx + 1), 4),
                np.around(total_loss_con / (idx + 1), 4),
                np.around(total_loss_disc / (idx + 1), 4)))
            pbar.update()

        # Save checkpoint per epoch
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'epoch': epoch,
        }, './model/cartoongan')

if __name__ == '__main__':
    train()