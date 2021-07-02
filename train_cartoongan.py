import click
import os
import torch
import random

from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np

from utils import init_device_seed
from datasets import CartoonGANDataset
from model_cartoongan import CartoonGANGenerator, CartoonGANDiscriminator
from losses import VGGPerceptualLoss


BATCH_SIZE = 16

@click.command()
@click.option('--load_model', type=click.BOOL, default=False)
def train(load_model):
    device = init_device_seed(1234)

    transform = transforms.Compose([
        transforms.RandomCrop((768, 768), pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        ])

    dataset = CartoonGANDataset('./data/cartoon_dataset', ['photo', 'cartoon'], transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    os.makedirs('./model', exist_ok=True)

    generator = CartoonGANGenerator().to(device)
    discriminator = CartoonGANGenerator().to(device)

    epoch = 0

    if load_model:
        checkpoint = torch.load('./model/cartoongan', map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        epoch = checkpoint['epoch']

    optimizer_gen = optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))
    
    criterion_gen = VGGPerceptualLoss()
    criterion_disc = nn.CrossEntropyLoss()

    while epoch <= 500:
        epoch += 1

        generator.train()
        discriminator.train()

        pbar = tqdm(range(len(dataloader)))
        pbar.set_description('Epoch {}'.format(epoch))
        total_loss_gen = .0
        total_loss_disc = .0

        # for idx, (img_photo, [img_cartoon, img_cartoon_blur]) in enumerate(dataloader):
        for idx, (img_photo, img_cartoon) in enumerate(dataloader):
            print(img_photo.shape)
            print(img_cartoon.shape)
            img_photo = img_photo.to(device, dtype=torch.float32)
            img_cartoon = img_cartoon.to(device, dtype=torch.float32)
            img_cartoon_blur = img_cartoon_blur.to(device, dtype=torch.float32)

            gen_photo = CartoonGANGenerator(img_photo)
            loss_con = criterion_gen(img_photo, gen_photo) * 10

            if epoch <= 100:
                optimizer_gen.zero_grad()
                loss_con.backward()
                optimizer_gen.step()
                total_loss_gen += loss_con.detach().cpu().numpy()
                pbar.set_postfix_str('gen_loss: ' + str(np.around(total_loss_gen / (idx + 1), 4)))
                continue
            
            label_gen = CartoonGANDiscriminator(gen_photo)
            label_cartoon = CartoonGANDiscriminator(img_cartoon)
            label_cartoon_blur = CartoonGANDiscriminator(img_cartoon_blur)
            
            loss_gen = criterion_disc(label_gen, torch.ones_like(label_gen)) + loss_con
            loss_disc = criterion_disc(label_gen, torch.zeros_like(label_gen)) + criterion_disc(label_cartoon, torch.ones_like(label_cartoon)) + criterion_disc(label_cartoon_blur, torch.zeros_like(label_cartoon_blur))

            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            total_loss_gen += loss_gen.detach().cpu().numpy()
            total_loss_disc += loss_disc.detach().cpu().numpy()
            pbar.set_postfix_str('gen_loss: {}, disc_loss: {}'.format(np.around(total_loss_gen / (idx + 1), 4), np.around(total_loss_disc / (idx + 1), 4)))
            pbar.update()

        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'epoch': epoch,
        }, './model/cartoongan')


if __name__ == '__main__':
    train()
