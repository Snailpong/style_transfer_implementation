import torch
from torch import nn


class Encoder1(torch.nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class Encoder2(torch.nn.Module):
    def __init__(self, channel_input):
        super(Encoder2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel_input, channel_input * 2, 3, 2, 1),
            nn.Conv2d(channel_input * 2, channel_input * 2, 3, 1, 1),
            nn.InstanceNorm2d(channel_input * 2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256)
        )

    def forward(self, x):
        return x + self.layers(x)


class Decoder(torch.nn.Module):
    def __init__(self, channel_input):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channel_input, channel_input // 2, 3, 2, 1),
            nn.Conv2d(channel_input // 2, channel_input // 2, 3, 1, 1),
            nn.InstanceNorm2d(channel_input / 2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class DiscriminatorLayer(torch.nn.Module):
    def __init__(self, channel_input, channel_middle):
        super(DiscriminatorLayer, self).__init__()
        nn.Conv2d(3, channel_input, channel_middle, 2, 1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(3, channel_middle, channel_middle * 2, 1, 1),
        nn.InstanceNorm2d(channel_middle * 2),
        nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.layers(x)


class CartoonGANGenerator(nn.Module):
    def __init__(self):
        super(CartoonGANGenerator, self).__init__()
        self.layers = nn.Sequential(
            Encoder1(),
            Encoder2(64),
            Encoder2(128),
            *[ResidualBlock() for i in range(8)],
            Decoder(256),
            Decoder(128),
            nn.Conv2d(64, 3, 7, 1, 3)
        )

    def forward(self, x):
        return self.layers(x)


class CartoonGANDiscriminator(torch.nn.Module):
    def __init__(self):
        super(CartoonGANDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            DiscriminatorLayer(32, 64),
            DiscriminatorLayer(128, 128),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.layers(x)