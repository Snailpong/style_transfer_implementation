import torch
from torch import nn

class ConvLayer(nn.Module):
    def __init__(self, channel_input, channel_output, kernal, stride, padding):
        super(ConvLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel_input, channel_output, kernal, stride, padding),
            nn.InstanceNorm2d(channel_output),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layers(x)


class ConvLayer2(nn.Module):
    def __init__(self, channel_input, channel_output, kernal, stride, padding):
        super(ConvLayer2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel_input, channel_output, kernal, stride, padding),
            nn.InstanceNorm2d(channel_output),
        )

    def forward(self, x):
        return self.layers(x)


class DeconvLayer(nn.Module):
    def __init__(self, channel_input, channel_output, kernal, stride, padding):
        super(DeconvLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channel_input, channel_output, kernal, stride, padding, output_padding=1),
            nn.InstanceNorm2d(channel_output),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            ConvLayer(3, 64, 7, 1, 3),
            ConvLayer(64, 128, 3, 2, 1),
            ConvLayer(128, 256, 3, 2, 1)
        )

    def forward(self, x):
        return self.layers(x)


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.layers = nn.Sequential(*[ConvLayer(256, 256, 3, 1, 1) for i in range(6)])

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            DeconvLayer(256, 128, 3, 2, 1),
            DeconvLayer(128, 64, 3, 2, 1),
            ConvLayer(64, 3, 7, 1, 3)
        )

    def forward(self, x):
        return self.layers(x)


class CycleGANGenerator(nn.Module):
    def __init__(self):
        super(CycleGANGenerator, self).__init__()
        self.layers = nn.Sequential(
            Encoder(),
            Transformer(),
            Decoder(),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class CycleGANDiscriminator(torch.nn.Module):
    def __init__(self):
        super(CycleGANDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            ConvLayer2(3, 64, 4, 2, 1),
            ConvLayer2(64, 128, 4, 2, 1),
            ConvLayer2(128, 256, 4, 2, 1),
            ConvLayer2(256, 512, 4, 1, 1),
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)