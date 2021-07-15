import torch
from torch import nn
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, channel_input, channel_output):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel_input, channel_output, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channel_output),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class DSConv(nn.Module):
    def __init__(self, channel_input, channel_output, stride):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel_input, channel_input, 3, stride, 1, groups=channel_input, bias=False),
            nn.Conv2d(channel_input, channel_output, 1, bias=False),
            nn.InstanceNorm2d(channel_output),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.layers(x)
        

class InvertedResidualBlock(nn.Module):
    def __init__(self, channels):
        super(InvertedResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(256, 512),
            DSConv(512, 512, 1), 
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(256)
        )

    def forward(self, x):
        return x + self.layers(x)


class DownConv(nn.Module):
    def __init__(self, channel_input, channel_output):
        super(DownConv, self).__init__()
        self.dsconv = DSConv(channel_input, channel_output, 2)
        self.resize = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            DSConv(channel_input, channel_output, 1)
        )

    def forward(self, x):
        return self.dsconv(x) + self.resize(x)


class UpConv(nn.Module):
    def __init__(self, channel_input, channel_output):
        super(UpConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            DSConv(channel_input, channel_output, 1)
        )

    def forward(self, x):
        return self.layers(x)


class DiscriminatorLayer(nn.Module):
    def __init__(self, channels):
        super(DiscriminatorLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class AnimeGANGenerator(nn.Module):
    def __init__(self):
        super(AnimeGANGenerator, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            DownConv(64, 128),
            ConvBlock(128, 128),
            DSConv(128, 128, 1),
            DownConv(128, 256),
            ConvBlock(256, 256),
            *[InvertedResidualBlock() for i in range(8)],
            ConvBlock(256, 256),
            UpConv(256, 128),
            DSConv(128, 128, 1),
            ConvBlock(128),
            UpConv(128, 64),
            ConvBlock(64),
            ConvBlock(64),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class AnimeGANDiscriminator(nn.Module):
    def __init__(self):
        super(AnimeGANDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DiscriminatorLayer(32),
            DiscriminatorLayer(64),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        return self.layers(x)


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg19_bn(pretrained=True)
        self.feature_extractor = vgg.features[:37]

        for child in self.feature_extractor.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, input):
        return self.feature_extractor(input)