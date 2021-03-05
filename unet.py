"""
Just a UNET with some dense concatination...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        c = F.elu(self.norm1(self.conv1(x)))
        c = F.elu(self.norm2(self.conv2(c)))
        return c

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, factor=2):
        super(Down, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=factor, padding=1)

    def forward(self, x):
        c = F.elu(self.down(x))
        return c

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, factor=2):
         super(Up, self).__init__()
         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        c = F.elu(self.up(x))
        return c

class BottleNeck(nn.Module):
    def __init__(self, channels):
         super(BottleNeck, self).__init__()
         self.conv = Down(channels, channels*2)
         self.block = BasicBlock(channels*2, channels*2)
         self.deconv = Up(channels*2, channels)

    def forward(self, x):
        c = F.elu(self.conv(x))
        c = self.block(c)
        c = F.elu(self.deconv(c))
        return c

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, init_channels=32):
        super(UNet, self).__init__()
        features = init_channels

        self.enc1 = BasicBlock(in_channels, features)
        self.down1 = Down(features, features*2)

        self.enc2 = BasicBlock(features*2, features*2)
        self.down2 = Down(features*2, features*4)

        self.enc3 = BasicBlock(features*4, features*4)
        self.down3 = Down(features*4, features*8)

        self.enc4 = BasicBlock(features*8, features*8)
        self.down4 = Down(features*8, features*16)

        self.bottleneck = BasicBlock(features*16, features*16)

        self.up4 = Up(features*16, features*8)
        self.dec4 = BasicBlock(features*8*2, features*8)

        self.up3 = Up(features*8, features*4)
        self.dec3 = BasicBlock(features*4*2, features*4)

        self.up2 = Up(features*4, features*2)
        self.dec2 = BasicBlock(features*2*2, features*2)
        
        self.up1 = Up(features*2, features)
        self.dec1 = BasicBlock(features*2, features)

        self.conv = nn.Sequential(nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1),
                                  nn.ELU(),
                                  nn.Conv2d(features//2, out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.down1(c1))
        c3 = self.enc3(self.down2(c2))
        c4 = self.enc4(self.down3(c3))

        bneck = self.bottleneck(self.down4(c4))

        dc4 = self.up4(bneck)
        dc4 = torch.cat((dc4, c4), dim=1)
        dc4 = self.dec4(dc4)

        dc3 = self.up3(dc4)
        dc3 = torch.cat((dc3, c3), dim=1)
        dc3 = self.dec3(dc3)

        dc2 = self.up2(dc3)
        dc2 = torch.cat((dc2, c2), dim=1)
        dc2 = self.dec2(dc2)

        dc1 = self.up1(dc2)
        dc1 = torch.cat((dc1, c1), dim=1)
        dc1 = self.dec1(dc1)

        out = F.softmax(self.conv(dc1), dim=1) # do we want softmax??

        return out
