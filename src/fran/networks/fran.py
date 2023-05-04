from functools import partial

import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from .blurpool import BlurPool

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxblurpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1),
            BlurPool(in_channels, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:

            self.up = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                BlurPool(in_channels // 2, stride=2),
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d( in_channels, in_channels // 2, kernel_size=4, stride=2),
                BlurPool(in_channels // 2, stride=2),

            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class Fran(nn.Module):
    def __init__(self,
                n_channels,
                dim=64,
                dim_mults=(1, 2, 4, 8), 
                bilinear=False):
        
        super(Fran, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        dims = list(map(lambda x: dim * x, dim_mults))
        factor = 2 if bilinear else 1

        self.inc = (DoubleConv(n_channels, dim))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for idx, in_dim in enumerate(dims):
            out_dim = in_dim*2
            if idx == len(dims)-1:
                out_dim = (in_dim * 2) // factor
            self.downs.append(Down(in_dim, out_dim))

        for idx, in_dim in enumerate(reversed(dims)):
            in_dim *= 2
            out_dim = in_dim // 2
            if idx == len(dims)-1:
                self.ups.append(Up(in_dim, out_dim, bilinear))
            else:
                self.ups.append(Up(in_dim, out_dim//factor, bilinear))
            

        self.outc = (OutConv(64, 3))

    def forward(self, x):

        down_hiddens = []
        print("input : ", x.shape)
        x = self.inc(x)
        print("conv1 : ", x.shape)

        down_hiddens.append(x)
        for down_block in self.downs:
            x = down_block(x)
            print("down : ", x.shape)

            down_hiddens.append(x)
        last = down_hiddens.pop()
        print("last : ", last.shape)
        for up_block in self.ups:
            skip_feature = down_hiddens.pop()
            x = up_block(x, skip_feature)
            print("up : ", x.shape, skip_feature.shape)

        image = self.outc(x)

        return image