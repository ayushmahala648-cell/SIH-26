"""
Deep Learning Multi-Class Semantic Segmentation Network for Aerial Cadastral Mapping.
Supports both PyTorch native U-Net architecture and SMP backbones.
Predicts 5 cadastral classes:
  0: Background / Bare Ground
  1: Parcel Boundary
  2: Building Footprint
  3: Road / Corridor
  4: Vegetation / Greenery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling then double conv with skip connection"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CadastralUNet(nn.Module):
    """
    Lightweight, high-accuracy U-Net tailored for drone aerial cadastral mapping.
    Outputs class logits for [Background, Boundary, Building, Road, Vegetation].
    """
    CLASSES = ["background", "boundary", "building", "road", "vegetation"]
    NUM_CLASSES = len(CLASSES)

    def __init__(self, in_channels: int = 3, num_classes: int = 5, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(256, 512 // factor)

        self.up1 = UpBlock(512, 256 // factor, bilinear)
        self.up2 = UpBlock(256, 128 // factor, bilinear)
        self.up3 = UpBlock(128, 64 // factor, bilinear)
        self.up4 = UpBlock(64, 32, bilinear)
        
        # Dual heads: Semantic Class Head and Dedicated Boundary Affinity Head
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feat = self.up4(x, x1)

        logits = self.outc(feat)
        boundary_logit = self.boundary_head(feat)
        return logits, boundary_logit


def get_segmentation_model(device: str = "cpu") -> CadastralUNet:
    """Factory to instantiate and return model moved to target device."""
    model = CadastralUNet(in_channels=3, num_classes=5)
    model.to(device)
    model.eval()
    return model

