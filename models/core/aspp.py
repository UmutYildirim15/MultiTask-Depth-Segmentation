# ASPP (Atrous Spatial Pyramid Pooling), DeepLabV3+ style.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        size = x.shape[-2:]
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        gap_feat = F.interpolate(self.gap(x), size=size, mode="bilinear", align_corners=False)
        return self.proj(torch.cat([feat0, feat1, feat2, feat3, gap_feat], dim=1))