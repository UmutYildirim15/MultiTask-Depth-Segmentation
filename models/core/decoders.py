# Depth decoder building blocks used by the multi-task models.

import torch.nn as nn

class DepthFeatureExtractor(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthHead(nn.Sequential):

    def __init__(self, in_channels: int = 64, mid_channels: int = 32):
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )