#ResNet-50 encoder with an ASPP segmentation head (DeepLabV3+) and a lightweight convolutional depth head.

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from ..core.aspp import ASPP

DEFAULT_ASPP_CHANNELS = 256


def _convert_to_dilated(layer: nn.Module, dilation: int = 2) -> None:
    # Turns a ResNet stage's stride-2 convs into stride-1 dilated convs in place, keeping the output stride at 16 instead of letting it drop to 32.
    for m in layer.modules():
        if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
            m.stride = (1, 1)
            if m.kernel_size in ((3, 3), 3):
                m.dilation = (dilation, dilation)
                m.padding = (dilation, dilation)


class ResNet50MultiTask(nn.Module):
    def __init__(self, num_classes: int = 19):
        super().__init__()
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # stride 1, 256 ch
        self.layer2 = backbone.layer2  # stride 2, 512 ch
        self.layer3 = backbone.layer3  # stride 2, 1024 ch
        self.layer4 = backbone.layer4  # stride 2 -> converted to dilated below

        _convert_to_dilated(self.layer4, dilation=2)

        self.aspp = ASPP(in_channels=2048, out_channels=DEFAULT_ASPP_CHANNELS)
        self.seg_head = nn.Sequential(
            nn.Conv2d(DEFAULT_ASPP_CHANNELS, DEFAULT_ASPP_CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(DEFAULT_ASPP_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(DEFAULT_ASPP_CHANNELS, num_classes, kernel_size=1),
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(DEFAULT_ASPP_CHANNELS, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)

        seg_logits = self.seg_head(x)
        depth_logits = self.depth_head(x)

        seg_logits = F.interpolate(seg_logits, size=input_shape, mode="bilinear", align_corners=False)
        depth_logits = F.interpolate(depth_logits, size=input_shape, mode="bilinear", align_corners=False)
        return seg_logits, depth_logits