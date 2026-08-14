#MobileViT backbone with lightweight segmentation and depth heads.
import torch.nn as nn
from transformers import MobileViTModel


class MobileViTMultiTask(nn.Module):
    def __init__(self, num_classes: int = 19, pretrained_model: str = "apple/mobilevit-small"):
        super().__init__()
        self.backbone = MobileViTModel.from_pretrained(pretrained_model)

        in_channels = 640  # MobileViT-Small's last_hidden_state channel count

        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(self, x):
        features = self.backbone(pixel_values=x).last_hidden_state
        seg_logits = self.seg_head(features)
        depth_logits = self.depth_head(features)
        return seg_logits, depth_logits