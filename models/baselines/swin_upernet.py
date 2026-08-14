#Swin Transformer + UperNet backbone with a lightweight depth head.

import torch.nn as nn
from transformers import UperNetForSemanticSegmentation


class SwinUperNetMultiTask(nn.Module):
    def __init__(self, num_classes: int = 19, pretrained_model: str = "openmmlab/upernet-swin-tiny"):
        super().__init__()
        self.backbone = UperNetForSemanticSegmentation.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        seg_logits = self.backbone(pixel_values=x).logits
        depth_logits = self.depth_decoder(seg_logits)
        return seg_logits, depth_logits

    def get_param_groups(self, base_lr: float):
        # Backbone gets a lower rate (0.6x) than the decode head and the depth head, which start from scratch and need to catch up faster.
        return [
            {"params": self.backbone.backbone.parameters(), "lr": base_lr * 0.6},
            {"params": self.backbone.decode_head.parameters(), "lr": base_lr},
            {"params": self.depth_decoder.parameters(), "lr": base_lr},
        ]