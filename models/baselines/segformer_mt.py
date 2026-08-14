# SegFormer (mit-b1) with a plain convolutional depth head.


import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

DEFAULT_BACKBONE = "nvidia/mit-b1"
ENCODER_OUT_CHANNELS = 512  # last-stage embedding size for mit-b1 and larger


class SegFormerMultiTask(nn.Module):
    def __init__(self, pretrained: str = DEFAULT_BACKBONE, num_classes: int = 19):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(ENCODER_OUT_CHANNELS, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, pixel_values):
        # output_hidden_states=True is required to access the encoder's feature maps
        outputs = self.segformer(pixel_values=pixel_values, output_hidden_states=True)
        seg_logits = outputs.logits
        encoder_features = outputs.hidden_states[-1]
        depth_logits = self.depth_head(encoder_features)
        return seg_logits, depth_logits