# SGDA-Net: Segformer backbone with a semantic-guided depth attention branch.


import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

from .core.attention import SemanticGuidedAttention
from .core.decoders import DepthFeatureExtractor, DepthHead

DEFAULT_BACKBONE = "nvidia/mit-b0"


class SGDANet(nn.Module):
    def __init__(self, pretrained_model: str = DEFAULT_BACKBONE, num_classes: int = 19):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        self.depth_feat = DepthFeatureExtractor(in_channels=num_classes, out_channels=64)
        self.cross_attention = SemanticGuidedAttention(num_classes=num_classes)
        self.depth_out = DepthHead(in_channels=64, mid_channels=32)

    def forward(self, pixel_values):
        seg_logits = self.segformer(pixel_values=pixel_values).logits
        depth_features = self.depth_feat(seg_logits)
        guided_features = self.cross_attention(seg_logits, depth_features)
        depth_logits = self.depth_out(guided_features)
        return seg_logits, depth_logits