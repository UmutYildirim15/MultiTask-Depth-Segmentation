# Attention modules shared across the multi-task models.

import torch.nn as nn


class SemanticGuidedAttention(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(num_classes, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, semantic_logits, depth_features):
        attention_map = self.sigmoid(self.conv(semantic_logits))
        guided_depth = depth_features * attention_map
        return guided_depth + depth_features