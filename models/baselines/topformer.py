# TopFormer Proxy -- MobileNetV2 backbone with a small transformer context module.
# Not the original TopFormer architecture from the paper; this is a lightweight stand-in that mixes local MobileNetV2 features with a global transformer bottleneck, used as one of the thesis's high FPS baselines.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

CONTEXT_DIM = 128
CONTEXT_POOL_SIZE = 8
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
TRANSFORMER_FF_DIM = 256


class TopFormerProxy(nn.Module):
    def __init__(self, num_classes: int = 19):
        super().__init__()
        mobilenet = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT).features

        self.stem = mobilenet[0:4]     # 1/4,  24 ch
        self.stage1 = mobilenet[4:7]   # 1/8,  32 ch
        self.stage2 = mobilenet[7:14]  # 1/16, 96 ch

        self.global_pool = nn.AdaptiveAvgPool2d((CONTEXT_POOL_SIZE, CONTEXT_POOL_SIZE))
        self.proj = nn.Conv2d(96, CONTEXT_DIM, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CONTEXT_DIM, nhead=TRANSFORMER_HEADS, dim_feedforward=TRANSFORMER_FF_DIM,
            dropout=0.1, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_LAYERS)

        fused_channels = 96 + CONTEXT_DIM
        self.seg_head = nn.Sequential(
            nn.Conv2d(fused_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(fused_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        input_shape = x.shape[-2:]

        x4 = self.stem(x)
        x8 = self.stage1(x4)
        x16 = self.stage2(x8)  # (B, 96, H/16, W/16)

        # global context -> pool down to a small grid, run it through a transformer
        glob = self.proj(self.global_pool(x16))  # (B, 128, 8, 8)
        b, c, h, w = glob.shape
        glob_attn = self.transformer(glob.view(b, c, -1).permute(0, 2, 1))
        glob_attn = glob_attn.permute(0, 2, 1).view(b, c, h, w)

        glob_up = F.interpolate(glob_attn, size=x16.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([x16, glob_up], dim=1)  # (B, 224, H/16, W/16)

        seg_logits = F.interpolate(self.seg_head(fused), size=input_shape, mode="bilinear", align_corners=False)
        depth_logits = F.interpolate(self.depth_head(fused), size=input_shape, mode="bilinear", align_corners=False)

        return seg_logits, depth_logits