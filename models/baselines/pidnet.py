#PIDNet-style baseline for single-task semantic segmentation.


import torch
import torch.nn as nn
import torch.nn.functional as F


class PIDNet(nn.Module):
    def __init__(self, num_classes: int = 19):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # P: proposal branch, keeps the stem resolution
        self.p_branch = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # I: integration branch, wider receptive field via stride + dilation
        self.i_branch = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # D: detail branch, produces a boundary-aware gating mask
        self.d_branch = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        stem_feat = self.stem(x)

        p_feat = self.p_branch(stem_feat)
        i_feat = self.i_branch(stem_feat)
        d_mask = self.d_branch(stem_feat)

        i_feat_up = F.interpolate(i_feat, size=p_feat.shape[2:], mode="bilinear", align_corners=False)
        i_feat_guided = i_feat_up * (1 - d_mask)

        fused = torch.cat([p_feat, i_feat_guided], dim=1)
        fused = self.fusion_conv(fused)

        return self.classifier(fused)