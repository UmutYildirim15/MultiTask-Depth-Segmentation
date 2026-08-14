# Loss functions shared across the segmentation + depth multi-task models."""

import torch
import torch.nn as nn


class SILogLoss(nn.Module):

    def __init__(self, variance_focus: float = 0.85):
        super().__init__()
        self.variance_focus = variance_focus

    def forward(self, pred, target):
        valid_mask = target > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = torch.clamp(pred[valid_mask], min=1e-3)
        target_valid = torch.clamp(target[valid_mask], min=1e-3)
        diff = torch.log(pred_valid) - torch.log(target_valid)

        return 10.0 * torch.sqrt(
            torch.mean(diff ** 2) - self.variance_focus * (torch.mean(diff) ** 2)
        )


class EnhancedDepthLoss(nn.Module):

    def __init__(self, variance_focus: float = 0.85, gradient_weight: float = 0.5):
        super().__init__()
        self.variance_focus = variance_focus
        self.gradient_weight = gradient_weight

    def forward(self, pred, target):
        valid_mask = target > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = torch.clamp(pred[valid_mask], min=1e-3)
        target_valid = torch.clamp(target[valid_mask], min=1e-3)
        diff = torch.log(pred_valid) - torch.log(target_valid)
        silog = 10.0 * torch.sqrt(
            torch.mean(diff ** 2) - self.variance_focus * (torch.mean(diff) ** 2)
        )

        error = pred - target
        dx = error[:, :, :, 1:] - error[:, :, :, :-1]
        dy = error[:, :, 1:, :] - error[:, :, :-1, :]
        mask_x = valid_mask[:, :, :, 1:] & valid_mask[:, :, :, :-1]
        mask_y = valid_mask[:, :, 1:, :] & valid_mask[:, :, :-1, :]

        grad_x = torch.mean(torch.abs(dx)[mask_x]) if mask_x.sum() > 0 else 0.0
        grad_y = torch.mean(torch.abs(dy)[mask_y]) if mask_y.sum() > 0 else 0.0

        return silog + self.gradient_weight * (grad_x + grad_y)


class UncertaintyLoss(nn.Module):

    def __init__(self, ignore_index: int = 255, depth_loss_fn: nn.Module = None):
        super().__init__()
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_depth = nn.Parameter(torch.zeros(1))
        self.seg_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.depth_loss_fn = depth_loss_fn if depth_loss_fn is not None else SILogLoss()

    def forward(self, seg_pred, seg_gt, depth_pred, depth_gt):
        loss_seg = self.seg_loss_fn(seg_pred, seg_gt)
        loss_depth = self.depth_loss_fn(depth_pred, depth_gt)

        precision_seg = torch.exp(-self.log_var_seg)
        precision_depth = torch.exp(-self.log_var_depth)

        return (
            precision_seg * loss_seg + self.log_var_seg +
            precision_depth * loss_depth + self.log_var_depth
        )