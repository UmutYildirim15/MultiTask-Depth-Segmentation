import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes=19, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.detach().cpu().numpy().flatten().astype(np.int64)
        target = target.detach().cpu().numpy().flatten().astype(np.int64)

        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        pred[pred >= self.num_classes] = 0

        indices = self.num_classes * target + pred
        cm_flat = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_mat += cm_flat.reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.confusion_mat.astype(np.float64)
        diag = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - diag

        iou = np.where(union > 0, diag / union, np.nan)
        miou = float(np.nanmean(iou))
        pixel_acc = float(diag.sum() / cm.sum()) if cm.sum() > 0 else 0.0

        return {
            "mIoU": miou,
            "pixel_acc": pixel_acc,
            "per_class_iou": iou,
        }


class BinnedDepthMetrics:
    def __init__(self, min_depth=1e-3, max_depth=80.0):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bins = [(0, 15), (15, 45), (45, 80)]
        self.results = {f"{b[0]}-{b[1]}m": {"abs_rel_sum": 0.0, "rmse_sq_sum": 0.0, "count": 0} for b in self.bins}
        self.abs_rel_sum = self.rmse_sum = self.d1_sum = 0.0
        self.n_batches = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = torch.clamp(pred.detach().cpu(), min=self.min_depth)
        target = target.detach().cpu()

        valid = (target > self.min_depth) & (target <= self.max_depth)
        if valid.sum() == 0:
            return

        p, t = pred[valid], target[valid]
        thresh = torch.max(p / t, t / p)

        self.d1_sum += float((thresh < 1.25).float().mean())
        self.abs_rel_sum += float(((p - t).abs() / t).mean())
        self.rmse_sum += float(torch.sqrt(((p - t) ** 2).mean()))
        self.n_batches += 1

        for b in self.bins:
            b_min, b_max = b
            v_bin = (target > max(self.min_depth, b_min)) & (target <= min(self.max_depth, b_max))
            if v_bin.sum() > 0:
                pb, tb = pred[v_bin], target[v_bin]
                self.results[f"{b_min}-{b_max}m"]["abs_rel_sum"] += float(((pb - tb).abs() / tb).sum())
                self.results[f"{b_min}-{b_max}m"]["rmse_sq_sum"] += float(((pb - tb) ** 2).sum())
                self.results[f"{b_min}-{b_max}m"]["count"] += v_bin.sum().item()

    def compute(self):
        n = self.n_batches if self.n_batches > 0 else 1
        global_mets = {
            "AbsRel": self.abs_rel_sum / n,
            "RMSE": self.rmse_sum / n,
            "d1": self.d1_sum / n
        }

        binned_mets = {}
        for b in self.bins:
            key = f"{b[0]}-{b[1]}m"
            count = self.results[key]["count"]
            binned_mets[key] = {
                "AbsRel": self.results[key]["abs_rel_sum"] / count if count > 0 else 0.0,
                "RMSE": np.sqrt(self.results[key]["rmse_sq_sum"] / count) if count > 0 else 0.0
            }
        return global_mets, binned_mets