import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

SYNTHIA_TO_TRAINID = {
    0: 255, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
    6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11,
}


class SynthiaVideoDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.rgb_dir = os.path.join(root_dir, 'RGB')
        self.mask_dir = os.path.join(root_dir, 'GT/LABELS')
        self.depth_dir = os.path.join(root_dir, 'Depth/Depth')

        self.images = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])

        transform_ops = [
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]

        if is_train:
            transform_ops.insert(1, A.HorizontalFlip(p=0.5))

        self.transform = A.Compose(
            transform_ops,
            additional_targets={'image2': 'image', 'mask2': 'mask', 'depth2': 'mask'}
        )

    def __len__(self):
        return len(self.images) - 1

    def _load_mask(self, img_name):
        mask_path = os.path.join(self.mask_dir, img_name.replace('.png', '_labelTrainIds.png'))
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, img_name)

        raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            return np.zeros((512, 512), dtype=np.int64)

        class_mask = (raw[:, :, 2] & 0x00FF).astype(np.uint8) if raw.ndim == 3 else (raw & 0x00FF).astype(np.uint8)
        remapped = np.full_like(class_mask, 255, dtype=np.int64)

        for synthia_id, train_id in SYNTHIA_TO_TRAINID.items():
            remapped[class_mask == synthia_id] = train_id

        return remapped

    def _load_depth(self, img_name):
        depth_bgr = cv2.imread(os.path.join(self.depth_dir, img_name))
        if depth_bgr is None:
            return np.zeros((512, 512), dtype=np.float32)

        depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        R, G, B = depth_rgb[:, :, 0], depth_rgb[:, :, 1], depth_rgb[:, :, 2]
        depth = 5000.0 * (R + G * 256.0 + B * 256.0 * 256.0) / ((256.0 ** 3) - 1.0)
        depth[depth > 80.0] = 0.0
        return depth

    def __getitem__(self, idx):
        img1_name, img2_name = self.images[idx], self.images[idx + 1]

        img1 = cv2.cvtColor(cv2.imread(os.path.join(self.rgb_dir, img1_name)), cv2.COLOR_BGR2RGB)
        mask1, depth1 = self._load_mask(img1_name), self._load_depth(img1_name)

        img2 = cv2.cvtColor(cv2.imread(os.path.join(self.rgb_dir, img2_name)), cv2.COLOR_BGR2RGB)
        mask2, depth2 = self._load_mask(img2_name), self._load_depth(img2_name)

        transformed = self.transform(
            image=img1, masks=[mask1, depth1],
            image2=img2, mask2=mask2, depth2=depth2
        )

        return {
            'img1': transformed['image'],
            'mask1': transformed['masks'][0].to(torch.long),
            'depth1': transformed['masks'][1].unsqueeze(0).to(torch.float32),
            'img2': transformed['image2'],
            'mask2': transformed['mask2'].to(torch.long),
            'depth2': transformed['depth2'].unsqueeze(0).to(torch.float32)
        }


def get_video_dataloader(root_dir, batch_size=4, is_train=True):
    dataset = SynthiaVideoDataset(root_dir, is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=2, pin_memory=True)