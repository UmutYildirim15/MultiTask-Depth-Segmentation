import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import get_transforms

# Maps raw SYNTHIA class ids (as stored in the label PNGs) to the contiguous train ids expected by the model. Anything not listed here is left as ignore.
SYNTHIA_TO_TRAINID = {
    0: 255, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
    6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11,
}

IGNORE_INDEX = 255
DEPTH_SCALE = 5000.0  # matches the SYNTHIA depth encoding convention
MAX_DEPTH = 80.0  # meters, anything beyond this is treated as invalid

# Human-readable names for the 12 active train ids. The model is trained with 19 output channels for compatibility with the Cityscapes-style label space, but only ids 0-11 used here.
CLASS_NAMES = [
    "sky", "building", "road", "sidewalk", "fence", "vegetation",
    "pole", "car", "sign", "pedestrian", "cyclist", "misc",
    "unused_12", "unused_13", "unused_14", "unused_15",
    "unused_16", "unused_17", "unused_18",
]


class SynthiaDataset(Dataset):

#Expected directory layout under "root_dir":
#    RGB/            input color images
#    GT/LABELS/      class id label PNGs
#    Depth/Depth/    depth-encoded PNGs

    def __init__(self, root_dir: str, is_train: bool = True, image_size: int = 512):
        self.rgb_dir = os.path.join(root_dir, "RGB")
        self.mask_dir = os.path.join(root_dir, "GT/LABELS")
        self.depth_dir = os.path.join(root_dir, "Depth/Depth")
        self.images = sorted(
            f for f in os.listdir(self.rgb_dir) if f.endswith(".png") and not f.startswith(".")
        )
        self.transform = get_transforms(is_train, image_size)

    def __len__(self) -> int:
        return len(self.images)

    def _load_mask(self, img_name: str, fallback_shape: tuple) -> np.ndarray:
        # some SYNTHIA exports use a "_labelTrainIds" suffix, others keep the
        # raw label under the same filename as the RGB image
        candidates = [
            os.path.join(self.mask_dir, img_name.replace(".png", "_labelTrainIds.png")),
            os.path.join(self.mask_dir, img_name),
        ]

        for path in candidates:
            if not os.path.exists(path):
                continue
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            # class id is encoded in the R channel (index 2, since cv2 reads BGR)
            channel = raw[:, :, 2] if raw.ndim == 3 else raw
            class_mask = (channel & 0x00FF).astype(np.uint8)
            remapped = np.full_like(class_mask, IGNORE_INDEX)
            for synthia_id, train_id in SYNTHIA_TO_TRAINID.items():
                remapped[class_mask == synthia_id] = train_id
            return remapped

        # no label file found for this frame, fall back to an all-ignore mask sized to match the original RGB frame
        return np.full(fallback_shape, IGNORE_INDEX, dtype=np.uint8)

    def _load_depth(self, img_name: str) -> np.ndarray:
        depth_bgr = cv2.imread(os.path.join(self.depth_dir, img_name))
        depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)

        r = depth_rgb[:, :, 0].astype(np.float32)
        g = depth_rgb[:, :, 1].astype(np.float32)
        b = depth_rgb[:, :, 2].astype(np.float32)

        depth = DEPTH_SCALE * (r + g * 256.0 + b * 256.0 * 256.0) / ((256.0 ** 3) - 1.0)
        depth[depth > MAX_DEPTH] = 0.0
        return depth

    def __getitem__(self, idx: int):
        img_name = self.images[idx]

        img_bgr = cv2.imread(os.path.join(self.rgb_dir, img_name))
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = self._load_mask(img_name, fallback_shape=image.shape[:2])
        depth = self._load_depth(img_name)

        transformed = self.transform(image=image, masks=[mask, depth])
        seg_mask = transformed["masks"][0].to(torch.long)
        depth_map = transformed["masks"][1].unsqueeze(0).to(torch.float32)

        return transformed["image"], seg_mask, depth_map


def get_dataloader(
    root_dir: str,
    batch_size: int = 8,
    is_train: bool = True,
    image_size: int = 512,
    num_workers: int = 2,
) -> DataLoader:
    dataset = SynthiaDataset(root_dir, is_train=is_train, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
    )