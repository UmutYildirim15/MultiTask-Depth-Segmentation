import numpy as np
import cv2
import matplotlib.pyplot as plt

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]
], dtype=np.uint8)


def colorize_segmentation_mask(mask_array: np.ndarray) -> np.ndarray:
    mask_array = np.clip(mask_array, 0, len(CITYSCAPES_PALETTE) - 1)
    return CITYSCAPES_PALETTE[mask_array]


def colorize_depth_map(depth_array: np.ndarray, max_depth: float = 80.0) -> np.ndarray:
    depth_norm = np.clip(depth_array / max_depth, 0, 1)
    depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)


def plot_depth_row_profile(standard_depth_arr: np.ndarray, sgda_depth_arr: np.ndarray, row_index: int):
    profile_std = standard_depth_arr[row_index, :]
    profile_sgda = sgda_depth_arr[row_index, :]
    x = np.arange(len(profile_std))

    plt.figure(figsize=(10, 5))
    plt.plot(x, profile_std, color='red', label='Standard Model', linewidth=2)
    plt.plot(x, profile_sgda, color='green', label='SGDA-Net', linewidth=2)

    plt.title(f'Depth Boundary Profile (Row: {row_index})', fontweight='bold')
    plt.xlabel('Horizontal Pixel Index')
    plt.ylabel('Depth Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()